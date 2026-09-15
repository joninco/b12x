"""Four-rank DCP candidate publication with deterministic peer-load selection.

Each rank publishes float32 [row,K,2] score/global-id pairs to its own IPC
slab. The selector reads peers with ordinary global loads and writes an exact
top-k set into caller-owned int32 output. Only signal words are stored remotely.
Output order is repeatable for fixed input positions but is not score-sorted.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from cutlass import Uint32

from b12x._lib.utils import current_cuda_stream
from ._cuda_ipc import CudaRTLibrary
from ._dcp_cute_common import signal_bytes
from ._dcp_topk_pull_cute import (
    BLOCKS,
    _pointer,
    precompile_candidate_publication,
    precompile_peer_topk,
    select_peer_topk,
)
from .pcie_dcp_topk import (
    _IPCChannel,
    _release_failed_allocations,
    _tensor_from_cuda_pointer,
)
from .pcie_oneshot import (
    PCIeOneshotAllReduce,
    _align_up,
    _is_current_stream_capturing,
    _normalize_device,
)

PAYLOAD_OFFSET = _align_up(signal_bytes(BLOCKS), 256)


@torch.library.custom_op("b12x::dcp_peer_topk_merge", mutates_args=("out", "state"))
def _merge_op(
    packed: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    peer_slabs: list[int],
    rank: int,
    topk: int,
    max_rows: int,
) -> None:
    if (
        packed.ndim != 3
        or packed.shape[1:] != (topk, 2)
        or packed.dtype != torch.float32
        or not packed.is_contiguous()
        or not packed.is_cuda
        or out.device != packed.device
        or state.device != packed.device
        or out.shape != packed.shape[:2]
        or out.dtype != torch.int32
        or not out.is_contiguous()
        or state.dtype != torch.uint8
        or not state.is_contiguous()
        or len(peer_slabs) != 4
        or rank not in range(4)
        or state.numel() < PAYLOAD_OFFSET + packed.numel() * packed.element_size()
        or not 1 <= packed.shape[0] <= max_rows
        or topk not in (512, 1024, 2048)
        or state.numel() != PAYLOAD_OFFSET + max_rows * topk * 8
    ):
        raise ValueError("DCP candidate tensors or peer geometry are incompatible")
    if peer_slabs[rank] != state.data_ptr() or any(
        p <= 0 or p % 256 for p in peer_slabs
    ):
        raise ValueError("DCP candidate state must alias this rank's aligned IPC slab")
    if any(
        torch._C._overlaps(a, b)
        for a, b in ((packed, out), (packed, state), (out, state))
    ):
        raise ValueError(
            "DCP candidate input, output and channel state must not overlap"
        )
    rows, topk, _ = packed.shape
    with torch.cuda.device(packed.device):
        destination = _pointer(Uint32, state.data_ptr() + PAYLOAD_OFFSET)
        source = _pointer(Uint32, packed.data_ptr())
        precompile_candidate_publication(rank, topk, packed.device.index)(
            tuple(_pointer(Uint32, p) for p in peer_slabs),
            source,
            destination,
            rows,
            current_cuda_stream(),
        )
        select_peer_topk(tuple(p + PAYLOAD_OFFSET for p in peer_slabs), out, topk * 2)


@_merge_op.register_fake
def _merge_fake(
    packed: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    peer_slabs: list[int],
    rank: int,
    topk: int,
    max_rows: int,
) -> None:
    pass


class PCIeDCPTopKPull(_IPCChannel):
    """Own one channel for a serially replayed graph of four-rank top-k merges.

    Construct collectively on a CPU process group before graph capture. The
    capacity is fixed, while live rows 1 through capacity remain launch arguments.
    The capacity sizes this rank's published slab (max_rows x K score/id pairs);
    publication strides 16 CTAs over the live rows and selection runs one block
    per live row, so every positive capacity shares one compiled geometry.
    Keep the channel alive while its graph exists; close collectively only after
    all graph work has completed. Independent graphs require independent channels.
    No allocation, peer mapping, or compiler resolution occurs during replay.
    """

    def __init__(self, *, process_group, device, max_rows: int = 16, topk: int = 2048):
        device = _normalize_device(device)
        if dist.get_world_size(process_group) != 4:
            raise ValueError("DCP candidate pull requires a four-rank process group")
        if max_rows < 1 or topk not in (512, 1024, 2048):
            raise ValueError(
                "DCP candidate pull requires a positive row capacity and "
                "K=512/1024/2048"
            )
        if _is_current_stream_capturing(device):
            raise RuntimeError("Construct DCP candidate channels before graph capture")
        self.rank = dist.get_rank(process_group)
        self.max_rows, self.topk = max_rows, topk
        self.slab_bytes = PAYLOAD_OFFSET + max_rows * topk * 2 * 4
        self._captured = False
        self._capture_depth = 0
        precompile_candidate_publication(self.rank, topk, device.index)
        precompile_peer_topk(topk, 4, device.index)
        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device.index)
        owned = []
        try:
            slab = PCIeOneshotAllReduce._allocate_shared_buffer(
                process_group, self.slab_bytes, zero_fill=True, ipc=ipc
            )
            owned.append(slab)
            self._init_channel(
                device=device,
                exchange_group=process_group,
                ipc=ipc,
                owned_buffers=owned,
                stream_affine=True,
            )
            self._peer_slabs = list(slab.peer_ptrs)
            self._state = _tensor_from_cuda_pointer(
                slab.local_ptr, (self.slab_bytes,), dtype=torch.uint8, device=device
            )
        except Exception:
            _release_failed_allocations(owned, ipc)
            raise

    @contextmanager
    def capture(self):
        """Claim this channel before entering its sole CUDA graph capture."""
        if self._closed or self._captured:
            raise RuntimeError(
                "DCP candidate channel is closed or already owns a graph"
            )
        self._bind_stream()
        self._captured = True
        self._capture_depth = 1
        try:
            yield self
        finally:
            self._capture_depth = 0

    def merge(self, packed: torch.Tensor, out: torch.Tensor) -> None:
        """Publish and select exact candidates with no intermediate gathered copy.

        Packed ids must be unique per row across ranks and exactly representable
        in float32. Negative ids denote absent candidates; NaN scores are unsupported.
        Inputs and output must not overlap the channel state or each other.
        """
        if torch.compiler.is_compiling():
            _merge_op(
                packed,
                out,
                self._state,
                self._peer_slabs,
                self.rank,
                self.topk,
                self.max_rows,
            )
            return
        if self._closed:
            raise RuntimeError("DCP candidate channel is closed")
        if (
            packed.ndim != 3
            or packed.shape[1:] != (self.topk, 2)
            or not 1 <= packed.shape[0] <= self.max_rows
            or packed.dtype != torch.float32
            or not packed.is_contiguous()
            or out.dtype != torch.int32
            or out.shape != packed.shape[:2]
            or not out.is_contiguous()
            or packed.device != self.device
            or out.device != self.device
        ):
            raise ValueError("DCP candidate payload or output geometry is incompatible")
        if _is_current_stream_capturing(self.device) and not self._capture_depth:
            raise RuntimeError("Capture DCP candidate merge inside channel.capture()")
        self._bind_stream()
        _merge_op(
            packed,
            out,
            self._state,
            self._peer_slabs,
            self.rank,
            self.topk,
            self.max_rows,
        )

    def _free_ipc_exports(self) -> None:
        self._state = None
        super()._free_ipc_exports()
