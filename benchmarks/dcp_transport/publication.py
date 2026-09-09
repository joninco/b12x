"""Local IPC publication and peer-load LSE merge prototypes.

Status: research-only. One channel belongs to one serially replayed CUDA graph.
Publication fences preceding consumers before overwriting the local slab, then
fences its writes before consumers read peer slabs. Only signal words are
stored remotely; all payload stores target the publishing rank's allocation.
"""

from contextlib import contextmanager
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
from b12x.comm.pcie._dcp_cute_common import block_pair_barrier, signal_bytes
from b12x.comm.pcie.pcie_dcp_topk import (
    _IPCChannel,
    _release_failed_allocations,
    _tensor_from_cuda_pointer,
)
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce, _align_up


CAPACITY = 16
BLOCKS = 16
Q_OFFSET = _align_up(signal_bytes(BLOCKS), 256)
O_OFFSET = Q_OFFSET + CAPACITY * 8 * 576 * 2
LSE_OFFSET = O_OFFSET + CAPACITY * 32 * 512 * 2
CANDIDATE_OFFSET = LSE_OFFSET + CAPACITY * 32 * 4
SLAB_BYTES = CANDIDATE_OFFSET + CAPACITY * 2048 * 2 * 4


class _Publish:
    def __init__(self, rank, words0, words1):
        self.rank, self.words0, self.words1 = rank, words0, words1

    @cute.jit
    def __call__(
        self,
        signals: tuple,
        source0: cute.Pointer,
        source1: cute.Pointer,
        destination0: cute.Pointer,
        destination1: cute.Pointer,
        rows: Int32,
        stream: CUstream,
    ):
        self.publish(
            signals, source0, source1, destination0, destination1, rows
        ).launch(grid=(BLOCKS, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def publish(
        self,
        signals: tuple,
        source0: cute.Pointer,
        source1: cute.Pointer,
        destination0: cute.Pointer,
        destination1: cute.Pointer,
        rows: Int32,
    ):
        block_pair_barrier(
            signals,
            self_signal=signals[self.rank],
            rank=self.rank,
            world_size=4,
            max_blocks=BLOCKS,
        )
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        start = Int64(block) * Int64(256) + Int64(thread)
        for offset in range(start, Int64(rows) * Int64(self.words0), BLOCKS * 256):
            (destination0 + offset).store((source0 + offset).load())
        if cutlass.const_expr(self.words1):
            for offset in range(start, Int64(rows) * Int64(self.words1), BLOCKS * 256):
                (destination1 + offset).store((source1 + offset).load())
        block_pair_barrier(
            signals,
            self_signal=signals[self.rank],
            rank=self.rank,
            world_size=4,
            max_blocks=BLOCKS,
        )


class _PeerLSEMerge:
    def __init__(self, rank):
        self.rank = rank

    @cute.jit
    def __call__(
        self,
        partials: tuple,
        lses: tuple,
        out: cute.Pointer,
        rows: Int32,
        stream: CUstream,
    ):
        self.merge(partials, lses, out).launch(
            grid=(rows, 8, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def merge(self, partials: tuple, lses: tuple, out: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        global_head = Int64(self.rank * 8) + Int64(head)
        lse_offset = Int64(row) * Int64(32) + global_head
        maximum = Float32(-float("inf"))
        values = []
        for peer in cutlass.range_constexpr(4):
            value = (lses[peer] + lse_offset).load()
            values.append(value)
            if value > maximum and value < Float32(float("inf")):
                maximum = value
        weights = []
        denominator = Float32(0)
        for peer in cutlass.range_constexpr(4):
            weight = Float32(0)
            if values[peer] > Float32(-float("inf")) and values[peer] < Float32(
                float("inf")
            ):
                weight = cute.math.exp2(
                    (values[peer] - maximum) * Float32(1.4426950408889634),
                    fastmath=True,
                )
            weights.append(weight)
            denominator += weight
        for column in range(thread, 512, 256):
            offset = (Int64(row) * Int64(32) + global_head) * Int64(512) + Int64(column)
            result = Float32(0)
            for peer in cutlass.range_constexpr(4):
                if weights[peer] > Float32(0):
                    result += Float32((partials[peer] + offset).load()) * weights[peer]
            if denominator > Float32(0):
                result /= denominator
            out_offset = (Int64(row) * Int64(8) + Int64(head)) * Int64(512) + Int64(
                column
            )
            (out + out_offset).store(BFloat16(result))


def _ptr(dtype, address):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)


@cache
def _publisher(rank, words0, words1, device_index):
    key = (rank, words0, words1, device_index)
    raise_if_kernel_resolution_frozen("cute.compile", target=_Publish, cache_key=key)
    with torch.cuda.device(device_index):
        return b12x_compile(
            _Publish(rank, words0, words1),
            tuple(_ptr(Uint32, 16) for _ in range(4)),
            *(_ptr(Uint32, 16) for _ in range(4)),
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "benchmark.dcp_transport.publish", 1, *key
            ),
        )


@cache
def _merger(rank, device_index):
    key = (rank, device_index)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=_PeerLSEMerge, cache_key=key
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _PeerLSEMerge(rank),
            tuple(_ptr(BFloat16, 16) for _ in range(4)),
            tuple(_ptr(Float32, 16) for _ in range(4)),
            _ptr(BFloat16, 16),
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "benchmark.dcp_transport.peer_lse", 1, *key
            ),
        )


class PublicationChannel(_IPCChannel):
    def __init__(self, group, device):
        if dist.get_world_size(group) != 4:
            raise ValueError("Publication prototype requires four ranks per group")
        device = torch.device(device)
        self.rank = dist.get_rank(group)
        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device.index)
        owned = []
        try:
            slab = PCIeOneshotAllReduce._allocate_shared_buffer(
                group, SLAB_BYTES, zero_fill=True, ipc=ipc
            )
            owned.append(slab)
            self._init_channel(
                device=device,
                exchange_group=group,
                ipc=ipc,
                owned_buffers=owned,
                stream_affine=True,
            )
            self.signals = tuple(_ptr(Uint32, p) for p in slab.peer_ptrs)
            self.query_pointers = tuple(p + Q_OFFSET for p in slab.peer_ptrs)
            self.partial_pointers = tuple(
                _ptr(BFloat16, p + O_OFFSET) for p in slab.peer_ptrs
            )
            self.lse_pointers = tuple(
                _ptr(Float32, p + LSE_OFFSET) for p in slab.peer_ptrs
            )
            self.candidate_pointers = tuple(
                p + CANDIDATE_OFFSET for p in slab.peer_ptrs
            )
            self._destinations = {
                name: _ptr(Uint32, slab.local_ptr + offset)
                for name, offset in (
                    ("query", Q_OFFSET),
                    ("partial", O_OFFSET),
                    ("lse", LSE_OFFSET),
                    ("candidates", CANDIDATE_OFFSET),
                )
            }
            self.local_query = _tensor_from_cuda_pointer(
                slab.local_ptr + Q_OFFSET,
                (CAPACITY, 8, 576),
                dtype=torch.bfloat16,
                device=device,
            )
            self._captured = False
            self.prepare()
        except Exception:
            _release_failed_allocations(owned, ipc)
            raise

    def prepare(self):
        for words0, words1 in ((2304, 0), (4096, 0), (2304, 4096), (8192, 32)):
            _publisher(self.rank, words0, words1, self.device.index)
        _merger(self.rank, self.device.index)

    @contextmanager
    def capture(self):
        if self._captured:
            raise RuntimeError("Publication channel already owns a captured graph")
        self._captured = True
        yield self

    def publish(self, kind, first, second=None):
        self._bind_stream()
        layouts = {
            "query": (2304, 0, "query", "query"),
            "candidates": (4096, 0, "candidates", "candidates"),
            "shared": (2304, 4096, "query", "candidates"),
            "partial": (8192, 32, "partial", "lse"),
        }
        words0, words1, dest0, dest1 = layouts[kind]
        rows = first.shape[0]
        second = first if second is None else second
        if (
            self._closed
            or not 1 <= rows <= CAPACITY
            or first.numel() * first.element_size() != rows * words0 * 4
            or (words1 and second.numel() * second.element_size() != rows * words1 * 4)
            or any(
                t.device != self.device or not t.is_contiguous() or t.data_ptr() % 16
                for t in (first, second)
            )
        ):
            raise ValueError("Publication payload geometry or storage is invalid")
        _publisher(self.rank, words0, words1, self.device.index)(
            self.signals,
            _ptr(Uint32, first.data_ptr()),
            _ptr(Uint32, second.data_ptr()),
            self._destinations[dest0],
            self._destinations[dest1],
            rows,
            current_cuda_stream(),
        )

    def combine(self, out):
        if (
            out.shape[1:] != (8, 512)
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            raise ValueError(
                "Peer LSE merge requires contiguous BF16 [rows,8,512] output"
            )
        _merger(self.rank, self.device.index)(
            self.partial_pointers,
            self.lse_pointers,
            _ptr(BFloat16, out.data_ptr()),
            out.shape[0],
            current_cuda_stream(),
        )
