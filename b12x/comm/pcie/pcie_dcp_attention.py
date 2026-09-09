"""Opaque graph-owned DCP query gather and LSE reduce-scatter.

The channel uses PCIeDCPA2A's local staging and ordinary peer loads. Its mutable
signal view is an explicit Torch operator argument so the compiler preserves
the ordering of exchanges. Runtime handles are process-local; construct channels
at model load and retain them for the lifetime of their captured graphs.
"""

from contextlib import contextmanager
from itertools import count
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist

from .pcie_dcp_a2a import PCIeDCPA2APool, _SIGNAL_BYTES
from .pcie_dcp_topk import _tensor_from_cuda_pointer
from .pcie_oneshot import _normalize_device
from ._dcp_attention_metadata import mask_local_lse, precompile_local_lse_mask


_HANDLES = count(1)
_CHANNELS: WeakValueDictionary = WeakValueDictionary()


def _channel(handle: int, state: torch.Tensor):
    channel = _CHANNELS.get(handle)
    if channel is None or channel._closed:
        raise RuntimeError("DCP attention channel has been closed or released")
    if (
        state.device != channel.device
        or state.dtype != torch.uint8
        or state.shape != channel._state.shape
        or state.data_ptr() != channel._state.data_ptr()
    ):
        raise ValueError("DCP attention signal state does not belong to the channel")
    return channel


@torch.library.custom_op("b12x::dcp_attention_query", mutates_args=("out", "state"))
def _query_op(
    query: torch.Tensor, out: torch.Tensor, state: torch.Tensor, handle: int
) -> None:
    channel = _channel(handle, state)
    if query.dtype != torch.bfloat16 or torch._C._overlaps(query, out):
        raise ValueError("DCP query requires BF16 input and disjoint output storage")
    channel._pool.all_gather_heads(query, out, channel_id=channel.channel_id)


@_query_op.register_fake
def _query_fake(query, out, state, handle) -> None:
    pass


@torch.library.custom_op("b12x::dcp_attention_combine", mutates_args=("out", "state"))
def _combine_op(
    partial: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    handle: int,
) -> None:
    channel = _channel(handle, state)
    if partial.dtype != torch.bfloat16 or any(
        torch._C._overlaps(tensor, out) for tensor in (partial, lse)
    ):
        raise ValueError(
            "DCP combine requires BF16 partials and disjoint output storage"
        )
    channel._pool.lse_reduce_scatter(
        partial, lse, out, channel_id=channel.channel_id, is_lse_base_on_e=True
    )


@_combine_op.register_fake
def _combine_fake(partial, lse, out, state, handle) -> None:
    pass


@torch.library.custom_op(
    "b12x::dcp_attention_combine_masked", mutates_args=("out", "state", "masked_lse")
)
def _combine_masked_op(
    partial: torch.Tensor,
    lse: torch.Tensor,
    local_seq_lens: torch.Tensor,
    masked_lse: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    handle: int,
) -> None:
    _channel(handle, state)
    if any(
        torch._C._overlaps(masked_lse, tensor)
        for tensor in (partial, lse, local_seq_lens, out, state)
    ):
        raise ValueError("DCP masked LSE scratch must not alias inputs or outputs")
    mask_local_lse(lse, local_seq_lens, masked_lse)
    _combine_op(partial, masked_lse, out, state, handle)


@_combine_masked_op.register_fake
def _combine_masked_fake(
    partial, lse, local_seq_lens, masked_lse, out, state, handle
) -> None:
    pass


class PCIeDCPAttention:
    """Own precompiled query/combine exchanges for one serially replayed graph.

    Construct collectively on a four-rank CPU group with a semantic channel id
    shared by all ranks. Geometry and capacity are fixed; live rows are runtime
    launch arguments. Enter capture() before torch.cuda.graph(). Different
    graphs require different instances even if they use the same CUDA stream.
    Query and combine outputs are caller-owned and must not alias inputs.
    """

    def __init__(
        self,
        *,
        process_group,
        device,
        channel_id: str,
        max_rows: int = 16,
        local_heads: int = 8,
        query_dim: int = 576,
        output_dim: int = 512,
    ):
        self.device = _normalize_device(device)
        if dist.get_world_size(process_group) != 4 or not 1 <= max_rows <= 16:
            raise ValueError(
                "DCP attention requires four ranks and capacity 1 through 16"
            )
        self.channel_id = channel_id
        self._closed = False
        self._pool = PCIeDCPA2APool.from_process_group(
            process_group=process_group,
            device=self.device,
            max_batch_size=max_rows,
            total_heads=local_heads * 4,
            head_dim=output_dim,
            query_head_dim=query_dim,
        )
        try:
            self._pool.prepare_channels((channel_id,))
            self._pool.prepare_graph_all_gather_heads(channel_id=channel_id)
            self._pool.prepare_graph_lse_reduce_scatter(
                dtype=torch.bfloat16, channel_id=channel_id
            )
            runtime = self._pool.for_stream(channel_id=channel_id)
            precompile_local_lse_mask(local_heads * 4, self.device.index)
            self.allocated_bytes = (
                runtime._staging1_ptrs[runtime.rank]
                + runtime._slot_bytes
                - runtime._signal_ptrs[runtime.rank]
            )
            self._state = _tensor_from_cuda_pointer(
                runtime._signal_ptrs[runtime.rank],
                (_SIGNAL_BYTES,),
                dtype=torch.uint8,
                device=self.device,
            )
        except Exception:
            self._pool.close()
            raise
        self._handle = next(_HANDLES)
        _CHANNELS[self._handle] = self

    def query(self, query: torch.Tensor, out: torch.Tensor) -> None:
        """Gather local BF16 heads into caller-owned [rows,4*heads,query_dim]."""
        _query_op(query, out, self._state, self._handle)

    def combine(
        self, partial: torch.Tensor, lse: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Reduce partials using natural-log LSE and return this rank's heads.

        Every empty local query row must carry -inf LSE for each head. Such
        partials may contain NaN; their weight is zero. An all-empty row emits
        zero. Per-request multi-token queries each occupy an independent row.
        """
        _combine_op(partial, lse, out, self._state, self._handle)

    def combine_masked(
        self,
        partial: torch.Tensor,
        lse: torch.Tensor,
        local_seq_lens: torch.Tensor,
        masked_lse: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Mask each empty local query row before reducing its partial output.

        local_seq_lens contains one int32 causal local length per query, including
        separate rows of an MTP request. masked_lse is caller-owned float32 scratch.
        """
        _combine_masked_op(
            partial, lse, local_seq_lens, masked_lse, out, self._state, self._handle
        )

    @contextmanager
    def capture(self):
        """Claim a single graph identity collectively before CUDA capture."""
        if self._closed:
            raise RuntimeError("DCP attention channel is closed")
        with self._pool.capture(channel_id=self.channel_id):
            yield self

    def close(self) -> None:
        """Collectively release IPC mappings after all graph work completes."""
        if self._closed:
            return
        self._pool.close()
        self._closed = True
        self.allocated_bytes = 0
        _CHANNELS.pop(self._handle, None)
        self._state = None
