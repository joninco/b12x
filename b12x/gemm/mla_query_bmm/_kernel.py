"""CuTe BF16 query BMM with FP32 MMA and explicit runtime operand strides."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    get_ptr_as_int64,
    ld_global_b16,
    st_global_u32,
)
from b12x.gemm._shared.mxfp8_bmm import pack_f32x2_to_bfloat2_rn


class QueryBmmKernel:
    """One 128-thread CTA computes 16 rows and 32 latent columns per head."""

    @cute.jit
    def __call__(
        self,
        a: cute.Pointer,
        b: cute.Pointer,
        c: cute.Pointer,
        rows: Int64,
        ah: Int64,
        am: Int64,
        bh: Int64,
        bk: Int64,
        bn: Int64,
        ch: Int64,
        cm: Int64,
        stream: cuda.CUstream,
    ):
        a_flat = cute.make_tensor(
            a, cute.make_layout((7 * ah + (rows - 1) * am + 192,))
        )
        b_flat = cute.make_tensor(
            b, cute.make_layout((7 * bh + 191 * bk + 511 * bn + 1,))
        )
        c_flat = cute.make_tensor(
            c, cute.make_layout((7 * ch + (rows - 1) * cm + 512,))
        )
        self.kernel(a_flat, b_flat, c_flat, rows, ah, am, bh, bk, bn, ch, cm).launch(
            grid=((rows + 15) // 16, 16, 8),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _pair(self, tensor: cute.Tensor, offset: Int64, stride: Int64) -> Uint32:
        lo = ld_global_b16(get_ptr_as_int64(tensor, offset))
        hi = ld_global_b16(get_ptr_as_int64(tensor, offset + stride))
        return lo | (hi << Uint32(16))

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        rows: Int64,
        ah: Int64,
        am: Int64,
        bh: Int64,
        bk: Int64,
        bn: Int64,
        ch: Int64,
        cm: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row_tile, col_tile, head = cute.arch.block_idx()
        lane = Int32(tid) % 32
        warp = Int32(tid) // 32
        group = lane // 4
        pair = lane % 4
        row0 = Int64(row_tile) * 16 + Int64(group)
        row1 = row0 + 8
        column = Int64(col_tile) * 32 + Int64(warp) * 8 + Int64(group)
        accum = cute.make_rmem_tensor((4,), Float32)
        accum.fill(0.0)
        # PTX m16n8k16 fragments: A pairs at (g,2t),(g+8,2t),
        # (g,2t+8),(g+8,2t+8); B pairs at (2t,g),(2t+8,g).
        for step in cutlass.range_constexpr(12):
            k = Int64(step * 16) + Int64(pair) * 2
            a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if row0 < rows:
                a0 = self._pair(a, Int64(head) * ah + row0 * am + k, Int64(1))
                a2 = self._pair(a, Int64(head) * ah + row0 * am + k + 8, Int64(1))
            if row1 < rows:
                a1 = self._pair(a, Int64(head) * ah + row1 * am + k, Int64(1))
                a3 = self._pair(a, Int64(head) * ah + row1 * am + k + 8, Int64(1))
            b0 = self._pair(b, Int64(head) * bh + k * bk + column * bn, bk)
            b1 = self._pair(b, Int64(head) * bh + (k + 8) * bk + column * bn, bk)
            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                accum[0],
                accum[1],
                accum[2],
                accum[3],
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
            )
            accum[0], accum[1], accum[2], accum[3] = d0, d1, d2, d3
        out_col = Int64(col_tile) * 32 + Int64(warp) * 8 + Int64(pair) * 2
        if row0 < rows:
            st_global_u32(
                get_ptr_as_int64(c, Int64(head) * ch + row0 * cm + out_col),
                pack_f32x2_to_bfloat2_rn(accum[0], accum[1]),
            )
        if row1 < rows:
            st_global_u32(
                get_ptr_as_int64(c, Int64(head) * ch + row1 * cm + out_col),
                pack_f32x2_to_bfloat2_rn(accum[2], accum[3]),
            )
