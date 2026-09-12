"""Strided BF16 GLM query BMM with caller-owned output and FP32 accumulation.

This fixed-geometry one-shot operation supports eight heads, 192 query features
and 512 latent features. Row counts and operand strides remain runtime values.
It preserves BF16 weights and does not allocate or assemble the rotary tail.
"""

from .api import can_implement, prewarm, run

__all__ = ["can_implement", "prewarm", "run"]
