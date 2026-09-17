"""The individual operations a transformer block is made of.

Plain functions over explicit tensors. Nothing here holds state, and nothing
here knows which layer it belongs to -- the caller passes the weights in.
"""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalization over the last dimension.

        rms = sqrt(mean(x**2) + eps)
        out = (x / rms) * weight

    Rescales each position's vector to a consistent magnitude so that values
    stay in a sane range across 28 blocks of residual additions. Unlike
    LayerNorm there is no mean subtraction and no bias -- only the divide and
    the learned per-channel scale.

    Args:
        x: (..., n). Any rank; only the last dimension is normalized, and every
           position is normalized independently of the others.
        weight: (n,) learned scale, broadcast across all leading dimensions.
        eps: added *inside* the square root, so an all-zero input gives
           sqrt(eps) rather than a division by zero.

    Returns:
        Same shape as x.

    Notes:
        - Used at two sizes: 1024 in the block, 128 per head for q_norm/k_norm.
          Write it against the last dimension and both work. `keepdim=True` on
          the mean is what keeps the broadcast right.
        - torch.rsqrt(v) computes 1/sqrt(v) in one step; HF uses it here.
        - HF casts to float32 inside this function before squaring, because
          squaring in bf16 loses precision that rsqrt then amplifies. We are
          already in float32 everywhere, so that cast is a no-op for us -- but
          it is the reason the operation is written that way upstream.
    """
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return (x / rms) * weight 
