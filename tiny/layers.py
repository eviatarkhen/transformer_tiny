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


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the rotation angles for a set of positions.

        inv_freq = 1 / (theta ** (arange(0, head_dim, 2) / head_dim))   # 64 values
        angles   = outer(positions, inv_freq)                           # (seq, 64)
        table    = cat((angles, angles), dim=-1)                        # (seq, 128)
        cos, sin = table.cos(), table.sin()

    Each of the 64 frequencies rotates one pair of dimensions. Pair i is
    (x[i], x[i + head_dim//2]) -- the two *halves*, not adjacent dimensions.
    The angles are duplicated by that cat so both members of a pair are handed
    the same angle.

    Args:
        positions: (seq,) integer positions. A tensor rather than a length, so
            incremental decoding can pass e.g. tensor([100]) with no offset
            parameter.
        head_dim: size of one head, 128 here. Must be even.
        theta: config.rope_theta, 1e6 here. Larger spreads the frequencies over
            a longer range.

    Returns:
        (cos, sin), each (seq, head_dim). Broadcasts against
        (batch, heads, seq, head_dim) unchanged.

    Notes:
        - torch.outer, or positions[:, None] * inv_freq[None, :].
        - Cast positions to float before multiplying; an integer tensor gives
          integer angles.
    """
    raise NotImplementedError


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension, negating the second.

        [x0 .. x63, x64 .. x127]  ->  [-x64 .. -x127, x0 .. x63]

    This is the partner term of the rotation: combined as
    `x * cos + rotate_half(x) * sin` it applies a 2-D rotation to each pair
    (x[i], x[i+64]). See apply_rope.
    """
    raise NotImplementedError


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate each head's vector by its position's angle.

        out = x * cos + rotate_half(x) * sin

    which, written out for one pair, is the standard 2-D rotation:

        out[i]    = x[i] * cos - x[i+64] * sin
        out[i+64] = x[i+64] * cos + x[i] * sin

    Why this is worth doing: a dot product depends on the angle *between* two
    vectors. Rotating the query at position m and the key at position n means
    their dot product depends only on (m - n) -- relative distance falls out of
    the geometry instead of being learned.

    Applied to queries and keys only, never to values: it changes where a token
    looks, not what it retrieves.

    Args:
        x: (batch, heads, seq, head_dim).
        cos, sin: (seq, head_dim) from rope_cos_sin. Broadcasting handles the
            batch and head dimensions.

    Returns:
        Same shape as x.
    """
    raise NotImplementedError
