import torch


def project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply one of the checkpoint's bias-free linear layers.
    """
    return x @ weight.T


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalization over the last dimension.

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
    """
    inv_rms = (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return x * inv_rms * weight


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the rotation angles for a set of positions.

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
    """
    inv_freq = 1 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    table = torch.cat([angles, angles], dim=-1)
    return table.cos(), table.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension, negating the second.

        [x0 .. x63, x64 .. x127]  ->  [-x64 .. -x127, x0 .. x63]

    This is the partner term of the rotation: combined as
    `x * cos + rotate_half(x) * sin` it applies a 2-D rotation to each pair
    (x[i], x[i+64]). See apply_rope.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate each head's vector by its position's angle.

    Args:
        x: (batch, heads, seq, head_dim).
        cos, sin: (seq, head_dim) from rope_cos_sin. Broadcasting handles the
            batch and head dimensions.

    Returns:
        Same shape as x.
    """
    return x * cos + rotate_half(x) * sin
