import torch


def project(
    x: torch.Tensor, weight: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Apply one of the checkpoint's bias-free linear layers.

    Args:
        x: (..., in).
        weight: (out, in) as stored, or an int8 (q, scale) pair from
            quantize_rows.

    Returns:
        (..., out).

    Notes:
        The int8 path calls a fused kernel that dequantizes inside the
        multiply. Spelling it out -- x @ (q * scale).T -- rebuilds the whole
        float matrix every call and measured 2-10x *slower* than not
        quantizing at all. The kernel only multiplies; rounding and scales,
        the decisions, are quantize_rows'. test_quantize checks it against
        the spelled-out version.
    """
    if isinstance(weight, tuple):
        q, scale = weight
        out = torch._weight_int8pack_mm(x.reshape(-1, x.shape[-1]), q, scale)
        return out.view(*x.shape[:-1], -1)
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
        - Computed in float32 whatever x's dtype, then cast back before the
          weight, exactly as HF's Qwen3RMSNorm. In bf16 the mean of 1024
          squares -- up to ~4e7 once the stream reaches 6,592 -- would keep
          only 3 significant digits. A no-op when x is already float32.
    """
    x32 = x.to(torch.float32)
    inv_rms = (x32.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x32 * inv_rms).to(x.dtype) * weight


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
        (cos, sin), each (seq, head_dim), float32 and on positions' device.
        Always float32: in bf16 an angle near 1000 has a resolution of 4
        radians. apply_rope casts the finished tables to x's dtype.
    """
    inv_freq = 1 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device) / head_dim))
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
            batch and head dimensions. Cast to x's dtype here: float32 tables
            times bf16 x would promote the result -- q and k -- to float32.

    Returns:
        Same shape and dtype as x.
    """
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin
