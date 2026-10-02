import torch

from tiny.attention import attention
from tiny.config import Config
from tiny.layers import rms_norm
from tiny.mlp import mlp
from tiny.weights import layer_prefix


def block(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    layer: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """One decoder block: an attention half, then an MLP half.

    Args:
        x: (batch, seq, hidden_size).
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture.
        layer: which block's weights to read.
        cos, sin: (seq, head_dim) rotation tables, forwarded to attention.

    Returns:
        (batch, seq, hidden_size), the same shape as x.

    Notes:
        post_attention_layernorm is NOT applied to attention's output. It is
        the pre-norm of the MLP half; the name only says it comes after the
        attention half in source order.
    """
    p = layer_prefix(layer)
    eps = config.rms_norm_eps

    normed = rms_norm(x, w[p + "input_layernorm.weight"], eps)
    x = x + attention(normed, w, config, layer, cos, sin)

    normed = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
    return x + mlp(normed, w, layer)
