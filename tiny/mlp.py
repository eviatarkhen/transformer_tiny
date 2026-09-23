import torch

from tiny.layers import project
from tiny.weights import layer_prefix


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU-gated elementwise product: silu(gate) * up, where silu(x) = x * sigmoid(x)."""
    return torch.nn.functional.silu(gate) * up


def mlp(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Position-wise gated feed-forward network (SwiGLU) for one block.

    Args:
        x: (batch, seq, hidden_size).
        w: the full weight dict, keyed exactly as the checkpoint is.
        layer: which block's weights to read.

    Returns:
        (batch, seq, hidden_size), the same shape as x.
    """
    p = layer_prefix(layer)
    gate = project(x, w[p+"mlp.gate_proj.weight"])
    up = project(x, w[p+"mlp.up_proj.weight"])
    h = swiglu(gate, up)
    return project(h, w[p+"mlp.down_proj.weight"])    