import torch
from safetensors.torch import load_file

from tiny.config import MODEL_ID, model_dir


def load_weights(
    model_id: str = MODEL_ID,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """The checkpoint as a dict of tensors, keyed exactly as the file is.

    The one place dtype and device are chosen: every tensor tiny/ creates
    later takes both from the weights or the activations, never a default.

    Args:
        dtype: float32 (the default) for testing -- parity at 1e-4 needs it.
            bfloat16 for speed: the file's own format, so nothing is lost, and
            half the bytes to read per token.
        device: "cpu" (the default), or "mps" for the Apple GPU.
    """
    raw = load_file(model_dir(model_id) / "model.safetensors")
    return {name: t.to(device=device, dtype=dtype) for name, t in raw.items()}


def layer_prefix(i: int) -> str:
    return f"model.layers.{i}."
