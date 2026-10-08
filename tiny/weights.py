import torch
from safetensors.torch import load_file

from tiny.config import MODEL_ID, model_dir
from tiny.quantize import quantize_weights


def load_weights(
    model_id: str = MODEL_ID,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    int8: bool = False,
) -> dict[str, torch.Tensor]:
    """The checkpoint as a dict of tensors, keyed exactly as the file is.

    The one place dtype, device and quantization are chosen: every tensor
    tiny/ creates later takes dtype and device from the weights or the
    activations, and project recognizes a quantized weight by what it is, a
    (q, scale) pair -- there is no flag that could disagree with the data.

    Args:
        dtype: float32 (the default) for testing -- parity at 1e-4 needs it.
            bfloat16 for speed: the file's own format, so nothing is lost, and
            half the bytes to read per token.
        device: "cpu" (the default), or "mps" for the Apple GPU.
        int8: store the projection matrices and the LM head as int8 with
            per-row scales (see tiny/quantize.py). dtype then sets the
            activations and the scales.
    """
    raw = load_file(model_dir(model_id) / "model.safetensors")
    w = {name: t.to(device=device, dtype=dtype) for name, t in raw.items()}
    return quantize_weights(w) if int8 else w


def layer_prefix(i: int) -> str:
    return f"model.layers.{i}."
