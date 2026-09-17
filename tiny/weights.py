import torch
from safetensors.torch import load_file

from tiny.config import MODEL_ID, model_dir


def load_weights(model_id: str = MODEL_ID) -> dict[str, torch.Tensor]:
    raw = load_file(model_dir(model_id) / "model.safetensors")
    return {name: t.to(torch.float32) for name, t in raw.items()}


def layer_prefix(i: int) -> str:
    return f"model.layers.{i}."
