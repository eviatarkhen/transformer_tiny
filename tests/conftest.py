import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny.config import MODEL_ID, Config, model_dir
from tiny.weights import load_weights


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.from_json(model_dir() / "config.json")


@pytest.fixture(scope="session")
def weights() -> dict[str, torch.Tensor]:
    return load_weights()


# --- the oracle -------------------------------------------------------------
# HuggingFace's own implementation, which each component we write is compared
# against. This is the one place in the project allowed to use transformers
# modeling code.


@pytest.fixture(scope="session")
def hf_model():
    """HuggingFace's Qwen3, float32, eval mode.

    float32 is not optional here: bf16 rounding noise on a matmul this size is
    around 1e-1, which would swamp the 1e-4 tolerance parity tests use.

    Submodules are reachable for per-component comparison:
        hf_model.model.layers[0].input_layernorm
        hf_model.model.layers[0].self_attn.q_norm
        hf_model.model.layers[0].mlp
        hf_model.model.norm
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)
