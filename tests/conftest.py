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


@pytest.fixture
def hf_model_eager(hf_model):
    """The same HF model, switched to the eager attention path for one test.

    Eager is plain matmul -> softmax -> matmul, the same arithmetic tiny/ does,
    so whole-stack comparisons against it are exact (0.0). The default sdpa
    kernel sums in a different order, and over 28 chained blocks that drifts
    to ~1e-3 -- past the 1e-4 tolerance with no bug present.

    Restored to the original implementation afterwards, because hf_model is
    session-scoped and every other test expects the default.
    """
    original = hf_model.config._attn_implementation
    hf_model.set_attn_implementation("eager")
    try:
        yield hf_model
    finally:
        hf_model.set_attn_implementation(original)
