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


# --- bf16 and MPS (step 10) -------------------------------------------------


@pytest.fixture(scope="session")
def weights_bf16() -> dict[str, torch.Tensor]:
    return load_weights(dtype=torch.bfloat16)


@pytest.fixture(scope="session")
def hf_model_bf16_eager():
    """HF's Qwen3 in bf16 on the eager path: the oracle for our bf16 path.

    A second copy rather than a cast of hf_model, which every float32 test
    shares. Eager because it is the same matmul-softmax-matmul we wrote, and
    with HF's float32 islands (RMSNorm, softmax, RoPE tables) matched, our
    bf16 forward equals it exactly on CPU -- so the bf16 path keeps the 1e-4
    tolerance instead of the ~1.0 that bf16-vs-float32 would need.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    return model
