import pytest
import torch

from tiny.attention import causal_mask, allocate_cache
from tiny.generate import greedy
from tiny.layers import rms_norm
from tiny.model import forward
from tiny.weights import load_weights

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")

# --- bf16: the precision islands, localized ---------------------------------


def test_rms_norm_bf16_matches_hf_bf16(config, weights_bf16, hf_model_bf16_eager):
    """bf16 in, bf16 out, equal to HF's Qwen3RMSNorm in bf16.

    Input at the stream's real scale (thousands, see step 6), where squaring
    in bf16 would lose most of the mean: catches a norm that skips the
    float32 island, or casts back after the weight instead of before.
    """
    torch.manual_seed(0)
    x = (torch.randn(2, 5, config.hidden_size) * 1000).to(torch.bfloat16)
    layer = 3
    weight = weights_bf16[f"model.layers.{layer}.input_layernorm.weight"]

    got = rms_norm(x, weight, config.rms_norm_eps)
    with torch.no_grad():
        expected = hf_model_bf16_eager.model.layers[layer].input_layernorm(x)

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), expected.float(), atol=1e-4, rtol=1e-5)


def test_causal_mask_follows_the_scores_dtype():
    """A float32 mask added to bf16 scores promotes them to float32 -- the
    crash that started step 10 was this, one matmul later. Batch and head
    dimensions in front are allowed; the mask is (q_len, k_len)."""
    scores = torch.zeros(2, 4, 3, 6, dtype=torch.bfloat16)

    mask = causal_mask(scores)

    assert mask.dtype == torch.bfloat16
    assert torch.equal(mask.float(), causal_mask(torch.zeros(3, 6)))


def test_allocate_cache_follows_like(config):
    """The buffers take dtype and device from `like`, never the default.

    A float32 buffer would silently upcast every bf16 key written into it,
    and the next matmul against bf16 queries would raise. No bf16 test runs
    with a cache, so this is the only place that would show it off-MPS.
    """
    cache = allocate_cache(config, 1, 8, like=torch.zeros(1, dtype=torch.bfloat16))

    assert all(s["k"].dtype == s["v"].dtype == torch.bfloat16 for s in cache)


# --- bf16: the whole model --------------------------------------------------


def test_forward_bf16_matches_hf_bf16(config, weights_bf16, hf_model_bf16_eager):
    """Our bf16 forward equals HF's bf16 eager forward, logit for logit.

    Measured 0.0: same operations in the same dtypes in the same order. The
    tolerance is therefore the float32 one, not bf16's -- against our own
    float32 the same logits differ by 0.71, which would hide every bug the
    float32 suite exists to catch. Catches any island missing (RMSNorm,
    softmax, RoPE tables built in bf16) and any tensor left in float32.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 32))

    with torch.no_grad():
        got = forward(ids, weights_bf16, config, torch.arange(32))
        expected = hf_model_bf16_eager(ids).logits

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), expected.float(), atol=1e-4, rtol=1e-5)


# --- MPS --------------------------------------------------------------------


@needs_mps
def test_load_weights_to_mps():
    """Every tensor lands on the requested device and dtype."""
    w = load_weights(dtype=torch.bfloat16, device="mps")

    assert all(t.device.type == "mps" and t.dtype == torch.bfloat16 for t in w.values())


@needs_mps
def test_forward_on_mps_matches_cpu(config, weights):
    """float32 on the GPU equals float32 on the CPU.

    Catches any tensor tiny/ creates on the default device (the RoPE
    frequencies, the mask) -- those raise -- and anything that runs but
    computes differently. Measured 6.4e-5: the GPU sums in its own order.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 32))
    w_mps = {k: v.to("mps") for k, v in weights.items()}

    with torch.no_grad():
        expected = forward(ids, weights, config, torch.arange(32))
        got = forward(ids.to("mps"), w_mps, config, torch.arange(32, device="mps"))

    assert got.device.type == "mps"
    torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-5)


@needs_mps
def test_greedy_on_mps_matches_cpu(config, weights, hf_tokenizer):
    """Cached greedy on the GPU picks the same ids as on the CPU.

    The positions greedy builds itself are the one device decision forward's
    test cannot see. float32 only: in bf16, a near-tie between two logits can
    legitimately flip, so equal ids would be luck, not a contract.
    """
    ids = hf_tokenizer("The capital of France is", return_tensors="pt").input_ids
    w_mps = {k: v.to("mps") for k, v in weights.items()}

    with torch.no_grad():
        expected = greedy(ids, weights, config, 20)
        got = greedy(ids.to("mps"), w_mps, config, 20)

    assert torch.equal(got.cpu(), expected)
