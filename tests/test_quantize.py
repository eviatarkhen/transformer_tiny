import pytest
import torch

from tiny.layers import project
from tiny.model import forward, lm_head
from tiny.quantize import LM_HEAD_INT8, quantize_rows
from tiny.weights import load_weights

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")

# Plain prose, no names or numbers the model could have memorized verbatim.
TEXT = (
    "The river rises in the northern hills and flows south for nearly two hundred "
    "miles before it reaches the sea. For centuries it carried timber and grain to "
    "the coastal towns, and the bridges built across it shaped the roads of the "
    "region. In the nineteenth century, engineers straightened several of its bends "
    "to make room for railways, which changed how often the valley flooded. Today "
    "the river is cleaner than it has been in a hundred years, and fish have "
    "returned to stretches where none were seen for decades."
)


def perplexity(ids, w, config):
    """exp(mean cross-entropy of each next token): how surprised the model is
    by real text. Lower is better; 1.0 would be certainty."""
    with torch.no_grad():
        logits = forward(ids, w, config, torch.arange(ids.shape[1], device=ids.device))
    nll = torch.nn.functional.cross_entropy(logits[0, :-1].float().cpu(), ids[0, 1:].cpu())
    return nll.exp().item()


# --- quantize_rows: the scheme, on numbers you can check by hand ------------


def test_quantize_rows_hand_example():
    """Row 0 is the worked example: max 1.27 -> scale 0.01 -> [50, -127, 2].

    Row 1 is all zeros, which would divide by a zero scale without the floor.
    Per-row scaling is not tested here -- one shared scale gives the same
    answer on these rows; the error-bound test below is the one that needs it.
    """
    w = torch.tensor([[0.50, -1.27, 0.02], [0.0, 0.0, 0.0]])

    q, scale = quantize_rows(w)

    assert q.dtype == torch.int8
    assert torch.equal(q, torch.tensor([[50, -127, 2], [0, 0, 0]], dtype=torch.int8))
    torch.testing.assert_close(scale[0], torch.tensor(0.01), atol=1e-7, rtol=0)
    assert torch.isfinite(scale).all() and scale[1] > 0


def test_quantize_rows_error_is_at_most_half_a_step():
    """Every weight comes back within scale/2 of itself, every q is in
    -127..127, and the largest |w| of each row maps to exactly +-127.

    Rows span four orders of magnitude, so a single scale for the whole
    matrix would round the small rows to nothing and fail the bound.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 300) * torch.logspace(-4, 0, 64)[:, None]

    q, scale = quantize_rows(w)
    back = q.float() * scale[:, None]

    assert (back - w).abs().le(scale[:, None] / 2 + 1e-9).all()
    assert q.abs().max() <= 127
    assert torch.equal(q.abs().amax(dim=-1), torch.full((64,), 127, dtype=torch.int8))


def test_quantize_rows_keeps_the_scale_in_the_weights_dtype():
    """bf16 weights give bf16 scales: the fused kernel needs the scale in the
    activations' dtype. A float32 scale is invisible on the float32 path,
    which every other test here runs."""
    _, scale = quantize_rows(torch.randn(8, 32, dtype=torch.bfloat16))

    assert scale.dtype == torch.bfloat16


# --- project: the fused kernel, checked against the spelled-out arithmetic --


def test_project_int8_matches_dequantize_then_multiply():
    """The fused kernel computes x @ (q * scale).T, with the batch dimensions
    folded and unfolded correctly. Same arithmetic, so float32-tight.

    This is what lets the kernel be used without being trusted.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 5, 1024)
    q, scale = quantize_rows(torch.randn(3072, 1024) * 0.02)

    got = project(x, (q, scale))
    expected = x @ (q.float() * scale[:, None]).T

    assert got.shape == (2, 5, 3072)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_project_int8_uses_the_fused_kernel():
    """A performance test that is deterministic: the int8 path must run the
    fused kernel, never a plain matmul.

    Dequantize-then-multiply gives the same numbers -- the test above cannot
    tell -- while rebuilding the float matrix every call, measured 2-10x
    slower than not quantizing at all.
    """
    q, scale = quantize_rows(torch.randn(64, 32))
    x = torch.randn(1, 3, 32)

    with torch.profiler.profile() as prof:
        project(x, (q, scale))

    ops = {e.key for e in prof.key_averages()}
    assert "aten::_weight_int8pack_mm" in ops
    assert "aten::mm" not in ops and "aten::matmul" not in ops


# --- quantize_weights and lm_head: what gets quantized, and is it read ------


def test_quantize_weights_quantizes_exactly_the_projections(config, weights, weights_int8):
    """All 7 projections of all 28 layers become (q, scale) pairs; norms and
    the embedding table are the very same tensors; the int8 lm_head copy is
    added; and the input dict is left alone."""
    pairs = [k for k, v in weights_int8.items() if isinstance(v, tuple)]

    assert len(pairs) == 7 * config.num_hidden_layers + 1
    assert all(k.endswith("_proj.weight") for k in pairs if k != LM_HEAD_INT8)
    assert weights_int8["model.embed_tokens.weight"] is weights["model.embed_tokens.weight"]
    assert weights_int8["model.norm.weight"] is weights["model.norm.weight"]
    assert LM_HEAD_INT8 in weights_int8 and LM_HEAD_INT8 not in weights
    assert isinstance(weights["model.layers.0.self_attn.q_proj.weight"], torch.Tensor)


def test_load_weights_int8_quantizes(config):
    """load_weights(int8=True) is quantize_weights applied at load, in the
    requested dtype -- the flag reaches the weights, and the scales follow
    dtype."""
    w = load_weights(dtype=torch.bfloat16, int8=True)

    q, scale = w["model.layers.0.mlp.down_proj.weight"]
    assert q.dtype == torch.int8 and scale.dtype == torch.bfloat16
    assert LM_HEAD_INT8 in w


def test_lm_head_reads_the_int8_copy_when_present():
    """With LM_HEAD_INT8 present lm_head uses it; without, the full table.

    The int8 copy here is deliberately all zeros, so reading it is visible;
    in real use it holds the same table and the two would be hard to tell
    apart -- which is why this cannot be left to parity.
    """
    table = torch.randn(10, 4)
    x = torch.randn(1, 2, 4)
    zeros = (torch.zeros(10, 4, dtype=torch.int8), torch.ones(10))

    assert torch.equal(lm_head(x, {"model.embed_tokens.weight": table, LM_HEAD_INT8: zeros}), torch.zeros(1, 2, 10))
    torch.testing.assert_close(lm_head(x, {"model.embed_tokens.weight": table}), x @ table.T)


# --- the whole model: tolerance from the measured gap -----------------------


def test_int8_logits_stay_near_float32(config, weights, weights_int8):
    """Relative logit error vs float32 sits between 1% and 15%.

    Measured on these ids: per-row int8 9.5%; per-matrix scales 22%; bf16
    alone 3.1%. 15% sits in the gap between the scheme's noise and the
    nearest wrong scheme. The lower bound rules out quantize_weights
    quietly returning the float weights.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 32))

    with torch.no_grad():
        ref = forward(ids, weights, config, torch.arange(32))
        got = forward(ids, weights_int8, config, torch.arange(32))

    rel = ((got - ref).norm() / ref.norm()).item()
    assert 0.01 < rel < 0.15


def test_int8_perplexity_matches_float32(config, weights, weights_int8, hf_tokenizer):
    """On real text, int8 is as good as float32: perplexity within 2%.

    Measured: float32 23.85, int8 per-row 23.63 (-0.9%, noise), per-matrix
    26.17 (+9.7%). The quality claim in its own terms, unlike the logit
    error above, which is about the numbers.
    """
    ids = hf_tokenizer(TEXT, return_tensors="pt").input_ids

    ref = perplexity(ids, weights, config)
    got = perplexity(ids, weights_int8, config)

    assert abs(got / ref - 1) < 0.02


@needs_mps
def test_int8_on_mps_matches_cpu(config, weights_int8, hf_tokenizer):
    """The int8 kernel on the GPU gives the CPU's perplexity, to 0.1%.

    Perplexity rather than max logit diff: on random ids one logit differs
    by 0.30 (float noise from the massive activation at layer 2, amplified),
    while perplexity agrees to 4 digits (23.6288 vs 23.6293) and greedy text
    is identical.
    """
    ids = hf_tokenizer(TEXT, return_tensors="pt").input_ids
    to_mps = lambda v: tuple(t.to("mps") for t in v) if isinstance(v, tuple) else v.to("mps")
    w_mps = {k: to_mps(v) for k, v in weights_int8.items()}

    cpu = perplexity(ids, weights_int8, config)
    mps = perplexity(ids.to("mps"), w_mps, config)

    assert abs(mps / cpu - 1) < 1e-3
