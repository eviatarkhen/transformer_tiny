"""Step 2: rms_norm matches HuggingFace, and behaves the way the math says.

Two kinds of test here, and both earn their place:

  - *Parity* against HF proves we compute the same thing they do. It catches
    everything, but only tells you "different", never why.
  - *Property* tests (scale invariance, unit RMS) check the maths directly, with
    no reference implementation involved. They still hold if HF changes, and a
    property failure usually points straight at which term is wrong.

Delete each `pytest.skip(...)` as you implement that test.

For comparisons use torch.testing.assert_close(ours, theirs, atol=1e-4,
rtol=1e-5) rather than a bare ==. It reports the largest difference and how
many elements disagree, which is what you actually want when a parity test
fails.
"""

import pytest
import torch

from tiny.layers import rms_norm
from tiny.weights import layer_prefix


@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_matches_hf_1024(config, weights, hf_model, scale):
    """Parity with HF's Qwen3RMSNorm at hidden_size, on random input.

    Use layer 0's input_layernorm weight from our dict, and the matching
    hf_model.model.layers[0].input_layernorm as the oracle -- they hold the
    same numbers, so any difference is in the arithmetic.

    Seed the random input (torch.manual_seed) so a failure is reproducible.
    Run the oracle under torch.no_grad().
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.hidden_size) * scale

    tiny_rms = rms_norm(x, weights[layer_prefix(0)+"input_layernorm.weight"], config.rms_norm_eps)
    with torch.no_grad():
        hf_rms = hf_model.model.layers[0].input_layernorm(x)
    torch.testing.assert_close(tiny_rms, hf_rms, atol=1e-4, rtol=1e-5)


def test_matches_hf_per_head_128(config, weights, hf_model):
    """Parity at head_dim on a 4-D tensor, against q_norm.

    Shape the input (batch, seq, num_attention_heads, head_dim) and use layer
    0's self_attn.q_norm. This is the case step 4 depends on: the same function
    must normalize the last dimension whatever the rank. If keepdim was
    forgotten, this test fails while the 2-D one passes.
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.num_attention_heads, config.head_dim)

    tiny_rms = rms_norm(x, weights[layer_prefix(0)+"self_attn.q_norm.weight"], config.rms_norm_eps)
    with torch.no_grad():
        hf_rms = hf_model.model.layers[0].self_attn.q_norm(x)
    torch.testing.assert_close(tiny_rms, hf_rms, atol=1e-4, rtol=1e-5)


def test_scale_invariance(config, weights):
    """rms_norm(c * x) == rms_norm(x) for a positive constant c.

    Dividing by the RMS cancels any constant factor, so magnitude is discarded
    and only the direction survives. Needs no oracle: it follows from the
    definition.

    Not exact in floating point -- use assert_close, not ==. eps also breaks
    the invariance very slightly, so keep c modest (2, 10) rather than 1e8.
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.num_attention_heads, config.head_dim)

    prefix = layer_prefix(0)
    rms = rms_norm(x, weights[prefix+"self_attn.q_norm.weight"], config.rms_norm_eps)
    rms_scaled = rms_norm(x*7, weights[prefix+"self_attn.q_norm.weight"], config.rms_norm_eps)
    
    torch.testing.assert_close(rms, rms_scaled, atol=1e-4, rtol=1e-5)


def test_unit_rms_when_weight_is_ones(config):
    """With weight = ones, each row of the output has RMS ~= 1.

    This is the operation's whole purpose stated directly: whatever magnitude
    went in, a standard one comes out. Compute sqrt(mean(out**2)) over the last
    dimension and compare to 1.

    Do not use a weight from the checkpoint here -- the learned scale is exactly
    what breaks unit RMS.
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.hidden_size)
    out = rms_norm(x, torch.ones(config.hidden_size), config.rms_norm_eps)
    torch.testing.assert_close(out.pow(2).mean(-1).sqrt(), torch.ones(1, seq), atol=1e-4, rtol=1e-5)


def test_zero_input_is_finite(config, weights):
    """An all-zeros input produces zeros, not NaN.

    Without eps this is 0/0. It is the only reason eps exists, and it is one
    line to prove. torch.isfinite(...).all()
    """
    out = rms_norm(torch.zeros(1, config.hidden_size), weights[layer_prefix(0)+"input_layernorm.weight"], config.rms_norm_eps)
    assert torch.isfinite(out).all()
