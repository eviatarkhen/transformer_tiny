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
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.hidden_size) * scale

    tiny_out = rms_norm(x, weights[layer_prefix(0)+"input_layernorm.weight"], config.rms_norm_eps)
    with torch.no_grad():
        hf_out = hf_model.model.layers[0].input_layernorm(x)
    torch.testing.assert_close(tiny_out, hf_out, atol=1e-4, rtol=1e-5)


def test_matches_hf_per_head_128(config, weights, hf_model):
    """Parity at head_dim on a 4-D tensor, against q_norm.

    This is the case step 4 depends on: the same function
    must normalize the last dimension whatever the rank. If keepdim was
    forgotten, this test fails while the 2-D one passes.
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.num_attention_heads, config.head_dim)

    tiny_out = rms_norm(x, weights[layer_prefix(0)+"self_attn.q_norm.weight"], config.rms_norm_eps)
    with torch.no_grad():
        hf_out = hf_model.model.layers[0].self_attn.q_norm(x)
    torch.testing.assert_close(tiny_out, hf_out, atol=1e-4, rtol=1e-5)


def test_scale_invariance(config, weights):
    """rms_norm(c * x) == rms_norm(x) for a positive constant c.

    Dividing by the RMS cancels any constant factor, so magnitude is discarded
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.num_attention_heads, config.head_dim)

    norm_weights = weights[layer_prefix(0)+"self_attn.q_norm.weight"]
    out = rms_norm(x, norm_weights, config.rms_norm_eps)
    out_scaled = rms_norm(x*7, norm_weights, config.rms_norm_eps)
    torch.testing.assert_close(out, out_scaled, atol=1e-4, rtol=1e-5)


def test_unit_rms_when_weight_is_ones(config):
    """With weight = ones, each row of the output has RMS ~= 1.

    This is the operation's whole purpose stated directly: whatever magnitude
    went in, a standard one comes out.
    """
    seq = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq, config.hidden_size)
    out = rms_norm(x, torch.ones(config.hidden_size), config.rms_norm_eps)
    torch.testing.assert_close(out.pow(2).mean(-1).sqrt(), torch.ones(1, seq), atol=1e-4, rtol=1e-5)


def test_zero_input_is_finite(config, weights):
    """An all-zeros input produces zeros, not NaN.

    Without eps this is 0/0. It is the only reason eps exists
    """
    out = rms_norm(torch.zeros(1, config.hidden_size), weights[layer_prefix(0)+"input_layernorm.weight"], config.rms_norm_eps)
    assert torch.isfinite(out).all()