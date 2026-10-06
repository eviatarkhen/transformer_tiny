import pytest
import torch

from tiny.model import block

LAYERS = [0, 1, 13, 27]
SHAPES = [(1, 1), (2, 7), (1, 32)]


def run(x, weights, config, layer):
    """Our block, at positions 0..seq-1."""
    return block(x, weights, config, layer, torch.arange(x.shape[1]))


def hf_block(x, hf_model, layer):
    """The oracle: HF's own Qwen3DecoderLayer on the same input.

    Returns a bare tensor in transformers v5 -- unlike self_attn, which
    returns (output, weights).
    """
    positions = torch.arange(x.shape[1]).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(x, positions)
    with torch.no_grad():
        return hf_model.model.layers[layer](
            x, position_embeddings=(cos, sin), attention_mask=None
        )


# --- the whole block --------------------------------------------------------


@pytest.fixture(scope="module")
def real_stream(hf_model, hf_tokenizer):
    """The residual stream as each layer actually receives it on a real prompt.

    hidden_states[i] is exactly the input to layer i for i in 0..27 (verified:
    layers[i](hidden_states[i]) == hidden_states[i + 1], max diff 0.0). The last
    entry, hidden_states[28], is NOT layer 27's output -- it has already been
    through the final norm.
    """
    ids = hf_tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        return hf_model.model(ids, output_hidden_states=True).hidden_states


@pytest.mark.parametrize("layer", LAYERS)
def test_matches_hf_on_the_real_stream(config, weights, hf_model, real_stream, layer):
    """The primary test: parity with HF's Qwen3DecoderLayer, on real input.

    This is the only test that can catch the wiring bugs left in this step --
    the two norms swapped, or the MLP half given the attention half's norm --
    because every function involved stays individually correct.

    Real input rather than randn because deep layers are only well-conditioned
    in distribution. On unit-scale random input, layer 27's
    post_attention_layernorm weight (max 192) and MLP amplify attention's
    ~2.5e-5 sdpa-vs-matmul drift to 1.1e-3, while every function involved is
    exact given identical input. On the stream layer 27 actually sees -- values
    up to ~6500 in four channels -- the same comparison is 3e-5.
    """
    x = real_stream[layer]

    got = run(x, weights, config, layer)
    expected = hf_block(x, hf_model, layer)

    assert got.shape == x.shape
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("batch,seq", SHAPES)
def test_matches_hf_across_shapes(config, weights, hf_model, batch, seq):
    """Parity at layer 0 over shapes the real prompt does not cover.

    batch > 1, and seq = 1 -- the degenerate case a KV cache will rely on.
    Layer 0 because it stays well-conditioned under randn (measured 7e-7),
    which the deep layers do not -- see the test above.
    """
    torch.manual_seed(0)
    x = torch.randn(batch, seq, config.hidden_size)

    got = run(x, weights, config, 0)
    expected = hf_block(x, hf_model, 0)

    assert got.shape == x.shape
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)
