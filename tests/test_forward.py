import torch

from tiny.model import forward


def hf_logits(ids, hf_model, position_ids=None):
    """The oracle: HF's full Qwen3ForCausalLM on the same ids."""
    with torch.no_grad():
        return hf_model(ids, position_ids=position_ids).logits


def test_matches_hf(config, weights, hf_model_eager):
    """The whole model equals HF's, logit for logit, at every position.

    The end-to-end test, and the only one that can catch wiring between the
    four pieces -- the final norm skipped, the pieces called in the wrong
    order -- since each piece is correct on its own and tested on its own.

    Against eager, not the default sdpa: on this input sdpa drifts to 1.1e-4
    over 28 blocks, past the 1e-4 tolerance with no bug present. Eager is
    exact (0.0).
    """
    batch, seq = 2, 7
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (batch, seq))

    got = forward(ids, weights, config, torch.arange(seq))

    assert got.shape == (batch, seq, config.vocab_size)
    torch.testing.assert_close(got, hf_logits(ids, hf_model_eager), atol=1e-4, rtol=1e-5)


def test_matches_hf_at_non_default_positions(config, weights, hf_model_eager):
    """forward passes its positions argument through, not arange(seq).

    The test above passes exactly arange(seq), so a forward that ignored its
    argument and rebuilt the positions would pass it -- and then break the KV
    cache in step 9. Gaps of 3 rather than an offset, because RoPE depends only
    on relative distance (see test_run_blocks).
    """
    batch, seq = 2, 7
    torch.manual_seed(1)
    ids = torch.randint(0, config.vocab_size, (batch, seq))
    positions = torch.arange(seq) * 3

    got = forward(ids, weights, config, positions)
    expected = hf_logits(ids, hf_model_eager, position_ids=positions.expand(batch, -1))

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)
