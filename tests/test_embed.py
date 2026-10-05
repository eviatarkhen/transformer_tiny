import torch

from tiny.model import embed


def hf_embed(ids, hf_model):
    """The oracle: HF's own nn.Embedding lookup on the same ids."""
    with torch.no_grad():
        return hf_model.model.embed_tokens(ids)


def test_matches_hf(config, weights, hf_model):
    """embed returns exactly HF's embedding for every token id.

    Exact equality, not the project's usual 1e-4 tolerance: a lookup does no
    arithmetic, so there is no rounding noise to tolerate, and a tolerance
    would only weaken the claim.

    Random ids drawn from the whole vocabulary, not a small range: a
    transposed table indexes fine for ids below 1024 and only fails above it.
    batch != seq so that swapping the two axes changes the shape and the shape
    assertion names the bug.
    """
    batch, seq = 2, 7
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (batch, seq))

    got = embed(ids, weights)

    assert got.shape == (batch, seq, config.hidden_size)
    assert torch.equal(got, hf_embed(ids, hf_model))
