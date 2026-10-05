import torch

from tiny.model import lm_head


def hf_lm_head(x, hf_model):
    """The oracle: HF's lm_head, an nn.Linear tied to the embedding table."""
    with torch.no_grad():
        return hf_model.lm_head(x)


def test_matches_hf(config, weights, hf_model):
    """lm_head equals HF's head, at every position.

    The shape assertion pins "every position, not just the last": generation
    only needs the last row, but slicing is the caller's job, and the full
    output is what lets every position be compared at once.
    """
    batch, seq = 2, 7
    torch.manual_seed(0)
    x = torch.randn(batch, seq, config.hidden_size)

    got = lm_head(x, weights)

    assert got.shape == (batch, seq, config.vocab_size)
    torch.testing.assert_close(got, hf_lm_head(x, hf_model), atol=1e-4, rtol=1e-5)


def test_reads_the_tied_embedding_table(config, weights, hf_model):
    """lm_head works on a checkpoint that does not store lm_head.weight.

    Guards the design decision recorded in lm_head's docstring. This file
    stores the tied table twice, identically, so the parity test above cannot
    tell which key is read -- reading lm_head.weight passes it. Many tied
    checkpoints omit the duplicate, and on those that version raises KeyError.
    """
    without_duplicate = {k: v for k, v in weights.items() if k != "lm_head.weight"}
    torch.manual_seed(1)
    x = torch.randn(1, 3, config.hidden_size)

    got = lm_head(x, without_duplicate)

    torch.testing.assert_close(got, hf_lm_head(x, hf_model), atol=1e-4, rtol=1e-5)
