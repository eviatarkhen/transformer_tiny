import torch

from tiny.model import embed, run_blocks


def hf_stack(ids, hf_model, position_ids=None):
    """The oracle: what HF's last decoder layer returns, before the final norm.

    HF's hidden_states cannot supply this -- its last entry has already been
    through model.norm -- so a forward hook on layers[-1] records the output
    as the full model runs.
    """
    captured = {}
    hook = hf_model.model.layers[-1].register_forward_hook(
        lambda module, inputs, output: captured.update(out=output)
    )
    try:
        with torch.no_grad():
            hf_model(ids, position_ids=position_ids)
    finally:
        hook.remove()
    return captured["out"]


def test_matches_hf(config, weights, hf_model_eager):
    """The stream after all 28 blocks equals HF's, on random token ids.

    Random ids are realistic input here, unlike the random vectors that broke
    deep-layer parity in step 6: they go through embed, so the stack sees real
    embedding rows.

    Catches every way of running the wrong blocks -- too few (running 26 still
    predicted ' Paris', with the stream off by 5,110), the wrong order, or a
    stream that is not threaded from one block to the next.
    """
    batch, seq = 2, 7
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (batch, seq))

    got = run_blocks(embed(ids, weights), weights, config, torch.arange(seq))
    expected = hf_stack(ids, hf_model_eager)

    assert got.shape == (batch, seq, config.hidden_size)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


def test_matches_hf_at_non_default_positions(config, weights, hf_model_eager):
    """The positions argument is used, and used correctly.

    Guards the design decision that the caller supplies positions. A version
    that rebuilt arange(seq) internally would pass the test above -- which
    passes exactly arange(seq) -- and then break the KV cache in step 9, where
    one new token can sit at position 100.

    The gaps must differ from 1, not just the offset: RoPE depends only on
    relative distance, so 100..106 behaves like 0..6 (measured: 1.7e-3 apart
    after 28 blocks, float noise in the angles) and would barely move the
    output. Spacing the positions 3 apart moves it by 133.
    """
    batch, seq = 2, 7
    torch.manual_seed(1)
    ids = torch.randint(0, config.vocab_size, (batch, seq))
    positions = torch.arange(seq) * 3

    got = run_blocks(embed(ids, weights), weights, config, positions)
    expected = hf_stack(
        ids, hf_model_eager, position_ids=positions.expand(batch, -1)
    )

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)
