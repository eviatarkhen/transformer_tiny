from types import SimpleNamespace

import pytest
import torch

from tiny.attention import attend, causal_mask, allocate_cache, update_cache
from tiny.model import forward

# --- update_cache: pure bookkeeping, toy tensors, no model ------------------

# allocate_cache reads only these three fields, so toy sizes need no real Config.
TOY = SimpleNamespace(num_hidden_layers=3, num_key_value_heads=2, head_dim=4)


def test_update_cache_fills_an_empty_slot():
    """The first call writes k, v at the start of the layer's buffers and
    returns a view of exactly that much; no other layer is touched.

    k and v hold different numbers, so storing them swapped fails here.
    """
    cache = allocate_cache(TOY, 1, 8, like=torch.zeros(1))
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    got_k, got_v = update_cache(cache, 1, k, v)

    assert torch.equal(got_k, k) and torch.equal(got_v, v)
    assert cache[1]["len"] == 3
    assert cache[0]["len"] == 0 and cache[2]["len"] == 0


def test_update_cache_appends_in_place_without_copying():
    """A second call writes after the old tokens, old first, into the SAME
    buffer: its address never changes, which is the whole point of
    preallocating -- torch.cat would allocate and copy the history.

    Writing at slot 0 again (a forgotten len) overwrites the old tokens and
    fails the first equality; new-before-old fails both.
    """
    cache = allocate_cache(TOY, 1, 8, like=torch.zeros(1))
    address = cache[0]["k"].data_ptr()
    k_old, v_old = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k_new, v_new = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    update_cache(cache, 0, k_old, v_old)

    got_k, got_v = update_cache(cache, 0, k_new, v_new)

    assert got_k.shape == (1, 2, 4, 4)
    assert torch.equal(got_k[:, :, :3], k_old) and torch.equal(got_k[:, :, 3:], k_new)
    assert torch.equal(got_v[:, :, :3], v_old) and torch.equal(got_v[:, :, 3:], v_new)
    assert cache[0]["k"].data_ptr() == address and got_k.data_ptr() == address


def test_update_cache_refuses_to_overflow():
    """Past max_len the write would land in an empty slice and fail with an
    opaque shape error; it says what happened instead."""
    cache = allocate_cache(TOY, 1, 4, like=torch.zeros(1))
    update_cache(cache, 0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

    with pytest.raises(ValueError, match="holds 4 tokens"):
        update_cache(cache, 0, torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))


# --- causal_mask and attend when q is shorter than k ------------------------


def test_causal_mask_rectangular():
    """3 new queries against 6 keys: each query is one of the LAST 3 positions.

    Query row i sits at position 3 + i, so it may see keys 0..3+i. The single-
    row case has nothing to mask, and equal lengths give the familiar square.
    """
    inf = float("-inf")
    expected = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, inf, inf],
        [0.0, 0.0, 0.0, 0.0, 0.0, inf],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    assert torch.equal(causal_mask(torch.zeros(3, 6)), expected)
    assert torch.equal(causal_mask(torch.zeros(1, 6)), torch.zeros(1, 6))
    assert torch.equal(causal_mask(torch.zeros(3, 3)), expected[:, 3:])


@pytest.mark.parametrize("q_len", [1, 3])
def test_attend_short_q_equals_the_last_rows_of_full_attention(q_len):
    """Attending with only the last q_len queries gives the last q_len rows.

    This is the KV cache's promise at the level of attend. q_len=3 is the case
    that needs the shifted mask: with one row there is nothing to mask, so the
    old square-mask-plus-broadcast passed by luck.
    """
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)

    full = attend(q, k, v)
    got = attend(q[:, :, -q_len:], k, v)

    assert got.shape == (1, 2, q_len, 8)
    torch.testing.assert_close(got, full[:, :, -q_len:], atol=1e-6, rtol=0)


# --- forward with a cache, on the real model --------------------------------


def cached_logits(ids, weights, config, chunks):
    """Feed ids through forward in consecutive chunks, sharing one cache.

    Returns the concatenated logits and the cache.
    """
    cache = allocate_cache(config, 1, ids.shape[1], like=weights["model.embed_tokens.weight"])
    rows, start = [], 0
    for size in chunks:
        end = start + size
        rows.append(forward(ids[:, start:end], weights, config, torch.arange(start, end), cache))
        start = end
    return torch.cat(rows, dim=1), cache


def test_cached_forward_matches_uncached(config, weights):
    """Prefill, then a 2-token chunk, then one token at a time: every row of
    logits equals the one-shot forward over the whole sequence.

    Catches a new token fed at the wrong position, keys stored before RoPE,
    old/new tokens joined in the wrong order, and a missing or wrong mask in
    the 2-token chunk. Measured drift with no bug is 5e-5: the matmuls sum in
    a different order for differently shaped inputs.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        expected = forward(ids, weights, config, torch.arange(8))
        got, _ = cached_logits(ids, weights, config, [4, 2, 1, 1])

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


def test_cache_holds_every_token_once_with_kv_heads(config, weights):
    """After 4 + 1 tokens every layer has filled exactly 5 slots, at 8 heads.

    The length catches a token stored twice. The head count pins the design
    decision to store k and v at kv_heads (8), not copied up to q's 16.
    """
    torch.manual_seed(1)
    ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        _, cache = cached_logits(ids, weights, config, [4, 1])

    expected = (1, config.num_key_value_heads, 5, config.head_dim)
    assert len(cache) == config.num_hidden_layers
    for slot in cache:
        assert slot["len"] == 5
        assert slot["k"].shape == expected and slot["v"].shape == expected
