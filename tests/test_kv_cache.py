import pytest
import torch

from tiny.attention import attend, causal_mask, update_cache
from tiny.model import forward

# --- update_cache: pure bookkeeping, toy tensors, no model ------------------


def test_update_cache_fills_an_empty_slot():
    """The first call for a layer stores k, v as given, and touches no other slot.

    k and v hold different numbers, so storing them swapped fails here.
    """
    cache = [None, None, None]
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    got_k, got_v = update_cache(cache, 1, k, v)

    assert torch.equal(got_k, k) and torch.equal(got_v, v)
    assert cache[1][0] is got_k and cache[1][1] is got_v
    assert cache[0] is None and cache[2] is None


def test_update_cache_appends_new_tokens_after_the_old():
    """A second call joins along the token axis, old tokens first.

    Joining the other way round (new first) keeps every shape and every number,
    and would silently pair each key with the wrong position's mask column.
    """
    cache = [None]
    k_old, v_old = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k_new, v_new = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    update_cache(cache, 0, k_old, v_old)

    got_k, got_v = update_cache(cache, 0, k_new, v_new)

    assert got_k.shape == (1, 2, 4, 4)
    assert torch.equal(got_k[:, :, :3], k_old) and torch.equal(got_k[:, :, 3:], k_new)
    assert torch.equal(got_v[:, :, :3], v_old) and torch.equal(got_v[:, :, 3:], v_new)
    assert torch.equal(cache[0][0], got_k) and torch.equal(cache[0][1], got_v)


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
    cache = [None] * config.num_hidden_layers
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
    """After 4 + 1 tokens every layer stores exactly 5 positions, at 8 heads.

    The length catches a token stored twice -- parity alone would not, if the
    duplicate happened to be harmless. The head count pins the design decision
    to store before repeat_kv (8 heads, not 16).
    """
    torch.manual_seed(1)
    ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        _, cache = cached_logits(ids, weights, config, [4, 1])

    expected = (1, config.num_key_value_heads, 5, config.head_dim)
    assert len(cache) == config.num_hidden_layers
    for k, v in cache:
        assert k.shape == expected and v.shape == expected
