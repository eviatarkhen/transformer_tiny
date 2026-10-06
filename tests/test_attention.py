import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.qwen3.modeling_qwen3 import repeat_kv as hf_repeat_kv

from tiny.attention import (
    attend,
    attention,
    causal_mask,
    merge_heads,
    project,
    repeat_kv,
    split_heads,
)
from tiny.weights import layer_prefix

LAYERS = [0, 1, 13, 27]
SHAPES = [(1, 1), (1, 5), (1, 32), (2, 7), (3, 16)]


def run(x, weights, config, layer):
    """Our attention, at positions 0..seq-1."""
    return attention(x, weights, config, layer, torch.arange(x.shape[1]))


def hf_attention(x, hf_model, layer):
    """The oracle: HF's own Qwen3Attention on the same input."""
    attn = hf_model.model.layers[layer].self_attn
    positions = torch.arange(x.shape[1]).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(x, positions)
    with torch.no_grad():
        out, _ = attn(x, position_embeddings=(cos, sin), attention_mask=None)
    return out


# --- the pieces -------------------------------------------------------------


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("name", ["q_proj", "k_proj", "v_proj", "o_proj"])
def test_project_matches_hf(config, weights, hf_model, layer, name):
    """Every projection equals HF's nn.Linear on the same input.

    All four, not just q_proj: k_proj and v_proj are square (1024x1024), so a
    missing .T there gives the right shape and wrong numbers. q_proj and o_proj
    are rectangular and would raise instead.
    """
    attn = hf_model.model.layers[layer].self_attn
    module = getattr(attn, name)

    torch.manual_seed(0)
    x = torch.randn(1, 5, module.in_features)

    with torch.no_grad():
        expected = module(x)
    got = project(x, weights[layer_prefix(layer) + f"self_attn.{name}.weight"])

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


def test_split_heads_matches_hf_layout(config):
    """split_heads reproduces HF's .view(..., -1, head_dim).transpose(1, 2)."""
    batch, seq = 2, 5
    torch.manual_seed(0)
    x = torch.randn(batch, seq, config.num_attention_heads * config.head_dim)

    expected = x.view(batch, seq, -1, config.head_dim).transpose(1, 2)
    got = split_heads(x, config.num_attention_heads, config.head_dim)

    assert got.shape == (batch, config.num_attention_heads, seq, config.head_dim)
    assert torch.equal(got, expected)


def test_merge_heads_inverts_split_heads(config):
    """The round trip is the identity, for both head counts."""
    batch, seq = 2, 5
    for heads in (config.num_attention_heads, config.num_key_value_heads):
        torch.manual_seed(0)
        x = torch.randn(batch, seq, heads * config.head_dim)
        assert torch.equal(merge_heads(split_heads(x, heads, config.head_dim)), x)


def test_merge_heads_is_not_a_bare_reshape(config):
    """merge_heads must transpose before reshaping.

    Reshaping (batch, heads, seq, head_dim) straight to (batch, seq, hidden) is
    legal and silently interleaves heads across positions. This asserts the two
    differ, so that dropping the transpose fails loudly.
    """
    batch, seq = 1, 5
    torch.manual_seed(0)
    x = torch.randn(batch, config.num_attention_heads, seq, config.head_dim)

    bare = x.reshape(batch, seq, config.num_attention_heads * config.head_dim)
    assert not torch.equal(merge_heads(x), bare)


def test_repeat_kv_matches_hf(config):
    """Equals HF's repeat_kv, which is the expand/reshape spelling."""
    batch, seq = 2, 5
    n_rep = config.num_attention_heads // config.num_key_value_heads

    torch.manual_seed(0)
    x = torch.randn(batch, config.num_key_value_heads, seq, config.head_dim)

    assert torch.equal(repeat_kv(x, n_rep), hf_repeat_kv(x, n_rep))


def test_repeat_kv_pairs_adjacent_heads(config):
    """kv head i must serve query heads i*n_rep .. (i+1)*n_rep - 1.

    Tiling (x.repeat) gives the identical shape but pairs kv head i with query
    heads i, i + kv_heads, ... -- the wrong grouping, and invisible to a shape
    check. The second clause rules that out explicitly.
    """
    batch, seq = 1, 5
    n_rep = config.num_attention_heads // config.num_key_value_heads

    torch.manual_seed(0)
    x = torch.randn(batch, config.num_key_value_heads, seq, config.head_dim)
    out = repeat_kv(x, n_rep)

    assert out.shape == (batch, config.num_attention_heads, seq, config.head_dim)
    for i in range(config.num_key_value_heads):
        for j in range(n_rep):
            assert torch.equal(out[:, i * n_rep + j], x[:, i])

    tiled = x.repeat(1, n_rep, 1, 1)
    assert tiled.shape == out.shape
    assert not torch.equal(out, tiled)


def test_causal_mask_structure():
    """Zero on and below the diagonal, -inf strictly above.

    Indexed with a boolean rather than via .triu(1), which would zero the lower
    triangle and make an is-neginf check on the whole tensor trivially false.
    """
    seq = 6
    mask = causal_mask(seq)
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

    assert mask.shape == (seq, seq)
    assert bool(torch.isneginf(mask[future]).all())
    assert torch.equal(mask[~future], torch.zeros(int((~future).sum())))


def test_causal_mask_renormalizes_after_softmax():
    """Softmax over masked scores still sums to 1, and the future is exactly 0.

    This is why the mask is additive and applied before the softmax rather than
    zeroing weights after it: masking inside the normalization keeps each row a
    probability distribution.
    """
    seq = 6
    torch.manual_seed(0)
    scores = torch.randn(1, 2, seq, seq)
    w = torch.softmax(scores + causal_mask(seq), dim=-1)

    torch.testing.assert_close(w.sum(-1), torch.ones(1, 2, seq), atol=1e-6, rtol=0)
    assert torch.equal(w[0, 0].triu(1), torch.zeros(seq, seq))


# --- attend: the scoring block, on toy sizes and no oracle ---------------


def test_attend_scale_is_one_over_sqrt_head_dim():
    """The 1/sqrt(head_dim) scale, pinned by hand-computed softmax weights.

    head_dim=4 so the scale is exactly 0.5. Query 1 scores 2.0 against key 0 and
    1.0 against key 1; scaled those become 1.0 and 0.5. Dropping the scale, or
    using 1/head_dim instead, changes the weights and fails here -- with no
    reference to HuggingFace.
    """
    q = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    v = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    q = torch.cat([k[:1], q])[None, None]
    k = k[None, None]
    v = v[None, None]

    expected_row1 = torch.softmax(torch.tensor([2.0, 1.0]) * 0.5, dim=-1)
    got = attend(q, k, v)[0, 0, 1]

    torch.testing.assert_close(got, expected_row1 @ v[0, 0], atol=1e-6, rtol=0)


def test_attend_first_position_returns_its_own_value():
    """Position 0 can only attend to itself, so its output is exactly v[0].

    True for any q and k, since the mask leaves that row a single allowed entry
    and softmax of one element is 1.0. Catches a missing mask and a softmax over
    the wrong dimension.
    """
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn(2, 3, 5, 8)

    torch.testing.assert_close(attend(q, k, v)[:, :, 0], v[:, :, 0], atol=1e-6, rtol=0)


def test_attend_with_constant_values_returns_that_constant():
    """If every position holds the same v, the blend is that v whatever q and k are.

    Only holds if each weight row sums to 1, so this pins the softmax to dim=-1;
    a softmax over dim=-2 normalizes columns and fails.
    """
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    row = torch.randn(8)
    v = row.expand(1, 2, 6, 8).contiguous()

    torch.testing.assert_close(attend(q, k, v), v, atol=1e-6, rtol=0)


def test_attend_is_causal():
    """Changing v at the last position must not move any earlier output."""
    torch.manual_seed(2)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)

    base = attend(q, k, v)
    changed = v.clone()
    changed[:, :, -1] = torch.randn(8)

    assert torch.equal(base[:, :, :-1], attend(q, k, changed)[:, :, :-1])
    assert not torch.allclose(base[:, :, -1], attend(q, k, changed)[:, :, -1])


def test_attend_preserves_shape():
    """Output matches q's shape, including when k and v carry fewer rows of
    information than q has heads (post repeat_kv they are the same shape)."""
    torch.manual_seed(3)
    q = torch.randn(2, 4, 7, 16)
    k = torch.randn(2, 4, 7, 16)
    v = torch.randn(2, 4, 7, 16)

    assert attend(q, k, v).shape == q.shape


# --- the whole layer --------------------------------------------------------


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("batch,seq", SHAPES)
def test_matches_hf(config, weights, hf_model, layer, batch, seq):
    """The primary test: parity with HF's Qwen3Attention.

    Swept over layers and shapes because a weight-key bug can be invisible at
    layer 0, and seq=1 is the degenerate case a KV cache will rely on.
    """
    torch.manual_seed(0)
    x = torch.randn(batch, seq, config.hidden_size)

    got = run(x, weights, config, layer)
    expected = hf_attention(x, hf_model, layer)

    assert got.shape == x.shape
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


def test_matches_hf_eager(config, weights, hf_model):
    """Parity against the explicit matmul-softmax-matmul path, not sdpa.

    hf_model runs sdpa by default, a fused kernel. eager_attention_forward is
    the same arithmetic we wrote, so it is the stricter comparison.
    """
    batch, seq = 1, 6
    attn = hf_model.model.layers[0].self_attn

    torch.manual_seed(1)
    x = torch.randn(batch, seq, config.hidden_size)
    positions = torch.arange(seq)

    with torch.no_grad():
        hf_cos, hf_sin = hf_model.model.rotary_emb(x, positions.unsqueeze(0))
        shape = (batch, seq, -1, config.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
        mask = causal_mask(seq)[None, None]
        out, _ = eager_attention_forward(
            attn, q, k, v, mask, scaling=config.head_dim**-0.5
        )
        expected = attn.o_proj(out.reshape(batch, seq, -1))

    got = run(x, weights, config, 0)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


def test_is_causal(config, weights):
    """Changing the last token must not move any earlier output.

    Needs no oracle. The second clause matters: without it, a function that
    ignored x entirely would satisfy the first.
    """
    batch, seq = 1, 8
    torch.manual_seed(2)
    x = torch.randn(batch, seq, config.hidden_size)
    base = run(x, weights, config, 0)

    last_changed = x.clone()
    last_changed[:, -1] = torch.randn(config.hidden_size)
    assert torch.equal(base[:, :-1], run(last_changed, weights, config, 0)[:, :-1])

    first_changed = x.clone()
    first_changed[:, 0] = torch.randn(config.hidden_size)
    assert not torch.allclose(
        base[:, -1], run(first_changed, weights, config, 0)[:, -1], atol=1e-4
    )


def test_uses_the_positions_argument(config, weights):
    """The positions passed in must actually be used.

    Guards the design decision that positions come from the caller: an
    implementation that rebuilt them internally from arange(seq) would ignore
    these arguments and return the same output for both, breaking the KV cache
    at step 9 while every parity test above still passed.

    The gaps must differ, not just the offset. A uniform shift (0..4 vs
    100..104) leaves every relative distance unchanged, and RoPE depends only
    on those -- so it produces byte-identical output and tests nothing.
    """
    batch, seq = 1, 5
    torch.manual_seed(3)
    x = torch.randn(batch, seq, config.hidden_size)

    out_contiguous = attention(x, weights, config, 0, torch.arange(seq))
    out_spread = attention(x, weights, config, 0, torch.arange(seq) * 3)

    assert not torch.allclose(out_contiguous, out_spread, atol=1e-4)
