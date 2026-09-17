import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from tiny.layers import apply_rope, rope_cos_sin


def test_cos_sin_matches_hf(config, hf_model):
    """Our angle table equals HF's rotary_emb output."""
    seq = 5
    positions = torch.arange(seq)
    hf_cos, hf_sin = hf_model.model.rotary_emb(torch.zeros(1, seq, config.hidden_size),
                                               positions.unsqueeze(0))
    cos, sin = rope_cos_sin(positions, config.head_dim, config.rope_theta)
    torch.testing.assert_close(cos, hf_cos[0], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(sin, hf_sin[0], atol=1e-4, rtol=1e-5)


def test_apply_rope_matches_hf(config, hf_model):
    """apply_rope on a (batch, heads, seq, head_dim) tensor equals HF's
    apply_rotary_pos_emb, using HF's own cos/sin so only the rotation is
    under test."""
    seq = 5
    positions = torch.arange(seq)
    dummy = torch.zeros(1, seq, config.hidden_size)
    hf_cos, hf_sin = hf_model.model.rotary_emb(dummy, positions.unsqueeze(0))

    torch.manual_seed(0)
    q = torch.randn(1, config.num_attention_heads, seq, config.head_dim)
    k = torch.randn(1, config.num_key_value_heads, seq, config.head_dim)
    q_out, _ = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    x_out = apply_rope(q, hf_cos[0], hf_sin[0])
    torch.testing.assert_close(x_out, q_out, atol=1e-4, rtol=1e-5)


def test_pairs_share_an_angle(config):
    """cos[:, i] == cos[:, i + head_dim // 2], and likewise for sin.

    The two dimensions of a rotated pair must be given the same angle. This is
    what the cat((angles, angles)) duplication is for, and it is the thing that
    breaks if you build the table with arange(head_dim) instead of
    arange(0, head_dim, 2).
    """
    seq = 5
    cos, sin = rope_cos_sin(torch.arange(seq), config.head_dim, config.rope_theta)
    h = config.head_dim // 2
    assert torch.equal(cos[:, :h], cos[:, h:])
    assert torch.equal(sin[:, :h], sin[:, h:])


def test_position_zero_is_identity(config):
    """At position 0 the angle is 0, so cos=1, sin=0 and the vector is
    unchanged. Catches an off-by-one in how positions index the table."""
    seq = 5
    cos, sin = rope_cos_sin(torch.arange(seq), config.head_dim, config.rope_theta)

    torch.manual_seed(0)
    x = torch.randn(1, config.num_attention_heads, seq, config.head_dim)
    out = apply_rope(x, cos, sin)

    assert torch.equal(out[:, :, 0], x[:, :, 0])


def test_preserves_norm(config):
    """A rotation never changes a vector's length, at any position.

    Checked per pair, not just per head: a bug that rotated (i, i+1) instead of
    (i, i+64) would still preserve the whole-vector norm, so the per-head check
    alone would miss it.
    """
    seq = 5
    half = config.head_dim // 2
    cos, sin = rope_cos_sin(torch.arange(seq), config.head_dim, config.rope_theta)

    torch.manual_seed(0)
    x = torch.randn(1, config.num_attention_heads, seq, config.head_dim)
    out = apply_rope(x, cos, sin)

    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)

    def pair_norms(t):
        return (t[..., :half].pow(2) + t[..., half:].pow(2)).sqrt()

    torch.testing.assert_close(pair_norms(out), pair_norms(x), atol=1e-5, rtol=1e-5)


def test_depends_only_on_relative_position(config):
    """The property the whole design exists to produce.

        dot( rope(q, m), rope(k, n) )  depends only on  m - n

    A query at position 3 attending to a key at position 7 sees exactly what a
    query at 13 sees looking at 17. That is what makes "four tokens back" a
    reusable pattern instead of something the model must relearn at every
    absolute position.

    Needs no oracle -- it follows from the construction.
    """
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, config.head_dim)
    k = torch.randn(1, 1, 1, config.head_dim)

    def score(m: int, n: int) -> torch.Tensor:
        """Attention score between q at position m and k at position n."""
        cos_m, sin_m = rope_cos_sin(
            torch.tensor([m]), config.head_dim, config.rope_theta
        )
        cos_n, sin_n = rope_cos_sin(
            torch.tensor([n]), config.head_dim, config.rope_theta
        )
        return (apply_rope(q, cos_m, sin_m) * apply_rope(k, cos_n, sin_n)).sum()

    # Same gap of 4, shifted by 10: identical score.
    torch.testing.assert_close(score(3, 7), score(13, 17), atol=1e-4, rtol=1e-4)

    # A different gap must give a different score -- without this, an apply_rope
    # that ignored position entirely and returned x would pass the assertion
    # above.
    assert not torch.isclose(score(3, 7), score(3, 8), atol=1e-4)
