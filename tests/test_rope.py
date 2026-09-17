"""Step 3: RoPE matches HuggingFace, and has the property it exists for.

The oracle:

    position_ids = torch.arange(seq).unsqueeze(0)          # (1, seq)
    dummy = torch.zeros(1, seq, config.hidden_size)        # only read for dtype
    cos, sin = hf_model.model.rotary_emb(dummy, position_ids)   # (1, seq, 128)

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

Ours returns (seq, head_dim), so compare against hf_cos[0].

Delete each pytest.skip as you go.
"""

import pytest
import torch

from tiny.layers import apply_rope, rope_cos_sin


def test_cos_sin_matches_hf(config, hf_model):
    """Our angle table equals HF's rotary_emb output."""
    pytest.skip("TODO")


def test_apply_rope_matches_hf(config, hf_model):
    """apply_rope on a (batch, heads, seq, head_dim) tensor equals HF's
    apply_rotary_pos_emb, using HF's own cos/sin so only the rotation is
    under test."""
    pytest.skip("TODO")


def test_pairs_share_an_angle(config):
    """cos[:, i] == cos[:, i + head_dim // 2], and likewise for sin.

    The two dimensions of a rotated pair must be given the same angle. This is
    what the cat((angles, angles)) duplication is for, and it is the thing that
    breaks if you build the table with arange(head_dim) instead of
    arange(0, head_dim, 2).
    """
    pytest.skip("TODO")


def test_position_zero_is_identity(config):
    """At position 0 the angle is 0, so cos=1, sin=0 and the vector is
    unchanged. Catches an off-by-one in how positions index the table."""
    pytest.skip("TODO")


def test_preserves_norm(config):
    """A rotation never changes a vector's length, at any position.

    Check per head: x.norm(dim=-1) before and after. Also worth checking one
    pair on its own, since a bug that rotates pair (i, i+1) instead of
    (i, i+64) still preserves the total norm -- the whole-vector check alone
    would not catch it.
    """
    pytest.skip("TODO")


def test_depends_only_on_relative_position(config):
    """The point of the whole exercise.

        dot( rope(q, m), rope(k, n) )  ==  dot( rope(q, m+d), rope(k, n+d) )

    Absolute positions cancel; only the gap m-n survives. Take one random q and
    one random k, rotate them at (3, 7) and again at (13, 17), and compare the
    dot products.

    Needs no oracle: it follows from the construction, and it is the property
    the design exists to produce.
    """
    pytest.skip("TODO")
