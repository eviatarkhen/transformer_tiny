import torch

from tiny.config import Config
from tiny.layers import apply_rope, rms_norm
from tiny.weights import layer_prefix


def project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply one of the checkpoint's bias-free linear layers.

    Weights are stored (out_features, in_features), so the matmul needs the
    transpose. k_proj and v_proj are square, so omitting it there yields the
    right shape and wrong numbers rather than an error.
    """
    return x @ weight.T


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """(batch, seq, heads * head_dim) -> (batch, heads, seq, head_dim)."""
    batch, seq, _ = x.shape
    return x.reshape(batch, seq, heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(batch, heads, seq, head_dim) -> (batch, seq, heads * head_dim).

    The exact inverse of split_heads. The transpose is not optional: reshaping
    straight from (batch, heads, seq, head_dim) is legal and silently
    interleaves heads across positions.
    """
    batch, heads, seq, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, seq, heads * head_dim)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, kv_heads, seq, head_dim) -> (batch, kv_heads * n_rep, seq, head_dim).

    Under GQA each key/value head is shared by n_rep query heads. The copies are
    adjacent, so kv head i serves query heads i*n_rep .. (i + 1)*n_rep - 1.
    Tiling instead (repeat) gives the same shape and pairs the wrong heads.
    """
    return x.repeat_interleave(n_rep, dim=1)


def causal_mask(seq: int) -> torch.Tensor:
    """(seq, seq) additive mask: 0 where attending is allowed, -inf above the diagonal.

    Added to the scores *before* the softmax, so forbidden positions contribute
    nothing to the denominator and the allowed ones renormalize to sum to 1.
    """
    return torch.full((seq, seq), float("-inf")).triu(1)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal scaled dot-product attention.

    Args:
        q, k, v: (batch, heads, seq, head_dim), with k and v already expanded to
            q's head count by repeat_kv.

    Returns:
        (batch, heads, seq, head_dim).

    Notes:
        The scale is read off q's own last dimension rather than passed in, so
        this needs no config and can be unit tested on toy sizes.

        The mask is square because q and k are the same length here. A KV cache
        breaks that assumption -- q_len becomes 1 while k_len stays long -- so
        this is the line step 9 has to revisit.
    """
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    weights = torch.softmax(scores + causal_mask(q.shape[-2]), dim=-1)
    return weights @ v


def attention(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    layer: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Grouped-query causal self-attention for one block.

    Args:
        x: (batch, seq, hidden_size).
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture; every shape below is derived from it.
        layer: which block's weights to read.
        cos, sin: (seq, head_dim) rotation tables from rope_cos_sin. Passed in
            rather than built here because all 28 layers share one table, and
            because once a KV cache exists only the caller knows the true
            positions -- seq=1 can mean position 100.

    Returns:
        (batch, seq, hidden_size), the same shape as x.
    """
    p = layer_prefix(layer) + "self_attn."
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads

    q = split_heads(project(x, w[p + "q_proj.weight"]), heads, config.head_dim)
    k = split_heads(project(x, w[p + "k_proj.weight"]), kv_heads, config.head_dim)
    v = split_heads(project(x, w[p + "v_proj.weight"]), kv_heads, config.head_dim)

    q = rms_norm(q, w[p + "q_norm.weight"], config.rms_norm_eps)
    k = rms_norm(k, w[p + "k_norm.weight"], config.rms_norm_eps)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    n_rep = heads // kv_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    return project(merge_heads(attend(q, k, v)), w[p + "o_proj.weight"])
