import torch

from tiny.config import Config
from tiny.layers import apply_rope, rms_norm, project, rope_cos_sin
from tiny.weights import layer_prefix


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


def update_cache(
    cache: list[tuple[torch.Tensor, torch.Tensor] | None],
    layer: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join this turn's keys and values to the layer's stored ones.

    Args:
        cache: one slot per layer. A slot is None until the layer's first call,
            then (k, v) for every token seen so far. MODIFIED IN PLACE: calling
            twice with the same k_new stores those tokens twice.
        layer: which slot to read and replace.
        k_new, v_new: (batch, heads, new, head_dim), the keys and values of the
            tokens in this call -- the whole prompt at prefill, one token after.

    Returns:
        (k, v), each (batch, heads, stored + new, head_dim): the old tokens
        first, then the new ones. The same pair is what the slot now holds.
    """
    if cache[layer] is not None:
        k_old, v_old = cache[layer]
        k_new = torch.cat([k_old, k_new], dim=-2)
        v_new = torch.cat([v_old, v_new], dim=-2)
    cache[layer] = (k_new, v_new)
    return k_new, v_new


def causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """(q_len, k_len) additive mask for scores (..., q_len, k_len): 0 where
    attending is allowed, -inf where not.

    Added to the scores *before* the softmax, so forbidden positions contribute
    nothing to the denominator and the allowed ones renormalize to sum to 1.

    The q rows are the *last* q_len of the k_len positions. Without a cache the
    two are equal and this is the familiar square with -inf above the diagonal.
    With one, q is 1 new token against k_len stored ones: the diagonal shifts
    right by k_len - q_len, and a single row has nothing to mask at all.

    Takes the scores themselves, not their lengths, so the mask inherits their
    dtype and device through new_full and cannot be built wrong: a float32
    mask silently promotes bf16 scores, and a CPU mask cannot be added to MPS
    scores at all. -inf exists in bf16, so the mask stays exact.
    """
    q_len, k_len = scores.shape[-2:]
    return scores.new_full((q_len, k_len), float("-inf")).triu(k_len - q_len + 1)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal scaled dot-product attention.

    Args:
        q: (batch, heads, q_len, head_dim).
        k, v: (batch, heads, k_len, head_dim), already expanded to q's head
            count by repeat_kv. k_len >= q_len: with a cache, k and v also hold
            every earlier token, and q's rows are the last q_len of them.

    Returns:
        (batch, heads, q_len, head_dim).

    Notes:
        The scale is read off q's own last dimension rather than passed in, so
        this needs no config and can be unit tested on toy sizes.

        The softmax runs in float32 and is cast back, as in HF's eager path.
        Measured redundant: PyTorch's bf16 softmax already accumulates in
        float32 internally, bitwise identical on CPU and MPS. Kept explicit
        because it is the decision, and a backend that did not do it
        silently would lose small weights to rounding.
    """
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    scores = scores + causal_mask(scores)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return weights @ v


def attention(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    layer: int,
    positions: torch.Tensor,
    cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
) -> torch.Tensor:
    """Grouped-query causal self-attention for one block.

    Args:
        x: (batch, seq, hidden_size).
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture; every shape below is derived from it.
        layer: which block's weights to read.
        positions: (seq,) integer position of each token in x, on x's
            device. Passed in because once a KV cache exists only the caller
            knows them -- seq=1 can mean position 100. The rotation tables
            are built from them here: they are this function's detail, and
            recomputing a (seq, head_dim) table per layer is noise next to
            the matmuls.
        cache: optional, see update_cache. None means no cache, exactly as
            before. Given, this layer's keys and values are stored after RoPE
            and before repeat_kv: a key's rotation depends only on its own
            position, so it is final once made, and storing 8 heads rather
            than 16 halves the memory.

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

    cos, sin = rope_cos_sin(positions, config.head_dim, config.rope_theta)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if cache is not None:
        k, v = update_cache(cache, layer, k, v)

    n_rep = heads // kv_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    return project(merge_heads(attend(q, k, v)), w[p + "o_proj.weight"])
