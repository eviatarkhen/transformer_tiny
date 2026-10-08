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


# One slot per layer: {"k": buffer, "v": buffer, "len": slots filled so far}.
Cache = list[dict]


def allocate_cache(
    config: Config, batch: int, max_len: int, like: torch.Tensor
) -> Cache:
    """Allocate every layer's key and value buffers once, at full size.

    Args:
        config: architecture -- layers, kv heads, head_dim.
        batch: prompts in the batch, 1 for greedy.
        max_len: the most tokens the cache will ever hold. greedy passes
            prompt length + max_new_tokens, one spare: the last token it
            generates is never fed back through the model.
        like: any tensor with the model's dtype and device, e.g. the
            embedding table. The buffers copy both, as causal_mask does.

    Returns:
        num_hidden_layers slots, each with k and v buffers of shape
        (batch, kv_heads, max_len, head_dim) and len 0. Held in full from the
        start: 229 KB per token for this model in float32, 115 KB in bf16.
    """
    shape = (batch, config.num_key_value_heads, max_len, config.head_dim)
    return [
        {"k": like.new_zeros(shape), "v": like.new_zeros(shape), "len": 0}
        for _ in range(config.num_hidden_layers)
    ]


def update_cache(
    cache: Cache,
    layer: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write this turn's keys and values into the layer's buffers, in place.

    Each call copies only the new tokens. The previous design joined old and
    new with torch.cat, which allocates a fresh tensor and copies the whole
    history every turn: 688 MB per turn at 3,000 tokens, to add 229 KB.

    Args:
        cache: from allocate_cache. MODIFIED IN PLACE: calling twice with the same
            k_new stores those tokens twice.
        layer: which slot to write.
        k_new, v_new: (batch, kv_heads, new, head_dim), the keys and values
            of the tokens in this call -- the whole prompt at prefill, one
            token after.

    Returns:
        (k, v), each (batch, kv_heads, filled, head_dim): views of the filled
        part of the buffers, old tokens first. A view, not a copy.

    Notes:
        The write position is a Python int kept in the slot, not read from
        positions: reading a tensor's value with .item() makes the CPU wait
        for the GPU, 28 times a turn on MPS.
    """
    slot = cache[layer]
    start = slot["len"]
    end = start + k_new.shape[-2]
    if end > slot["k"].shape[-2]:
        raise ValueError(f"cache holds {slot['k'].shape[-2]} tokens, asked for {end}")
    slot["k"][:, :, start:end] = k_new
    slot["v"][:, :, start:end] = v_new
    slot["len"] = end
    return slot["k"][:, :, :end], slot["v"][:, :, :end]


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
    """Causal scaled dot-product attention, grouped-query.

    Args:
        q: (batch, heads, q_len, head_dim).
        k, v: (batch, kv_heads, k_len, head_dim), with heads a multiple of
            kv_heads. k_len >= q_len: with a cache, k and v also hold every
            earlier token, and q's rows are the last q_len of them.

    Returns:
        (batch, heads, q_len, head_dim).

    Notes:
        Under GQA each kv head serves n_rep = heads // kv_heads adjacent query
        heads: kv head i serves query heads i*n_rep .. (i+1)*n_rep - 1. Rather
        than copying k and v up to q's head count (the old repeat_kv, which
        rewrote the whole cache, doubled, every layer, every turn), q is
        regrouped: (batch, heads, ...) viewed as (batch, kv_heads, n_rep, ...)
        is exactly that adjacent pairing. Tiling instead -- (n_rep, kv_heads)
        -- would give the same shapes and pair the wrong heads.

        For the mask the scores are viewed as (batch, kv_heads, n_rep, q_len,
        k_len): the last two dimensions are still (q_len, k_len), so
        causal_mask needs no change.

        The scale is read off q's own last dimension rather than passed in, so
        this needs no config and can be unit tested on toy sizes.

        The softmax runs in float32 and is cast back, as in HF's eager path.
        Measured redundant: PyTorch's bf16 softmax already accumulates in
        float32 internally, bitwise identical on CPU and MPS. Kept explicit
        because it is the decision, and a backend that did not do it
        silently would lose small weights to rounding.
    """
    batch, heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    n_rep = heads // kv_heads

    # The n_rep query heads of a group stacked as rows: one plain matmul per
    # kv head. No broadcasting -- matmul would implement it by copying k.
    q = q.reshape(batch, kv_heads, n_rep * q_len, head_dim)
    scores = (q @ k.transpose(-2, -1)) * head_dim ** -0.5

    scores = scores.view(batch, kv_heads, n_rep, q_len, -1)
    scores = scores + causal_mask(scores)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    weights = weights.view(batch, kv_heads, n_rep * q_len, -1)
    return (weights @ v).view(batch, heads, q_len, head_dim)


def attention(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    layer: int,
    positions: torch.Tensor,
    cache: Cache | None = None,
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
        cache: optional, from allocate_cache. None means no cache. Given, this
            layer's keys and values are stored after RoPE, at kv_heads: a
            key's rotation depends only on its own position, so it is final
            once made, and 8 heads rather than 16 halve the memory.

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

    return project(merge_heads(attend(q, k, v)), w[p + "o_proj.weight"])
