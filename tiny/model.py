import torch

from tiny.attention import attention
from tiny.config import Config
from tiny.layers import project, rms_norm, rope_cos_sin
from tiny.mlp import mlp
from tiny.weights import layer_prefix


def block(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    layer: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """One decoder block: an attention half, then an MLP half.

    Args:
        x: (batch, seq, hidden_size).
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture.
        layer: which block's weights to read.
        cos, sin: (seq, head_dim) rotation tables, forwarded to attention.

    Returns:
        (batch, seq, hidden_size), the same shape as x.

    Notes:
        post_attention_layernorm is NOT applied to attention's output. It is
        the pre-norm of the MLP half; the name only says it comes after the
        attention half in source order.
    """
    p = layer_prefix(layer)
    eps = config.rms_norm_eps

    normed = rms_norm(x, w[p + "input_layernorm.weight"], eps)
    x = x + attention(normed, w, config, layer, cos, sin)

    normed = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
    return x + mlp(normed, w, layer)


def embed(
    input_ids: torch.Tensor,
    w: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return one embedding vector per token id.

    Args:
        input_ids: (batch, seq) tokens input ids.
        w: the full weight dict, keyed exactly as the checkpoint is.

    Returns:
        (batch, seq, hidden_size) initial state
    """
    return w["model.embed_tokens.weight"][input_ids]


def run_blocks(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Run all layers blocks

    Args:
        x: (batchm seq, hidden_size) initial hidden state.
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture.
        positions: (seq,) integer positions. A tensor rather than a length, so
            incremental decoding can pass e.g. tensor([100]) with no offset
            parameter.

    Returns:
       (batch, seq, hidden_size) residual state
    """
    cos, sin = rope_cos_sin(positions, config.head_dim, config.rope_theta)
    for layer in range(config.num_hidden_layers):
        x = block(x, w, config, layer, cos, sin)

    return x


def final_norm(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
) -> torch.Tensor:
    """Normalize the stream once more before the LM head.
    It leaves the last block with values in the thousands.

    Args:
        x: (batchm seq, hidden_size) final hidden state.
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture.

    Returns:
        (batch, seq, hidden_size) normalized stream
    """
    return rms_norm(x, w["model.norm.weight"], config.rms_norm_eps)


def lm_head(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Score every vocabulary token at every position.

    Each logit is the dot product of a position's final vector with one
    token's embedding row: how closely the model's output points at that
    token.

    Reads the embedding table rather than lm_head.weight, because Qwen3-0.6B
    ties the two (config.tie_word_embeddings). This file happens to store a
    duplicate lm_head.weight, but many tied checkpoints do not; an untied
    model would need lm_head.weight instead.

    Args:
        x: (batch, seq, hidden_size), the stream after final_norm.
        w: the full weight dict, keyed exactly as the checkpoint is.

    Returns:
        (batch, seq, vocab_size) logits for every position, not just the
        last. Generation needs only the last row; slicing is the caller's job.
    """
    return project(x, w["model.embed_tokens.weight"])


def forward(
    ids: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    positions: torch.Tensor,
) -> torch.Tensor:
    """The whole model: token ids in, next-token logits out.

    Args:
        ids: (batch, seq) integer token ids, as the tokenizer returns them.
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture.
        positions: (seq,) the position of each token in ids. Required rather
            than defaulting to arange(seq): for a fresh prompt the caller
            passes exactly that, but once a KV cache exists a single new token
            can sit at position 100, and only the caller knows.

    Returns:
        (batch, seq, vocab_size) logits. Row i scores the token that follows
        position i, having seen positions 0..i only.
    """
    x = embed(ids, w)
    x = run_blocks(x, w, config, positions)
    x = final_norm(x, w, config)
    return lm_head(x, w)
