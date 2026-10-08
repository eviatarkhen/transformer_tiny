import torch

# Where quantize_weights puts the int8 copy of the tied embedding table for
# lm_head. Not a checkpoint key: it exists only after quantize_weights, so a
# plain checkpoint still reads the embedding table, as step 7 decided.
LM_HEAD_INT8 = "lm_head.int8"


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8, one scale per output row.

        scale = max|row| / 127          q = round(w / scale), in -127..127

    so each weight comes back as q * scale, off by at most scale / 2. One
    scale per row rather than per matrix because a few large weights would
    stretch a shared scale: on layer 13's down_proj the largest weight is 32x
    the median, and per-matrix scaling quadruples the output error (4.6% vs
    1.1% per row).

    Args:
        weight: (out, in), float32 or bf16 -- one row per output, as stored.

    Returns:
        q: (out, in) int8.
        scale: (out,) in weight's dtype, the dtype the fused kernel needs it
            in to match the activations. q is rounded against the stored
            scale, so q * scale is the best this pair can reconstruct.
    """
    w = weight.to(torch.float32)
    max_abs = w.abs().amax(dim=-1)
    # An all-zero row has max 0; any positive scale reconstructs it exactly.
    scale = (max_abs / 127).clamp_min(1e-12).to(weight.dtype)
    q = (w / scale.to(torch.float32)[:, None]).round().clamp(-127, 127)
    return q.to(torch.int8), scale


def quantize_weights(w: dict[str, torch.Tensor]) -> dict:
    """The weight dict, with every projection matrix as an int8 (q, scale) pair.

    Quantized: the seven *_proj.weight matrices of every layer, and the tied
    embedding table, as a separate copy under LM_HEAD_INT8 -- lm_head reads
    all 151,936 rows every turn, about a quarter of the model's bytes.

    Kept as they are: the embedding table itself (the lookup reads one row
    per token, negligible), and every norm weight (1-D, tiny).

    Returns a new dict; w is not modified. Same dtype and device as w.
    """
    out = {}
    for name, t in w.items():
        out[name] = quantize_rows(t) if name.endswith("_proj.weight") else t
    out[LM_HEAD_INT8] = quantize_rows(w["model.embed_tokens.weight"])
    return out
