# 01 — What's actually in the checkpoint

*Draft by Claude, pending my edit pass.*

## A checkpoint is a dictionary

`model.safetensors`, 1.4 GB, is a flat `dict[str, Tensor]`. No code, no structure, just
named numbers. The architecture lives in Python; only the values live in the file.

That's what makes parity testing possible at all: HuggingFace's code and ours read the
*same numbers*. Arrange them in the same order of operations and the outputs are
identical. Differ anywhere and they aren't.

## 311 tensors = 3 + 28 × 11

The arithmetic is exact. Three global tensors:

```
model.embed_tokens.weight   (151936, 1024)   the lookup table
model.norm.weight           (1024,)          final normalization
lm_head.weight              (151936, 1024)   scores over the vocabulary
```

And eleven per layer, keyed `model.layers.{i}.`:

```
self_attn.q_proj / k_proj / v_proj / o_proj    the four attention projections
self_attn.q_norm / k_norm                      per-head QK normalization
mlp.gate_proj / up_proj / down_proj            the three MLP projections
input_layernorm                                before attention
post_attention_layernorm                       before the MLP
```

The 28 identical layers are the "one small function applied 28 times" made visible.

**Naming trap:** `post_attention_layernorm` is applied **before the MLP**. The name says
where it sits in the sequence (after attention), not what it precedes.

## Weights are stored transposed

Every projection is backwards from what you'd expect:

```
q_proj.weight   (2048, 1024)     but q_proj maps 1024 → 2048
gate_proj       (3072, 1024)     but it maps 1024 → 3072
down_proj       (1024, 3072)     but it maps 3072 → 1024
```

PyTorch convention: `nn.Linear` stores `(out_features, in_features)` and computes
`y = x @ W.T`. To apply a stored weight by hand you need the **transpose**.

And the dangerous part: `k_proj.weight` is `(1024, 1024)`. **Square.** Forget the
transpose there and the matmul runs fine, produces the right shape, and is wrong.

That's the recurring failure mode of this whole project: *wrong orientation or wrong
rank, no exception, plausible-looking output.* It has now appeared at least six times.
It's the reason for testing every component against HF rather than assembling the model
and hoping.

## Tied embeddings, and a surprise

`tie_word_embeddings: true` means the model uses one matrix in both directions: row *i*
turns token *i* into a vector on the way in, and scores token *i* on the way out. Saves
155M parameters — a quarter of this model.

But the file stores **both copies anyway**:

```
identical values?  True
same storage?      False
```

~300 MB of redundancy that HF ignores at load time. Here they happen to agree. Nothing
guarantees that, so the code reads `embed_tokens` and honours the config flag.

## Precision

The file is **bfloat16**; we cast everything to **float32** on load.

Not because bf16 is wrong — the model was *trained* in bf16, those are the real weights.
It's because parity testing needs it. The same matmul done both ways:

```
max abs difference, bf16 vs fp32:  0.3643
our parity tolerance:              0.0001
```

Rounding error 3,600× larger than the threshold. A genuinely broken component would sit
comfortably inside that noise.

bf16 → fp32 is lossless: every bf16 value is exactly representable in fp32. We're not
inventing precision, just not shedding more with each operation.

Surprise finding: on this Mac's CPU, bf16 matmul is **855× slower** than fp32
(1009 ms vs 1.18 ms) — PyTorch has no optimized bf16 kernel for this path and falls back
to something naive. That's a property of this backend, not of bf16; on MPS or a GPU with
bf16 tensor cores it's the other way round.
