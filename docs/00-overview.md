# 00 — The shape of the whole thing

*Draft by Claude, pending my edit pass.*

## What inference actually is

```
text → tokens → embeddings → [ 28 × block ] → final norm → logits → next token
                                                                       ↓
                                                              append, repeat
```

**1. Text becomes integers.** `"The capital of France is"` → `[785, 6722, 315, 9625, 374]`.
Each integer is a *token*, roughly a word or word-piece. The tokenizer owns this and we
treat it as a black box.

Decoding them one at a time is instructive:

```
['The', ' capital', ' of', ' France', ' is']
```

The leading space belongs to the *following* token, not the preceding one. That's why
slicing text on token boundaries goes wrong if you assume otherwise.

**2. Each integer becomes a vector.** A lookup table with 151,936 rows (one per possible
token) and 1024 columns. Token 785 means "take row 785". Five tokens in, so we now hold a
matrix of shape `(5, 1024)`.

**3. That matrix passes through 28 blocks, and its shape never changes.** Block 1 takes
`(5, 1024)` and returns `(5, 1024)`. So does block 28. Same shape signature, same
structure, different weights. The whole model is one small function applied 28 times.

That matrix is called the **residual stream**, because each block *adds* to it rather
than replacing it:

```
x = x + attention(x)
x = x + mlp(x)
```

**4. Each block does exactly two things:**

- **Attention** — positions look at each other and exchange information.
- **MLP** — each position processes its own vector alone, no communication.

The single most useful fact in here: **attention is the only place information moves
between positions.** Everything else in the entire model operates on one position at a
time, blind to its neighbours. Normalization, the MLP, the projections — all per-token.

**5. Vectors become token scores.** After block 28, multiply by the lookup table from
step 2, transposed, giving `(5, 151936)`. Row *i* holds a score for every token that
could follow position *i*. These scores are **logits**.

**6. Pick one, append, repeat.** Take the last row, argmax → `" Paris"`. Append it, now
6 tokens, run everything again.

## The waste in step 6

To produce one new token we recompute all the previous positions from scratch. That's
what the KV cache (step 9) fixes, and it's why it's filed as a structural change rather
than an optimization.

## Baseline

HF's own implementation, CPU, float32, no KV cache:

```
"The capital of France is" + 10 tokens  →  21-30 tok/s
output: "Paris. The capital of Italy is Rome. The"
```

Wide variance across runs on a single untimed measurement — worth remembering when
phase 4 claims something got faster.

## The model we're building

Qwen3-0.6B. 28 layers, hidden_size 1024, 16 query heads / 8 key-value heads,
head_dim **128** (deliberately *not* 1024/16 = 64), RMSNorm, RoPE, SwiGLU, no biases
anywhere, tied embeddings, 151,936 vocab.
