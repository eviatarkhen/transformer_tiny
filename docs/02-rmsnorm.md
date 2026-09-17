# 02 — RMSNorm

*Draft by Claude, pending my edit pass.*

## The problem

The residual stream flows through 28 blocks, and each block *adds* to it:

```
x = x + attention(x)
x = x + mlp(x)
```

Fifty-six additions, nothing controlling magnitude. The drift compounds — larger inputs
to a matmul give larger outputs, which get added back in. Left alone, activations reach a
scale where softmax saturates and float precision degrades.

The fix is to rescale the vector to a consistent size before each sub-layer reads it.

## What it computes

For one token's vector `x` of length `n`:

```
rms = sqrt( mean(x²) + eps )
out = (x / rms) * weight
```

- `mean(x²)` — over the **last dimension only**
- `eps` (1e-6) — *inside* the square root, so an all-zero vector gives `sqrt(eps)` rather
  than dividing by zero
- `weight` — a learned per-channel scale, shape `(n,)`, applied *after* normalizing

## "Divide by size"

Take `[3, 4]`. Its length is 5. Divide by 5 → `[0.6, 0.8]`, whose length is 1.
That's all normalization is: measure the size, divide by it, result has size 1.

RMS is just a different ruler — `sqrt(mean of squares)` instead of `sqrt(sum of squares)`.
For `[1, 2, 3, 4]`:

```
squares:         1, 4, 9, 16
mean:            30/4 = 7.5
sqrt:            2.739                    ← "the size"
divide:          0.365, 0.730, 1.095, 1.461
measure again:   1.0                      ← size is now 1
```

## Per row, independently

Three tokens, same direction, magnitudes 100× apart:

```
x = [[ 1.0,  2.0,  3.0,  4.0]
     [10.0, 20.0, 30.0, 40.0]
     [ 0.1,  0.2,  0.3,  0.4]]

mean of squares (keepdim):  [[7.5], [750.0], [0.075]]    ← one per ROW
x / rms:                    [[0.365, 0.730, 1.095, 1.461]
                             [0.365, 0.730, 1.095, 1.461]
                             [0.365, 0.730, 1.095, 1.461]]
```

All three rows come out **identical**. Magnitude discarded, direction kept. That's
**scale invariance**, and it's a property test that needs no reference implementation.

Token 1's huge values had no effect on token 0's result. Each row is processed alone.

**Why it must be that way:** if normalization mixed positions, token 0's output would
depend on token 5, and during generation a position could see the future. Attention is
the one place allowed to move information between positions, and it's carefully masked.

## The `keepdim` trap

`keepdim=True` keeps the mean shaped `(3, 1)` instead of `(3,)`. `(3,1)` broadcasts down
rows; `(3,)` broadcasts across columns.

On a non-square tensor PyTorch raises. On a **square** one — 4 tokens × 4 dims, exactly
what happens inside `q_norm` — it goes quiet and divides each *column* by a token's RMS.
Same shape out, no exception, completely different numbers.

## vs LayerNorm

LayerNorm: `(x − mean) / std * g + b`. RMSNorm drops the mean subtraction and the bias.
The centering wasn't earning its cost. Cheaper, works as well, and every current model
uses it.

## Where it's used — five places, two sizes

| where | size |
|---|---|
| `input_layernorm` — before attention | 1024 |
| `post_attention_layernorm` — before the MLP | 1024 |
| `model.norm` — after all 28 blocks | 1024 |
| `q_norm` — per attention head | **128** |
| `k_norm` — per attention head | **128** |

Written against the last dimension, one function handles `(seq, 1024)` and
`(batch, seq, heads, 128)` with no rank-specific logic.

## Things learned while testing this

- **`eps` placement is invisible at normal magnitudes.** `sqrt(v+eps)` and `sqrt(v)+eps`
  differ by ~1e-6 on unit-scale input — far under any useful tolerance, and both give
  finite output on zeros. Five passing tests missed it. Only a deliberately tiny input
  (`mean(x²)` comparable to `eps`) separates them, where they differ by **1.5**.
  Generalizes: *a term that is negligible in the common case needs a test case where it
  is not negligible.*
- **A passing suite means "no test caught anything", not "the code is correct."** Asking
  *which wrong version would still pass this?* is how you find the gaps.
- **Memory-bandwidth bound.** `x * rsqrt(v)` vs `x / sqrt(v)` measured identical
  (4.6 µs both). Instruction-count optimizations that don't reduce data movement do
  nothing here.
