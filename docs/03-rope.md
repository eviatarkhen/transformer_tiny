# 03 — RoPE (rotary position embedding)

*Draft by Claude, pending my edit pass.*

## Where it fits

RoPE is **not** a stage in the model. It doesn't appear in the block diagram at all — it's
a helper used *inside* attention.

```
one block:
    rms_norm  →  ATTENTION  ←  RoPE lives in here
                             ↓
    rms_norm  →  MLP
```

Inside attention each token produces three vectors: a **query** (what am I looking for),
a **key** (what do I offer), a **value** (what I'll hand over). RoPE rotates the queries
and the keys — the two used for *comparing* — and leaves the values alone.

## The problem: attention cannot see word order

Not an approximation. Attention is genuinely, completely blind to position.

Three tokens A, B, C through a plain attention layer, then reordered to C, A, B:

```
attention(A,B,C)                    attention(C,A,B)
[[-0.273,  0.193, -0.191,  0.256]   [[-0.058,  0.242,  0.025,  0.283]   ← C
 [-0.103,  0.566, -0.182,  0.143]    [-0.273,  0.193, -0.191,  0.256]   ← A
 [-0.058,  0.242,  0.025,  0.283]]   [-0.103,  0.566, -0.182,  0.143]]  ← B
```

Every row identical, just reordered (max difference `1.5e-08`, float noise). A's output
is the same whether A was first or second.

This follows from the mechanism: attention scores every pair from their vectors alone,
`q_i · k_j`, and nothing in that expression knows where *i* and *j* sit.

**So a transformer with no positional signal cannot distinguish "the dog bit the man"
from "the man bit the dog".** Position must be injected deliberately. It is not emergent.

## The old fix and why it was replaced

GPT-2 added a learned vector per position. It works, but: fixed maximum length, nothing
in the representation says position 5 and 6 are adjacent, and language mostly cares about
**relative** distance anyway — "the adjective two words back" is a reusable pattern,
"the word at absolute position 847" almost never is.

## RoPE's idea: rotate instead of add

A dot product depends on the angle *between* two vectors. Rotate the query by an angle
proportional to its position *m*, and the key by one proportional to *n*:

```
before:   q · k  ∝  cos(a − b)
after:    q · k  ∝  cos( (a + mθ) − (b + nθ) )
                 =  cos( (a − b) + (m − n)θ )
```

The absolute positions **cancel**. Only `m − n` survives.

A query at position 3 attending to a key at position 7 sees exactly what a query at 13
sees looking at 17. Relative position falls out of the geometry instead of being learned.

## Scaling to 128 dimensions

Split the 128 numbers into **64 pairs**, treat each pair as a 2-D vector, rotate each by
its own frequency. Like 64 clock hands moving at different speeds — fast hands
distinguish nearby positions precisely, slow hands carry long-range information.

```
inv_freq[i] = theta ** (−2i/128)      theta = 1e6
```

| pair i | inv_freq | full turn every |
|---|---|---|
| 0 | 1.000e+00 | 6.3 tokens |
| 16 | 3.162e-02 | 199 tokens |
| 32 | 1.000e-03 | 6,283 tokens |
| 63 | 1.241e-06 | 5,063,256 tokens |

## The pairing detail that matters

HuggingFace pairs dimension `i` with dimension `i + 64` — the two **halves**. The
original RoPE paper (and most tutorials) pairs adjacent dimensions `(0,1), (2,3)`.

Both are valid rotations, and both work *if training and inference agree*. These weights
were trained with the half-split, so anything else produces plausible-looking garbage.

## The implementation

Build two tables of shape `(seq, 128)` where the 64 angles are **duplicated**:

```
cos[m, i] = cos[m, i+64] = cos(m · inv_freq[i])
```

Then with `rotate_half(x) = [−x[64:], x[:64]]`:

```
out = x * cos + rotate_half(x) * sin
```

The duplication is what lets one elementwise multiply serve both members of every pair.

## Free properties worth testing

- **Position 0 is the identity** — angle 0, so `cos=1`, `sin=0`. Bit-exact.
- **Norm is preserved** — it's a rotation. Check *per pair*, not just per vector: a bug
  rotating `(i, i+1)` instead of `(i, i+64)` still preserves the total norm.
- **Only relative distance matters** — the property the whole design exists for.

## Things learned while testing this

- **A test asserting "these two are equal" needs a companion clause ruling out the
  trivial implementation.** `test_depends_only_on_relative_position` would have been
  satisfied by an `apply_rope` that ignored position entirely and returned `x` — every
  score equal, test green. The fix was a second assertion that a *different* gap gives a
  *different* score.
- **A property test claims its property and nothing more.** Swapping `sin` and `cos`
  legitimately *passes* the relative-position test, because it's still a rotation linear
  in position. The position-0 test is what catches it.
- **Read the failure numbers as a diagnostic.** "100% of elements mismatched, max diff
  4.34" means the formula is structurally wrong — re-read the expression. A handful of
  mismatched elements would mean an edge case instead.
- **`rope_cos_sin` should be called once per forward pass, not once per layer.** It
  depends only on positions, head_dim and theta. Rebuilding it in each of the 28 blocks
  wastes 225 µs per token at seq=5 — where the table build (8 µs) actually exceeds the
  rotation itself (7.1 µs). That's why HF hangs `rotary_emb` off the model, not the
  attention module.
