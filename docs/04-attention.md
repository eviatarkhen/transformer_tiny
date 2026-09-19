# A very good example tutorial
Complete Example (Matching Real Qwen3 Pattern)

Setup:
- batch = 1
- seq = 3
- hidden_size = 6
- head_dim = 3 (fixed for all heads)
- q_heads = 4 → q projects to 4 × 3 = 12 (2× hidden_size)
- kv_heads = 2 → kv projects to 2 × 3 = 6 (1× hidden_size)

---

Step 1: Project to q, k, v

Input: (batch=1, seq=3, hidden=6)

q = q_proj(input)
    (1, 3, 6) → (1, 3, 12)     [projects to 2× hidden_size]

k = k_proj(input)
    (1, 3, 6) → (1, 3, 6)      [projects to hidden_size]

v = v_proj(input)
    (1, 3, 6) → (1, 3, 6)      [projects to hidden_size]

---

Step 2: Reshape into heads

q: (1, 3, 12) reshape to (1, 3, 4, 3)
   [batch, seq, q_heads=4, head_dim=3]

k: (1, 3, 6)  reshape to (1, 3, 2, 3)
   [batch, seq, kv_heads=2, head_dim=3]

v: (1, 3, 6)  reshape to (1, 3, 2, 3)
   [batch, seq, kv_heads=2, head_dim=3]

---

Step 3: Normalize q and k (per-head RMSNorm)

q: (1, 3, 4, 3) → (1, 3, 4, 3)   [normalized on last dim]
k: (1, 3, 2, 3) → (1, 3, 2, 3)   [normalized on last dim]
v: (1, 3, 2, 3)  [unchanged]

---

Step 4: Transpose to put heads first

q: (1, 3, 4, 3) → (1, 4, 3, 3)
   [batch, q_heads=4, seq, head_dim=3]

k: (1, 3, 2, 3) → (1, 2, 3, 3)
   [batch, kv_heads=2, seq, head_dim=3]

v: (1, 3, 2, 3) → (1, 2, 3, 3)

---

Step 5: Apply RoPE

q: (1, 4, 3, 3) → (1, 4, 3, 3)   [rotated by position]
k: (1, 2, 3, 3) → (1, 2, 3, 3)   [rotated by position]
v: (1, 2, 3, 3)  [unchanged]

---

Step 6: Broadcast k to match q's head count

k_rope: (1, 2, 3, 3) broadcast → (1, 4, 3, 3)
        [each of 2 kv-heads covers 2 q-heads]

        k_rope[0, 0, :, :] = k_rope[0, 1, :, :] = original kv-head 0
        k_rope[0, 2, :, :] = k_rope[0, 3, :, :] = original kv-head 1

v:      (1, 2, 3, 3) broadcast → (1, 4, 3, 3)

---

Step 6b: Scores = q @ k^T

q_rope:  (1, 4, 3, 3)
k_rope:  (1, 4, 3, 3)  [after broadcast]

For each head, compute q @ k^T:
(3, 3) @ (3, 3) = (3, 3)
↓
q @ k^T of shape (3, 3) for each of 4 heads
↓
scores: (1, 4, 3, 3)  [batch, q_heads, seq_q, seq_k]

---

Step 6c: Scale, mask, softmax

scores = scores / sqrt(head_dim)
       = scores / sqrt(3)
       = scores / 1.732...

scores: (1, 4, 3, 3)  [divided by √3]

Add causal mask: (3, 3) matrix with -∞ in upper triangle
Softmax over last dimension (key positions)

weights: (1, 4, 3, 3)  [each query position has a probability distribution over keys]

---

Step 7: Blend values

attended = weights @ v

weights:  (1, 4, 3, 3)
v:        (1, 4, 3, 3)  [after broadcast]

(1, 4, 3, 3) @ (1, 4, 3, 3) = (1, 4, 3, 3)

attended: (1, 4, 3, 3)  [batch, q_heads, seq, head_dim]

---

Step 8: Concatenate heads

attended: (1, 4, 3, 3) → (1, 3, 4*3=12)
          [batch, seq, q_heads * head_dim = 12]

---

Step 9: Output projection

o_proj: (1, 3, 12) → (1, 3, 6)
        [projects from 2×hidden back to hidden]

output: (1, 3, 6)  [batch, seq, hidden_size]

---

Key insight now clear:

┌───────────────────┬──────────────────────┐
│     Dimension     │        Value         │
├───────────────────┼──────────────────────┤
│ input hidden_size │ 6                    │
├───────────────────┼──────────────────────┤
│ q projects to     │ 12 (2× hidden)       │
├───────────────────┼──────────────────────┤
│ k projects to     │ 6 (1× hidden)        │
├───────────────────┼──────────────────────┤
│ v projects to     │ 6 (1× hidden)        │
├───────────────────┼──────────────────────┤
│ head_dim          │ 3 (same for q and k) │
├───────────────────┼──────────────────────┤
│ q_heads           │ 4 (12/3)             │
├───────────────────┼──────────────────────┤
│ kv_heads          │ 2 (6/3)              │
└───────────────────┴──────────────────────┘
---

# Decomposition and testability

The lesson of this step, and the one most worth keeping.

## What the mutation pass showed

After the tests were green, we broke `attention` twelve different ways and recorded
which tests went red. Nothing survived — but the *pattern* of what caught what was the
interesting part. Seven of the twelve were caught **only** by the two tests that compare
against HuggingFace. Every other test stayed green.

Splitting those twelve by where the bug lived explains it completely:

- Mutations inside a **helper** (`split_heads`, `merge_heads`, `repeat_kv`,
  `causal_mask`) were caught by that helper's own unit test. Four for four.
- Mutations inside **`attention` itself** — the scale, the softmax dimension, the mask —
  were caught by nothing but the oracle.

So ordinary unit-test discipline was working perfectly. The problem was that `attention`
was one nine-stage unit, and its only possible unit test is "compare the final 1024-wide
vector to HuggingFace". That detects a bug but says nothing about *where* it is.

## The fix was not more tests

It was a smaller unit. These three lines lived inline in `attention`:

    scores = (q @ k.transpose(-2, -1)) * config.head_dim**-0.5
    weights = torch.softmax(scores + causal_mask(x.shape[1]), dim=-1)
    attended = weights @ v

Pulled out into `attend(q, k, v)`, they became testable on toy sizes — `head_dim=4`,
`seq=2`, hand-computed softmax weights, no model and no oracle. Re-running the same
mutations afterwards:

| mutation | caught without the oracle |
|---|---|
| no 1/sqrt(d) scale | `test_attend_scale_is_one_over_sqrt_head_dim` |
| scale 1/d instead of 1/sqrt(d) | `test_attend_scale_is_one_over_sqrt_head_dim` |
| no causal mask | `test_attend_first_position_returns_its_own_value`, `test_attend_is_causal` |
| softmax over dim=-2 | three of the `attend` tests |

Same tests-to-code ratio, far better localization. **Testability follows decomposition:
if a bug can only be caught end to end, the unit is too big.**

## What stays uncatchable, correctly

Two mutations are still caught by the oracle alone, and always will be:

- the `q_norm` and `k_norm` weights swapped
- RoPE applied to `v` as well as to q and k

Every individual function is correct in both cases. They are **wiring** bugs — the
right parts connected wrongly — and no unit test can see that by construction. That is
what the integration test is for, and here the integration test is the HF comparison.
A suite where the end-to-end test covers the wiring and nothing else is the healthy
shape, not a gap.

## The one test carrying its own weight alone

`test_uses_the_cos_sin_arguments` is the only test that catches a mutation the oracle
misses: recomputing `cos`/`sin` inside `attention` from `arange(seq)`. Every parity test
still passes, because parity always feeds positions 0..seq-1. It would break silently at
step 9, when a cached token at position 100 arrives with seq=1.

A corollary worth remembering: a mutation matrix is only as good as the mutations you
think to write. The first pass here missed both this one and the dropped `.T` in
`project` — the same blind spot the tests themselves have.