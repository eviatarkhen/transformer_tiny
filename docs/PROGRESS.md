# Progress

State of the project. Rules live in `CLAUDE.md`; this file is status only.
Updated at the close of every step.

**Session-start ritual:** read `CLAUDE.md` → read this file → `uv run pytest` →
continue from "Next action".

---

## Next action

**Step 6: `block`.** Not started; theory not yet delivered. The two components
now exist and are both exact against the oracle, so step 6 is pure wiring:
`input_layernorm` -> attention -> residual add -> `post_attention_layernorm` ->
mlp -> residual add. Oracle is `hf_model.model.layers[0]`, the whole
`Qwen3DecoderLayer`, which takes the same `position_embeddings=(cos, sin)`
argument `self_attn` did -- **check its return type before writing the spec**,
`self_attn` returned a tuple and `mlp` returned a bare tensor.

Per step 4's lesson, expect this step's bugs to be *wiring* bugs -- the two
norms swapped, a residual dropped, the pre-norm applied to the wrong branch --
which unit tests cannot catch by construction. Plan the tests accordingly.

Deliver the theory first, then agree the design, then Eviatar implements.

## Step status

| # | Step | State | Proven by |
|---|---|---|---|
| 0 | Environment + oracle | **done** | `scripts/smoke.py` runs |
| 1 | `config.py` + `weights.py` | **done** | `tests/test_weights.py`, 7 tests |
| 2 | `rms_norm` | **done** | `tests/test_rmsnorm.py`, 6 tests |
| 3 | `rope` | **done** | `tests/test_rope.py`, 6 tests |
| 4 | `attention` | **done** | `tests/test_attention.py`, 51 tests |
| 5 | `mlp` | **done** | `tests/test_mlp.py`, 4 tests |
| 6 | `block` | next | — |
| 7 | full model | not started | — |
| 8 | greedy generation | not started | — |
| 9 | KV cache | not started | — |
| 10 | optimize | not started | — |
| 11 | sampling | not started | — |

## Environment (verified 2026-09-16)

- Python 3.12.14 in `.venv` (system Python is 3.14 — too new for torch wheels)
- torch 2.14.0, MPS available
- **transformers 5.17.0**
- Model already in HF cache: `Qwen/Qwen3-0.6B`, single 1.4 GB safetensors file,
  snapshot `c1899de289a04d12100db370d81485cdf75e47ca`

## Verified facts about the HF oracle

Checked directly in the installed `transformers/models/qwen3/modeling_qwen3.py`,
not assumed:

- Classes: `Qwen3RMSNorm`, `Qwen3MLP`, `Qwen3RotaryEmbedding`, `Qwen3Attention`,
  `Qwen3DecoderLayer`, `Qwen3Model`, `Qwen3ForCausalLM`.
- `q_norm` / `k_norm` are `Qwen3RMSNorm(head_dim)` — applied **per head**, after
  reshaping to `(..., num_heads, head_dim)` and **before** the transpose and before RoPE.
- Attention order is: project → view into heads → q_norm/k_norm → transpose →
  RoPE → attend → reshape → o_proj.
- `Qwen3Attention.forward` takes `position_embeddings=(cos, sin)` precomputed by the
  model-level `rotary_emb`, not a position index.

## Baseline (step 0)

HF oracle, CPU, float32, no KV cache, `"The capital of France is"` + 10 tokens:
**21-30 tok/s** across runs. Output: `Paris. The capital of Italy is Rome. The`
Wide variance on a single untimed run — phase 4 comparisons need repeated runs.

## Testing method that is working

**Mutation testing.** After a component goes green, break it deliberately several
ways and record which tests catch which mutation. It found a real hole in the
rms_norm suite that five passing tests missed. Keep doing this at every step.

rms_norm mutation matrix (final, after adding the small-input parity case):

| mutation | caught by |
|---|---|
| no keepdim | all four |
| no sqrt | all four |
| eps outside sqrt | **only** parity at input scale 1e-4 |
| weight ignored | parity only |
| mean not squared | parity, unit_rms |

`test_zero_input_is_finite` caught none of these; it only fires if eps is dropped
entirely.

### rope mutation matrix

| mutation | caught by |
|---|---|
| identity apply_rope | apply_hf, relative |
| sin/cos swapped | apply_hf, pos0 |
| rotate_half without negation | apply_hf, norm, relative |
| adjacent (interleaved) pairing | all but pos0 |
| no duplication / 128 frequencies | all but pos0 |
| positions off by one | cos_sin_hf, apply_hf, pos0 |

Two lessons recorded from it:

- `test_depends_only_on_relative_position` only catches an identity `apply_rope`
  because of its second clause (a *different* gap must give a *different* score).
  Whenever a test asserts "these two things are equal", ask what trivial
  implementation makes everything equal, and add the clause that rules it out.
- "sin/cos swapped" legitimately passes the relative-position test: it is still a
  rotation linear in position, so the property genuinely still holds. A property
  test claims its property and nothing more.

## Step 5: design decisions

- `mlp(x, w, layer) -> y` in `tiny/mlp.py`. **No `config` parameter**, unlike
  `attention` -- every shape falls out of the weight matrices and nothing in
  `Config` is read. A deliberate departure from the `f(x, weights, config)`
  convention in `CLAUDE.md`: a parameter that is never read is a false claim.
- `project` moved from `attention.py` down into `layers.py`. It is a generic
  primitive, not attention-specific, and `attention.py` already imports
  `layers.py`, so leaving it put would have made `mlp` depend on `attention`.
- One helper, `swiglu(gate, up)` = `silu(gate) * up`. Everything subtle lives
  there and is testable on hand-computed numbers with no weights and no model.
  `silu` itself stays inlined: the interesting mutation (activation applied to
  `up` instead of `gate`) is only visible one level up.
- The file layout now reads `layers.py` = primitives, `attention.py` / `mlp.py`
  = components, and step 6's `model.py` calls both. `CLAUDE.md` updated to match.

## Step 5 mutation matrix

Seven mutations, none survived. Four tests.

| mutation | caught by |
|---|---|
| `silu` -> `relu` | `swiglu_matches_hand_computed_values`, both parity |
| `silu` -> `sigmoid` (plain GLU) | `swiglu_matches_hand_computed_values`, both parity |
| no activation (`gate * up`) | `swiglu_matches_hand_computed_values`, both parity |
| gate/up swapped at the call site | parity only -- wiring, as predicted |
| `layer` ignored, always block 0 | `matches_hf[27]` **only** |
| elementwise `*` -> `@` | all four, three as `RuntimeError` |
| positions mixed (`up.mean(dim=1)`) | `is_position_wise`, both parity |

Readings:

- **`matches_hf[27]` is load-bearing alone.** Parametrizing parity over two
  layers cost one line and is the only thing that catches a hardcoded prefix.
- **`is_position_wise` catches nothing parity misses**, because parity is total.
  It earns its place by localizing (it fails on the property, naming the cause,
  rather than as a tensor diff) and by stating the property in the file.
- The whole suite is 4 tests where attention needed 51, and the reason is
  measurable rather than a matter of taste: all three MLP matrices are
  rectangular, so the failure mode that dominated step 4 -- right shape, wrong
  numbers -- cannot occur here. Every shape mistake raises.
- One test vector does nearly all the work: `gate = [1, -1, 0]` against
  `up = [2, -2, 2]` separates five different wrong activations. The negative and
  the zero entries are what make it strong; on all-positive inputs `relu`,
  `sigmoid` and no-activation-at-all are indistinguishable from the truth.


## Gotchas

Running list of things that actually bit us, and how they were found.
The highest-value section in this file — these are exactly what gets lost.

- **The checkpoint stores `lm_head.weight` *and* `model.embed_tokens.weight`, and
  `torch.equal` on the two is True.** Tied embeddings are a literal duplicate copy
  here despite `tie_word_embeddings: true`. Summing the file gives 751.6M
  parameters; the model is 596.0M. Step 7 must not double-count, and can read
  either key.
- **The MLP is position-wise mathematically, but not bit-for-bit.** One token
  alone vs. the same token inside a 5-token batch differs by 5.96e-6 -- the matmul
  kernel picks a different reduction order for a different-shaped input. A
  *permutation* of the sequence keeps the shape and is exactly 0.0. When asserting
  an invariance, prefer the spelling that leaves the shape alone.
- **Calling HF's own kernel makes parity exactly 0.0, and that is a coincidence,
  not a property.** `F.silu` is the same call `Qwen3MLP` makes, so the MLP matches
  at 0.0; hand-writing `gate * sigmoid(gate)` gives 7.2e-7. Never assert `== 0.0`
  on parity -- it encodes which kernel was called, not correctness.
- **`k_proj` and `v_proj` are square (1024x1024), so a missing `.T` is silent.**
  Weights are stored `(out, in)`, so a projection is `x @ W.T`. Measured across all
  four: `q_proj` (2048,1024) and `o_proj` (1024,2048) *raise* on `x @ W`, but
  `k_proj` and `v_proj` return the correct shape with values off by 7.741 and
  4.799. Writing all four at once and only checking the end hides two of them.
  This is the argument for validating stage by stage.
- **`.T` on a tensor of rank > 2 reverses *every* dimension**, turning
  (1,16,5,128) into (128,5,16,1). It is deprecated and currently only *warns*, so
  it does not announce itself. Use `transpose(-2, -1)` to swap the last two axes.
- **`transpose(-2,-1)` and `transpose(1,2)` are not interchangeable**, and which is
  correct depends on the layout at that line. On `(batch, seq, heads, head_dim)`
  you want `transpose(1, 2)`; on `(batch, heads, seq, head_dim)` heading into the
  scores matmul you want `transpose(-2, -1)`. Track the shape, not the axis number.
- **Under sdpa the attention call returns `None` for the weights.** Confirmed at
  `transformers/integrations/sdpa_attention.py:170` (`return attn_output, None`).
  To inspect or compare the softmax weights, go through `eager_attention_forward`,
  which takes the module as its first argument and returns its output already
  transposed to (batch, seq, heads, head_dim).
- **`.triu(1)` zeroes the lower triangle rather than removing it.** So
  `isneginf(mask.triu(1)).all()` is False for a correct mask -- those zeros are not
  -inf. Index with a boolean instead. Cost one red test that looked like an
  implementation bug and was not.
- **A uniform position shift is a no-op for RoPE, by construction.** A test
  comparing positions 0-4 against 100-104 and expecting a difference got
  byte-identical output: every *relative* distance is unchanged, and RoPE depends
  only on those. To make the output move, change the gaps (`arange(seq) * 3`), not
  the offset.
- **transformers is v5, checkpoint is from v4.51.** Module paths and call signatures
  differ from most tutorials and from memory. Always check the installed source before
  writing a spec against it.
- **`head_dim` (128) != `hidden_size / num_heads` (64)** in Qwen3. Code that derives
  head_dim by division is wrong here. HF itself uses
  `getattr(config, "head_dim", hidden_size // num_heads)`.
- **`assert_close` raises, `torch.equal`/`allclose` return.** Wrapping assert_close
  in `assert ... is None` is a no-op. Know which style a helper uses.
- **`dataclasses.fields()` returns `Field` objects, not strings** — use `.name`.
- **eps placement is invisible at normal magnitudes.** `sqrt(v+eps)` vs
  `sqrt(v)+eps` differ by ~1e-6 on unit-scale input, far under any useful
  tolerance. Only a deliberately tiny input (mean(x^2) comparable to eps) separates
  them. Generalizes: a term that is negligible in the common case needs a test case
  where it is not negligible.
- **rms_norm is memory-bandwidth bound**, so rsqrt-vs-divide measured identical
  (4.6us both, at (1,5,1024)). Instruction-count optimizations that do not reduce
  data movement do nothing here.
- **A discarded return value is a test that always passes.** `torch.equal(a, b)`
  without `assert` is legal Python and silently green. Same shape as the
  `tokenizer.decode(...)` slip in smoke.py. Check every new test by feeding it
  deliberately broken input and confirming it goes red.
- **`json.loads` takes JSON text, `json.load` takes a file.** And a dataclass never
  validates its annotations: `head_dim=["head_dim"]` (a missing `data`) produced a
  `list` where an `int` was declared, with no error until much later.
- **`tokenizer.decode()` accepts a 2D tensor without raising** and returns a *list of
  strings* instead of a string. Pass `ids[0]`. First instance of the project's dominant
  failure mode: wrong rank, no exception, silently different result.

## Step 3 theory: delivered so far

Covered with Eviatar already:

1. **Attention is permutation equivariant** -- demonstrated empirically, not asserted.
   Three tokens A,B,C through a plain attention layer, then reordered to C,A,B: every
   output row identical, just reordered (max diff 1.5e-08). So a transformer with no
   positional signal cannot tell "dog bit man" from "man bit dog". Position must be
   injected deliberately; it is not emergent.
2. **Why not GPT-2's learned absolute position vectors**: fixed max length, nothing in
   the representation says position 5 and 6 are adjacent, and language mostly cares
   about *relative* distance anyway.
3. **RoPE's core idea**: a dot product depends on the angle *between* two vectors.
   Rotate q by an angle proportional to its position m and k by one proportional to n,
   and `cos((a-b) + (m-n)*angle)` -- absolute positions cancel, only m-n survives.
4. **The frequency spectrum** (head_dim=128, theta=1e6 -> 64 frequencies):

   | pair i | inv_freq | full turn every |
   |---|---|---|
   | 0 | 1.000e+00 | 6.3 tokens |
   | 8 | 1.778e-01 | 35.3 tokens |
   | 16 | 3.162e-02 | 198.7 tokens |
   | 32 | 1.000e-03 | 6,283 tokens |
   | 48 | 3.162e-05 | 198,692 tokens |
   | 63 | 1.241e-06 | 5,063,256 tokens |

   Fast pairs distinguish nearby positions precisely; slow pairs carry long-range
   information. Like 64 clock hands at different speeds.

All of the above is now delivered, plus:

5. `rotate_half` implements the 2-D rotation: `out = x*cos + rotate_half(x)*sin`
   expands to `out[i] = x[i]cos - x[i+64]sin` and `out[i+64] = x[i+64]cos + x[i]sin`,
   which is only correct because `cos[i+64] == cos[i]` -- the reason for the
   `cat((angles, angles))` duplication.
6. RoPE applies to q and k only, never v: it changes the addressing, not the payload.
7. Order inside attention: project -> reshape to heads -> q_norm/k_norm -> transpose
   -> RoPE -> scores -> o_proj.

Oracle behaviour verified directly:
- `rotary_emb(dummy, position_ids)` -> cos, sin each (batch, seq, 128)
- `cos[:, 0] == cos[:, 64]` is True
- at position 0, cos=1 and sin=0, so RoPE is the identity there
- norm is preserved at every position

Superseded (was: still to cover):

- How `rotate_half` implements a 2D rotation of the pair (x_i, x_{i+64}), and why
  cos/sin are duplicated via `cat((freqs, freqs), -1)`.
- Why RoPE is applied to q and k only, never to v (it steers *where* attention looks,
  not *what* is retrieved).
- Where it sits in the order: after q_norm/k_norm, before the scores; per head, on the
  last dimension.
- The design: proposed `rope_cos_sin(positions, head_dim, theta) -> (cos, sin)` each
  (seq, head_dim), and `apply_rope(x, cos, sin)` with x as (batch, heads, seq, head_dim)
  so (seq, head_dim) broadcasts. Taking `positions` as a tensor rather than a length
  means step 9's KV cache needs no offset parameter.
- Test plan: parity of cos/sin vs HF `rotary_emb`; parity of apply_rope vs HF
  `apply_rotary_pos_emb`; property tests for norm preservation, position 0 being the
  identity, and the relative-position property
  (dot(rope(q,m), rope(k,n)) depends only on m-n).

## Verified facts about HF's RoPE (read from the installed v5 source)

Needed before writing the step 3 spec; do not trust tutorials, which usually
describe the *other* pairing convention.

- `inv_freq = 1 / (theta ** (arange(0, head_dim, 2) / head_dim))` -> 64 frequencies
  for head_dim=128.
- `Qwen3RotaryEmbedding.forward` builds `freqs = outer(positions, inv_freq)` of shape
  (batch, seq, 64), then `emb = cat((freqs, freqs), -1)` -> (batch, seq, 128), and
  returns `cos = emb.cos()`, `sin = emb.sin()`. `attention_scaling` is 1.0 for
  default rope, so it can be ignored.
- `rotate_half(x)` splits the last dim in **halves**: `cat((-x[..., 64:], x[..., :64]))`.
  So dimension i is paired with dimension i+64 -- NOT the interleaved (0,1),(2,3)
  pairing from the original RoPE paper. Both are valid; only one matches these weights.
- `apply_rotary_pos_emb(q, k, cos, sin)` = `q * cos + rotate_half(q) * sin`, with
  cos/sin unsqueezed at dim 1 to broadcast over heads.

## Note on docs/

`00-overview.md`, `01-weights.md`, `02-rmsnorm.md` and `03-rope.md` exist but were
**drafted by Claude at Eviatar's request**, not written by him. Each is marked as a
draft pending his edit pass. The working agreement still has him writing these in his
own words -- treat the drafts as a starting point for that, not as a replacement.

## Step 4 oracle: settled, calling the module directly works

Measured directly, not assumed. Forward hooks are NOT needed.

```python
attn = hf_model.model.layers[0].self_attn
cos, sin = hf_model.model.rotary_emb(x, torch.arange(seq).unsqueeze(0))
with torch.no_grad():
    out, attn_weights = attn(x, position_embeddings=(cos, sin), attention_mask=mask)
```

- `x` is `(batch, seq, hidden_size)`; the call returns a 2-tuple, output first.
- `hf_model.config._attn_implementation` is **`"sdpa"`** by default.
- An additive float mask is `torch.full((seq, seq), -inf).triu(1)[None, None]`,
  shape (1, 1, seq, seq), 0 where allowed and -inf where forbidden.
- **Under sdpa, `attention_mask=None` gives output identical to that explicit causal
  mask (max diff 0.0), and both are genuinely causal**: changing only the last token
  leaves every earlier position's output bit-identical. So the oracle is causal
  whether or not a mask is passed, and our implementation must be causal to match it.
- Not measured: whether `attention_mask=None` is also causal under the `eager`
  implementation. Do not assume it is. If the test ever switches implementation,
  re-check.

## Step 4: design decisions

- `attention(x, w, config, layer, cos, sin) -> y`. Weights come from the full dict
  via `layer_prefix(layer) + "self_attn."`, bound once as `p`.
- **`cos`/`sin` are supplied by the caller, never rebuilt inside.** All 28 layers
  share one table, and once a KV cache exists only the caller knows the true
  positions -- seq=1 can mean position 100. This is the step 3 decision
  (`rope_cos_sin` takes a positions *tensor*) finally paying off.
- Helpers, each with a name from the domain: `project`, `split_heads`,
  `merge_heads`, `repeat_kv`, `causal_mask`, `attend`.
- `split_heads` folds in the transpose, so `rms_norm` runs *after* it rather than
  before as in HF. Equivalent, because `rms_norm` reduces over the last axis and
  `transpose(1, 2)` does not move it. Confirmed by exact-0.0 eager parity.
- Scale is `* head_dim**-0.5`, matching HF's literal operation rather than
  `/ sqrt(head_dim)`. Mathematically identical; using HF's spelling took eager
  parity from 1.4e-06 to exactly 0.0.
- `attend` reads its scale off `q.shape[-1]` rather than taking config, so it can
  be unit tested at head_dim=4 with hand-computed weights and no model.
- Lives in `tiny/attention.py`, not `tiny/layers.py` as the CLAUDE.md layout
  sketch had it.

## Step 4 mutation matrix

Twelve mutations, none survived. The interesting result was *which* tests caught
them: seven were caught **only** by the two HF parity tests. Splitting by where the
bug lived explained it -- every mutation to a *helper* was caught by that helper's
own unit test, and every mutation caught by parity alone lived inline in
`attention` itself.

That is a localization problem, not a detection one, and the fix was a smaller
unit rather than more tests. Extracting `attend` (scores, scale, mask, softmax,
blend) moved four of them:

| mutation | caught without the oracle |
|---|---|
| merge_heads without the transpose | `merge_heads_inverts_split_heads` |
| split_heads with transpose(-2,-1) | `split_heads_matches_hf_layout` |
| repeat_kv tiling instead of interleaving | `repeat_kv_matches_hf`, `repeat_kv_pairs_adjacent_heads` |
| mask `triu(0)` | `causal_mask_structure`, `is_causal` |
| project missing `.T` | `project_matches_hf` |
| attend: no 1/sqrt(d) scale | `attend_scale_is_one_over_sqrt_head_dim` |
| attend: scale 1/d | `attend_scale_is_one_over_sqrt_head_dim` |
| attend: no mask | `attend_first_position_returns_its_own_value`, `attend_is_causal` |
| attend: softmax dim=-2 | three `attend` tests |
| cos/sin recomputed internally | `uses_the_cos_sin_arguments` -- **oracle misses this entirely** |
| q_norm / k_norm weights swapped | nothing -- oracle only |
| rope applied to v as well | nothing -- oracle only |

Three lessons recorded, the first two new:

- **Testability follows decomposition.** If a bug can only be caught end to end,
  the unit is too big. More tests would not have fixed it; a smaller function did.
- **Wiring bugs are uncatchable by unit tests, correctly.** Swapped norm weights
  and RoPE-on-v leave every individual function correct. That is what the
  integration test is for. A suite where the end-to-end test covers the wiring and
  nothing else is the healthy shape, not a gap.
- **A mutation matrix is only as good as the mutations you think to write.** The
  first pass missed both the dropped `.T` in `project` and the internally-recomputed
  cos/sin -- the second being the one mutation the oracle cannot see at all.

The long-form version is in `docs/04-attention.md` under "Decomposition and
testability".

## Rule change (2026-09-23): the no-black-box line moved

`CLAUDE.md`'s hard rule banned `torch.nn` wholesale, which made `F.silu` illegal
while `torch.softmax` and `torch.rsqrt` were explicitly allowed -- three objects of
the same kind, split by which namespace PyTorch files them under. Eviatar
challenged it; the challenge was right.

The rule now bans what hides a **decision**: weights, a reduction axis, an `eps`, a
mask, a head layout. `F.silu` has none of those and is allowed by name. Everything
that mattered -- `nn.Linear`, `nn.LayerNorm`, `F.scaled_dot_product_attention` --
still fails the new test, each for several reasons at once.

Recorded because the general lesson outlives the specific call: a proxy rule
(namespace) stood in for the real one (hidden decisions), and the proxy misfired on
the first case that distinguished them.

## Working agreement change (2026-09-19)

Recorded in `CLAUDE.md`. Claude may now write unit **test** code, but only once
Eviatar has named the specific units and said what to assert about each; Claude
does not pick the units or add assertions beyond the brief. Implementations in
`tiny/` remain his. Two project skills were added to carry this:
`.claude/skills/implementation-feedback` and `.claude/skills/testing-feedback`.

Caveat for whoever reads this next: `tests/test_attention.py` was written *before*
that rule existed, with Claude choosing the units. It is the one file in `tests/`
not produced under the current agreement.

## Open questions

- `docs/05-mlp.md` was never written -- step 5 went theory -> design -> implement
  with beat 3 skipped. The theory, the worked SwiGLU example and the mutation
  matrix are in the session, not in the repo.
- `tests/test_attention.py` imports `project` from `tiny.attention`, which now
  only re-exports it from `tiny.layers`. Harmless today, breaks the moment
  `attention` stops needing it.

## Design decisions (step 1)

- `Config` is a frozen dataclass built by `Config.from_json(path)`; `model_dir()` wraps
  `snapshot_download` so no cache path is hardcoded.
- `load_weights()` returns a plain `dict[str, Tensor]` keyed exactly as the file is,
  cast to float32. No wrapper object — nothing between the code and the numbers.
- `layer_prefix(i)` -> `"model.layers.{i}."` keeps that string in one place.
- Tests derive every expected shape from `config` fields, never from literals.

## Decisions log

- **Inference only** — no training, optimizer, loss, or dataset.
- **Qwen3-0.6B** over GPT-2 (obsolete arch), SmolLM2-135M (fallback if too slow),
  DeepSeek-V2-Lite (MLA + MoE, a sequel).
- **float32 everywhere** for parity testing; max-abs-diff < 1e-4.
- **Plain functions**, not classes: `f(x, weights, config) -> y`.
- Tokenizer and chat template out of scope; use `AutoTokenizer`.
