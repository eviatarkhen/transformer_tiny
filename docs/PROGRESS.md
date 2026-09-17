# Progress

State of the project. Rules live in `CLAUDE.md`; this file is status only.
Updated at the close of every step.

**Session-start ritual:** read `CLAUDE.md` → read this file → `uv run pytest` →
continue from "Next action".

---

## Next action

**Step 3: `rope`.** Theory delivery was interrupted partway. Resume by finishing
the theory (see "Step 3 theory" below for what has and has not been covered), then
agree the design, then Eviatar implements `tiny/layers.py` + `tests/test_rope.py`.
No step 3 code exists yet.

---

## Step status

| # | Step | State | Proven by |
|---|---|---|---|
| 0 | Environment + oracle | **done** | `scripts/smoke.py` runs |
| 1 | `config.py` + `weights.py` | **done** | `tests/test_weights.py`, 7 tests |
| 2 | `rms_norm` | **done** | `tests/test_rmsnorm.py`, 6 tests |
| 3 | `rope` | not started | — |
| 4 | `attention` | not started | — |
| 5 | `mlp` | not started | — |
| 6 | `block` | not started | — |
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

## Gotchas

Running list of things that actually bit us, and how they were found.
The highest-value section in this file — these are exactly what gets lost.

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

Still to cover before he implements:

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

Eviatar has not yet written the per-step notes (`docs/00-overview.md`,
`01-weights.md`, `02-rmsnorm.md`). Only PROGRESS.md exists. Do not assume the notes
are there when resuming.

## Open questions

- Oracle style for *attention* (step 4): `Qwen3Attention.forward` needs
  `position_embeddings`, an explicit causal mask and a `Cache`, so constructing it
  standalone is awkward. Likely answer: run the full HF model once with forward hooks
  and capture each submodule's (input, output) pair. Decide at step 4.
  RMSNorm and MLP are simple enough to instantiate directly.

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
