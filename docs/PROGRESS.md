# Progress

State of the project. Rules live in `CLAUDE.md`; this file is status only.
Updated at the close of every step.

**Session-start ritual:** read `CLAUDE.md` → read this file → `uv run pytest` →
continue from "Next action".

---

## Next action

**Step 11: sampling.** Not started; theory not yet delivered. Step 10 closed
2026-10-08 with three pieces (bf16 + MPS, preallocated cache, int8); int4
and `torch.compile` left as sequels.

Step 10's final speed, 100-token prompt + 300 tokens (MPS runs vary ~15% run to run):

| | float weights | int8 weights |
|---|---|---|
| CPU f32 activations | 42.7 | **55.9** |
| CPU bf16 activations | 30.3 | 39.1 |
| MPS f32 activations | 50.3 | **72.5** (best) |
| MPS bf16 activations | 59.3 | 64.7 |

**`torch.compile` withdrawn** by Eviatar (2026-10-08), before theory. The
notes below on why it was next are kept for a later return.

Why compile was next: MPS gained nothing from piece 2 (43.4 -> 43.4 tok/s f32)
and only 1.16x over CPU from piece 1, against 1.9x on its matmuls alone.
~600 small ops per turn, each with a fixed GPU launch cost (2.8 us vs 0.7 us
on CPU), dominate there. Fusion attacks exactly that. Shapes are now static
per buffer, except the filled-length slice `[:, :, :len]`, which grows every
turn -- expect compile to recompile on it, or to need the whole buffer
attended with a positions-based mask. That is the design question.

Benchmark with `scripts/decode.py` (`--dtype bf16 --device mps`,
`--no-cache`, `--torch-profile`). Speed to beat, 100-token prompt + 300
tokens: **CPU f32 43.2, CPU bf16 31.5, MPS f32 43.4, MPS bf16 47.9 tok/s.**

## Step status

| # | Step | State | Proven by |
|---|---|---|---|
| 0 | Environment + oracle | **done** | `scripts/smoke.py` runs |
| 1 | `config.py` + `weights.py` | **done** | `tests/test_weights.py`, 7 tests |
| 2 | `rms_norm` | **done** | `tests/test_rmsnorm.py`, 6 tests |
| 3 | `rope` | **done** | `tests/test_rope.py`, 6 tests |
| 4 | `attention` | **done** | `tests/test_attention.py`, 51 tests |
| 5 | `mlp` | **done** | `tests/test_mlp.py`, 4 tests |
| 6 | `block` | **done** | `tests/test_block.py`, 7 tests |
| 7 | full model | **done** | 5 files, 8 tests: `test_embed`, `test_run_blocks`, `test_final_norm`, `test_lm_head`, `test_forward` |
| 8 | greedy generation | **done** | `tests/test_generate.py`, 2 tests |
| 9 | KV cache | **done** | `tests/test_kv_cache.py`, 7 tests; `test_generate` now 4 (both cache modes) |
| 10 | optimize | **done** (bf16 + MPS, preallocated cache, int8) | `test_dtype_device` 7, `test_quantize` 10 |
| 11 | sampling | next | — |

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

## Step 10, piece 3: int8 weight-only quantization

**Written by Claude, all decisions Claude's**, at Eviatar's request ("you
write and make all decisions"). `CLAUDE.md` amended to bring quantization in
scope; `torch.compile` withdrawn.

Measured first: `torch._weight_int8pack_mm` (fused dequantize-in-the-matmul)
on 84 3072x1024 matrices -- CPU 4.59 ms vs 8.04 f32 / 5.24 bf16; MPS 1.22 vs
3.98 / 2.13. Spelled-out `x @ (q * scale).T` is 17-22 ms: 2-10x *slower*
than not quantizing.

Decisions:

- **`tiny/quantize.py`**: `quantize_rows(weight) -> (q int8, scale)`,
  symmetric, one scale per output row, `scale = max|row| / 127` floored at
  1e-12 for all-zero rows, kept in the weight's dtype (the kernel needs the
  activations' dtype), q rounded against the stored scale.
  `quantize_weights(w)` returns a new dict. Reached through
  **`load_weights(..., int8=True)`** (added on Eviatar's question), so dtype,
  device and quantization are chosen in one place. Not in `Config`: that
  describes the checkpoint, and the same checkpoint runs in every format.
- **Quantized: the 7 `*_proj.weight` of every layer, and the LM head** as a
  separate int8 copy under `LM_HEAD_INT8 = "lm_head.int8"` -- a key no
  checkpoint has, so step 7's tied-table reading is unchanged for plain
  weights. The embedding lookup keeps the full table (one row per token).
  Norms untouched. The file's duplicate `lm_head.weight` stays, unused.
- **`project(x, weight)` takes a tensor or a `(q, scale)` pair**: one
  `isinstance` branch in one place, no call site changed. The `w` dict
  annotations (`dict[str, Tensor]`) are now loose -- values can be pairs.
- **The private kernel accepted**: rounding and scales are ours; a test
  checks it against the spelled-out arithmetic.

Quality, measured: logits relative error vs float32 on random ids -- int8
per-row 9.5%, per-matrix 22%, bf16 3.1%. Perplexity on a fixed 99-token
paragraph -- float32 23.85, bf16 23.81, **int8 23.63** (noise), per-matrix
26.17. Greedy text on the France prompt identical in every variant. LM head
quantization adds almost nothing (9.46% -> 9.51%).

### Step 10 piece 3 mutation matrix

| mutation | caught by |
|---|---|
| one scale for the whole matrix | error bound, logits band, perplexity |
| truncate instead of round | error bound, logits band, perplexity |
| 255 levels | hand example, error bound, logits band, perplexity |
| no zero-row floor | **only** hand example |
| scale in float32, not the weight's dtype | **none at first** -- equivalent on f32; added `keeps_the_scale_in_the_weights_dtype` |
| no int8 lm_head copy | **only** the selection test (quality barely moves) |
| embedding table quantized too | selection, logits, perplexity, MPS |
| only attention projections quantized | **only** the selection test |
| lm_head ignores the int8 copy | **only** lm_head_reads_the_int8_copy |
| project: dequantize then multiply | **none at first** -- same numbers, 2-10x slower; added `uses_the_fused_kernel`, a profiler-based performance test |
| project: scale ignored | kernel parity, logits, perplexity, MPS |

The second added test is the first deterministic performance test in the
suite: it asserts which kernel runs, not how fast.


## Step 10, piece 2: preallocated cache, no repeat_kv

**Written by Claude**, code and tests, as piece 1. Measured first, decode
turns only (CPU f32): cache copies (`cat` in `update_cache`, `copy_` from
`repeat_interleave`) were 12% of a turn at 100 context, 29% at 1,000, **44%
at 3,000** -- more than the weight matmuls. At 3,000 tokens the cache is
688 MB, copied every turn to add 229 KB.

Design:

- **`allocate_cache(config, batch, max_len, like)`** (Eviatar renamed it
  from `new_cache`): every layer's k and v buffers allocated once, dtype and
  device from `like` via `new_zeros`. A slot is `{"k", "v", "len"}`. greedy
  sizes it prompt + max_new_tokens -- one spare, since the last generated
  token is never fed back (the "max_len 1 short" mutant is equivalent).
- **`update_cache` keeps its signature**: writes only the new tokens at
  `len`, returns a *view* of the filled part, raises ValueError on overflow.
  The write position is a Python int, **not** read from `positions` as first
  proposed: `.item()` on an MPS tensor makes the CPU wait for the GPU, 28
  times a turn. `positions` stays for RoPE only.
- **`repeat_kv` deleted; `attend` groups q instead.** `q.reshape(batch,
  kv_heads, n_rep * q_len, head_dim)` is the adjacent pairing. Folded into
  rows rather than broadcast against `k.unsqueeze(2)`: matmul implements a
  broadcast batch dim by expanding and copying, which is the copy being
  removed. Scores are viewed `(..., n_rep, q_len, k_len)` for the mask, so
  `causal_mask` is untouched.
- **Stream tolerance in `test_run_blocks` raised to 1e-3**, with the reason
  in the file. The stream reaches ~1,923, where one float32 step is 2.3e-4;
  1e-4 held only by bit-exactness with HF eager's matmul shapes. Grouped
  attend measured 7e-7 vs the expanded spelling, 1.5e-4 on the stream,
  4e-5 on the logits (which keep 1e-4).

Result, decode turns: 37.5 -> 44.9 tok/s at 100 context, 25.9 -> 38.5 at
1,000, **13.9 -> 26.0 at 3,000**; copies 44% -> 0.7%. Whole runs: CPU f32
36.0 -> 43.2, CPU bf16 21.2 -> 31.5, MPS unchanged (overhead-bound).

### Step 10 piece 2 mutation matrix

| mutation | caught by |
|---|---|
| len never advanced | all three update_cache, cache length, cached forward, greedy[cache] |
| whole buffer returned, not the filled part | both update_cache, cached forward, greedy[cache] |
| k and v swapped on write | both update_cache, cached forward, greedy[cache] |
| overflow check removed | **only** refuses_to_overflow |
| buffers ignore `like` | only greedy_on_mps -- **then a hole off-Mac**; added `test_allocate_cache_follows_like` |
| one buffer shared by all layers | cache length, cached forward, greedy, MPS greedy |
| attend groups query heads tiled | attend_groups_adjacent + 34 parity tests |
| mask on the folded scores | 41 tests |
| greedy max_len 2 short | greedy[cache], MPS greedy |
| greedy max_len 1 short | **none -- equivalent** (one spare slot) |


## Step 10, piece 1: bf16 and MPS

**Written by Claude**, code and tests, at Eviatar's request ("you write
everthing"). Beat 3 skipped. Theory went: how profiling works at three levels
(cProfile / torch.profiler / OS: `time -l`, `sample`, py-spy, Instruments),
how the candidates were ranked (map each op to our code, Amdahl's bound, then
bytes-per-multiply), how a tolerance is set (noise floor < tolerance <
smallest bug).

Profile of a cached turn (CPU f32, 100 context): `mm` 68% (exactly 197 per
turn = 28 x 7 + lm_head), `copy_`+`cat` 12%, `bmm`+softmax 7%. One token reads
all 596M weights: 2.4 GB / 25 ms = ~95 GB/s. Memory-bound, by inference --
not measured with hardware counters (no Xcode on this machine).

Design:

- **`load_weights(dtype=float32, device="cpu")`** is the one place both are
  chosen. Defaults unchanged, so every float32 test is untouched. float32
  stays the testing dtype: bf16 noise (~0.7 on the logits) would hide the
  bugs the 1e-4 suite catches.
- **Every tensor tiny/ creates inherits from an input**, never a default:
  `rope_cos_sin` from `positions.device`; `causal_mask(scores)` via
  `scores.new_full` (on Eviatar's challenge -- it first took `dtype`/`device`
  arguments); greedy's positions from `ids.device`. Factory functions
  (`full`, `arange`) default to CPU float32; ops on existing tensors promote.
- **Precision islands matched to HF**, read from the installed source:
  `rms_norm` computes in float32 and casts back *before* the weight; softmax
  `dtype=float32` then cast back; RoPE tables built in float32, cast to x's
  dtype inside `apply_rope` (moved there from `attention`, also on his
  challenge).
- **CPU bf16 kept bit-exact with HF**, slow, on Eviatar's call. Upcasting the
  attention bmm to float32 is 6x faster at 100 context but changes 2 in
  ~11,000 scores by one bf16 step. MPS bf16 is the fast path.

Measured: our bf16 equals HF bf16 eager **exactly (0.0)** on CPU, so the bf16
path is tested at the float32 tolerance. MPS f32 vs CPU f32: 6.4e-5.

### Step 10 piece 1 mutation matrix

| mutation | caught by |
|---|---|
| rms_norm without the float32 island | rms_norm_bf16, forward_bf16 |
| rms_norm casts back after the weight | rms_norm_bf16, forward_bf16 |
| RoPE angles rounded to bf16 | 32 tests, every float32 parity |
| cos/sin not cast to x's dtype | forward_bf16 |
| mask dtype ignored | causal_mask dtype test |
| RoPE inv_freq on default device | both MPS parity |
| mask device ignored | both MPS parity |
| greedy positions on default device | **only** greedy_on_mps |
| load_weights ignores device / dtype | load_to_mps / three bf16 tests |
| softmax without the float32 island | **none -- equivalent**: PyTorch's bf16 softmax accumulates in float32, bitwise equal on CPU and MPS |
| attend not passing the mask's dtype (pre-`new_full` design) | **none -- equivalent**: float32 mask promotes exactly, island casts back |


## Step 9: KV cache

**Written by Claude** at Eviatar's explicit request, as in steps 6 and 8;
tests chosen by Claude at his request ("test all and write the tests"). Beat 3
(notes) skipped at his request. Theory delivered in small pieces; what landed
was the score grid as a picture (rows = queries, columns = keys), and the
tamper experiment below, not prose.

Design, agreed beat 2:

- **The cache is plain data**: `cache = [None] * num_hidden_layers`, a slot
  per layer holding `(k, v)`. Eviatar rejected a `KVCache` class as foreign
  to a codebase with no classes, and two parallel function families as four
  duplicated pairs. Mutated **in place**, documented as such.
- **`update_cache(cache, layer, k_new, v_new) -> (k, v)`** in
  `tiny/attention.py` holds all the bookkeeping, so it tests on toy tensors.
- **`cache=None` added to `attention`, `block`, `run_blocks`, `forward`.**
  None is exactly the old behaviour: no existing call site or test changed.
- **Stored after RoPE, before `repeat_kv`.** A key's rotation depends only on
  its own position, so it is final once made; 8 heads instead of 16 halves
  the memory.
- **`causal_mask(q_len, k_len=None)`**, diagonal shifted by `k_len - q_len`;
  `attend` passes both lengths. Not needed for greedy's shapes -- the old
  square mask worked by broadcasting a (1, 1) zero across the row -- but kept
  on Eviatar's call: "i dont like luck". It breaks (RuntimeError) the moment
  more than one new token meets a non-empty cache.
- **One `greedy`, `use_cache=True` default** (as HF). Eviatar asked for one
  implementation over a separate `greedy_cached`. The two modes differ in one
  line: `start`, the first position forward has not seen, advances only with
  a cache. `use_cache=False` is the benchmark baseline.

Verified: both modes id-for-id equal to each other and to HF's generate
(10/20/30 tokens, chat prompt). Cached vs uncached logits drift 5e-5 (7e-5 on
random ids) with no bug: different input shapes, different summation order.

Why old queries are not cached, as finally landed: corrupting an old token's
q leaves the new token's output unchanged (0.0000); corrupting its k or v
moves it (0.04, 3.04). Old rows of the grid exist only to make the old tokens'
next-layer K and V -- once those are stored, the rows have no reader.

### Step 9 mutation matrix

| mutation | caught by |
|---|---|
| new tokens joined before old | update_cache append, cached forward, greedy[cache] |
| slot never written | both update_cache, cache length, cached forward, greedy[cache] |
| k and v swapped in the slot | both update_cache, cached forward, greedy[cache] |
| keys stored before RoPE | cached forward, greedy[cache] -- wiring, end to end only |
| keys stored after repeat_kv | **only** the cache-shape test (equivalent output) |
| mask diagonal not shifted | mask test, both attend, cached forward, greedy[cache] |
| attend passes q_len only (the "luck" mask) | **only** attend[q_len=3] and cached forward's 2-token chunk |
| greedy feeds the new token at position 0 | greedy[cache] |
| greedy never advances start (tokens stored twice) | greedy[cache] |
| use_cache ignored, always caching | greedy[no cache] |

Not tried: batch > 1 (greedy refuses it), calling forward twice by hand with
one cache. The "luck" row is why a multi-token chunk after prefill is in the
forward test: single-token decoding cannot see the mask at all.


## Step 8: greedy generation

**Written by Claude** at Eviatar's explicit request, tests chosen by Claude at
his request (as for `lm_head` and `forward`). Theory delivered; beat 2 skipped.

Design, in `tiny/generate.py`:

- `greedy(ids, w, config, max_new_tokens, stop_ids=()) -> ids`. Returns the
  **whole sequence**, prompt then new tokens, the same contract as HF's
  `generate`, so the two compare directly.
- **Stop ids come from the caller**, not `Config`. `config.json` lists one stop
  token (151645); `generation_config.json` lists two (151645, 151643). Reading
  them from `Config` would silently miss one. A stop token is **kept** in the
  output, as HF does.
- **One prompt only**, enforced with `ValueError`. Batched prompts finish at
  different times and need padding -- out of scope.
- Positions are `arange(ids.shape[1])` every turn: correct without a cache,
  the line step 9 replaces.

Verified: 20 tokens on the plain prompt and a 4-token chat answer
(`'HELLO!<|im_end|>'`, stopped on 151645) both identical to HF's
`generate(do_sample=False)`, id for id.

Tests use the **default sdpa** oracle, not eager: generation compares argmax
choices, which flip only if two logits are within ~1e-4, and the closest
top-two gap over 30 steps on the prompt measured 0.024. If a future prompt has
a near-tie, switch to `hf_model_eager`.

### Step 8 mutation matrix

| mutation | `matches_hf_generate` | `stops_at_a_stop_token` |
|---|---|---|
| reads the first row of logits | caught | caught |
| argmin instead of argmax | caught | caught |
| loop one token short | **only this** | -- |
| stop ids ignored | -- | **only this** |
| stop checked before appending (token dropped) | -- | **only this** |
| positions stop advancing after the prompt | caught | caught |

Each test owns its concern. The positions mutation was artificial (clamped);
the realistic version only exists once there is a cache.


## Step 7: who wrote what

Split into five pieces at Eviatar's request, guided one at a time.

| piece | implementation | tests |
|---|---|---|
| `embed` | Eviatar | his plan, Claude wrote it |
| `run_blocks` | Eviatar (two bugs found in review: no `return`, 26 of 28 blocks) | Claude chose, at his request |
| `final_norm` | Eviatar | Claude chose, at his request |
| `lm_head` | **Claude**, at his explicit request | Claude chose, at his request |
| `forward` | **Claude**, at his explicit request | Claude chose, at his request |

The `CLAUDE.md` rule is unchanged; these are per-piece exceptions he made.

## Step 7: design decisions

- Five functions in `tiny/model.py`: `embed`, `run_blocks`, `final_norm`,
  `lm_head`, and `forward`, which is four lines reading in pipeline order.
- `run_blocks` and `forward` take **`positions`**, required, no `arange`
  default. Step 4's decision carried to the top: a default would reintroduce
  the bug the positions tests exist to catch.
- `cos`/`sin` built **once** in `run_blocks`, shared by all 28 blocks.
  **Superseded 2026-10-06** (after step 8, on Eviatar's objection that the
  tables were built far above their only user): `block` and `attention` now
  take `positions`, and `attention` builds its own tables. `positions` is the
  real dependency; cos/sin are attention's detail, and rebuilding a
  (seq, head_dim) table per layer costs nothing next to the matmuls. Step 7's
  "rebuilt inside the loop" equivalent mutant is now the actual design. The
  attention test `test_uses_the_cos_sin_arguments` became
  `test_uses_the_positions_argument`, same assertion.
- `lm_head` reads **`model.embed_tokens.weight`**, not the duplicate
  `lm_head.weight`: it states the model is tied, and survives tied checkpoints
  that omit the duplicate. No branch on `tie_word_embeddings` -- it would be
  dead, untested code here.
- `lm_head` returns logits for **every** position; slicing to the last row is
  the caller's job.
- `final_norm` kept as its own function; in `forward` it reads fine.

## Step 7: testing decisions

- **`embed` asserts exact equality** (`torch.equal`), departing from the 1e-4
  convention on purpose: a lookup does no arithmetic, so there is nothing to
  tolerate.
- **Whole-stack tests use HF's eager path** via the `hf_model_eager` fixture in
  `conftest.py`, which toggles the shared model with `set_attn_implementation`
  and restores it. Against eager, `forward` is **0.0** on every logit. Against
  the default sdpa it is 6.0e-5 on the prompt and **1.1e-4 on random (2, 7)
  ids -- past the tolerance with no bug present.** The restore was verified by
  running `test_run_blocks` before `test_block` (which relies on sdpa's
  causal `mask=None`): 60 passed.
- HF's stack output (before the final norm) is captured with a **forward hook**
  on `hf_model.model.layers[-1]`; `hidden_states` cannot supply it.
- HF accepts custom `position_ids`, so the positions tests are true parity at
  gapped positions (0, 3, 6, ...), not just "the output changes".
- Random token ids are realistic input for every piece here, because they go
  through `embed` first.

## Step 7 mutation matrices

`embed`: transposed table (IndexError), ids off by one, batch/seq swapped
(shape), ids ignored, first sentence only -- all caught. `lm_head.weight` key
passes: an **equivalent mutant**, identical tables in this file.

`run_blocks`: 26 blocks, reversed order, always layer 0, stream not threaded
-- caught by both tests. Positions rebuilt internally -- caught **only** by
the non-default-positions test. cos/sin rebuilt inside the loop passes: an
equivalent mutant (28x slower, same answer).

`final_norm`: no norm, wrong key, weight ignored -- caught at both scales. eps
hardcoded 1e-5, eps dropped -- caught **only** at input scale 1e-3.

`lm_head`: transpose forgotten (raises), last position only, softmax applied,
1/sqrt(hidden) scaling -- caught by both. `lm_head.weight` key -- caught
**only** by the tied-table test, which drops that key. The same code change
that was an equivalent mutant in `embed` is a regression here, because the
docstring promised it.

`forward`: final norm skipped, norm before the blocks, last position only,
block output discarded -- caught by both. Positions rebuilt -- caught **only**
by the non-default-positions test.

Lessons recorded:

- **"Predicts Paris" is a useless test.** Running 26 of 28 blocks moved the
  stream by 5,110 and still predicted `' Paris'`; applying softmax keeps the
  argmax. Only numeric parity catches either.
- **Equivalent mutants are a category, not a hole.** Different code, same
  answer for every input. Note them and move on -- unless a test describes a
  context where they differ, as the tied-table test does.


## Step 6: who wrote it

`tiny/model.py` and `tests/test_block.py` were written by **Claude**, at Eviatar's
explicit authorization for this step only (2026-09-30), and reviewed by him before
commit. Beat 2 was skipped as part of that. This is a one-step exception, not a
change to the standing rule in `CLAUDE.md`.

## Step 6: design decisions

- `block(x, w, config, layer, cos, sin) -> y` in `tiny/model.py` (now
  `positions` instead of `cos, sin`; see step 7's superseded note). Takes `config`
  (unlike `mlp`) because it reads `rms_norm_eps` and forwards to `attention`.
- **The flow is written out plainly**, four lines that mirror HF's forward:
  normalize a copy, run the sublayer, add back; twice.
- A helper `pre_norm_residual(x, norm_weight, eps, sublayer)` was tried first and
  **rejected by Eviatar on readability**: passing attention and mlp in as lambdas
  hid the flow in the one function people read to understand the architecture.
  The trade, measured: the helper let four bugs (residual dropped, post-norm,
  norm on the sublayer's output, no norm) be caught by toy-sublayer unit tests
  with no model. Without it, all four still fail **all seven** parity tests --
  detection unchanged, only localization lost. Inside a block, norm and residual
  placement *is* the wiring, and wiring is what parity is for. Recorded because
  it is the first case where testability-follows-decomposition lost to
  readability, and correctly so.

## Step 6 oracle facts (measured)

- `hf_model.model.layers[i](x, position_embeddings=(cos, sin), attention_mask=None)`
  returns a **bare tensor** in v5. `self_attn` returns a tuple, `mlp` a bare
  tensor: three siblings, two conventions.
- The residual branches are corrections, not replacements: on layer 0, attention
  and mlp each contribute ~19% of the stream's norm.
- **Massive activations**: the stream's norm goes 1.8 (embedding) -> 20.4 -> 28.8
  -> 6592.5 at layer 3 and stays there. At layer 10 the largest value is 6529.2
  and the median magnitude 0.541; channels 35, 13, 1, 277 hold 99.8% of the
  squared magnitude. `input_layernorm` brings the max back to 17.1. This is why
  each branch needs its own pre-norm.

## Step 6 mutation matrix

Nine mutations, none survived. Seven tests: parity on the real stream at layers
0/1/13/27, parity on random input at layer 0 over three shapes.

| mutation | caught by |
|---|---|
| residual dropped | all seven parity |
| post-norm | all seven parity |
| norm on the sublayer's output | all seven parity |
| no norm | all seven parity |
| the two norms swapped | all seven parity |
| mlp half reuses `input_layernorm` | all seven parity |
| mlp half fed the pre-attention stream | all seven parity |
| norm weights always from layer 0 | real-stream parity at 1/13/27 **only** |
| mlp always reads layer 0 | real-stream parity at 1/13/27 **only** |

As in step 5, sweeping parity over more than one layer is what catches a
hardcoded layer index; nothing else can.


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

- **`torch._weight_int8pack_mm` requires N % 32 == 0 and K % 32 == 0.**
  Every Qwen3-0.6B matrix qualifies (vocab 151,936 = 32 x 4,748); a model
  with an odd vocab size would raise.
- **Max-abs logit diff can mislead where perplexity does not.** int8 on MPS
  vs CPU: one logit 0.30 apart on random ids (float noise born at layer 2's
  massive activation, amplified), yet perplexity 23.6288 vs 23.6293 and
  identical greedy text. Pick the metric that states the claim.
- **A benchmark that reuses one matrix measures the cache, not memory.** One
  3072x1024 float32 matrix (12.6 MB) fits in the M5 Pro's cache: bf16 looked
  *slower* than float32. 84 distinct matrices (1 GB) gave the real answer,
  bf16 1.54x faster. Memory-bound benchmarks need data bigger than cache.
- **CPU bf16 `bmm` has no fast kernel here.** The attention score matmuls
  went 10.5 -> 148 ms per 10 turns in bf16; weight `mm` improved 183 -> 132.
  Net: CPU bf16 is slower than float32, and worse with longer context.
- **MPS is asynchronous.** Python queues work and returns; without
  `torch.mps.synchronize()` before reading the clock, a timer measures
  queueing. The torch.profiler CPU table has the same blind spot.
- **zsh does not word-split unquoted variables.** `for a in "--dtype bf16"`
  then `cmd $a` passes one argument; argparse rejects it. Use `${=a}`.
- **`model_dir()` calls `snapshot_download`, which contacts the Hub even
  when the model is cached.** On 2026-10-06 the Hub was unreachable and
  `load_weights()` hung for minutes, then raised `RemoteProtocolError`. Set
  `HF_HUB_OFFLINE=1` to read straight from the cache.
- **A (1, 1) mask broadcasts silently across a (1, k_len) score row.** The
  square causal mask was "correct" for single-token cached decoding by
  accident. Any shape bug that happens to broadcast passes; test the shape
  that does not.
- **HF's `generate()` samples unless told not to.** The checkpoint's
  `generation_config.json` sets `do_sample: true`. A greedy oracle must pass
  `do_sample=False`.
- **A uniform position shift is exact in theory, approximate after 28 blocks.**
  The step 4 gotcha below measured 0..4 vs 100..104 as byte-identical at the
  attention level. Through the whole stack, 0..6 vs 100..106 differ by 1.7e-3:
  float32 noise in the larger rotation angles, amplified by depth. Still
  negligible next to a real change of gaps (133).
- **Absolute 1e-4 parity fails on deep layers fed random input, with no bug.**
  Block parity at layer 27 on `randn` was 1.1e-3. Localized stage by stage: our
  mlp given HF's exact input is 0.0; attention's ~2.5e-5 sdpa-vs-matmul drift is
  amplified ~40x by layer 27's `post_attention_layernorm` weight (max **192**) and
  an mlp output reaching ~1900. On the real stream layer 27 actually receives,
  the same comparison is 3e-5. Deep layers are only well-conditioned in
  distribution: feed them real hidden states, not noise.
  **Proven to be kernel arithmetic, not a bug:** against HF's *eager* attention
  (the same matmul-softmax-matmul we wrote) with an explicit causal mask, the
  block is **exactly 0.0** -- random input at layer 27, every one of the 28
  layers alone on the real stream, and all 28 chained end to end. The sdpa
  numbers for the same cases are 1.2e-3, 3.0e-5 and 4.4e-4 (3e-7 relative). The
  whole diff is sdpa summing in a different order, amplified by large weights.
- **Under `eager`, `attention_mask=None` is NOT causal.** Measured at layer 0:
  `mask=None` vs an explicit causal mask differ by 0.49, and changing the last
  token moves position 0's output by 0.088. Under sdpa, `None` means causal.
  Same argument, opposite meaning, depending on a config string. Any test that
  uses eager must pass `torch.full((seq, seq), -inf).triu(1)[None, None]`.
- **`hidden_states` has 29 entries and the last one is post-final-norm.**
  `layers[i](hidden_states[i]) == hidden_states[i+1]` exactly for i in 0..26, but
  `hidden_states[28]` is `model.norm(layers[27](...))`. Easy to mistake for the
  last block's output.
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
- ~~Not measured~~ **Measured 2026-10-02: under `eager`, `attention_mask=None` is
  NOT causal** (diff 0.49 vs an explicit mask; see Gotchas). Our implementation
  matches eager-with-causal-mask exactly, 0.0, through all 28 layers.

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

- `docs/05-mlp.md` was never written -- beat 3 was skipped in step 5.
  `docs/06-block.md` exists (2026-10-02) and covers the sdpa-vs-eager testing
  findings. The theory, the worked SwiGLU example and the mutation
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
