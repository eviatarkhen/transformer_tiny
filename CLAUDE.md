# transformer_tiny

Qwen3-0.6B **inference** written from scratch, by hand, as a learning exercise.
Not a shipping project. The goal is understanding, so the process matters more than
the speed.

**Read `docs/PROGRESS.md` next** — this file holds rules, that file holds state.

## Who does what

Eviatar writes **every line** of `tiny/`, and decides everything that goes in `tests/`.
He is a CS / engineering manager, fluent in software but not in ML terminology.

Claude writes: theory explanations, function stubs (signature + docstring only), test
specs described in prose, code reviews, `CLAUDE.md`, and `docs/PROGRESS.md`.

Claude does **not** write implementations in `tiny/`. If asked for "just this one
function", offer a stub and a worked explanation instead.

**Tests are different.** Claude may write unit test code, but only once Eviatar has
named the specific units and said what about them to test. Claude does not pick the
units, and does not invent extra assertions beyond what was asked. Absent that brief,
the answer is still a prose test spec, not code. The HuggingFace oracle wiring in
`tests/conftest.py` is reference code and always Claude's.

The point of the constraint is that deciding *what is worth asserting* is the part
that teaches; typing out `assert_close` is not.

## The six beats of every step

1. **Theory** — explain what the piece is and why it exists. Lead with tensor shapes
   and intuition; use math only where it genuinely clarifies. Define terminology on
   first use. Never assume ML vocabulary.
2. **Design** — agree on function signatures and shapes *before* any code is written.
3. **Notes** — Eviatar writes `docs/NN-name.md` in his own words.
4. **Implement** — Eviatar writes the implementation. He then names the units to
   cover and what each should assert; Claude writes those tests.
5. **Review** — Eviatar asks for a review; Claude reviews the module and the tests
   together and suggests changes.
6. **Record** — Claude updates `docs/PROGRESS.md` and commits. A step is not closed
   until this happens.

Do not run ahead. Explain step N, stop, and wait. Never explain step N+1 while he is
still working on N.

## Hard rule: no black boxes

In `tiny/`, nothing from `torch.nn` and nothing that implements a transformer concept.
Banned: `nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention`,
`F.scaled_dot_product_attention`, and anything from `transformers` modeling code.

Allowed: raw tensor ops — `matmul`/`@`, `softmax`, `exp`, `rsqrt`, `mean`, `cat`,
`stack`, `reshape`, `view`, `transpose`, `arange`, indexing.

Claude uses `torch.nn` and `transformers` freely in `tests/` — that is the oracle.

## The model

Qwen3-0.6B, already in the local HF cache. Verified from its `config.json`:

| | |
|---|---|
| layers | 28 |
| hidden_size | 1024 |
| heads | 16 query / 8 key-value (GQA) |
| head_dim | **128** — *not* hidden_size/num_heads (= 64). Never assume they match. |
| intermediate_size | 3072 (SwiGLU) |
| norm | RMSNorm, eps 1e-6 |
| positions | RoPE, theta 1e6 |
| vocab | 151,936, **tied** embeddings (LM head *is* the embedding matrix) |
| biases | none, anywhere |
| rope_scaling / sliding_window | none |

Consequence of head_dim=128: attention projections are not square.
q_proj 1024→2048, k_proj 1024→1024, v_proj 1024→1024, o_proj 2048→1024.

## Testing convention

HuggingFace's own Qwen3 implementation is the oracle. Every component is compared
numerically against the matching HF module on identical input.

- **float32 everywhere**, including forcing the HF oracle to float32, even though the
  checkpoint is bfloat16. Otherwise bf16 rounding noise swamps the comparison and a real
  bug is indistinguishable from arithmetic drift.
- Compare **max absolute difference < 1e-4**. Never `==`.
- Fixed seeds. The HF model fixture is session-scoped — loading it is slow.

## Layout

```
tiny/config.py     # dataclass parsed from the model's config.json
tiny/weights.py    # safetensors -> dict of tensors
tiny/layers.py     # rms_norm, rope, attention, mlp
tiny/model.py      # block, forward
tiny/generate.py   # sampling loop
```

Plain functions over explicit weight tensors, not classes with hidden state.
A component is `f(x, weights, config) -> y`. Every data dependency stays visible.

## Out of scope, deliberately

Training, the BPE tokenizer (use `AutoTokenizer`), chat templates, quantization,
flash attention. Each is a fine sequel; none is needed to understand the transformer.

## Commands

```
uv run pytest                  # the ground truth
uv run python scripts/smoke.py # HF reference output
```
