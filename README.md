# transformer_tiny

[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) inference, written from scratch
in plain PyTorch tensor ops, as a learning exercise. No `nn.Linear`, no
`scaled_dot_product_attention`, no `transformers` modeling code -- every reduction
axis, epsilon, mask and head layout is written out by hand. The goal is
understanding, not speed, though it ended up reasonably quick.

```
text -> tokens -> embedding -> 28 x [ RMSNorm -> attention (GQA, RoPE, q/k-norm) -> RMSNorm -> SwiGLU ] -> norm -> logits
```

## What is in it

| | |
|---|---|
| `tiny/layers.py` | `project`, `rms_norm`, RoPE |
| `tiny/attention.py` | grouped-query causal attention, KV cache |
| `tiny/mlp.py` | SwiGLU feed-forward |
| `tiny/model.py` | block, embedding, LM head, `forward` |
| `tiny/generate.py` | greedy generation, with or without the cache |
| `tiny/quantize.py` | int8 weight-only quantization, per-row scales |

Components are plain functions over an explicit weight dict -- `f(x, weights, config) -> y`
-- not classes with hidden state.

## Correctness

HuggingFace's own Qwen3 is the oracle: every component is compared numerically
against the matching HF module on identical input, in float32, to 1e-4. The full
model matches HF's eager path exactly (0.0) in float32 and in bf16. After each
component went green it was broken deliberately -- mutation testing -- to find what
the tests could not see; the matrices are in [`docs/PROGRESS.md`](docs/PROGRESS.md).

```
uv run pytest        # 119 tests
```

## Running it

```
uv run python scripts/chat.py                          # chat in the terminal (MPS, int8 weights)
uv run python scripts/decode.py --dtype bf16 --device mps --int8
uv run python scripts/decode.py --torch-profile        # op table + trace.json for ui.perfetto.dev
```

Decode speed on an Apple M5 Pro, 100-token prompt + 300 tokens:

| | float weights | int8 weights |
|---|---|---|
| CPU, float32 | 42.7 tok/s | 55.9 tok/s |
| MPS, float32 | 50.3 tok/s | **72.5 tok/s** |

int8 leaves perplexity unchanged (23.85 -> 23.63 on a fixed paragraph). Sampling is
deferred: generation is greedy only.

## How it was built

Step by step -- weights, RMSNorm, RoPE, attention, MLP, block, full model, greedy
generation, KV cache, then optimization (bf16, MPS, preallocated cache, int8) -- with
Claude as tutor and reviewer. [`CLAUDE.md`](CLAUDE.md) holds the working rules,
[`docs/PROGRESS.md`](docs/PROGRESS.md) the state, decisions, gotchas, and who wrote
what at each step. `docs/NN-*.md` are the learning notes.

## Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/). The `Qwen/Qwen3-0.6B` checkpoint
(1.4 GB) downloads into the HuggingFace cache on first use. The scripts then run
offline; the tests reach the Hub unless `HF_HUB_OFFLINE=1` is set.

## License

MIT
