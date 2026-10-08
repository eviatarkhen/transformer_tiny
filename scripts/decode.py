"""A fixed workload for profiling: prefill a prompt, then decode with the KV cache.

    uv run python scripts/decode.py                 # just time it
    uv run python scripts/decode.py --torch-profile # + op table and trace.json
    uv run python scripts/decode.py --no-cache      # the step 8 path, for contrast
    uv run python scripts/decode.py --dtype bf16 --device mps
    uv run python scripts/decode.py --int8          # int8 weights, see tiny/quantize.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `tiny`
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model_dir() otherwise contacts the Hub

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tiny.attention import allocate_cache
from tiny.config import Config, model_dir
from tiny.model import forward
from tiny.weights import load_weights

parser = argparse.ArgumentParser()
parser.add_argument("--prompt-len", type=int, default=100)
parser.add_argument("--tokens", type=int, default=50, help="decode turns to run")
parser.add_argument("--no-cache", action="store_true")
parser.add_argument("--torch-profile", action="store_true")
parser.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
parser.add_argument("--int8", action="store_true", help="int8 weights, --dtype activations")
args = parser.parse_args()

t0 = time.perf_counter()
config = Config.from_json(model_dir() / "config.json")
dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
w = load_weights(dtype=dtype, device=args.device, int8=args.int8)
print(f"load:    {time.perf_counter() - t0:6.2f} s")

torch.manual_seed(0)
ids = torch.randint(0, config.vocab_size, (1, args.prompt_len)).to(args.device)
max_len = args.prompt_len + args.tokens + 1
cache = None if args.no_cache else allocate_cache(config, 1, max_len, like=w["model.embed_tokens.weight"])


def decode():
    """Prefill, then one token per turn -- greedy's loop, without the stop check."""
    seq, start = ids, 0
    for _ in range(args.tokens + 1):
        with record_function("turn"):
            logits = forward(seq[:, start:], w, config, torch.arange(start, seq.shape[1], device=args.device), cache)
        seq = torch.cat([seq, logits[0, -1].argmax().view(1, 1)], dim=1)
        if cache is not None:
            start = seq.shape[1] - 1


with torch.no_grad():
    t0 = time.perf_counter()
    if args.torch_profile:
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            decode()
    else:
        decode()
    if args.device == "mps":
        torch.mps.synchronize()  # the GPU runs asynchronously: wait before reading the clock
    elapsed = time.perf_counter() - t0

print(f"decode:  {elapsed:6.2f} s, {args.tokens / elapsed:5.1f} tok/s (includes the prefill)")

if args.torch_profile:
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    prof.export_chrome_trace("trace.json")
    print("timeline written to trace.json -- open it at https://ui.perfetto.dev")
