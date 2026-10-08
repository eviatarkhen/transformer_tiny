"""Chat with tiny/'s Qwen3-0.6B in the terminal.

    uv run python scripts/chat.py                      # MPS, int8: the fastest setup
    uv run python scripts/chat.py --device cpu --no-int8
    uv run python scripts/chat.py --think              # let the model reason first

Greedy decoding (sampling is deferred), so answers are deterministic and can
loop. Each turn re-runs the whole conversation as the prefill -- simple, and
fast enough at chat lengths. Ctrl-D or "exit" to quit, "/reset" to clear.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `tiny`
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model_dir() otherwise contacts the Hub

import torch
from transformers import AutoTokenizer

from tiny.attention import allocate_cache
from tiny.config import MODEL_ID, Config, model_dir
from tiny.model import forward
from tiny.weights import load_weights

STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> -- from generation_config.json

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "mps"], default="mps" if torch.backends.mps.is_available() else "cpu")
parser.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
parser.add_argument("--no-int8", action="store_true", help="full-precision weights")
parser.add_argument("--think", action="store_true", help="enable Qwen3's <think> reasoning")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
config = Config.from_json(model_dir() / "config.json")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
w = load_weights(dtype=dtype, device=args.device, int8=not args.no_int8)
print(f"tiny Qwen3-0.6B | {args.device}, {args.dtype}, {'full' if args.no_int8 else 'int8'} weights"
      f"{', thinking' if args.think else ''} | Ctrl-D to quit, /reset to clear\n")


def reply(messages):
    """Greedy-decode one assistant turn, printing each token as it is chosen."""
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=args.think, return_tensors="pt"
    )
    ids = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to(args.device)
    cache = allocate_cache(config, 1, ids.shape[1] + args.max_new_tokens, like=w["model.embed_tokens.weight"])

    new, start, out, printed = ids, 0, [], ""
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            positions = torch.arange(start, start + new.shape[1], device=args.device)
            token = forward(new, w, config, positions, cache)[0, -1].argmax()
            if token.item() in STOP_IDS:
                break
            out.append(token.item())
            text = tok.decode(out, skip_special_tokens=True)  # decode the whole reply: tokens can split characters
            print(text[len(printed):], end="", flush=True)
            printed = text
            start += new.shape[1]
            new = token.view(1, 1)
    elapsed = time.perf_counter() - t0
    print(f"\n\033[2m[{len(out)} tokens, {len(out) / elapsed:.1f} tok/s]\033[0m\n")
    return printed


messages = []
while True:
    try:
        user = input("\033[1myou>\033[0m ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if user in ("exit", "quit"):
        break
    if user == "/reset":
        messages = []
        print("(cleared)\n")
        continue
    if not user:
        continue
    messages.append({"role": "user", "content": user})
    print("\033[1mqwen>\033[0m ", end="", flush=True)
    messages.append({"role": "assistant", "content": reply(messages)})
