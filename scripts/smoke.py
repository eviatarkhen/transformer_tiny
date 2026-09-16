import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 10


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

    print(model)

    inputs = tokenizer(PROMPT, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0]
    print(f"\nprompt ids: {prompt_ids.tolist()}")
    print(f"per token:  {[tokenizer.decode(i) for i in prompt_ids]}")

    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    elapsed = time.perf_counter() - start

    # generate() stops early at EOS, so count what came back instead of
    # assuming MAX_NEW_TOKENS.
    new_tokens = generated_ids.shape[1] - prompt_ids.shape[0]

    print(f"\n{tokenizer.decode(generated_ids[0])}")
    print(f"\n{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s")
    print(f"embed_tokens: {tuple(model.model.embed_tokens.weight.shape)}")
    print(f"layers:       {len(model.model.layers)}")


if __name__ == "__main__":
    main()
