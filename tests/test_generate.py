import pytest
import torch

from tiny.generate import greedy

STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> -- from generation_config.json


def hf_greedy(ids, hf_model, max_new_tokens):
    """The oracle: HF's generate, forced greedy.

    do_sample=False is not optional. This checkpoint's generation_config.json
    sets do_sample=True, so a bare generate() samples and returns different
    text. HF stops on both STOP_IDS by default, from that same file.
    """
    with torch.inference_mode():
        return hf_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)


@pytest.mark.parametrize("use_cache", [True, False])
def test_matches_hf_generate(config, weights, hf_model, hf_tokenizer, use_cache):
    """Twenty greedy tokens, identical to HF's, id for id.

    Exact equality because token ids are integers. Default sdpa rather than
    eager is fine here: an argmax only flips if two logits sit within the
    ~1e-4 drift of each other, and the closest top-two gap over 30 steps on
    this prompt measured 0.024.

    Also pins the return contract: prompt followed by the new tokens, and
    exactly max_new_tokens of them when no stop token appears.

    Run with and without the KV cache: both paths must give HF's ids.
    """
    ids = hf_tokenizer("The capital of France is", return_tensors="pt").input_ids
    max_new_tokens = 20

    got = greedy(ids, weights, config, max_new_tokens, STOP_IDS, use_cache)

    assert got.shape == (1, ids.shape[1] + max_new_tokens)
    assert torch.equal(got, hf_greedy(ids, hf_model, max_new_tokens))


@pytest.mark.parametrize("use_cache", [True, False])
def test_stops_at_a_stop_token(config, weights, hf_model, hf_tokenizer, use_cache):
    """Generation ends on the first stop token, and keeps it, as HF does.

    The prompt is a chat turn with thinking switched off, which the model
    answers in a few tokens and then closes with <|im_end|> (151645) -- well
    before the limit, so stopping is what ends the loop, not the count.
    """
    chat = (
        "<|im_start|>user\nSay hello in one word.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    ids = hf_tokenizer(chat, return_tensors="pt").input_ids
    max_new_tokens = 30

    got = greedy(ids, weights, config, max_new_tokens, STOP_IDS, use_cache)

    assert got.shape[1] < ids.shape[1] + max_new_tokens
    assert got[0, -1].item() in STOP_IDS
    assert torch.equal(got, hf_greedy(ids, hf_model, max_new_tokens))
