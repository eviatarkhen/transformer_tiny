import torch

from tiny.config import Config
from tiny.model import forward


def greedy(
    ids: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    max_new_tokens: int,
    stop_ids: tuple[int, ...] = (),
    use_cache: bool = True,
) -> torch.Tensor:
    """Extend a prompt one token at a time, always taking the top-scoring token.

    `start` is the first position forward has not yet seen. Without a cache it
    stays 0: every turn reruns the whole sequence and reads only its last row
    of logits. With one, the first turn (the prefill) runs the whole prompt
    and fills the cache, and every turn after feeds only the newest token, at
    its true position. Same output either way, id for id; use_cache=False is
    kept as the benchmark baseline.

    Args:
        ids: (1, seq) prompt token ids. One prompt only: prompts in a batch
            would stop at different times and need padding, which is out of
            scope.
        w: the full weight dict, keyed exactly as the checkpoint is.
        config: architecture.
        max_new_tokens: upper bound on how many tokens to append.
        stop_ids: token ids that end generation early. A stop token is kept in
            the output, as HF's generate does. For Qwen3-0.6B these are 151645
            (<|im_end|>) and 151643 (<|endoftext|>), from
            generation_config.json -- config.json lists only the first.
        use_cache: keep each layer's keys and values between turns. True by
            default, as in HF's generate.

    Returns:
        (1, seq + k) the prompt followed by the k <= max_new_tokens generated
        tokens.
    """
    if ids.shape[0] != 1:
        raise ValueError(f"greedy takes one prompt, got a batch of {ids.shape[0]}")

    cache = [None] * config.num_hidden_layers if use_cache else None
    start = 0
    for _ in range(max_new_tokens):
        logits = forward(ids[:, start:], w, config, torch.arange(start, ids.shape[1]), cache)
        token = logits[0, -1].argmax()
        ids = torch.cat([ids, token.view(1, 1)], dim=1)
        if token.item() in stop_ids:
            break
        if use_cache:
            start = ids.shape[1] - 1

    return ids
