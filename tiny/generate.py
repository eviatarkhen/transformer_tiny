import torch

from tiny.config import Config
from tiny.model import forward


def greedy(
    ids: torch.Tensor,
    w: dict[str, torch.Tensor],
    config: Config,
    max_new_tokens: int,
    stop_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    """Extend a prompt one token at a time, always taking the top-scoring token.

    Every step reruns forward over the whole sequence and reads only its last
    row of logits. That recomputation is the waste step 9's KV cache removes.

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

    Returns:
        (1, seq + k) the prompt followed by the k <= max_new_tokens generated
        tokens.
    """
    if ids.shape[0] != 1:
        raise ValueError(f"greedy takes one prompt, got a batch of {ids.shape[0]}")

    for _ in range(max_new_tokens):
        logits = forward(ids, w, config, torch.arange(ids.shape[1]))
        token = logits[0, -1].argmax()
        ids = torch.cat([ids, token.view(1, 1)], dim=1)
        if token.item() in stop_ids:
            break

    return ids
