import pytest
import torch

from tiny.mlp import mlp, swiglu

LAYERS = [0, 27]


def hf_mlp(x, hf_model, layer):
    """The oracle: HF's own Qwen3MLP on the same input.

    Unlike self_attn this returns a plain tensor, not a tuple.
    """
    with torch.no_grad():
        return hf_model.model.layers[layer].mlp(x)


# --- the gate ---------------------------------------------------------------


def test_swiglu_matches_hand_computed_values():
    """silu(gate) * up on numbers worked out by hand.

    The test vector is chosen so that a wrong activation cannot slip through:
        gate = 1  -> silu(1) = 0.731059, the ordinary case
        gate = -1 -> silu(-1) = -0.268941, negative and NOT clamped to zero,
                     which is what separates SiLU from ReLU
        gate = 0  -> silu(0) = 0, the closed gate, whose channel contributes
                     nothing however large up is

    Keep the negative and the zero. On an all-positive vector relu(gate) * up,
    sigmoid(gate) * up and gate * up are all indistinguishable from the truth.
    """
    gate = torch.tensor([1.0, -1.0, 0.0])
    up = torch.tensor([2.0, -2.0, 2.0])

    expected = torch.tensor([1.462117, 0.537883, 0.0])

    torch.testing.assert_close(swiglu(gate, up), expected, atol=1e-6, rtol=1e-5)


# --- the component ----------------------------------------------------------


@pytest.mark.parametrize("layer", LAYERS)
def test_matches_hf(config, weights, hf_model, layer):
    """The whole MLP equals HF's Qwen3MLP on the same input.

    Layer 27 is not redundant with layer 0: it is the only thing here that
    catches a prefix which ignores its `layer` argument and always reads the
    first block's weights.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 7, config.hidden_size)

    expected = hf_mlp(x, hf_model, layer)
    got = mlp(x, weights, layer)

    assert got.shape == x.shape
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


def test_is_position_wise(config, weights):
    """Permuting the tokens permutes the output, exactly.

    The defining property of the MLP: it is the half of the block where tokens
    do not talk to each other, so each position's output depends on that
    position's input alone. Equality is exact rather than approximate because
    the permuted input has the same shape, so the matmul takes the same path.
    (The other spelling of this test -- one token alone vs. the same token
    inside a batch -- drifts by ~6e-6 for exactly that reason.)
    """
    torch.manual_seed(0)
    x = torch.randn(1, 5, config.hidden_size)
    perm = torch.tensor([4, 0, 3, 1, 2])

    assert torch.equal(mlp(x[:, perm], weights, 0), mlp(x, weights, 0)[:, perm])
