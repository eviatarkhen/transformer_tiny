import pytest
import torch

from tiny.model import final_norm


def hf_final_norm(x, hf_model):
    """The oracle: HF's model.norm, the RMSNorm between the last block and the head."""
    with torch.no_grad():
        return hf_model.model.norm(x)


@pytest.mark.parametrize("scale", [1e3, 1e-3])
def test_matches_hf(config, weights, hf_model, scale):
    """final_norm equals HF's model.norm, at two input scales.

    1e3 is the realistic case: the stream leaving the last block has values in
    the thousands (measured max ~1,800).

    1e-3 exists for eps. At realistic scale mean(x^2) is ~1e6 and eps (1e-6)
    vanishes, so a wrong eps value -- hardcoded instead of read from config --
    is invisible. At 1e-3, mean(x^2) is ~1e-6, the same size as eps, and the
    value matters. Same lesson as the eps-placement case in test_rmsnorm.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 7, config.hidden_size) * scale

    got = final_norm(x, weights, config)

    assert got.shape == x.shape
    torch.testing.assert_close(got, hf_final_norm(x, hf_model), atol=1e-4, rtol=1e-5)
