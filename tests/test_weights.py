import dataclasses
import json

import torch

from tiny.config import model_dir
from tiny.weights import layer_prefix

GLOBAL_KEYS = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def expected_layer_shapes(config):
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    return {
        'self_attn.q_proj.weight': (q, config.hidden_size),
        'self_attn.k_proj.weight': (kv, config.hidden_size),
        'self_attn.v_proj.weight': (kv, config.hidden_size),
        'self_attn.o_proj.weight': (config.hidden_size, q),
        'mlp.gate_proj.weight':    (config.intermediate_size, config.hidden_size),
        'mlp.up_proj.weight':      (config.intermediate_size, config.hidden_size),
        'mlp.down_proj.weight':    (config.hidden_size, config.intermediate_size),
        'self_attn.k_norm.weight': (config.head_dim, ),
        'self_attn.q_norm.weight': (config.head_dim, ),
        'input_layernorm.weight':  (config.hidden_size, ),
        'post_attention_layernorm.weight': (config.hidden_size, )
    }


def test_config_matches_json(config):
    with open(model_dir() / 'config.json') as f:
        file_config = json.load(f)

    for field in dataclasses.fields(config):
        assert getattr(config, field.name) == file_config[field.name]


def test_head_dim_is_not_hidden_over_heads(config):
    assert config.head_dim != config.hidden_size / config.num_attention_heads


def test_all_keys_present(config, weights):
    """3 global tensors + 11 per layer, for num_hidden_layers layers.
    """
    tensors = 0
    for global_key in GLOBAL_KEYS:
        assert global_key in weights
        tensors += 1

    fields = expected_layer_shapes(config).keys()
    for i in range(config.num_hidden_layers):
        prefix = layer_prefix(i)
        for field in fields:
            assert prefix + field in weights
            tensors += 1

    assert tensors == len(weights)

        
def test_shapes(config, weights):
    """Every tensor has the shape config implies.
    """
    assert weights['model.embed_tokens.weight'].shape == (config.vocab_size, config.hidden_size)
    assert weights['model.norm.weight'].shape == (config.hidden_size, )

    layer_shapes = expected_layer_shapes(config)
    for i in range(config.num_hidden_layers):
        prefix = layer_prefix(i)
        for name, value in layer_shapes.items():
            assert weights[prefix + name].shape == value


def test_all_float32(weights):
    """The file is bfloat16; everything we hand out is float32.

    Parity tests compare against HF in float32, and bf16 rounding noise is large
    enough to hide a real bug.
    """
    for weight in weights.values():
        assert weight.dtype == torch.float32


def test_embeddings_are_tied(config, weights):
    """lm_head.weight and embed_tokens.weight hold identical values.

    The checkpoint stores both copies (~300 MB of redundancy) even though the
    config says they are tied. Use torch.equal -- these are the same numbers,
    not the result of a computation, so no tolerance is appropriate.

    Guard the assertion on config.tie_word_embeddings.
    """
    assert torch.equal(weights['model.embed_tokens.weight'], weights['lm_head.weight'])


def test_no_nan_or_inf(weights):
    """Rules out a corrupted download before it can masquerade as a bug in
    attention three steps from now."""
    for weight in weights.values():
        assert torch.isfinite(weight).all()