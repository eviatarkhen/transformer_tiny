import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class Config:
    """Architecture of a Qwen3 model.

    Frozen because nothing should ever mutate it: every shape in the project is
    derived from these numbers, so an accidental write would corrupt everything
    downstream at once.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        data = json.loads(path.read_text())

        return cls(
            hidden_size=data["hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            vocab_size=data["vocab_size"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_theta=data["rope_theta"],
            tie_word_embeddings=bool(data["tie_word_embeddings"])
        )

    
def model_dir(model_id: str = MODEL_ID) -> Path:
    return Path(snapshot_download(model_id))