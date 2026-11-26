"""Data integrity helpers for the TextWorld RL pipeline."""

import json
from pathlib import Path


def score_delta(current_score: float, next_score: float) -> float:
    """Return the reward as the increase of the game score."""
    return float(next_score) - float(current_score)


def validate_history(history, step):
    """Validate a trajectory history against the current step index."""
    if not isinstance(history, list):
        raise ValueError(
            f"History must be a list of steps, got {type(history).__name__}"
        )
    current = int(step)
    if current < 0:
        raise ValueError(f"Invalid step index: {step}")
    if not history:
        if current != 0:
            raise ValueError(
                f"Empty history at step {current}"
            )
        return
    last = history[-1]
    if not isinstance(last, dict):
        raise ValueError("History entries must be dicts")
    if int(last.get("step", -1)) != current - 1:
        raise ValueError(
            f"History last step {last.get('step')} does not match "
            f"expected {current - 1} for step {current}"
        )


def validate_adapter_base(adapter_path, base_model_name):
    """Validate that a LoRA adapter was trained from the expected base model."""
    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    base = config.get("base_model_name_or_path")
    if base != base_model_name:
        raise ValueError(
            f"Adapter {adapter_path} was trained from {base}, "
            f"expected {base_model_name}"
        )
