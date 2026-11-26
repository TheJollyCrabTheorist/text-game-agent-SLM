import json
import argparse
from data_integrity import validate_history
from pathlib import Path

from evaluate_games import (
    LEGACY_SYSTEM_PROMPT,
    build_legacy_user_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEVEL1_REPLAY = (
    PROJECT_ROOT / "data/level2/replay_level1/train.jsonl"
)
LEVEL2_CORRECTIONS = (
    PROJECT_ROOT / "data/level2/corrections/train.jsonl"
)

LEVEL2_MANIFEST = PROJECT_ROOT / "data/level2/splits.json"
LEVEL2_TRAJECTORIES = PROJECT_ROOT / "data/level2/trajectories"

OUTPUT_DIR = PROJECT_ROOT / "data/level2/finetuning"


def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_validation_examples():
    """Build validation examples for the level 2 fine-tuning."""
    manifest = json.loads(LEVEL2_MANIFEST.read_text())

    validation_ids = {
        game["game_id"]
        for game in manifest["games"]
        if game["split"] == "validation"
    }

    examples = []
    missing_games = []
    unsuccessful_games = []
    invalid_actions = []

    for game_id in sorted(validation_ids):
        path = LEVEL2_TRAJECTORIES / f"{game_id}.json"

        if not path.exists():
            missing_games.append(game_id)
            continue

        trajectory = json.loads(path.read_text())

        if not trajectory.get("success") or not trajectory.get("won"):
            unsuccessful_games.append(game_id)
            continue

        steps = trajectory.get("steps", [])

        for index, step in enumerate(steps):
            action = step.get("action")

            if not action:
                continue

            if action not in step.get("admissible_commands", []):
                invalid_actions.append({
                    "game_id": game_id,
                    "step": index,
                    "action": action,
                })
                continue

            previous_steps = steps[:index]

            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": LEGACY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_legacy_user_prompt(
                            step,
                            previous_steps,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": action,
                    },
                ],
                "prompt_step": step,
                "previous_steps": previous_steps,
                "game_id": game_id,
                "step": index,
                "split": "validation",
                "source": "level2_expert",
                "was_correction": False,
            })

    report = {
        "expected_games": len(validation_ids),
        "games_used": len({
            example["game_id"] for example in examples
        }),
        "num_examples": len(examples),
        "missing_games": missing_games,
        "unsuccessful_games": unsuccessful_games,
        "invalid_actions": invalid_actions,
    }

    if missing_games or unsuccessful_games or invalid_actions:
        raise RuntimeError(
            f"Validation dataset integrity check failed: {report}"
        )

    return examples, report


def main():
    """Run the level 2 fine-tuning dataset preparation pipeline."""
    global LEVEL1_REPLAY, LEVEL2_CORRECTIONS, LEVEL2_MANIFEST, LEVEL2_TRAJECTORIES, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=LEVEL1_REPLAY)
    parser.add_argument("--corrections", type=Path, default=LEVEL2_CORRECTIONS)
    parser.add_argument("--manifest", type=Path, default=LEVEL2_MANIFEST)
    parser.add_argument("--trajectories", type=Path, default=LEVEL2_TRAJECTORIES)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    LEVEL1_REPLAY, LEVEL2_CORRECTIONS, LEVEL2_MANIFEST, LEVEL2_TRAJECTORIES, OUTPUT_DIR = (p.resolve() for p in (args.replay, args.corrections, args.manifest, args.trajectories, args.output))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Level 1 replay.
    replay_examples = read_jsonl(LEVEL1_REPLAY)

    # Level 2 trajectory-aligned corrections.
    correction_examples = read_jsonl(LEVEL2_CORRECTIONS)

    # Integrity checks.
    for example in replay_examples + correction_examples:
        validate_history(example["previous_steps"], example["step"])
    for example in replay_examples:
        assert example["split"] == "train"
        assert example["messages"][0]["content"] == LEGACY_SYSTEM_PROMPT

    for example in correction_examples:
        assert example["split"] == "train"
        assert example["messages"][0]["content"] == LEGACY_SYSTEM_PROMPT

    num_corrections = sum(
        example.get("was_correction", False)
        for example in correction_examples
    )

    # Combine training sources.
    train_examples = replay_examples + correction_examples

    validation_examples, validation_report = (
        build_validation_examples()
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    validation_path = OUTPUT_DIR / "validation.jsonl"

    with train_path.open("w") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    with validation_path.open("w") as f:
        for example in validation_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    report = {
        "experiment": "level2_replay_and_corrective_finetuning",
        "prompt_version": "legacy",
        "training": {
            "level1_replay_examples": len(replay_examples),
            "level2_aligned_examples": len(correction_examples),
            "level2_corrections": num_corrections,
            "total_examples": len(train_examples),
            "level1_games": len({
                example["game_id"] for example in replay_examples
            }),
            "level2_games": len({
                example["game_id"] for example in correction_examples
            }),
        },
        "validation": validation_report,
        "files": {
            "train": str(train_path),
            "validation": str(validation_path),
        },
    }

    report_path = OUTPUT_DIR / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print("=" * 60)
    print("LEVEL 2 FINE-TUNING DATASET")
    print("=" * 60)
    print(f"Level 1 replay       : {len(replay_examples)}")
    print(f"Level 2 aligned      : {len(correction_examples)}")
    print(f"Level 2 corrections  : {num_corrections}")
    print(f"Training total       : {len(train_examples)}")
    print()
    print(f"Validation games     : {validation_report['games_used']}")
    print(f"Validation examples : {len(validation_examples)}")
    print()
    print(f"Train      : {train_path}")
    print(f"Validation : {validation_path}")
    print(f"Report     : {report_path}")


if __name__ == "__main__":
    main()
