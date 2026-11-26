import json
from pathlib import Path

from evaluate_games import (
    PROJECT_ROOT,
    LEGACY_SYSTEM_PROMPT,
    PROGRESS_SYSTEM_PROMPT,
    build_progress_user_prompt,
)


LEVEL1_REPLAY = (
    PROJECT_ROOT / "data/level2/replay_level1_scored/train.jsonl"
)

LEVEL2_CORRECTIONS = (
    PROJECT_ROOT / "data/level2/corrections_3epochs_scored/train.jsonl"
)

LEVEL2_MANIFEST = PROJECT_ROOT / "data/level2/splits.json"
LEVEL2_TRAJECTORIES = PROJECT_ROOT / "data/level2/trajectories_scored"

OUTPUT_DIR = PROJECT_ROOT / "data/level2/finetuning_progress_corrected"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build_progress_example(example):
    """Rebuild an example using the progress-aware prompt."""

    step = example["prompt_step"]
    previous_steps = example.get("previous_steps", [])

    return {
        **example,
        "messages": [
            {
                "role": "system",
                "content": PROGRESS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_progress_user_prompt(
                    step,
                    previous_steps,
                ),
            },
            {
                "role": "assistant",
                "content": example["messages"][-1]["content"],
            },
        ],
    }


def build_validation_examples():
    manifest = json.loads(
        LEVEL2_MANIFEST.read_text(encoding="utf-8")
    )

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

        trajectory = json.loads(
            path.read_text(encoding="utf-8")
        )

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

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": LEGACY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "",
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
            }

            examples.append(
                build_progress_example(example)
            )

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    replay_examples = read_jsonl(LEVEL1_REPLAY)
    correction_examples = read_jsonl(LEVEL2_CORRECTIONS)

    for example in replay_examples + correction_examples:
        if example["split"] != "train":
            raise ValueError(
                f"Non-training example found: {example.get('game_id')}"
            )

        if example["messages"][0]["content"] != LEGACY_SYSTEM_PROMPT:
            raise ValueError(
                f"Unexpected source prompt: {example.get('game_id')}"
            )

    train_examples = [
        build_progress_example(example)
        for example in replay_examples + correction_examples
    ]

    validation_examples, validation_report = (
        build_validation_examples()
    )

    num_corrections = sum(
        example.get("was_correction", False)
        for example in correction_examples
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    validation_path = OUTPUT_DIR / "validation.jsonl"

    for path, examples in (
        (train_path, train_examples),
        (validation_path, validation_examples),
    ):
        with path.open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )

    report = {
        "experiment": "level2_progress_aware_finetuning",
        "prompt_version": "progress",
        "training": {
            "level1_replay_examples": len(replay_examples),
            "level2_aligned_examples": len(correction_examples),
            "level2_corrections": num_corrections,
            "total_examples": len(train_examples),
        },
        "validation": validation_report,
        "files": {
            "train": str(train_path.relative_to(PROJECT_ROOT)),
            "validation": str(validation_path.relative_to(PROJECT_ROOT)),
        },
    }

    report_path = OUTPUT_DIR / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 60)
    print("LEVEL 2 PROGRESS-AWARE DATASET")
    print("=" * 60)
    print(f"Level 1 replay       : {len(replay_examples)}")
    print(f"Level 2 aligned      : {len(correction_examples)}")
    print(f"Level 2 corrections  : {num_corrections}")
    print(f"Training total       : {len(train_examples)}")
    print()
    print(f"Validation games     : {validation_report['games_used']}")
    print(f"Validation examples  : {len(validation_examples)}")
    print()
    print(f"Train      : {train_path}")
    print(f"Validation : {validation_path}")
    print(f"Report     : {report_path}")


if __name__ == "__main__":
    main()
