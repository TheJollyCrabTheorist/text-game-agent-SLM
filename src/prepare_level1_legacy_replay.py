import json
import argparse
from pathlib import Path

from evaluate_games import (
    LEGACY_SYSTEM_PROMPT,
    build_legacy_user_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data/raw/splits.json"
TRAJECTORIES_DIR = PROJECT_ROOT / "data/raw/trajectories_scored"

OUTPUT_DIR = PROJECT_ROOT / "data/level2/replay_level1_scored"
OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"


def main():
    """Run the level 1 legacy replay preparation pipeline."""
    global TRAJECTORIES_DIR, OUTPUT_DIR, OUTPUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=TRAJECTORIES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    TRAJECTORIES_DIR = args.trajectories.resolve()
    OUTPUT_DIR = args.output.resolve()
    OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
    manifest = json.loads(MANIFEST_PATH.read_text())

    # Only use Level 1 training games.
    train_game_ids = set(manifest["splits"]["train"])

    examples = []
    missing_games = []
    invalid_actions = []
    unsuccessful_games = []

    for game_id in sorted(train_game_ids):
        trajectory_path = TRAJECTORIES_DIR / f"{game_id}.json"

        if not trajectory_path.exists():
            missing_games.append(game_id)
            continue

        trajectory = json.loads(trajectory_path.read_text())

        if not trajectory.get("success") or not trajectory.get("won"):
            unsuccessful_games.append(game_id)
            continue

        steps = trajectory.get("steps", [])

        for index, step in enumerate(steps):
            action = step.get("action")

            if not action:
                continue

            admissible = step.get("admissible_commands", [])

            if action not in admissible:
                invalid_actions.append(
                    {
                        "game_id": game_id,
                        "step": index,
                        "action": action,
                    }
                )
                continue

            previous_steps = steps[:index]

            user_prompt = build_legacy_user_prompt(
                step,
                previous_steps,
            )

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": LEGACY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
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
                "split": "train",
                "source": "level1_replay",
                "was_correction": False,
            }

            examples.append(example)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    report = {
        "num_expected_games": len(train_game_ids),
        "num_examples": len(examples),
        "num_games_used": len(
            {example["game_id"] for example in examples}
        ),
        "missing_games": missing_games,
        "unsuccessful_games": unsuccessful_games,
        "invalid_actions": invalid_actions,
        "output": str(OUTPUT_PATH),
    }

    report_path = OUTPUT_DIR / "collection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print("=" * 60)
    print("LEVEL 1 LEGACY REPLAY DATASET")
    print("=" * 60)
    print(f"Expected training games : {len(train_game_ids)}")
    print(f"Games used              : {report['num_games_used']}")
    print(f"Training examples       : {len(examples)}")
    print(f"Missing games           : {len(missing_games)}")
    print(f"Unsuccessful games      : {len(unsuccessful_games)}")
    print(f"Invalid actions         : {len(invalid_actions)}")
    print(f"\nDataset: {OUTPUT_PATH}")
    print(f"Report : {report_path}")

    if missing_games or unsuccessful_games or invalid_actions:
        print("\nWARNING: Some trajectories or actions were excluded.")


if __name__ == "__main__":
    main()
