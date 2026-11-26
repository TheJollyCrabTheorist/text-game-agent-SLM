import argparse
from copy import deepcopy
from data_integrity import validate_history
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_games import (
    LEGACY_SYSTEM_PROMPT,
    build_legacy_user_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAJECTORIES = PROJECT_ROOT / "data/level2/trajectories"
DEFAULT_EVALUATION = (
    PROJECT_ROOT / "evaluation/level2/zero_shot_legacy_train.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/level2/corrections"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value):
    """Normalize line endings and irrelevant surrounding whitespace."""
    if value is None:
        return ""

    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def state_key(step):
    """Build a strict key identifying the current environment state."""
    commands = step.get("admissible_commands") or []

    return (
        normalize_text(step.get("observation")),
        normalize_text(step.get("inventory")),
        tuple(commands),
    )


def load_expert_trajectories(directory):
    """Load only successful expert trajectories from the train split."""
    trajectories = {}

    for path in sorted(directory.glob("*.json")):
        data = load_json(path)

        if data.get("split") != "train":
            continue

        if not data.get("success") or not data.get("won"):
            raise ValueError(
                f"Expert trajectory is not successful: {path}"
            )

        game_id = data["game_id"]

        if game_id in trajectories:
            raise ValueError(f"Duplicate expert game: {game_id}")

        trajectories[game_id] = data

    return trajectories


def index_expert_steps(trajectory):
    """Index expert states and their corresponding actions."""
    index = defaultdict(list)

    for step in trajectory.get("steps", []):
        key = state_key(step)
        action = step.get("action")

        if not action:
            continue

        if action not in step.get("admissible_commands", []):
            raise ValueError(
                f"Expert action is not admissible in "
                f"{trajectory['game_id']} step {step.get('step')}: "
                f"{action}"
            )

        index[key].append(step)

    return index


def build_example(game_id, model_step, expert_step, previous_steps):
    """Build one supervised example using the exact legacy prompt."""
    
    validate_history(previous_steps, model_step["step"])
    required_score_fields = ("score", "max_score")

    for field in required_score_fields:
        if not isinstance(model_step.get(field), (int, float)):
            raise ValueError(
                f"Missing numeric {field} in model step "
                f"{game_id}, step {model_step.get('step')}"
            )

    prompt_step = {
        "objective": model_step["objective"],
        "observation": model_step["observation"],
        "inventory": model_step["inventory"],
        "admissible_commands": model_step["admissible_commands"],
        "score": float(model_step["score"]),
        "max_score": float(model_step["max_score"]),
    }

    user_prompt = build_legacy_user_prompt(
        prompt_step,
        previous_steps,
    )

    expert_action = expert_step["action"]

    return {
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
                "content": expert_action,
            },
        ],
        "prompt_step": prompt_step,
        "previous_steps": deepcopy(previous_steps),
        "game_id": game_id,
        "step": model_step["step"],
        "split": "train",
        "source": "level2_aligned",
        "model_action": model_step["action"],
        "expert_action": expert_action,
        "was_correction": model_step["action"] != expert_action,
    }


def collect(args):
    print("=== LOADING EXPERT TRAJECTORIES ===")

    expert_trajectories = load_expert_trajectories(
        args.trajectories
    )

    print(f"Successful train trajectories: {len(expert_trajectories)}")

    print("\n=== LOADING MODEL EVALUATION ===")

    evaluation = load_json(args.evaluation)

    if evaluation.get("split") != "train":
        raise ValueError(
            "Only the train split may be used to collect corrections."
        )

    if evaluation.get("prompt_version") != "legacy":
        raise ValueError(
            "Evaluation must use prompt_version='legacy'."
        )

    model_results = evaluation.get("results", [])

    if not model_results:
        raise ValueError("No model evaluation results found.")

    output_examples = []
    game_reports = []

    totals = Counter()

    for model_game in model_results:
        game_id = model_game["game_id"]

        if game_id not in expert_trajectories:
            print(f"WARNING: no expert trajectory for {game_id}")
            totals["missing_expert_games"] += 1
            continue

        expert_game = expert_trajectories[game_id]

        if normalize_text(model_game["objective"]) != normalize_text(
            expert_game["objective"]
        ):
            raise ValueError(
                f"Objective mismatch for {game_id}."
            )

        expert_index = index_expert_steps(expert_game)

        game_examples = []
        game_counts = Counter()
        previous_steps = []

        for model_step in model_game.get("steps", []):
            key = state_key(model_step)
            candidates = expert_index.get(key, [])

            if not candidates:
                game_counts["unmatched"] += 1
                previous_steps.append(model_step)
                continue

            expert_actions = {
                candidate["action"] for candidate in candidates
            }

            if len(expert_actions) != 1:
                game_counts["ambiguous"] += 1
                previous_steps.append(model_step)
                continue

            expert_step = candidates[0]
            expert_action = expert_step["action"]

            if expert_action not in model_step["admissible_commands"]:
                raise ValueError(
                    f"Expert action is not admissible in the matched "
                    f"model state: {game_id}, step {model_step['step']}"
                )

            example = build_example(
                game_id,
                model_step,
                expert_step,
                previous_steps,
            )

            game_examples.append(example)

            if example["was_correction"]:
                game_counts["corrections"] += 1
            else:
                game_counts["already_correct"] += 1

            game_counts["aligned"] += 1

            # The next prompt must contain the model's actual history.
            previous_steps.append(model_step)

        output_examples.extend(game_examples)
        totals.update(game_counts)

        game_reports.append({
            "game_id": game_id,
            "model_success": model_game["success"],
            "model_steps": model_game["num_steps"],
            "aligned": game_counts["aligned"],
            "already_correct": game_counts["already_correct"],
            "corrections": game_counts["corrections"],
            "unmatched": game_counts["unmatched"],
            "ambiguous": game_counts["ambiguous"],
        })

        print(
            f"{game_id}: "
            f"{game_counts['aligned']} aligned | "
            f"{game_counts['corrections']} corrections | "
            f"{game_counts['unmatched']} unmatched | "
            f"{game_counts['ambiguous']} ambiguous"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    dataset_path = args.output / "train.jsonl"

    with open(dataset_path, "w", encoding="utf-8") as file:
        for example in output_examples:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    report = {
        "experiment": "level2_trajectory_aligned_corrections",
        "prompt_version": "legacy",
        "source_split": "train",
        "num_model_games": len(model_results),
        "num_expert_games": len(expert_trajectories),
        "num_examples": len(output_examples),
        "aligned_states": totals["aligned"],
        "already_correct": totals["already_correct"],
        "corrections": totals["corrections"],
        "unmatched_states": totals["unmatched"],
        "ambiguous_states": totals["ambiguous"],
        "missing_expert_games": totals["missing_expert_games"],
        "games": game_reports,
    }

    report_path = args.output / "collection_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n=== COLLECTION SUMMARY ===")
    print(f"Aligned states:    {totals['aligned']}")
    print(f"Already correct:   {totals['already_correct']}")
    print(f"Corrections:       {totals['corrections']}")
    print(f"Unmatched states:  {totals['unmatched']}")
    print(f"Ambiguous states:  {totals['ambiguous']}")
    print(f"Examples written:  {len(output_examples)}")
    print(f"\nDataset: {dataset_path}")
    print(f"Report:  {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectory-aligned Level 2 corrections."
    )

    parser.add_argument(
        "--trajectories",
        type=Path,
        default=DEFAULT_TRAJECTORIES,
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=DEFAULT_EVALUATION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
