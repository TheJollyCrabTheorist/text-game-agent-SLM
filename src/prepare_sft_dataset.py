import json
import argparse
from pathlib import Path
from collections import Counter


TRAJECTORY_DIR = Path("data/raw/trajectories")
MANIFEST_PATH = Path("data/raw/splits.json")
OUTPUT_DIR = Path("data/processed")

SYSTEM_PROMPT = """You are an agent playing a text-based adventure game.

Your task is to choose the best next action to accomplish the game objective.

You must consider:
- the game objective and its ordered requirements,
- the interaction history and the outcomes of previous actions,
- the current observation,
- your inventory,
- the admissible commands.

Planning rules:
1. Treat the objective as an ordered sequence of requirements.
2. Prioritize the earliest requirement that has not yet been completed.
3. Do not skip a requirement to perform a later one.
4. Before moving to the next requirement, check whether the current one has actually been completed.
5. Use observations, inventory changes and previous action outcomes to track progress.
6. If an action fails, do not repeat it unless the situation has changed.
7. Do not assume that an action succeeded simply because it was admissible.

Action rules:
8. Choose exactly one command from the admissible commands.
9. Copy the command exactly as provided.
10. Do not invent commands.
11. Do not provide explanations or additional text.

Output only the selected command."""

def format_history(previous_steps):
    """Format previous interactions, including their observed outcomes."""

    if not previous_steps:
        return "No previous interactions."

    history = []

    for step in previous_steps:
        interaction = [
            f"Previous observation:\n{step['observation']}",
            f"Previous inventory:\n{step['inventory']}",
            f"Action taken:\n{step['action']}",
        ]

        if step.get("next_observation") is not None:
            interaction.append(
                "Resulting observation:\n"
                f"{step['next_observation']}"
            )

        if step.get("next_inventory") is not None:
            interaction.append(
                "Resulting inventory:\n"
                f"{step['next_inventory']}"
            )

        history.append("\n\n".join(interaction))

    return "\n\n---\n\n".join(history)

def build_user_prompt(step, previous_steps):
    """Build the user prompt, including the explicit game objective."""

    objective = step.get("objective")

    if not objective:
        raise ValueError(
            "Missing game objective. "
            "Check that oracle trajectories were recollected "
            "with objective=True in EnvInfos."
        )

    history = format_history(previous_steps)

    admissible = "\n".join(
        f"- {command}"
        for command in step["admissible_commands"]
    )

    return f"""GAME OBJECTIVE
{objective}

INTERACTION HISTORY
{history}

CURRENT OBSERVATION
{step["observation"]}

CURRENT INVENTORY
{step["inventory"]}

ADMISSIBLE COMMANDS
{admissible}

Choose the next action."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", choices=["legacy", "ordered"], default="ordered")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    if args.prompt_version == "legacy":
        from evaluate_games import LEGACY_SYSTEM_PROMPT as system_prompt, build_legacy_user_prompt as builder
    else:
        system_prompt, builder = SYSTEM_PROMPT, build_user_prompt
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    split_examples = {
        "train": [],
        "validation": [],
        "test": [],
    }

    skipped_games = []
    game_counts = Counter()

    for game in manifest["games"]:
        game_id = game["game_id"]
        split = game["split"]

        if split not in split_examples:
            raise ValueError(
                f"Unknown split '{split}' for game {game_id}"
            )

        path = TRAJECTORY_DIR / f"{game_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing trajectory: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        # Exclude unsuccessful oracle trajectories.
        if not data.get("success", False):
            skipped_games.append(game_id)
            continue

        objective = data.get("objective")

        if not objective:
            raise ValueError(
                f"{game_id}: missing trajectory objective. "
                "Recollect the oracle trajectory before preparing "
                "the SFT dataset."
            )

        steps = data["steps"]

        for index, step in enumerate(steps):

            required_fields = (
                "observation",
                "inventory",
                "admissible_commands",
                "action",
            )

            for field in required_fields:
                if step.get(field) is None:
                    raise ValueError(
                        f"{game_id}, step {index}: missing {field}"
                    )

            # Use the step-level objective when available,
            # otherwise inherit the trajectory-level objective.
            step_objective = step.get("objective") or objective

            if step_objective != objective:
                raise ValueError(
                    f"{game_id}, step {index}: "
                    "objective differs from trajectory objective."
                )

            # Ensure the objective is available to build_user_prompt.
            prompt_step = {
                **step,
                "objective": step_objective,
            }

            action = step["action"]
            admissible = step["admissible_commands"]

            if action not in admissible:
                raise ValueError(
                    f"{game_id}, step {index}: "
                    f"oracle action is not admissible: {action}"
                )

            user_prompt = builder(
                prompt_step,
                steps[:index],
            )

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
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
                "prompt_step": prompt_step,
                "previous_steps": steps[:index],
                "game_id": game_id,
                "step": index,
                "split": split,
            }

            split_examples[split].append(example)

            game_counts[(split, game_id)] += 1

    # Save one JSONL file per split.
    for split, examples in split_examples.items():
        output_path = output_dir / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )

    print("=" * 60)
    print("SFT DATASET PREPARATION")
    print("=" * 60)

    for split, examples in split_examples.items():
        unique_games = {
            example["game_id"] for example in examples
        }

        print(f"\n{split.upper()}")
        print("Games:", len(unique_games))
        print("Examples:", len(examples))
        print(
            "Output:",
            output_dir / f"{split}.jsonl"
        )

    print("\nSkipped unsuccessful games:", skipped_games)
    print("Total examples:", sum(map(len, split_examples.values())))

    # Display one example for inspection.
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING EXAMPLE")
    print("=" * 60)

    if split_examples["train"]:
        print(
            json.dumps(
                split_examples["train"][0],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print("No training examples were generated.")


if __name__ == "__main__":
    main()
