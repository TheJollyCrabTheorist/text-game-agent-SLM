import argparse
from data_integrity import score_delta, validate_adapter_base
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import textworld

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from textworld import EnvInfos

from collect_walkthrough import sanitize_observation
from prepare_sft_dataset import (
    SYSTEM_PROMPT as ORDERED_SYSTEM_PROMPT,
    build_user_prompt as build_ordered_user_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "models/qwen25-05b-textworld-qlora/best_adapter"
)

MANIFEST_PATH = PROJECT_ROOT / "data/raw/splits.json"

MAX_LENGTH = 1024
MAX_NEW_TOKENS = 64
MAX_STEPS = 50
REPETITION_PENALTY = 1.2


LEGACY_SYSTEM_PROMPT = """You are an agent playing a text-based adventure game.

Your task is to choose the best next action to accomplish the game objective.

You must consider:
- the game objective,
- the interaction history,
- the current observation,
- your inventory,
- the admissible commands.

Rules:
1. Choose exactly one command from the admissible commands.
2. Copy the command exactly as provided.
3. Do not invent commands.
4. Do not provide explanations or additional text.
5. Consider previous observations and actions when making decisions.
6. Prefer actions that make progress toward the objective.
7. Avoid repeating actions that have already failed or produced no progress.

Output only the selected command."""


NO_LOOP_SYSTEM_PROMPT = """You are an agent playing a text-based adventure game.

Your task is to choose the best next action to accomplish the game objective.

You must consider:
- the game objective,
- the interaction history,
- the current observation,
- your inventory,
- the admissible commands.

Rules:
1. Choose exactly one command from the admissible commands.
2. Copy the command exactly as provided.
3. Do not invent commands.
4. Do not provide explanations or additional text.
5. Consider previous observations and actions when making decisions.
6. Prefer actions that make progress toward the objective.
7. Avoid repeating actions that have already failed or produced no progress.

Additional rules for this episode:
8. Never repeat the same action consecutively.
9. If an action does not change the score or the game state, choose a different admissible action.
10. Keep the trajectory as short as possible.

Output only the selected command."""


def build_legacy_user_prompt(step, previous_steps):
    """
    Reproduce the original objective-aware prompt format.

    The legacy history contains previous observations,
    inventories and actions, but not their resulting states.
    """

    objective = step.get("objective")

    if not objective:
        raise ValueError("Missing game objective.")

    if not previous_steps:
        history = "No previous interactions."

    else:
        interactions = []

        for previous in previous_steps:
            interactions.append(
                f"""Previous observation:
{previous["observation"]}

Previous inventory:
{previous["inventory"]}

Action taken:
{previous["action"]}"""
            )

        history = "\n\n---\n\n".join(interactions)

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

def extract_score(step):
    """Return the pre-action score and max score for a trajectory step."""

    if "score" in step and "max_score" in step:
        return float(step["score"]), float(step["max_score"])

    observation = step.get("observation", "")
    match = re.search(r"(\d+)\s*/\s*(\d+)\s*$", observation.strip())

    if not match:
        raise ValueError(
            "Cannot recover score/max_score from trajectory step."
        )

    return float(match.group(1)), float(match.group(2))


def remove_score_suffix(observation):
    """Remove TextWorld's trailing score footer from an observation."""

    return re.sub(
        r"\s*\d+\s*/\s*\d+\s*$",
        "",
        observation.strip(),
    )


PROGRESS_SYSTEM_PROMPT = LEGACY_SYSTEM_PROMPT + """

You are also given the current quest score and maximum score.
Use this information to estimate how much of the objective has
already been completed.

Important:
- The score is a progress signal, not an instruction.
- It does not tell you by itself which subgoal remains.
- Always use the objective, current observation, inventory,
  interaction history and admissible commands to choose the next action.
- Never repeat a completed subgoal unless the current state requires it.
"""


def build_progress_user_prompt(step, previous_steps):
    """Build a legacy-style prompt with explicit score progress."""

    objective = step.get("objective")
    if not objective:
        raise ValueError("Missing game objective.")

    score, max_score = extract_score(step)

    if max_score <= 0:
        raise ValueError(f"Invalid max_score: {max_score}")

    progress = score / max_score

    if not previous_steps:
        history = "No previous interactions."
    else:
        interactions = []

        for previous in previous_steps:
            previous_score, previous_max_score = extract_score(previous)

            interactions.append(
                f"""Previous observation:
{remove_score_suffix(previous["observation"])}

Previous inventory:
{previous["inventory"]}

Score after reaching this state:
{previous_score:g} / {previous_max_score:g}

Action taken:
{previous["action"]}"""
            )

        history = "\n\n---\n\n".join(interactions)

    admissible = "\n".join(
        f"- {command}"
        for command in step["admissible_commands"]
    )

    return f"""GAME OBJECTIVE
{objective}

QUEST PROGRESS
Current score: {score:g} / {max_score:g}
Progress: {progress:.0%}

INTERACTION HISTORY
{history}

CURRENT OBSERVATION
{remove_score_suffix(step["observation"])}

CURRENT INVENTORY
{step["inventory"]}

ADMISSIBLE COMMANDS
{admissible}

Choose the next action."""

PROMPT_CONFIGS = {
    "legacy": {
        "system_prompt": LEGACY_SYSTEM_PROMPT,
        "user_prompt_builder": build_legacy_user_prompt,
    },
    "ordered": {
        "system_prompt": ORDERED_SYSTEM_PROMPT,
        "user_prompt_builder": build_ordered_user_prompt,
    },
    "progress": {
        "system_prompt": PROGRESS_SYSTEM_PROMPT,
        "user_prompt_builder": build_progress_user_prompt,
    },
    "no_loop": {
        "system_prompt": NO_LOOP_SYSTEM_PROMPT,
        "user_prompt_builder": build_legacy_user_prompt,
    },
}


def load_model(adapter_path=None, base_model_name=BASE_MODEL_NAME):
    """Load the base model and optionally a LoRA adapter."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if adapter_path is not None:
        validate_adapter_base(adapter_path, base_model_name)

    print("\n=== LOADING TOKENIZER ===")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print("\n=== LOADING BASE MODEL ===")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.float16,
    )

    if adapter_path is not None:

        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: {adapter_path}"
            )

        print("\n=== LOADING LoRA ADAPTER ===")
        print(adapter_path)

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

    else:
        print("\n=== BASE MODEL: NO ADAPTER ===")

    model.eval()
    model.config.use_cache = True

    print("\nModel loaded successfully.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model, tokenizer


@torch.inference_mode()
def generate_action(
    model,
    tokenizer,
    history,
    observation,
    inventory,
    admissible_commands,
    objective,
    prompt_version="legacy",
    score=None,
    max_score=None,
    repetition_penalty=REPETITION_PENALTY,
):
    """
    Generate the next action.

    If the prompt exceeds MAX_LENGTH, progressively remove
    the oldest history entries while preserving:

    - the system prompt,
    - the game objective,
    - the current observation,
    - the current inventory,
    - the admissible commands.
    """

    if prompt_version not in PROMPT_CONFIGS:
        raise ValueError(
            f"Unknown prompt version: {prompt_version}"
        )

    prompt_config = PROMPT_CONFIGS[prompt_version]

    system_prompt = prompt_config["system_prompt"]
    user_prompt_builder = prompt_config["user_prompt_builder"]

    current_step = {
        "objective": objective,
        "observation": observation,
        "inventory": inventory,
        "admissible_commands": admissible_commands,
        "score": score,
        "max_score": max_score,
    }

    # Preserve the original history.
    bounded_history = list(history)

    removed_history_steps = 0

    while True:

        user_prompt = user_prompt_builder(
            current_step,
            bounded_history,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[-1]

        # Remove the oldest interaction until the prompt fits.
        if prompt_length <= MAX_LENGTH:
            break

        if not bounded_history:
            raise ValueError(
                f"Current prompt alone exceeds MAX_LENGTH: "
                f"{prompt_length} > {MAX_LENGTH}. "
                "Consider shortening the current observation "
                "or admissible command list."
            )

        bounded_history.pop(0)
        removed_history_steps += 1

    if removed_history_steps:
        print(
            f"     Context limited: removed "
            f"{removed_history_steps} oldest history steps "
            f"({len(history)} -> {len(bounded_history)})."
        )

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0, input_ids.shape[-1]:]

    action = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return action


def evaluate_game(
    model,
    tokenizer,
    game_id,
    game_path,
    max_steps=MAX_STEPS,
    prompt_version="legacy",
    repetition_penalty=REPETITION_PENALTY,
):
    """Evaluate the model on one TextWorld game."""

    print("\n" + "=" * 65)
    print(f"EVALUATING {game_id}")
    print("=" * 65)

    request_infos = EnvInfos(
        admissible_commands=True,
        inventory=True,
        description=True,
        objective=True,
        won=True,
        lost=True,
        moves=True,
        score=True,
        max_score=True,
    )

    env = textworld.start(
        str(game_path),
        request_infos=request_infos,
    )

    history = []
    trajectory = []

    invalid_actions = 0

    try:

        state = env.reset()

        objective = getattr(state, "objective", "") or ""

        if not objective:
            raise ValueError(
                f"{game_id}: TextWorld did not provide "
                "a game objective."
            )

        print("Objective:", objective)

        for step in range(max_steps):

            observation = sanitize_observation(
                state.feedback
            )

            inventory = state.inventory

            admissible_commands = list(
                state.admissible_commands or []
            )

            if not admissible_commands:
                print("No admissible commands available.")
                break
            
            current_score = float(state.score)
            current_max_score = float(state.max_score)
            
            action = generate_action(
                model=model,
                tokenizer=tokenizer,
                history=history,
                observation=observation,
                inventory=inventory,
                admissible_commands=admissible_commands,
                objective=objective,
                prompt_version=prompt_version,
                score=current_score,
                max_score=current_max_score,
                repetition_penalty=repetition_penalty,
            )

            is_admissible = action in admissible_commands

            if not is_admissible:
                invalid_actions += 1

            print(
                f"[{step + 1:02d}] "
                f"Action: {action}"
            )

            print(
                f"     Admissible: {is_admissible}"
            )

            next_state, env_score, done = env.step(action)
            reward = score_delta(current_score, next_state.score)

            transition = {
                "step": step,
                "objective": objective,
                "score": current_score,
                "max_score": current_max_score,
                "observation": observation,
                "inventory": inventory,
                "admissible_commands": admissible_commands,
                "action": action,
                "is_admissible": is_admissible,
                "next_observation": sanitize_observation(
                    next_state.feedback
                ),
                "next_inventory": next_state.inventory,
                "next_score": float(next_state.score),
                "reward": float(reward),
                "done": bool(done),
                "won": bool(next_state.won),
                "lost": bool(next_state.lost),
            }

            trajectory.append(transition)

            history.append(transition)

            state = next_state

            if state.won or state.lost or done:
                break

        success = bool(state.won)

        result = {
            "game_id": game_id,
            "objective": objective,
            "success": success,
            "won": bool(state.won),
            "lost": bool(state.lost),
            "num_steps": len(trajectory),
            "invalid_actions": invalid_actions,
            "invalid_action_rate": (
                invalid_actions / len(trajectory)
                if trajectory else 0.0
            ),
            "max_score": float(state.max_score),
            "final_score": float(state.score),
            "reward_semantics": "score_delta",
            "total_reward": sum(
                transition["reward"]
                for transition in trajectory
            ),
            "steps": trajectory,
        }

    finally:
        env.close()

    print("\nGame summary:")
    print(f"  Success: {result['success']}")
    print(f"  Steps: {result['num_steps']}")
    print(f"  Invalid actions: {result['invalid_actions']}")

    return result


def evaluate_test_set(
    model,
    tokenizer,
    adapter_path=None,
    max_steps=MAX_STEPS,
    output_path=None,
    split="validation",
    prompt_version="legacy",
    base_model_name=BASE_MODEL_NAME,
    repetition_penalty=REPETITION_PENALTY,
):
    """
    Evaluate the model on a selected dataset split.

    Validation is the default to avoid accidentally reusing
    the test set during model selection.
    """

    if split not in ("validation", "test"):
        raise ValueError(
            f"Unsupported split: {split}. "
            "Choose 'validation' or 'test'."
        )

    if prompt_version not in PROMPT_CONFIGS:
        raise ValueError(
            f"Unsupported prompt version: {prompt_version}. "
            "Choose 'legacy', 'ordered' or 'progress'."
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    selected_ids = manifest["splits"][split]

    games_by_id = {
        game["game_id"]: game
        for game in manifest["games"]
    }

    print("\n" + "=" * 65)
    print("TEXTWORLD GAME EVALUATION")
    print("=" * 65)

    print(f"Evaluation split: {split}")
    print(f"Prompt version: {prompt_version}")
    print(f"Number of games: {len(selected_ids)}")
    print(f"Maximum steps per game: {max_steps}")

    results = []

    for index, game_id in enumerate(selected_ids):

        game = games_by_id[game_id]

        game_path = PROJECT_ROOT / game["game_path"]

        print(
            f"\n[{index + 1}/{len(selected_ids)}] "
            f"{game_id}"
        )

        result = evaluate_game(
            model=model,
            tokenizer=tokenizer,
            game_id=game_id,
            game_path=game_path,
            max_steps=max_steps,
            prompt_version=prompt_version,
            repetition_penalty=repetition_penalty,
        )

        results.append(result)

    num_games = len(results)

    successful_games = sum(
        result["success"]
        for result in results
    )

    total_steps = sum(
        result["num_steps"]
        for result in results
    )

    total_invalid = sum(
        result["invalid_actions"]
        for result in results
    )

    completion_rate = (
        successful_games / num_games
        if num_games else 0.0
    )

    invalid_action_rate = (
        total_invalid / total_steps
        if total_steps else 0.0
    )

    average_steps = (
        total_steps / num_games
        if num_games else 0.0
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": base_model_name,
        "reward_semantics": "score_delta",
        "repetition_penalty": repetition_penalty,
        "adapter": (
            str(adapter_path)
            if adapter_path is not None
            else None
        ),
        "split": split,
        "prompt_version": prompt_version,
        "num_games": num_games,
        "successful_games": successful_games,
        "completion_rate": completion_rate,
        "average_steps": average_steps,
        "total_steps": total_steps,
        "total_invalid_actions": total_invalid,
        "invalid_action_rate": invalid_action_rate,
        "max_steps_per_game": max_steps,
        "results": results,
    }

    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS")
    print("=" * 65)

    print(
        f"Quest completion: "
        f"{successful_games}/{num_games} "
        f"({completion_rate:.1%})"
    )

    print(f"Average steps per game: {average_steps:.2f}")

    print(
        f"Invalid actions: "
        f"{total_invalid}/{total_steps} "
        f"({invalid_action_rate:.1%})"
    )

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"\nResults saved to: {output_path}")

    return summary


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a TextWorld language model."
    )

    parser.add_argument(
        "--adapter",
        type=str,
        default=str(DEFAULT_ADAPTER),
        help="Path to LoRA adapter. Use 'none' for the base model.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL_NAME,
        help=(
            "Base model name, e.g. Qwen/Qwen2.5-0.5B-Instruct "
            "or Qwen/Qwen2.5-1.5B-Instruct."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum number of actions per game.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation/finetuned_games.json",
        help="Path to the output JSON file.",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Dataset split to evaluate. Defaults to validation.",
    )

    parser.add_argument(
        "--prompt-version",
        type=str,
        choices=["legacy", "ordered", "progress", "no_loop"],
        default="legacy",
        help="Prompt format matching the model's training configuration.",
    )

    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    args = parser.parse_args()
    if not (0 < args.repetition_penalty < float("inf")):
        parser.error("repetition penalty must be positive and finite")

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")

    adapter_path = (
        None
        if args.adapter.lower() == "none"
        else args.adapter
    )

    model, tokenizer = load_model(
        adapter_path,
        base_model_name=args.base_model,
    )

    evaluate_test_set(
        model=model,
        tokenizer=tokenizer,
        adapter_path=adapter_path,
        max_steps=args.max_steps,
        output_path=args.output,
        base_model_name=args.base_model,
        split=args.split,
        prompt_version=args.prompt_version,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
