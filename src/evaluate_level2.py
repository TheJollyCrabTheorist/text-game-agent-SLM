import argparse
import json
from pathlib import Path

from evaluate_games import (
    PROJECT_ROOT,
    BASE_MODEL_NAME,
    REPETITION_PENALTY,
    load_model,
    evaluate_game,
)

MANIFEST_PATH = PROJECT_ROOT / "data/level2/splits.json"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Level 1 model on Level 2 TextWorld games."
    )

    parser.add_argument(
        "--adapter",
        type=str,
        default=str(
            PROJECT_ROOT
            / "models/qwen25-05b-textworld-qlora-objective/best_adapter"
        ),
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL_NAME,
        help="Base model name, e.g. Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen2.5-1.5B-Instruct.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--prompt-version",
        choices=["legacy", "ordered", "progress", "no_loop"],
        default="legacy",
        help="Prompt format used to generate actions.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    
    parser.add_argument("--manifest",
        type=str,
        default="data/level2/splits.json",
        help="Path to the game manifest."
    )

    parser.add_argument(
        "--expert-trajectories-dir",
        type=str,
        default=None,
        help="Directory containing expert trajectories (optional)."
    )
    
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    args = parser.parse_args()
    if args.max_steps < 1 or not (0 < args.repetition_penalty < float("inf")):
        parser.error("max steps and repetition penalty must be positive (finite penalty)")

    adapter_path = (
        None if args.adapter.lower() == "none"
        else args.adapter
    )

    manifest_path = PROJECT_ROOT / args.manifest

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    expert_dir = (
        PROJECT_ROOT / args.expert_trajectories_dir
        if args.expert_trajectories_dir
        else manifest_path.parent / "trajectories"
    )

    has_expert_trajectories = expert_dir.exists()
    selected_ids = manifest["splits"][args.split]

    games_by_id = {
        game["game_id"]: game
        for game in manifest["games"]
    }

    model, tokenizer = load_model(
        adapter_path,
        base_model_name=args.base_model,
    )

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
            max_steps=args.max_steps,
            prompt_version=args.prompt_version,
            repetition_penalty=args.repetition_penalty,
        )

        # Attach expert reference when available.
        expert_path = expert_dir / f"{game_id}.json"

        if expert_path.exists():
            expert = json.loads(
                expert_path.read_text(encoding="utf-8")
            )

            if expert.get("success") and expert.get("won"):
                expert_steps = expert["num_steps"]

                result["expert_steps"] = expert_steps
                result["steps_above_expert"] = (
                    result["num_steps"] - expert_steps
                )
            else:
                result["expert_steps"] = None
                result["steps_above_expert"] = None

        results.append(result) 
    num_games = len(results)
    successful_games = sum(r["success"] for r in results)
    total_steps = sum(r["num_steps"] for r in results)
    total_invalid = sum(r["invalid_actions"] for r in results)
    expert_results = [
        r for r in results
        if r.get("expert_steps") is not None
    ]

    average_expert_steps = (
        sum(r["expert_steps"] for r in expert_results)
        / len(expert_results)
        if expert_results else None
    )

    average_steps_above_expert = (
        sum(r["steps_above_expert"] for r in expert_results)
        / len(expert_results)
        if expert_results else None
    )

    successful_expert_comparisons = [
        r for r in expert_results if r["success"]
    ]

    average_steps_above_expert_on_success = (
        sum(
            r["steps_above_expert"]
            for r in successful_expert_comparisons
        ) / len(successful_expert_comparisons)
        if successful_expert_comparisons else None
    )
    
    summary = {
        "experiment": f"level2_zero_shot_{args.prompt_version}",
        "prompt_version": args.prompt_version,
        "model": args.base_model,
        "reward_semantics": "score_delta",
        "repetition_penalty": args.repetition_penalty,
        "adapter": str(adapter_path) if adapter_path else None,
        "split": args.split,
        "num_games": num_games,
        "successful_games": successful_games,
        "completion_rate": (
            successful_games / num_games if num_games else 0
        ),
        "average_steps": (
            total_steps / num_games if num_games else 0
        ),
        "total_steps": total_steps,
        "total_invalid_actions": total_invalid,
        "invalid_action_rate": (
            total_invalid / total_steps if total_steps else 0
        ),
        "max_steps_per_game": args.max_steps,
        "results": results,
        "manifest": str(manifest_path),
        "expert_trajectories_dir": (
            str(expert_dir) if has_expert_trajectories else None
        ),
        "games_with_expert_reference": len(expert_results),
        "average_expert_steps": average_expert_steps,
        "average_steps_above_expert": average_steps_above_expert,
        "average_steps_above_expert_on_success": (
            average_steps_above_expert_on_success
        ),
    }

    output_path = (
        PROJECT_ROOT / args.output
        if args.output
        else PROJECT_ROOT
        / "evaluation"
        / "level2"
        / f"zero_shot_{args.prompt_version}_{args.split}.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("LEVEL 2 MODEL EVALUATION")
    print("=" * 60)
    print(f"Prompt version: {args.prompt_version}")
    print(
        f"Quest completion: {successful_games}/{num_games} "
        f"({summary['completion_rate']:.1%})"
    )
    print(f"Average steps: {summary['average_steps']:.2f}")
    print(
        f"Invalid actions: {total_invalid}/{total_steps} "
        f"({summary['invalid_action_rate']:.1%})"
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
