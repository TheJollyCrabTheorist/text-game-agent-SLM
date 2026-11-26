import argparse
import contextlib
import io
import json
from pathlib import Path

from collect_walkthrough import collect_walkthrough


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect_dataset(
    manifest_path,
    output_dir,
    max_steps=100,
    overwrite=False
):
    """Collect expert walkthrough trajectories from a dataset manifest."""
    manifest_path = PROJECT_ROOT / manifest_path
    output_dir = PROJECT_ROOT / output_dir

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run generate_games.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    games = manifest["games"]

    results = []

    print("=" * 65)
    print("TEXTWORLD ORACLE DATASET COLLECTION")
    print("=" * 65)
    print(f"Games to collect: {len(games)}")
    print(f"Maximum steps: {max_steps}")
    print(f"Output directory: {output_dir}")
    print()

    for i, game in enumerate(games):
        game_id = game["game_id"]
        game_path = PROJECT_ROOT / game["game_path"]
        split = game["split"]

        output_path = output_dir / f"{game_id}.json"

        print(
            f"[{i + 1:03d}/{len(games)}] "
            f"{game_id} | {split}"
        )

        if not game_path.exists():
            print(f"  ERROR: Game not found: {game_path}")
            results.append({
                "game_id": game_id,
                "split": split,
                "success": False,
                "error": "game_file_missing"
            })
            continue

        if output_path.exists() and not overwrite:
            print("  SKIPPED: trajectory already exists")

            with output_path.open(encoding="utf-8") as f:
                trajectory = json.load(f)

        else:
            try:
                # Suppress TextWorld's large ASCII banner and walkthrough
                # during batch collection. The cleaned observations are
                # still saved by collect_walkthrough().
                with contextlib.redirect_stdout(io.StringIO()):
                    collect_walkthrough(
                        game_path=game_path,
                        output_path=output_path,
                        max_steps=max_steps
                    )

                with output_path.open(encoding="utf-8") as f:
                    trajectory = json.load(f)

                # Attach experimental metadata.
                trajectory["split"] = split
                trajectory["seed"] = game["seed"]
                trajectory["generation_parameters"] = {
                    "world_size": game["world_size"],
                    "nb_objects": game["nb_objects"],
                    "quest_length": game["quest_length"]
                }

                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        trajectory,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )

            except Exception as error:
                print(f"  ERROR: {type(error).__name__}: {error}")

                results.append({
                    "game_id": game_id,
                    "split": split,
                    "success": False,
                    "error": str(error)
                })
                continue

        success = bool(trajectory.get("success", False))
        num_steps = trajectory.get("num_steps", 0)
        score = trajectory.get("total_reward", 0.0)

        print(
            f"  Success: {success} | "
            f"Steps: {num_steps} | "
            f"Score: {score}"
        )

        results.append({
            "game_id": game_id,
            "split": split,
            "success": success,
            "num_steps": num_steps,
            "total_reward": score
        })

    # Save a compact collection summary.
    summary = {
        "num_games": len(results),
        "successful_games": sum(
            r.get("success", False) for r in results
        ),
        "failed_games": sum(
            not r.get("success", False) for r in results
        ),
        "results": results
    }

    summary_path = output_dir / "collection_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("COLLECTION SUMMARY")
    print("=" * 65)

    print(f"Games processed: {summary['num_games']}")
    print(f"Successful: {summary['successful_games']}")
    print(f"Failed: {summary['failed_games']}")

    for split in ("train", "validation", "test"):
        split_results = [
            r for r in results
            if r["split"] == split
        ]

        if split_results:
            successes = sum(
                r.get("success", False)
                for r in split_results
            )

            print(
                f"{split.capitalize():12s}: "
                f"{successes}/{len(split_results)} successful"
            )

    print(f"\nSummary saved to: {summary_path}")


def main():
    """Run the dataset collection pipeline."""
    parser = argparse.ArgumentParser(
        description="Collect oracle trajectories for TextWorld games."
    )

    parser.add_argument(
        "--manifest",
        default="data/raw/splits.json"
    )

    parser.add_argument(
        "--output-dir",
        default="data/raw/trajectories"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    collect_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
