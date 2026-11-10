import argparse
import contextlib
import io
import json
from pathlib import Path

from collect_walkthrough import collect_walkthrough


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect_dataset(manifest_path, output_dir, max_steps=150, overwrite=False):
    manifest_path = PROJECT_ROOT / manifest_path
    output_dir = PROJECT_ROOT / output_dir

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    games = manifest["games"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 65)
    print("LEVEL 2 — ORACLE TRAJECTORY COLLECTION")
    print("=" * 65)
    print(f"Games: {len(games)}")
    print(f"Output: {output_dir}")
    print()

    for i, game in enumerate(games):
        game_id = game["game_id"]
        split = game["split"]
        game_path = PROJECT_ROOT / game["game_path"]
        output_path = output_dir / f"{game_id}.json"

        print(f"[{i + 1:03d}/{len(games)}] {game_id} | {split}")

        try:
            if not game_path.exists():
                raise FileNotFoundError(game_path)

            if output_path.exists() and not overwrite:
                print("  SKIPPED: trajectory already exists")
                with output_path.open(encoding="utf-8") as f:
                    trajectory = json.load(f)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    collect_walkthrough(
                        game_path=game_path,
                        output_path=output_path,
                        max_steps=max_steps
                    )

                with output_path.open(encoding="utf-8") as f:
                    trajectory = json.load(f)

            # Attach Level 2 experimental metadata.
            trajectory["split"] = split
            trajectory["seed"] = game["seed"]
            trajectory["generation_parameters"] = game[
                "generation_parameters"
            ]

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)

            success = bool(trajectory.get("success", False))

            result = {
                "game_id": game_id,
                "split": split,
                "success": success,
                "num_steps": trajectory.get("num_steps", 0),
                "total_reward": trajectory.get("total_reward", 0.0)
            }

            print(
                f"  Success: {success} | "
                f"Steps: {result['num_steps']} | "
                f"Reward: {result['total_reward']}"
            )

        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")

            result = {
                "game_id": game_id,
                "split": split,
                "success": False,
                "error": str(error)
            }

        results.append(result)

    summary = {
        "num_games": len(results),
        "successful_games": sum(r["success"] for r in results),
        "failed_games": sum(not r["success"] for r in results),
        "results": results
    }

    summary_path = output_dir / "collection_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("LEVEL 2 — COLLECTION SUMMARY")
    print("=" * 65)
    print(f"Games: {summary['num_games']}")
    print(f"Successful: {summary['successful_games']}")
    print(f"Failed: {summary['failed_games']}")

    for split in ("train", "validation", "test"):
        subset = [r for r in results if r["split"] == split]
        if subset:
            successful = sum(r["success"] for r in subset)
            print(f"{split.capitalize():12s}: {successful}/{len(subset)}")

    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Level 2 TextWorld oracle trajectories."
    )

    parser.add_argument(
        "--manifest",
        default="data/level2/splits.json"
    )
    parser.add_argument(
        "--output-dir",
        default="data/level2/trajectories"
    )
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    collect_dataset(
        args.manifest,
        args.output_dir,
        args.max_steps,
        args.overwrite
    )


if __name__ == "__main__":
    main()
