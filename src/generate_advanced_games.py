import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_splits(game_ids, seed=42):
    shuffled = game_ids.copy()
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }


def generate_game(game_id, seed, output_path, params, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to regenerate."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "tw-make", "custom",
        "--world-size", str(params["world_size"]),
        "--nb-objects", str(params["nb_objects"]),
        "--quest-min-length", str(params["quest_min_length"]),
        "--quest-max-length", str(params["quest_max_length"]),
        "--quest-min-breadth", str(params["quest_min_breadth"]),
        "--quest-max-breadth", str(params["quest_max_breadth"]),
        "--quest-min-depth", str(params["quest_min_depth"]),
        "--quest-max-depth", str(params["quest_max_depth"]),
        "--nb-parallel-quests", str(params["nb_parallel_quests"]),
        "--theme", params["theme"],
        "--seed", str(seed),
        "--output", str(output_path),
    ]

    if params["ambiguous_instructions"]:
        command.append("--ambiguous-instructions")

    if params["blend_descriptions"]:
        command.append("--blend-descriptions")

    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    if not output_path.exists():
        raise RuntimeError(f"TextWorld did not produce {output_path}")

    return {
        "game_id": game_id,
        "seed": seed,
        "game_path": str(output_path.relative_to(PROJECT_ROOT)),
        "generation_parameters": params.copy()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate an advanced TextWorld dataset for Level 2."
    )

    parser.add_argument("--num-games", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/level2", help="Output directory for games and splits.json.")
    parser.add_argument("--test-only", action="store_true", help="Assign all generated games to the test split.")

    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--nb-objects", type=int, default=15)

    parser.add_argument("--quest-min-length", type=int, default=5)
    parser.add_argument("--quest-max-length", type=int, default=8)

    parser.add_argument("--quest-min-breadth", type=int, default=2)
    parser.add_argument("--quest-max-breadth", type=int, default=3)

    parser.add_argument("--quest-min-depth", type=int, default=2)
    parser.add_argument("--quest-max-depth", type=int, default=4)

    parser.add_argument("--nb-parallel-quests", type=int, default=1)
    parser.add_argument("--theme", type=str, default="house")

    parser.add_argument("--ambiguous-instructions", action="store_true")
    parser.add_argument("--blend-descriptions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.num_games < 3:
        parser.error("--num-games must be at least 3.")

    if args.quest_min_length > args.quest_max_length:
        parser.error("Minimum quest length cannot exceed maximum.")

    if args.quest_min_breadth > args.quest_max_breadth:
        parser.error("Minimum quest breadth cannot exceed maximum.")

    if args.quest_min_depth > args.quest_max_depth:
        parser.error("Minimum quest depth cannot exceed maximum.")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    games_dir = output_dir / "games"
    manifest_path = output_dir / "splits.json"

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists. "
            "Use --overwrite to regenerate the dataset."
        )

    games_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "world_size": args.world_size,
        "nb_objects": args.nb_objects,
        "quest_min_length": args.quest_min_length,
        "quest_max_length": args.quest_max_length,
        "quest_min_breadth": args.quest_min_breadth,
        "quest_max_breadth": args.quest_max_breadth,
        "quest_min_depth": args.quest_min_depth,
        "quest_max_depth": args.quest_max_depth,
        "nb_parallel_quests": args.nb_parallel_quests,
        "theme": args.theme,
        "ambiguous_instructions": args.ambiguous_instructions,
        "blend_descriptions": args.blend_descriptions,
    }

    game_ids = [
        f"advanced_game_{i:04d}"
        for i in range(args.num_games)
    ]

    if args.test_only:
        splits = {
            "train": [],
            "validation": [],
            "test": game_ids.copy()
        }
    else:
        splits = create_splits(game_ids, args.split_seed)

    split_by_game = {
        game_id: split
        for split, ids in splits.items()
        for game_id in ids
    }

    print("=" * 65)
    print("LEVEL 2 — ADVANCED TEXTWORLD GENERATION")
    print("=" * 65)
    print(f"Games: {args.num_games}")
    print(f"Generation parameters: {params}")
    print()

    games = []

    for i, game_id in enumerate(game_ids):
        seed = args.base_seed + i
        output_path = games_dir / f"{game_id}.z8"

        print(
            f"[{i + 1:03d}/{args.num_games}] "
            f"Generating {game_id} (seed={seed})..."
        )

        metadata = generate_game(
            game_id=game_id,
            seed=seed,
            output_path=output_path,
            params=params,
            overwrite=args.overwrite
        )

        metadata["split"] = split_by_game[game_id]
        games.append(metadata)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "level": 2,
        "generator": "TextWorld tw-make custom",
        "num_games": args.num_games,
        "base_seed": args.base_seed,
        "split_seed": args.split_seed,
        "generation_parameters": params,
        "splits": splits,
        "games": games
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("GENERATION SUMMARY")
    print("=" * 65)

    for split, ids in splits.items():
        print(f"{split.capitalize():12s}: {len(ids)} games")

    print(f"\nManifest: {manifest_path}")
    print(f"Games:    {games_dir}")


if __name__ == "__main__":
    main()
