import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_splits(game_ids, seed=42):
    """Create reproducible train/validation/test splits by game."""

    shuffled_ids = game_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)

    n = len(shuffled_ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    return {
        "train": shuffled_ids[:n_train],
        "validation": shuffled_ids[n_train:n_train + n_val],
        "test": shuffled_ids[n_train + n_val:]
    }


def generate_game(
    game_id,
    seed,
    output_path,
    world_size,
    nb_objects,
    quest_length,
    overwrite=False
):
    """Generate one TextWorld game using tw-make."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Use --overwrite to regenerate it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "tw-make",
        "custom",
        "--world-size", str(world_size),
        "--nb-objects", str(nb_objects),
        "--quest-length", str(quest_length),
        "--seed", str(seed),
        "--output", str(output_path)
    ]

    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT
    )

    if not output_path.exists():
        raise RuntimeError(
            f"TextWorld did not produce the expected file: {output_path}"
        )

    return {
        "game_id": game_id,
        "seed": seed,
        "game_path": str(output_path.relative_to(PROJECT_ROOT)),
        "world_size": world_size,
        "nb_objects": nb_objects,
        "quest_length": quest_length
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible TextWorld dataset."
    )

    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--nb-objects", type=int, default=10)
    parser.add_argument("--quest-length", type=int, default=5)

    parser.add_argument(
        "--manifest",
        default="data/raw/splits.json"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    if args.num_games < 3:
        parser.error("--num-games must be at least 3.")

    manifest_path = PROJECT_ROOT / args.manifest
    games_dir = PROJECT_ROOT / "data/raw/games"

    games_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists. "
            "Use --overwrite to regenerate the dataset."
        )

    game_ids = [
        f"game_{i:04d}"
        for i in range(args.num_games)
    ]

    splits = create_splits(
        game_ids,
        seed=args.split_seed
    )

    split_by_game = {
        game_id: split
        for split, ids in splits.items()
        for game_id in ids
    }

    print("=" * 60)
    print("TEXTWORLD DATASET GENERATION")
    print("=" * 60)
    print(f"Games: {args.num_games}")
    print(f"Base seed: {args.base_seed}")
    print(f"Split seed: {args.split_seed}")
    print(f"World size: {args.world_size}")
    print(f"Objects: {args.nb_objects}")
    print(f"Quest length: {args.quest_length}")
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
            world_size=args.world_size,
            nb_objects=args.nb_objects,
            quest_length=args.quest_length,
            overwrite=args.overwrite
        )

        metadata["split"] = split_by_game[game_id]
        games.append(metadata)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "TextWorld tw-make custom",
        "num_games": args.num_games,
        "base_seed": args.base_seed,
        "split_seed": args.split_seed,
        "generation_parameters": {
            "world_size": args.world_size,
            "nb_objects": args.nb_objects,
            "quest_length": args.quest_length
        },
        "splits": splits,
        "games": games
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    for split, ids in splits.items():
        print(f"{split.capitalize():12s}: {len(ids)} games")

    print(f"\nManifest saved to: {manifest_path}")
    print(f"Games saved to: {games_dir}")


if __name__ == "__main__":
    main()
