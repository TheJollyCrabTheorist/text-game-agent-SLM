import json
from collections import Counter
from pathlib import Path

EVALUATIONS = {
    "Base": Path("evaluation/base"),
    "Old QLoRA": Path("evaluation/old_qlora"),
    "New QLoRA": Path("evaluation/objective_qlora"),
}


def load_evaluation(path):
    """Load an evaluation report from disk."""
    with path.open() as f:
        data = json.load(f)

    return {
        game["game_id"]: game
        for game in data["results"]
    }


def won(game):
    """Return whether a game was won."""
    return game.get("won", game.get("success", False))


def main():
    """Run the evaluation comparison pipeline."""
    evaluations = {
        name: load_evaluation(path)
        for name, path in EVALUATIONS.items()
    }

    # --------------------------------------------------
    # 1. Overall comparison
    # --------------------------------------------------
    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(f"{'Game':<15}", end="")
    for name in evaluations:
        print(f"{name:>16}", end="")
    print()

    game_ids = sorted(evaluations["New QLoRA"])

    for game_id in game_ids:
        print(f"{game_id:<15}", end="")
        for games in evaluations.values():
            game = games[game_id]
            status = "WIN" if won(game) else "FAIL"
            print(f"{status:>16}", end="")
        print()

    print("\n" + "-" * 65)

    for name, games in evaluations.items():
        successes = sum(won(g) for g in games.values())
        invalid = sum(g["invalid_actions"] for g in games.values())
        steps = sum(g["num_steps"] for g in games.values())

        print(
            f"{name:<12} "
            f"Wins: {successes:2d}/15 | "
            f"Steps: {steps:3d} | "
            f"Invalid actions: {invalid:3d}"
        )

    # --------------------------------------------------
    # 2. Games failed by the new model
    # --------------------------------------------------
    new_games = evaluations["New QLoRA"]

    failed_games = [
        game_id
        for game_id, game in new_games.items()
        if not won(game)
    ]

    print("\n" + "=" * 65)
    print(f"NEW QLoRA FAILURE ANALYSIS ({len(failed_games)} games)")
    print("=" * 65)

    if not failed_games:
        print("No failed games!")
        return

    for game_id in failed_games:
        print(f"\n{'=' * 55}")
        print(f"{game_id}")
        print("=" * 55)

        # Compare outcomes across models
        for name, games in evaluations.items():
            game = games[game_id]
            status = "WIN" if won(game) else "FAIL"
            print(
                f"{name:<12} {status:<6} "
                f"steps={game['num_steps']:<3} "
                f"invalid={game['invalid_actions']}"
            )

        game = new_games[game_id]

        print(f"\nObjective: {game['objective']}")
        print(f"Actions ({len(game['steps'])}):")

        actions = []

        for step in game["steps"]:
            action = step.get("action", "<missing>")
            admissible = step.get("is_admissible")

            marker = "" if admissible is not False else " [INVALID]"
            actions.append(action)

            print(f"  {step['step']:02d}: {action}{marker}")

        # Repeated actions
        counts = Counter(actions)
        repeated = [
            (action, count)
            for action, count in counts.items()
            if count > 1
        ]

        if repeated:
            print("\nRepeated actions:")
            for action, count in sorted(
                repeated, key=lambda x: -x[1]
            ):
                print(f"  {count}x {action}")

        # Invalid actions
        invalid = [
            step
            for step in game["steps"]
            if step.get("is_admissible") is False
        ]

        if invalid:
            print("\nInvalid actions:")
            for step in invalid:
                print(
                    f"  Step {step['step']}: "
                    f"{step['action']}"
                )


if __name__ == "__main__":
    main()
