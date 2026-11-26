import re
import argparse
from data_integrity import score_delta
import json
from pathlib import Path
from datetime import datetime, timezone

import textworld
import textworld.gym
from textworld import EnvInfos

def sanitize_observation(observation):
    """
    Remove TextWorld's initial walkthrough and ASCII banner.
    Preserve the actual game description and state.
    """

    # Remove the explicit walkthrough instructions.
    observation = re.sub(
        r"It's time to explore.*?you're the winner!",
        "",
        observation,
        flags=re.DOTALL
    )

    # Remove the ASCII banner if it appears at the beginning.
    if observation.lstrip().startswith("________"):
        observation = re.sub(
            r"\A\s*________.*?(?=\n\s*-= .+ =-)",
            "",
            observation,
            count=1,
            flags=re.DOTALL
        )

    return observation.strip()

def serialize_info(infos):
    """Convert TextWorld information into JSON-compatible values."""
    serialized = {}

    for key, value in infos.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            serialized[key] = list(value)
        else:
            serialized[key] = str(value)

    return serialized


def play_game(game_path, output_path, max_steps=50):

    """Play and record one TextWorld game."""
    game_path = Path(game_path)

    if not game_path.exists():
        raise FileNotFoundError(f"Game not found: {game_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Request additional information from TextWorld.
    request_infos = EnvInfos(
        admissible_commands=True,
        inventory=True,
        description=True,
        won=True,
        lost=True,
        score=True,
        max_score=True
    )

    # Register and initialize the environment.
    env_id = textworld.gym.register_game(
        str(game_path),
        request_infos=request_infos,
        max_episode_steps=max_steps
    )

    env = textworld.gym.make(env_id)

    trajectory = {
        "game_id": game_path.stem,
        "game_path": str(game_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_steps": max_steps,
        "steps": [],
        "success": False,
        "total_reward": 0.0,
        "num_steps": 0
    }

    try:
        observation, infos = env.reset()
        current_score = float(infos["score"])

        print("\n" + "=" * 60)
        print("TEXTWORLD GAME")
        print("=" * 60)

        print("\nInitial observation:")
        print(observation)

        print("\nInventory:")
        print(infos.get("inventory", "Not available"))

        for step in range(max_steps):

            admissible = infos.get("admissible_commands", [])

            print("\n" + "-" * 60)
            print(f"STEP {step + 1}/{max_steps}")
            print("-" * 60)

            print("\nAdmissible commands:")

            for i, command in enumerate(admissible[:30]):
                print(f"  {i}: {command}")

            command = input("\nYour command (or 'quit'): ").strip()

            if command.lower() == "quit":
                break

            if not command:
                print("Empty command ignored.")
                continue

            # Execute the action.
            next_observation, score, done, next_infos = env.step(command)

            # Record the transition.
            transition = {
                "step": len(trajectory["steps"]),
                "observation": sanitize_observation(observation),
                "inventory": infos.get("inventory"),
                "admissible_commands": admissible,
                "action": command,
                "next_observation": sanitize_observation(next_observation),
                "score": current_score,
                "next_score": float(score),
                "max_score": float(infos["max_score"]),
                "reward": score_delta(current_score, score),
                "done": bool(done),
                "won": next_infos.get("won"),
                "lost": next_infos.get("lost"),
                "next_inventory": next_infos.get("inventory")
            }

            trajectory["steps"].append(transition)

            print("\nObservation:")
            print(next_observation)

            print(f"\nCumulative score: {score}")

            print(f"Victory: {next_infos.get('won')}")
            print(f"Defeat: {next_infos.get('lost')}")

            observation = next_observation
            infos = next_infos
            current_score = float(score)

            if done:
                break

        trajectory["success"] = any(
            step["won"] is True
            for step in trajectory["steps"]
        )

        trajectory["total_reward"] = sum(s["reward"] for s in trajectory["steps"])
        trajectory["final_score"] = current_score
        trajectory["reward_semantics"] = "score_delta"

        trajectory["num_steps"] = len(trajectory["steps"])

    finally:
        env.close()

    # Save the complete trajectory.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            trajectory,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)

    print(f"Game: {game_path.name}")
    print(f"Success: {trajectory['success']}")
    print(f"Steps: {trajectory['num_steps']}")
    print(f"Total reward: {trajectory['total_reward']}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/trajectories/first_trajectory.json"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50
    )

    args = parser.parse_args()

    play_game(
        game_path=args.game,
        output_path=args.output,
        max_steps=args.max_steps
    )
