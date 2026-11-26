import argparse
from data_integrity import score_delta
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import textworld
from textworld import EnvInfos
from textworld.agents import WalkthroughAgent
from textworld.agents.walkthrough import WalkthroughDone


def sanitize_observation(observation):
    """
    Remove TextWorld's initial walkthrough and ASCII banner.
    Preserve the actual game description and state.
    """

    observation = re.sub(
        r"It's time to explore.*?you're the winner!",
        "",
        observation,
        flags=re.DOTALL
    )

    if observation.lstrip().startswith("________"):
        observation = re.sub(
            r"\A\s*________.*?(?=\n\s*-= .+ =-)",
            "",
            observation,
            count=1,
            flags=re.DOTALL
        )

    return observation.strip()


def collect_walkthrough(game_path, output_path, max_steps=100):

    game_path = Path(game_path)
    output_path = Path(output_path)

    if not game_path.exists():
        raise FileNotFoundError(game_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Request all information needed by the agent and the dataset.
    request_infos = EnvInfos(
        admissible_commands=True,
        inventory=True,
        description=True,
        objective=True,
        won=True,
        lost=True,
        moves=True,
        score=True,
        max_score=True
    )

    env = textworld.start(
        str(game_path),
        request_infos=request_infos
    )

    agent = WalkthroughAgent()

    try:
        agent.reset(env)
        state = env.reset()

        # The objective is constant throughout the game.
        objective = getattr(state, "objective", "") or ""

        trajectory = {
            "game_id": game_path.stem,
            "game_path": str(game_path),
            "agent": "WalkthroughAgent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objective": objective,
            "steps": [],
            "success": False,
            "total_reward": 0.0,
            "num_steps": 0
        }

        reward = 0.0
        env_score = float(state.score)
        total_reward = 0.0
        done = False

        print("\nStarting oracle agent...")
        print("Objective:", objective)
        print(state.feedback)

        for step in range(max_steps):

            if done:
                break

            # Ask the oracle for the next action.
            try:
                action = agent.act(state, env_score, done)
            except WalkthroughDone:
                trajectory["walkthrough_exhausted"] = True
                print("Walkthrough exhausted: no more oracle actions.")
                break

            admissible = getattr(
                state, "admissible_commands", []
            ) or []
            
            if action not in admissible:
                error_message = (
                    f"Oracle produced a non-admissible action "
                    f"at step {step}: {action!r}. "
                    f"Available commands: {admissible}"
                )

                print(f"\nORACLE ERROR: {error_message}")
                trajectory["oracle_error"] = error_message
                trajectory["walkthrough_exhausted"] = False
                break

            # Execute the action.
            current_score = float(state.score)
            next_state, env_score, done = env.step(action)
            reward = score_delta(current_score, next_state.score)

            total_reward += reward

            transition = {
                "step": step,
                "objective": objective,
                "observation": sanitize_observation(state.feedback),
                "inventory": state.inventory,
                "admissible_commands": list(admissible),
                "action": action,
                "next_observation": sanitize_observation(next_state.feedback),
                "next_inventory": next_state.inventory,
                "reward": float(reward),
                "done": bool(done),
                "won": bool(next_state.won),
                "lost": bool(next_state.lost),
                "score": current_score,
                "max_score": float(state.max_score),
                "next_score": float(next_state.score),
                "next_max_score": float(next_state.max_score),
            }

            trajectory["steps"].append(transition)

            print(
                f"[{step + 1:03d}] "
                f"Action: {action} | "
                f"Reward: {reward} | "
                f"Done: {done}"
            )

            state = next_state

        trajectory["environment_done"] = bool(done)
        trajectory["won"] = bool(state.won)
        trajectory["total_reward"] = total_reward
        trajectory["reward_semantics"] = "score_delta"
        trajectory["final_score"] = float(state.score)
        trajectory["num_steps"] = len(trajectory["steps"])
        trajectory["max_score"] = float(state.max_score)

        # Only an explicit TextWorld victory counts as success.
        trajectory["success"] = bool(state.won)

    finally:
        env.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            trajectory,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("ORACLE SUMMARY")
    print("=" * 55)

    print(f"Objective: {trajectory['objective']}")
    print(f"Success: {trajectory['success']}")
    print(f"Steps: {trajectory['num_steps']}")
    print(f"Reward: {trajectory['total_reward']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--game", required=True)

    parser.add_argument(
        "--output",
        default="data/raw/trajectories/oracle_trajectory.json"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100
    )

    args = parser.parse_args()

    collect_walkthrough(
        args.game,
        args.output,
        args.max_steps
    )
