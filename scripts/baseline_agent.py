"""
Baseline Agent for E-commerce Fulfillment Environment.

This script provides a simple heuristic-based agent that can be used
as a baseline for comparison with learned policies. The agent follows
simple rules to prioritize actions based on the current state.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from env.models import (
    Action,
    ActionType,
    AlertType,
    InquiryPriority,
    TASK_CONFIGS,
)
from env.environment import FulfillmentEnv


class BaselineAgent:
    """
    A simple heuristic-based agent for the fulfillment environment.

    Priority order:
    1. Handle urgent customer inquiries
    2. Process returns (to free up capacity)
    3. Restock low-stock items (prevent stockouts)
    4. Process orders (main revenue generator)
    5. Quality check / move orders through pipeline
    6. Idle (only if nothing else to do)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.action_log: list[dict[str, Any]] = []

    def select_action(self, observation: dict[str, Any]) -> Action:
        """
        Select an action based on the current observation.

        Args:
            observation: Current observation from the environment

        Returns:
            Action to take
        """
        action = self._decide(observation)

        if self.verbose:
            print(f"  Agent chose: {action.action_type.value}"
                  + (f" -> {action.target_id}" if action.target_id else ""))

        self.action_log.append({
            "step": observation["current_step"],
            "action": action.action_type.value,
            "target": action.target_id,
        })

        return action

    def _decide(self, obs: dict[str, Any]) -> Action:
        """Core decision logic."""

        # 1. Check for urgent alerts that need immediate attention
        alerts = obs.get("alerts", [])
        for alert in alerts:
            if alert["severity"] == "critical":
                if alert["type"] == AlertType.STOCKOUT.value:
                    # Critical stockout - restock immediately
                    product_id = self._find_product_for_alert(obs, alert)
                    if product_id:
                        return Action(ActionType.RESTOCK_ITEM, target_id=product_id)

        # 2. Handle urgent customer inquiries first
        if obs.get("urgent_inquiries", 0) > 0:
            return Action(ActionType.HANDLE_CUSTOMER)

        # 3. Process pending returns
        if obs.get("returns_pending", 0) > 0:
            return Action(ActionType.PROCESS_RETURN)

        # 4. Restock low-stock items (prevent future stockouts)
        low_stock = obs.get("low_stock_items", [])
        if low_stock:
            # Prioritize items that are most low on stock
            inventory = obs.get("inventory_levels", {})
            worst_item = min(
                low_stock,
                key=lambda pid: inventory.get(pid, {}).get("health", 1.0)
            )
            return Action(ActionType.RESTOCK_ITEM, target_id=worst_item)

        # 5. Process pending orders (main task)
        if obs.get("orders_pending", 0) > 0:
            return Action(ActionType.PROCESS_ORDER)

        # 6. Move orders through the pipeline
        if obs.get("orders_in_progress", 0) > 0:
            return Action(ActionType.QUALITY_CHECK)

        # 7. Handle any remaining customer inquiries
        if obs.get("customer_queue", 0) > 0:
            return Action(ActionType.HANDLE_CUSTOMER)

        # 8. Update inventory (minor optimization)
        if obs.get("current_step", 0) % 20 == 0:
            return Action(ActionType.UPDATE_INVENTORY)

        # 9. Expedite old orders if time is running low
        time_remaining = obs.get("time_remaining", 100)
        if time_remaining < 20 and obs.get("orders_pending", 0) > 0:
            return Action(ActionType.EXPEDITE_ORDER)

        # 10. Idle - nothing to do
        return Action(ActionType.IDLE)

    def _find_product_for_alert(self, obs: dict[str, Any], alert: dict[str, Any]) -> str | None:
        """Find a product ID associated with an alert."""
        # For stockout alerts, find any out of stock item
        if alert["type"] == AlertType.STOCKOUT.value:
            out_of_stock = obs.get("out_of_stock_items", [])
            if out_of_stock:
                return out_of_stock[0]
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the agent's actions."""
        action_counts: dict[str, int] = {}
        for entry in self.action_log:
            action = entry["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        return {
            "total_actions": len(self.action_log),
            "action_distribution": action_counts,
        }


def run_episode(
    env: FulfillmentEnv,
    agent: BaselineAgent,
    task_id: str,
    render: bool = False,
) -> dict[str, Any]:
    """
    Run a single episode with the baseline agent.

    Args:
        env: The environment
        agent: The agent
        task_id: Task to run
        render: Whether to print progress

    Returns:
        Episode results
    """
    obs = env.reset(task_id=task_id)
    obs_dict = obs.to_dict()

    total_reward = 0.0
    step = 0
    done = False

    if render:
        print(f"\n{'='*60}")
        print(f"Running {task_id}")
        print(f"{'='*60}")

    while not done:
        step += 1

        # Agent selects action
        action = agent.select_action(obs_dict)

        # Take step
        result = env.step(action)
        total_reward += result.reward
        done = result.done

        # Get new observation
        obs = env.get_observation()
        obs_dict = obs.to_dict()

        if render and step % 10 == 0:
            print(f"  Step {step:3d} | Reward: {result.reward:+.2f} | "
                  f"Total: {total_reward:+.2f} | "
                  f"Completed: {obs_dict['orders_completed']} | "
                  f"Failed: {obs_dict['orders_failed']}")

    # Calculate final score
    final_score = env.calculate_score()

    if render:
        print(f"\n  Episode Complete!")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Score: {final_score:.3f}")
        print(f"  Orders Completed: {obs_dict['orders_completed']}")
        print(f"  Orders Failed: {obs_dict['orders_failed']}")
        print(f"  Customer Satisfaction: {obs_dict['customer_satisfaction']:.2f}")

    return {
        "task_id": task_id,
        "total_steps": step,
        "total_reward": total_reward,
        "final_score": final_score,
        "orders_completed": obs_dict["orders_completed"],
        "orders_failed": obs_dict["orders_failed"],
        "customer_satisfaction": obs_dict["customer_satisfaction"],
        "total_revenue": obs_dict["total_revenue"],
        "agent_stats": agent.get_stats(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline agent on e-commerce fulfillment environment"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["task_easy", "task_medium", "task_hard", "all"],
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per task (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episode progress",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    tasks = ["task_easy", "task_medium", "task_hard"] if args.task == "all" else [args.task]

    all_results = []

    print("=" * 60)
    print("E-commerce Fulfillment Environment - Baseline Agent")
    print("=" * 60)

    for task_id in tasks:
        task_scores = []

        for episode in range(args.episodes):
            # Create new environment with seed for reproducibility
            env = FulfillmentEnv(seed=args.seed + episode)
            agent = BaselineAgent(verbose=False)

            result = run_episode(env, agent, task_id, render=args.render)
            task_scores.append(result["final_score"])
            all_results.append(result)

        avg_score = sum(task_scores) / len(task_scores)
        pass_threshold = TASK_CONFIGS[task_id].pass_threshold
        passed = avg_score >= pass_threshold

        print(f"\n{task_id} Summary:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Pass Threshold: {pass_threshold}")
        print(f"  Passed: {'✓' if passed else '✗'}")
        print(f"  Scores: {[f'{s:.3f}' for s in task_scores]}")

    # Overall summary
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)

    for task_id in tasks:
        task_results = [r for r in all_results if r["task_id"] == task_id]
        avg_score = sum(r["final_score"] for r in task_results) / len(task_results)
        passed = avg_score >= TASK_CONFIGS[task_id].pass_threshold
        print(f"  {task_id}: {avg_score:.3f} {'✓' if passed else '✗'}")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "results": all_results,
                "summary": {
                    task_id: {
                        "avg_score": sum(r["final_score"] for r in all_results if r["task_id"] == task_id)
                        / max(1, len([r for r in all_results if r["task_id"] == task_id])),
                        "passed": sum(r["final_score"] for r in all_results if r["task_id"] == task_id)
                        / max(1, len([r for r in all_results if r["task_id"] == task_id]))
                        >= TASK_CONFIGS[task_id].pass_threshold,
                    }
                    for task_id in tasks
                }
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return all_results


if __name__ == "__main__":
    main()