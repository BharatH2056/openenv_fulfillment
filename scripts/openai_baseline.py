"""
OpenAI Baseline Agent for E-commerce Fulfillment Environment.

This script uses the OpenAI API to run a language model as an agent
against the fulfillment environment. It reads API credentials from
the OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY=your-key-here
    python -m scripts.openai_baseline --task all --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from env.models import (
    Action,
    ActionType,
    TASK_CONFIGS,
)
from env.environment import FulfillmentEnv

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")
    print("This script requires OpenAI API access.")
    sys.exit(1)


# System prompt for the AI agent
SYSTEM_PROMPT = """You are an AI agent managing an e-commerce fulfillment center.
Your goal is to efficiently process orders, manage inventory, handle customer inquiries,
and process returns to maximize your score.

You will receive an observation of the current state and must choose one of the following actions:

AVAILABLE ACTIONS:
0. process_order - Process the next pending order (fulfill from inventory)
1. restock_item - Queue a restock for a low-inventory item (specify product_id)
2. handle_customer - Respond to a customer inquiry
3. process_return - Process a return request
4. expedite_order - Fast-track an urgent order
5. quality_check - Perform quality check on orders in progress
6. update_inventory - Update inventory records (audit)
7. idle - Skip this turn (only if nothing else to do)

PRIORITIES:
- Process pending orders first (main revenue source)
- Handle urgent customer inquiries
- Restock items that are low on stock (prevent stockouts)
- Process returns to free up capacity
- Avoid idling when there's work to do

RESPONSE FORMAT:
Respond with a JSON object containing:
{
    "action": <action_id>,
    "target_id": "<optional target product/order/inquiry id>",
    "reasoning": "<brief explanation of your choice>"
}

Make your decision based on the current observation state."""


class OpenAIAgent:
    """An agent that uses OpenAI's API to select actions."""

    def __init__(self, model: str = "gpt-4o-mini", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.action_log: list[dict[str, Any]] = []
        self._conversation_history: list[dict[str, str]] = []

    def reset(self):
        """Reset the conversation history for a new episode."""
        self._conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def select_action(self, observation: dict[str, Any]) -> Action:
        """
        Select an action using the OpenAI API.

        Args:
            observation: Current observation from the environment

        Returns:
            Action to take
        """
        # Format observation as a prompt
        prompt = self._format_observation(observation)

        # Add user message
        self._conversation_history.append({"role": "user", "content": prompt})

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._conversation_history,
                temperature=0.1,  # Low temperature for consistent decisions
                max_tokens=200,
            )

            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self._conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )

            # Parse the response
            action = self._parse_response(assistant_message, observation)

            if self.verbose:
                print(f"  AI chose: {action.action_type.value}"
                      + (f" -> {action.target_id}" if action.target_id else ""))
                if assistant_message:
                    try:
                        parsed = json.loads(assistant_message)
                        print(f"    Reasoning: {parsed.get('reasoning', 'N/A')}")
                    except json.JSONDecodeError:
                        print(f"    Raw response: {assistant_message[:100]}...")

            self.action_log.append({
                "step": observation["current_step"],
                "action": action.action_type.value,
                "target": action.target_id,
            })

            return action

        except Exception as e:
            if self.verbose:
                print(f"  API Error: {e}, falling back to heuristic")
            # Fall back to heuristic on API error
            return self._heuristic_fallback(observation)

    def _format_observation(self, obs: dict[str, Any]) -> str:
        """Format observation into a readable prompt."""
        return f"""Current State (Step {obs['current_step']}/{obs['max_steps']}):

ORDERS:
  - Pending: {obs['orders_pending']}
  - In Progress: {obs['orders_in_progress']}
  - Completed: {obs['orders_completed']}
  - Failed: {obs['orders_failed']}

INVENTORY:
  - Total products: {len(obs['inventory_levels'])}
  - Low stock items: {len(obs['low_stock_items'])} {obs['low_stock_items'][:3] if obs['low_stock_items'] else ''}
  - Out of stock: {len(obs['out_of_stock_items'])} {obs['out_of_stock_items'][:3] if obs['out_of_stock_items'] else ''}

CUSTOMER SERVICE:
  - Inquiries in queue: {obs['customer_queue']}
  - Urgent inquiries: {obs['urgent_inquiries']}

RETURNS:
  - Pending returns: {obs['returns_pending']}
  - Processing returns: {obs['returns_processing']}

ALERTS: {len(obs['alerts'])} active
  {chr(10).join(f'  - [{a["severity"]}] {a["message"]}' for a in obs['alerts'][:5]) if obs['alerts'] else '  None'}

METRICS:
  - Revenue: ${obs['total_revenue']:.2f}
  - Customer satisfaction: {obs['customer_satisfaction']:.2f}
  - Time remaining: {obs['time_remaining']} steps

What action should you take next? Respond with JSON."""

    def _parse_response(self, response: str, observation: dict[str, Any]) -> Action:
        """Parse the AI's response into an Action."""
        try:
            # Try to extract JSON from the response
            if "{" in response and "}" in response:
                # Extract JSON object
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                parsed = json.loads(json_str)

                action_id = parsed.get("action", 7)
                target_id = parsed.get("target_id")

                # Validate action_id
                if isinstance(action_id, int) and 0 <= action_id <= 7:
                    return Action(ActionType(action_id), target_id=target_id)
                elif isinstance(action_id, str):
                    return Action(ActionType(action_id), target_id=target_id)

        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: try to extract action from text
        if response:
            for action_type in ActionType:
                if action_type.value in response.lower():
                    return Action(action_type)

        # Default fallback
        return self._heuristic_fallback(observation)

    def _heuristic_fallback(self, obs: dict[str, Any]) -> Action:
        """Simple heuristic fallback when AI fails."""
        if obs.get("orders_pending", 0) > 0:
            return Action(ActionType.PROCESS_ORDER)
        if obs.get("customer_queue", 0) > 0:
            return Action(ActionType.HANDLE_CUSTOMER)
        if obs.get("returns_pending", 0) > 0:
            return Action(ActionType.PROCESS_RETURN)
        low_stock = obs.get("low_stock_items", [])
        if low_stock:
            return Action(ActionType.RESTOCK_ITEM, target_id=low_stock[0])
        return Action(ActionType.IDLE)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the agent's actions."""
        action_counts: dict[str, int] = {}
        for entry in self.action_log:
            action = entry["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        return {
            "total_actions": len(self.action_log),
            "action_distribution": action_counts,
            "model": self.model,
        }


def run_episode(
    env: FulfillmentEnv,
    agent: OpenAIAgent,
    task_id: str,
    render: bool = False,
) -> dict[str, Any]:
    """Run a single episode with the OpenAI agent."""
    agent.reset()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.to_dict()

    total_reward = 0.0
    step = 0
    done = False

    if render:
        print(f"\n{'='*60}")
        print(f"Running {task_id} with OpenAI Agent ({agent.model})")
        print(f"{'='*60}")

    while not done:
        step += 1

        # Agent selects action via OpenAI API
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
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run OpenAI agent on e-commerce fulfillment environment"
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
        default=3,
        help="Number of episodes per task (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o1-mini"],
        help="OpenAI model to use (default: gpt-4o-mini)",
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
    print(f"E-commerce Fulfillment Environment - OpenAI Agent")
    print(f"Model: {args.model}")
    print("=" * 60)

    for task_id in tasks:
        task_scores = []

        for episode in range(args.episodes):
            env = FulfillmentEnv(seed=args.seed + episode)
            agent = OpenAIAgent(model=args.model, verbose=args.render)

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
                "model": args.model,
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