"""
Gradio Web Interface for E-commerce Fulfillment Environment.

This app provides an interactive interface to:
1. Explore the environment and its tasks
2. Run the baseline agent with configurable parameters
3. Visualize results and metrics
4. Test custom agent policies
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.models import TASK_CONFIGS, TaskDifficulty
from env.environment import FulfillmentEnv

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)


def run_single_episode(task_id: str, seed: int, render_steps: bool) -> dict[str, Any]:
    """Run a single episode and return results."""
    env = FulfillmentEnv(seed=seed)
    obs = env.reset(task_id=task_id)
    obs_dict = obs.to_dict()

    total_reward = 0.0
    step = 0
    done = False
    step_log = []

    # Simple baseline agent logic
    while not done:
        step += 1

        # Baseline action selection
        action = select_baseline_action(obs_dict)

        # Take step
        result = env.step(action)
        total_reward += result.reward
        done = result.done

        # Log step if rendering
        if render_steps and step % 5 == 0:
            obs = env.get_observation()
            obs_dict = obs.to_dict()
            step_log.append(
                f"Step {step:3d} | Reward: {result.reward:+.2f} | "
                f"Total: {total_reward:+.2f} | "
                f"Completed: {obs_dict['orders_completed']} | "
                f"Failed: {obs_dict['orders_failed']}"
            )

        # Get new observation
        obs = env.get_observation()
        obs_dict = obs.to_dict()

    # Calculate final score
    final_score = env.calculate_score()
    passed = final_score >= TASK_CONFIGS[task_id].pass_threshold

    return {
        "task_id": task_id,
        "total_steps": step,
        "total_reward": round(total_reward, 3),
        "final_score": round(final_score, 3),
        "passed": passed,
        "orders_completed": obs_dict["orders_completed"],
        "orders_failed": obs_dict["orders_failed"],
        "customer_satisfaction": round(obs_dict["customer_satisfaction"], 3),
        "total_revenue": round(obs_dict["total_revenue"], 2),
        "step_log": step_log,
    }


def select_baseline_action(obs: dict[str, Any]) -> str:
    """Simple baseline action selection."""
    from env.models import Action, ActionType, AlertType

    # 1. Critical alerts
    alerts = obs.get("alerts", [])
    for alert in alerts:
        if alert["severity"] == "critical" and alert["type"] == AlertType.STOCKOUT.value:
            out_of_stock = obs.get("out_of_stock_items", [])
            if out_of_stock:
                return ActionType.RESTOCK_ITEM.value

    # 2. Urgent inquiries
    if obs.get("urgent_inquiries", 0) > 0:
        return ActionType.HANDLE_CUSTOMER.value

    # 3. Returns
    if obs.get("returns_pending", 0) > 0:
        return ActionType.PROCESS_RETURN.value

    # 4. Low stock
    low_stock = obs.get("low_stock_items", [])
    if low_stock:
        inventory = obs.get("inventory_levels", {})
        worst = min(low_stock, key=lambda pid: inventory.get(pid, {}).get("health", 1.0))
        return ActionType.RESTOCK_ITEM.value

    # 5. Process orders
    if obs.get("orders_pending", 0) > 0:
        return ActionType.PROCESS_ORDER.value

    # 6. Quality check
    if obs.get("orders_in_progress", 0) > 0:
        return ActionType.QUALITY_CHECK.value

    # 7. Customer inquiries
    if obs.get("customer_queue", 0) > 0:
        return ActionType.HANDLE_CUSTOMER.value

    # 8. Idle
    return ActionType.IDLE.value


def run_episodes(task_id: str, num_episodes: int, seed: int, render: bool) -> str:
    """Run multiple episodes and format results."""
    results = []
    for i in range(num_episodes):
        result = run_single_episode(task_id, seed + i, render)
        results.append(result)

    avg_score = sum(r["final_score"] for r in results) / len(results)
    passed = avg_score >= TASK_CONFIGS[task_id].pass_threshold

    output = f"""
{'='*60}
E-commerce Fulfillment Environment - Results
{'='*60}

Task: {TASK_CONFIGS[task_id].name}
Difficulty: {TASK_CONFIGS[task_id].difficulty.value.upper()}
Episodes: {num_episodes}
Pass Threshold: {TASK_CONFIGS[task_id].pass_threshold}

--- Episode Results ---
"""

    for i, r in enumerate(results):
        output += f"""
Episode {i+1}:
  Score: {r['final_score']:.3f}
  Steps: {r['total_steps']}
  Reward: {r['total_reward']:.2f}
  Orders Completed: {r['orders_completed']}
  Orders Failed: {r['orders_failed']}
  Customer Satisfaction: {r['customer_satisfaction']:.2f}
"""

    output += f"""
--- Summary ---
Average Score: {avg_score:.3f}
Passed: {'✓ YES' if passed else '✗ NO'}
"""

    if render and results:
        output += "\n--- Step Log (sampled) ---\n"
        output += "\n".join(results[0]["step_log"][:20])

    return output


def get_task_info(task_id: str) -> str:
    """Get detailed information about a task."""
    config = TASK_CONFIGS[task_id]
    return f"""
**Task:** {config.name}

**Description:** {config.description}

**Difficulty:** {config.difficulty.value.upper()}

**Configuration:**
- Max Steps: {config.max_steps}
- Pass Threshold: {config.pass_threshold}
- Initial Orders: {config.initial_orders}
- Order Arrival Rate: {config.order_arrival_rate}
- Inventory Size: {config.initial_inventory_size}
- Inquiry Arrival Rate: {config.inquiry_arrival_rate}
- Return Rate: {config.return_rate}
"""


def get_env_info() -> str:
    """Get general environment information."""
    return """
## E-commerce Fulfillment Environment

A real-world simulation where AI agents learn to manage order processing,
inventory management, and customer service tasks in an e-commerce setting.

### Action Space (8 actions)

| ID | Action | Description |
|----|--------|-------------|
| 0 | process_order | Process the next pending order |
| 1 | restock_item | Restock a low-inventory item |
| 2 | handle_customer | Respond to a customer inquiry |
| 3 | process_return | Process a return request |
| 4 | expedite_order | Fast-track an urgent order |
| 5 | quality_check | Perform quality check on orders |
| 6 | update_inventory | Update inventory records |
| 7 | idle | Wait/skip this turn |

### Observation Space

- **orders_pending**: Number of orders waiting to be processed
- **orders_in_progress**: Orders currently being fulfilled
- **orders_completed**: Successfully completed orders
- **orders_failed**: Orders that could not be fulfilled
- **inventory_levels**: Stock levels for each product
- **low_stock_items**: Items below reorder point
- **out_of_stock_items**: Items with zero stock
- **customer_queue**: Pending customer inquiries
- **urgent_inquiries**: High-priority inquiries
- **returns_pending**: Return requests to process
- **alerts**: Active system alerts
- **time_remaining**: Steps left in episode

### Reward Structure

- Order completed: +1.0
- Order failed: -0.5
- Customer satisfied: +0.5
- Customer escalated: -0.3
- Return processed: +0.3
- Return disputed: -0.4
- Stockout: -1.0
- Time penalty: -0.01 per step

### Tasks

1. **Easy**: Simple Order Processing - Focus on fulfilling orders
2. **Medium**: Inventory Management - Balance orders and stock levels
3. **Hard**: Full Operations - Handle all aspects of e-commerce
"""


# Build Gradio interface
with gr.Blocks(title="E-commerce Fulfillment Env", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏭 E-commerce Fulfillment Environment")
    gr.Markdown("Train and evaluate AI agents on real-world e-commerce operations")

    with gr.Tab("Environment Info"):
        gr.Markdown(get_env_info())

    with gr.Tab("Task Explorer"):
        task_dropdown = gr.Dropdown(
            choices=list(TASK_CONFIGS.keys()),
            value="task_easy",
            label="Select Task",
        )
        task_info = gr.Markdown()
        task_dropdown.change(fn=get_task_info, inputs=task_dropdown, outputs=task_info)
        demo.load(fn=get_task_info, inputs=task_dropdown, outputs=task_info)

    with gr.Tab("Run Agent"):
        with gr.Row():
            with gr.Column():
                run_task = gr.Dropdown(
                    choices=list(TASK_CONFIGS.keys()),
                    value="task_easy",
                    label="Task",
                )
                run_episodes_input = gr.Slider(
                    minimum=1, maximum=20, value=3, step=1,
                    label="Number of Episodes",
                )
                run_seed = gr.Number(value=42, precision=0, label="Random Seed")
                render_toggle = gr.Checkbox(value=True, label="Show Step-by-Step Log")
                run_button = gr.Button("Run Baseline Agent", variant="primary")

            with gr.Column():
                run_output = gr.Textbox(
                    label="Results",
                    lines=30,
                    max_lines=50,
                )

        run_button.click(
            fn=run_episodes,
            inputs=[run_task, run_episodes_input, run_seed, render_toggle],
            outputs=run_output,
        )

    with gr.Tab("API Usage"):
        gr.Markdown("""
## Using the Environment Programmatically

```python
from env.environment import FulfillmentEnv
from env.models import Action, ActionType

# Create environment
env = FulfillmentEnv(seed=42)

# Reset for a specific task
obs = env.reset(task_id="task_easy")

# Run episode
done = False
while not done:
    # Your agent's action selection logic
    action = Action(action_type=ActionType.PROCESS_ORDER)

    # Take step
    result = env.step(action)

    # Get observation
    obs = env.get_observation()

    if result.done:
        break

# Calculate final score
score = env.calculate_score()
print(f"Final Score: {score:.3f}")
```

### Available Tasks

- `task_easy`: Simple order processing (50 steps)
- `task_medium`: Inventory management (100 steps)
- `task_hard`: Full operations (200 steps)

### Action Types

```python
from env.models import ActionType

# All available actions:
ActionType.PROCESS_ORDER      # Process pending order
ActionType.RESTOCK_ITEM       # Restock inventory
ActionType.HANDLE_CUSTOMER    # Handle inquiry
ActionType.PROCESS_RETURN     # Process return
ActionType.EXPEDITE_ORDER     # Expedite order
ActionType.QUALITY_CHECK      # Quality check
ActionType.UPDATE_INVENTORY   # Inventory audit
ActionType.IDLE               # Do nothing
```
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )