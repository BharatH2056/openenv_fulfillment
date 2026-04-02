---
title: E-commerce Fulfillment Environment
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# E-commerce Fulfillment Environment

A real-world OpenEnv environment where AI agents learn to manage order processing, inventory management, and customer service tasks in an e-commerce fulfillment center.

## 🌟 Overview

This environment simulates the operations of an e-commerce fulfillment center, where an AI agent must:

- **Process customer orders** efficiently while managing inventory levels
- **Restock items** before they run out to prevent stockouts
- **Handle customer inquiries** with varying priority levels
- **Process returns** and manage refunds
- **Balance competing objectives** to maximize overall performance

The environment provides a realistic testbed for reinforcement learning agents, with meaningful reward signals, partial progress tracking, and graduated difficulty levels.

## 🎯 Key Features

- **Real-world task**: Simulates actual e-commerce operations, not a toy problem
- **3 difficulty levels**: Easy → Medium → Hard with increasing complexity
- **Meaningful rewards**: Partial progress signals at each step
- **8 distinct actions**: Process orders, restock, handle customers, process returns, etc.
- **Rich observations**: Inventory levels, order queues, alerts, customer satisfaction
- **Reproducible**: Seeded random generation for consistent evaluation
- **Zero external dependencies**: Uses only Python standard library
- **OpenEnv compliant**: Full `reset()` / `step()` / `state()` API

## 📦 Installation

### Quick Start (No Dependencies)

```bash
# Clone or download the repository
cd openenv_fulfillment

# Run the baseline agent
python -m scripts.baseline_agent --task all --episodes 5 --render
```

### With Docker

```bash
# Build and run
docker build -t fulfillment-env .
docker run fulfillment-env
```

### Hugging Face Spaces

The environment is deployed as an interactive Gradio app:

```bash
# Install Gradio
pip install gradio

# Run the web interface
python deploy/gradio/app.py
```

## 🎮 Environment API

### Basic Usage

```python
from env.environment import FulfillmentEnv
from env.models import Action, ActionType

# Create environment with optional seed for reproducibility
env = FulfillmentEnv(seed=42)

# Reset for a specific task
observation = env.reset(task_id="task_easy")

# Run episode
done = False
while not done:
    # Get observation as dictionary
    obs_dict = observation.to_dict()

    # Your agent's decision logic here
    action = Action(action_type=ActionType.PROCESS_ORDER)

    # Take step and get result
    result = env.step(action)

    # Get new observation
    observation = env.get_observation()

    if result.done:
        break

# Calculate final score (0.0 to 1.0)
score = env.calculate_score()
print(f"Final Score: {score:.3f}")
```

### Observation Space

The observation is a dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `orders_pending` | int | Orders waiting to be processed |
| `orders_in_progress` | int | Orders being fulfilled |
| `orders_completed` | int | Successfully completed orders |
| `orders_failed` | int | Orders that couldn't be fulfilled |
| `inventory_levels` | dict | Stock levels per product |
| `low_stock_items` | list | Products below reorder point |
| `out_of_stock_items` | list | Products with zero stock |
| `customer_queue` | int | Pending customer inquiries |
| `urgent_inquiries` | int | High-priority inquiries |
| `returns_pending` | int | Return requests to process |
| `alerts` | list | Active system alerts |
| `current_step` | int | Current step number |
| `max_steps` | int | Maximum steps in episode |
| `time_remaining` | int | Steps remaining |
| `total_revenue` | float | Total revenue earned |
| `customer_satisfaction` | float | Satisfaction metric (0-1) |
| `task_id` | str | Current task identifier |

### Action Space

8 discrete actions available:

| ID | Action | Description |
|----|--------|-------------|
| 0 | `process_order` | Process the next pending order |
| 1 | `restock_item` | Queue restock for low-inventory item |
| 2 | `handle_customer` | Respond to a customer inquiry |
| 3 | `process_return` | Process a return request |
| 4 | `expedite_order` | Fast-track an urgent order |
| 5 | `quality_check` | Move orders through fulfillment pipeline |
| 6 | `update_inventory` | Perform inventory audit |
| 7 | `idle` | Skip this turn |

### Reward Structure

| Event | Reward |
|-------|--------|
| Order completed | +1.0 |
| Order failed (no inventory) | -0.5 |
| Stockout penalty | -1.0 |
| Customer satisfied | +0.5 |
| Customer escalated | -0.3 |
| Return processed | +0.3 |
| Return disputed | -0.4 |
| Expedited order bonus | +0.3 |
| Time penalty (per step) | -0.01 |
| Idle penalty | -0.01 |

## 📋 Tasks

### Task 1: Simple Order Processing (Easy)

Focus on fulfilling orders accurately and efficiently.

- **Max Steps**: 50
- **Pass Threshold**: 0.7
- **Features**: Order processing only, no inventory management needed
- **Grading**: Order completion rate (60%), revenue efficiency (20%), speed (20%)

### Task 2: Inventory Management (Medium)

Balance order fulfillment with inventory management.

- **Max Steps**: 100
- **Pass Threshold**: 0.6
- **Features**: Orders + inventory restocking + occasional customer inquiries
- **Grading**: Order completion (30%), inventory health (30%), stockout prevention (20%), customer satisfaction (20%)

### Task 3: Full Operations Management (Hard)

Handle all aspects of e-commerce operations.

- **Max Steps**: 200
- **Pass Threshold**: 0.5
- **Features**: Everything - orders, inventory, customers, returns
- **Grading**: Order completion (25%), inventory health (20%), customer satisfaction (20%), return management (15%), revenue efficiency (10%), operational efficiency (10%)

## 🏃 Running the Baseline Agent

### Heuristic Baseline

```bash
# Run all tasks
python -m scripts.baseline_agent --task all --episodes 5 --seed 42 --render

# Run specific task
python -m scripts.baseline_agent --task task_easy --episodes 10 --render

# Save results to JSON
python -m scripts.baseline_agent --task all --output results.json
```

### OpenAI API Baseline

For a more sophisticated baseline using LLM reasoning:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Run with GPT-4o-mini (default)
python -m scripts.openai_baseline --task all --episodes 3 --render

# Use a more powerful model
python -m scripts.openai_baseline --task all --model gpt-4o --render

# Save results
python -m scripts.openai_baseline --task all --output openai_results.json
```

### Expected Baseline Performance

| Task | Heuristic Score | OpenAI Score | Pass Threshold |
|------|-----------------|--------------|----------------|
| Easy | ~0.65-0.80 | ~0.70-0.85 | 0.70 |
| Medium | ~0.50-0.65 | ~0.55-0.70 | 0.60 |
| Hard | ~0.40-0.55 | ~0.45-0.60 | 0.50 |

## 📁 Project Structure

```
openenv_fulfillment/
├── env/
│   ├── __init__.py          # Package exports
│   ├── models.py            # Typed data models (orders, inventory, etc.)
│   └── environment.py       # Main environment engine
├── scripts/
│   ├── __init__.py
│   ├── baseline_agent.py    # Heuristic baseline agent
│   └── openai_baseline.py   # OpenAI API baseline agent
├── deploy/
│   └── gradio/
│       ├── app.py           # Gradio web interface
│       └── requirements.txt  # Gradio dependencies
├── tests/
│   └── test_environment.py  # Unit tests
├── openenv.yaml             # OpenEnv configuration
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/ -v

# Quick validation
python -c "from env.environment import FulfillmentEnv; env = FulfillmentEnv(); obs = env.reset(); print('OK')"
```

## 🚀 Deployment

### Docker

```bash
# Build
docker build -t ecommerce-fulfillment .

# Run baseline agent
docker run ecommerce-fulfillment

# Run with custom arguments
docker run ecommerce-fulfillment python -m scripts.baseline_agent --task task_hard --episodes 10
```

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Upload the contents of `deploy/gradio/`
4. The app will automatically deploy

### Local Gradio Server

```bash
pip install gradio
python deploy/gradio/app.py
```

## 📊 Evaluation

The environment provides comprehensive evaluation metrics:

```python
# Get final score
score = env.calculate_score()

# Get detailed state
state = env.state()
print(f"Orders completed: {len(state.orders_completed)}")
print(f"Orders failed: {len(state.orders_failed)}")
print(f"Customer satisfaction: {state.customer_satisfaction:.2f}")
print(f"Total revenue: ${state.total_revenue:.2f}")

# Get task-specific grader
grader = env.get_task_grader()
score = grader.grade(state)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This environment was built for the OpenEnv Hackathon, demonstrating how to create real-world simulation environments for AI agent training and evaluation.