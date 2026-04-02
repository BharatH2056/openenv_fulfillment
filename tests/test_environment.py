"""
Unit tests for the E-commerce Fulfillment Environment.

Tests cover:
- Environment initialization and reset
- Step execution and state transitions
- Task-specific grading
- Edge cases and error handling
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import FulfillmentEnv, TaskGrader
from env.models import (
    Action,
    ActionType,
    OrderStatus,
    TASK_CONFIGS,
    EnvironmentState,
)


class TestEnvironmentInitialization(unittest.TestCase):
    """Test environment initialization and reset."""

    def test_create_environment(self):
        """Test that environment can be created."""
        env = FulfillmentEnv()
        self.assertIsNotNone(env)

    def test_reset_with_seed(self):
        """Test reset with a specific seed produces consistent results."""
        env1 = FulfillmentEnv(seed=42)
        env2 = FulfillmentEnv(seed=42)

        obs1 = env1.reset(task_id="task_easy")
        obs2 = env2.reset(task_id="task_easy")

        # Same seed should produce same initial state
        self.assertEqual(obs1.orders_pending_count, obs2.orders_pending_count)
        self.assertEqual(obs1.task_id, obs2.task_id)

    def test_reset_all_tasks(self):
        """Test reset works for all task types."""
        env = FulfillmentEnv()
        for task_id in TASK_CONFIGS:
            obs = env.reset(task_id=task_id)
            self.assertEqual(obs.task_id, task_id)
            self.assertIsNotNone(obs)

    def test_reset_invalid_task(self):
        """Test that reset with invalid task raises error."""
        env = FulfillmentEnv()
        with self.assertRaises(ValueError):
            env.reset(task_id="invalid_task")


class TestStepExecution(unittest.TestCase):
    """Test step execution and state transitions."""

    def test_step_process_order(self):
        """Test processing an order."""
        env = FulfillmentEnv(seed=42)
        obs = env.reset(task_id="task_easy")

        initial_pending = obs.orders_pending_count
        action = Action(action_type=ActionType.PROCESS_ORDER)
        result = env.step(action)

        # Reward should be returned
        self.assertIsInstance(result.reward, float)

        # State should be updated
        self.assertEqual(result.state.current_step, 1)

    def test_step_without_reset(self):
        """Test that step without reset raises error."""
        env = FulfillmentEnv()
        with self.assertRaises(RuntimeError):
            env.step(Action(action_type=ActionType.IDLE))

    def test_step_all_actions(self):
        """Test that all action types can be executed."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_hard")  # Use hard task for all features

        actions_to_test = [
            ActionType.PROCESS_ORDER,
            ActionType.RESTOCK_ITEM,
            ActionType.HANDLE_CUSTOMER,
            ActionType.PROCESS_RETURN,
            ActionType.EXPEDITE_ORDER,
            ActionType.QUALITY_CHECK,
            ActionType.UPDATE_INVENTORY,
            ActionType.IDLE,
        ]

        for action_type in actions_to_test:
            action = Action(action_type=action_type)
            result = env.step(action)
            self.assertFalse(result.done)

    def test_step_with_dict_action(self):
        """Test that actions can be passed as dictionaries."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        action = {"action_type": "process_order"}
        result = env.step(action)
        self.assertIsInstance(result.reward, float)

    def test_episode_termination(self):
        """Test that episodes terminate properly."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        done = False
        steps = 0
        max_steps = 1000  # Safety limit

        while not done and steps < max_steps:
            action = Action(action_type=ActionType.PROCESS_ORDER)
            result = env.step(action)
            done = result.done
            steps += 1

        # Episode should eventually end
        self.assertTrue(done or steps >= TASK_CONFIGS["task_easy"].max_steps)


class TestObservationSpace(unittest.TestCase):
    """Test observation space structure."""

    def test_observation_keys(self):
        """Test that observation contains all required keys."""
        env = FulfillmentEnv(seed=42)
        obs = env.reset(task_id="task_easy")
        obs_dict = obs.to_dict()

        required_keys = [
            "orders_pending",
            "orders_in_progress",
            "orders_completed",
            "orders_failed",
            "inventory_levels",
            "low_stock_items",
            "out_of_stock_items",
            "customer_queue",
            "urgent_inquiries",
            "returns_pending",
            "returns_processing",
            "alerts",
            "current_step",
            "max_steps",
            "time_remaining",
            "total_revenue",
            "customer_satisfaction",
            "task_id",
        ]

        for key in required_keys:
            self.assertIn(key, obs_dict, f"Missing key: {key}")

    def test_observation_types(self):
        """Test that observation values have correct types."""
        env = FulfillmentEnv(seed=42)
        obs = env.reset(task_id="task_easy")
        obs_dict = obs.to_dict()

        self.assertIsInstance(obs_dict["orders_pending"], int)
        self.assertIsInstance(obs_dict["inventory_levels"], dict)
        self.assertIsInstance(obs_dict["low_stock_items"], list)
        self.assertIsInstance(obs_dict["total_revenue"], float)
        self.assertIsInstance(obs_dict["customer_satisfaction"], float)


class TestTaskGrading(unittest.TestCase):
    """Test task-specific grading."""

    def test_grader_creates_for_all_tasks(self):
        """Test that graders can be created for all tasks."""
        for task_id in TASK_CONFIGS:
            grader = TaskGrader(task_id)
            self.assertIsNotNone(grader)

    def test_grading_empty_state(self):
        """Test grading with minimal state."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        # Immediately grade without doing anything
        state = env.state()
        score = env.calculate_score()

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_grading_after_episode(self):
        """Test grading after running an episode."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        # Run some steps
        for _ in range(10):
            action = Action(action_type=ActionType.PROCESS_ORDER)
            result = env.step(action)
            if result.done:
                break

        score = env.calculate_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_easy_grading_criteria(self):
        """Test that easy task grading focuses on order completion."""
        grader = TaskGrader("task_easy")

        # Create a state with good completion rate
        state = EnvironmentState()
        state.orders_completed = [1, 2, 3, 4, 5]  # 5 completed
        state.orders_failed = []  # 0 failed
        state.total_revenue = 500.0
        state.current_step = 10

        score = grader.grade(state)
        self.assertGreater(score, 0.5)  # Should be passing

    def test_medium_grading_criteria(self):
        """Test that medium task grading considers inventory health."""
        grader = TaskGrader("task_medium")

        state = EnvironmentState()
        state.orders_completed = [1, 2, 3]
        state.orders_failed = [4]
        state.total_revenue = 300.0
        state.current_step = 20

        # Add inventory with good health
        from env.models import InventoryItem
        state.inventory = {
            "PROD-001": InventoryItem(
                product_id="PROD-001",
                product_name="Test Product",
                category="test",
                current_stock=40,
                min_stock=5,
                max_stock=50,
                reorder_point=15,
                unit_cost=10.0,
            )
        }

        score = grader.grade(state)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestRewardStructure(unittest.TestCase):
    """Test reward calculations."""

    def test_order_completion_reward(self):
        """Test that completing orders gives positive reward."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        # Process orders until one completes
        total_reward = 0.0
        for _ in range(20):
            action = Action(action_type=ActionType.PROCESS_ORDER)
            result = env.step(action)
            total_reward += result.reward

            if result.info.get("result") == "order_completed":
                self.assertGreater(result.reward, 0)
                break

    def test_idle_penalty(self):
        """Test that idling gives small negative reward."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        action = Action(action_type=ActionType.IDLE)
        result = env.step(action)

        # Idle should give negative reward
        self.assertLess(result.reward, 0)


class TestReproducibility(unittest.TestCase):
    """Test that environment is reproducible with seeds."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        def run_episode(seed):
            env = FulfillmentEnv(seed=seed)
            env.reset(task_id="task_easy")
            rewards = []
            for _ in range(10):
                action = Action(action_type=ActionType.PROCESS_ORDER)
                result = env.step(action)
                rewards.append(result.reward)
            return rewards

        rewards1 = run_episode(123)
        rewards2 = run_episode(123)

        self.assertEqual(rewards1, rewards2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        def run_episode(seed):
            env = FulfillmentEnv(seed=seed)
            env.reset(task_id="task_easy")
            rewards = []
            for _ in range(10):
                action = Action(action_type=ActionType.PROCESS_ORDER)
                result = env.step(action)
                rewards.append(result.reward)
            return rewards

        rewards1 = run_episode(111)
        rewards2 = run_episode(222)

        # Results should differ (with high probability)
        self.assertNotEqual(rewards1, rewards2)


class TestStateAPI(unittest.TestCase):
    """Test the state() API."""

    def test_state_returns_copy(self):
        """Test that state() returns a copy, not the original."""
        env = FulfillmentEnv(seed=42)
        env.reset(task_id="task_easy")

        state1 = env.state()
        state1.current_step = 999
        
        # Test deep copy / state isolation
        if state1.orders_pending:
            state1.orders_pending[0].status = OrderStatus.CANCELLED

        state2 = env.state()
        self.assertNotEqual(state1.current_step, state2.current_step)
        if state1.orders_pending and state2.orders_pending:
            self.assertNotEqual(state1.orders_pending[0].status, state2.orders_pending[0].status)

    def test_state_before_reset(self):
        """Test that state() before reset raises error."""
        env = FulfillmentEnv()
        with self.assertRaises(RuntimeError):
            env.state()


if __name__ == "__main__":
    unittest.main()