"""
Main environment engine for the E-commerce Fulfillment Environment.

Implements the standard OpenEnv API: reset(), step(), state()
with full reward calculation and task-specific grading.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from env.models import (
    Alert,
    AlertType,
    Action,
    ActionType,
    CustomerInquiry,
    EnvironmentState,
    InquiryPriority,
    InquiryStatus,
    InquiryType,
    InventoryItem,
    Observation,
    Order,
    OrderItem,
    OrderStatus,
    ReturnRequest,
    ReturnStatus,
    StepResult,
    TASK_CONFIGS,
    TaskConfig,
)


class FulfillmentEnv:
    """
    E-commerce Fulfillment Environment.

    A real-world simulation where AI agents learn to manage order processing,
    inventory management, and customer service tasks in an e-commerce setting.

    The environment follows the standard OpenEnv API:
    - reset(task_id) -> Observation
    - step(action) -> StepResult
    - state() -> EnvironmentState
    """

    # Product catalog for generating inventory and orders
    PRODUCT_CATALOG: list[dict[str, Any]] = [
        {"product_id": "ELEC-001", "name": "Wireless Headphones", "category": "electronics", "price": 79.99, "cost": 40.00},
        {"product_id": "ELEC-002", "name": "Bluetooth Speaker", "category": "electronics", "price": 49.99, "cost": 25.00},
        {"product_id": "ELEC-003", "name": "USB-C Cable", "category": "electronics", "price": 12.99, "cost": 5.00},
        {"product_id": "ELEC-004", "name": "Phone Charger", "category": "electronics", "price": 24.99, "cost": 12.00},
        {"product_id": "ELEC-005", "name": "Power Bank", "category": "electronics", "price": 39.99, "cost": 20.00},
        {"product_id": "HOME-001", "name": "Coffee Maker", "category": "home", "price": 89.99, "cost": 45.00},
        {"product_id": "HOME-002", "name": "Desk Lamp", "category": "home", "price": 34.99, "cost": 17.00},
        {"product_id": "HOME-003", "name": "Throw Pillow", "category": "home", "price": 19.99, "cost": 8.00},
        {"product_id": "HOME-004", "name": "Kitchen Scale", "category": "home", "price": 29.99, "cost": 15.00},
        {"product_id": "HOME-005", "name": "Plant Pot Set", "category": "home", "price": 24.99, "cost": 10.00},
        {"product_id": "CLTH-001", "name": "T-Shirt", "category": "clothing", "price": 19.99, "cost": 8.00},
        {"product_id": "CLTH-002", "name": "Jeans", "category": "clothing", "price": 49.99, "cost": 22.00},
        {"product_id": "CLTH-003", "name": "Sneakers", "category": "clothing", "price": 79.99, "cost": 35.00},
        {"product_id": "CLTH-004", "name": "Jacket", "category": "clothing", "price": 89.99, "cost": 40.00},
        {"product_id": "CLTH-005", "name": "Socks Pack", "category": "clothing", "price": 14.99, "cost": 5.00},
        {"product_id": "BOOK-001", "name": "Novel Bestseller", "category": "books", "price": 15.99, "cost": 7.00},
        {"product_id": "BOOK-002", "name": "Cookbook", "category": "books", "price": 24.99, "cost": 10.00},
        {"product_id": "BOOK-003", "name": "Self-Help Guide", "category": "books", "price": 18.99, "cost": 8.00},
        {"product_id": "BOOK-004", "name": "Tech Manual", "category": "books", "price": 39.99, "cost": 18.00},
        {"product_id": "BOOK-005", "name": "Children's Book", "category": "books", "price": 12.99, "cost": 5.00},
    ]

    INQUIRY_TEMPLATES: list[dict[str, Any]] = [
        {"type": InquiryType.ORDER_STATUS, "priority": InquiryPriority.NORMAL},
        {"type": InquiryType.PRODUCT_QUESTION, "priority": InquiryPriority.LOW},
        {"type": InquiryType.SHIPPING_INQUIRY, "priority": InquiryPriority.NORMAL},
        {"type": InquiryType.COMPLAINT, "priority": InquiryPriority.HIGH},
        {"type": InquiryType.RETURN_REQUEST, "priority": InquiryPriority.HIGH},
        {"type": InquiryType.REFUND_REQUEST, "priority": InquiryPriority.URGENT},
    ]

    RETURN_REASONS: list[str] = [
        "Product damaged",
        "Wrong item received",
        "Not as described",
        "Changed mind",
        "Found better price",
        "Defective product",
    ]

    def __init__(self, seed: int | None = None):
        """Initialize the environment with optional random seed."""
        self.seed = seed
        self.rng = random.Random(seed)
        self._state: EnvironmentState | None = None
        self._task_config: TaskConfig | None = None
        self._restock_queue: dict[str, int] = {}  # product_id -> remaining steps
        self._order_queue: list[Order] = []  # Orders being prepared

    def reset(self, task_id: str = "task_easy") -> Observation:
        """
        Reset the environment to initial state for a specific task.

        Args:
            task_id: The task configuration to use (task_easy, task_medium, task_hard)

        Returns:
            Initial observation for the agent
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASK_CONFIGS.keys())}")

        # Reset random state for reproducibility
        if self.seed is not None:
            self.rng = random.Random(self.seed)

        self._task_config = TASK_CONFIGS[task_id]
        self._restock_queue = {}

        # Initialize state
        self._state = EnvironmentState(
            task_id=task_id,
            max_steps=self._task_config.max_steps,
            current_step=0,
        )

        # Generate initial inventory
        self._generate_inventory()

        # Generate initial orders
        self._generate_initial_orders()

        # Generate initial alerts if needed
        self._check_inventory_alerts()

        return self._get_observation()

    def step(self, action: Action | dict[str, Any]) -> StepResult:
        """
        Take a step in the environment.

        Args:
            action: An Action object or dict with action_type and optional target_id

        Returns:
            StepResult containing new state, reward, and done flags
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Convert dict to Action if needed
        if isinstance(action, dict):
            action_type = action.get("action_type")
            if isinstance(action_type, str):
                action_type = ActionType(action_type)
            action = Action(
                action_type=action_type,
                target_id=action.get("target_id"),
                parameters=action.get("parameters", {}),
            )

        # Execute action and calculate reward
        step_reward = 0.0
        action_info: dict[str, Any] = {"action": action.action_type.value}

        if action.action_type == ActionType.PROCESS_ORDER:
            reward, info = self._process_order(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.RESTOCK_ITEM:
            reward, info = self._restock_item(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.HANDLE_CUSTOMER:
            reward, info = self._handle_customer(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.PROCESS_RETURN:
            reward, info = self._process_return(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.EXPEDITE_ORDER:
            reward, info = self._expedite_order(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.QUALITY_CHECK:
            reward, info = self._quality_check(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.UPDATE_INVENTORY:
            reward, info = self._update_inventory(action)
            step_reward += reward
            action_info.update(info)

        elif action.action_type == ActionType.IDLE:
            step_reward -= 0.01  # Small penalty for idling
            action_info["reason"] = "idle"

        # Process restock queue
        self._process_restock_queue()

        # Generate new events
        self._generate_new_orders()
        self._generate_new_inquiries()
        self._generate_new_returns()

        # Check for alerts
        self._check_inventory_alerts()
        self._check_order_alerts()

        # Edge case: 1% chance for a random order cancellation by user
        if self._state.orders_pending and self.rng.random() < 0.01:
            order_to_cancel = self.rng.choice(self._state.orders_pending)
            self._state.orders_pending.remove(order_to_cancel)
            order_to_cancel.status = OrderStatus.CANCELLED
            self._state.orders_failed.append(order_to_cancel)

        # Operating costs
        self._state.total_operating_cost += 5.0  # Fixed facility cost
        if action.action_type != ActionType.IDLE:
            self._state.total_operating_cost += 2.0  # Variable labor cost

        # Advance time
        self._state.current_step += 1

        # Calculate time penalty
        step_reward -= 0.01

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self._state.current_step >= self._state.max_steps

        # Update customer satisfaction metric
        self._update_customer_satisfaction()

        # Build info dict
        info = {
            **action_info,
            "orders_completed": len(self._state.orders_completed),
            "orders_failed": len(self._state.orders_failed),
            "total_revenue": self._state.total_revenue,
            "customer_satisfaction": self._state.customer_satisfaction,
        }

        result = StepResult(
            state=self._state.copy(),
            reward=step_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        return result

    def state(self) -> EnvironmentState:
        """
        Get the current full environment state.

        Returns:
            Current EnvironmentState
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state.copy()

    def get_observation(self) -> Observation:
        """
        Get the current observation for the agent.

        Returns:
            Current Observation
        """
        return self._get_observation()

    def _get_observation(self) -> Observation:
        """Build observation from current state."""
        state = self._state

        # Inventory levels
        inventory_levels: dict[str, dict[str, Any]] = {}
        low_stock_items: list[str] = []
        out_of_stock_items: list[str] = []

        for product_id, item in state.inventory.items():
            inventory_levels[product_id] = {
                "stock": item.current_stock,
                "health": item.stock_health,
                "category": item.category,
            }
            if item.is_out_of_stock:
                out_of_stock_items.append(product_id)
            elif item.is_low_stock:
                low_stock_items.append(product_id)

        # Active alerts
        active_alerts = [
            {
                "type": alert.alert_type.value,
                "message": alert.message,
                "severity": alert.severity,
            }
            for alert in state.alerts
            if not alert.resolved
        ]

        # Count inquiries by priority
        urgent_inquiries = sum(
            1 for i in state.customer_inquiries
            if i.priority == InquiryPriority.URGENT and i.status == InquiryStatus.OPEN
        )

        # Returns counts
        returns_pending = sum(
            1 for r in state.return_requests
            if r.status == ReturnStatus.PENDING
        )
        returns_processing = sum(
            1 for r in state.return_requests
            if r.status in (ReturnStatus.APPROVED, ReturnStatus.PROCESSING)
        )

        return Observation(
            orders_pending_count=len(state.orders_pending),
            orders_in_progress_count=len(state.orders_in_progress),
            orders_completed_count=len(state.orders_completed),
            orders_failed_count=len(state.orders_failed),
            inventory_levels=inventory_levels,
            low_stock_items=low_stock_items,
            out_of_stock_items=out_of_stock_items,
            customer_inquiries_count=len(state.customer_inquiries),
            urgent_inquiries_count=urgent_inquiries,
            returns_pending_count=returns_pending,
            returns_processing_count=returns_processing,
            active_alerts=active_alerts,
            current_step=state.current_step,
            max_steps=state.max_steps,
            time_remaining=state.max_steps - state.current_step,
            total_revenue=state.total_revenue,
            total_operating_cost=state.total_operating_cost,
            operating_profit=state.operating_profit,
            customer_satisfaction=state.customer_satisfaction,
            task_id=state.task_id,
        )

    def _generate_inventory(self):
        """Generate initial inventory based on task config."""
        num_items = self._task_config.initial_inventory_size

        for i, product in enumerate(self.PRODUCT_CATALOG[:num_items]):
            max_stock = self.rng.randint(30, 60)
            self._state.inventory[product["product_id"]] = InventoryItem(
                product_id=product["product_id"],
                product_name=product["name"],
                category=product["category"],
                current_stock=self.rng.randint(15, max_stock),
                min_stock=5,
                max_stock=max_stock,
                reorder_point=self.rng.randint(10, 20),
                unit_cost=product["cost"],
                supplier_lead_time=self.rng.randint(2, 5),
            )

    def _generate_initial_orders(self):
        """Generate initial pending orders."""
        for i in range(self._task_config.initial_orders):
            order = self._generate_order()
            self._state.orders_pending.append(order)

    def _generate_order(self) -> Order:
        """Generate a single random order."""
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        customer_id = f"CUST-{uuid.uuid4().hex[:6].upper()}"

        # Random number of items (1-4)
        num_items = self.rng.randint(1, 4)
        items = []

        available_products = list(self._state.inventory.keys())
        if not available_products:
            # Fallback to catalog
            available_products = [p["product_id"] for p in self.PRODUCT_CATALOG]

        # Use Pareto distribution (80/20 rule) simulation for authentic data demand
        weights = [1.0 / ((i + 1) ** 1.5) for i in range(len(available_products))]
        chosen_products = self.rng.choices(
            available_products,
            weights=weights,
            k=num_items
        )
        selected_products = list(set(chosen_products))

        for product_id in selected_products:
            product = next(
                (p for p in self.PRODUCT_CATALOG if p["product_id"] == product_id),
                {"name": product_id, "price": 29.99}
            )
            items.append(OrderItem(
                product_id=product_id,
                product_name=product.get("name", product_id),
                quantity=self.rng.randint(1, 3),
                unit_price=product.get("price", 29.99),
            ))

        # Some orders are expedited
        is_expedited = self.rng.random() < 0.15
        current_step = self._state.current_step

        return Order(
            order_id=order_id,
            customer_id=customer_id,
            items=items,
            status=OrderStatus.PENDING,
            priority="expedited" if is_expedited else "normal",
            created_at=current_step,
            promised_by=current_step + (self.rng.randint(3, 7) if not is_expedited else self.rng.randint(1, 3)),
        )

    def _generate_new_orders(self):
        """Generate new orders based on arrival rate."""
        if self.rng.random() < self._task_config.order_arrival_rate:
            order = self._generate_order()
            self._state.orders_pending.append(order)

            if order.priority == "expedited":
                self._state.alerts.append(Alert(
                    alert_id=f"ALERT-{uuid.uuid4().hex[:6].upper()}",
                    alert_type=AlertType.EXPEDITED_ORDER,
                    message=f"Expedited order {order.order_id} received",
                    severity="warning",
                    created_at=self._state.current_step,
                    related_entity_id=order.order_id,
                ))

    def _generate_new_inquiries(self):
        """Generate new customer inquiries based on arrival rate."""
        if self._task_config.inquiry_arrival_rate <= 0:
            return

        if self.rng.random() < self._task_config.inquiry_arrival_rate:
            template = self.rng.choice(self.INQUIRY_TEMPLATES)

            # Try to link to an existing order
            order_id = None
            all_orders = (
                self._state.orders_pending
                + self._state.orders_in_progress
                + self._state.orders_completed
            )
            if all_orders:
                order_id = self.rng.choice(all_orders).order_id

            inquiry = CustomerInquiry(
                inquiry_id=f"INQ-{uuid.uuid4().hex[:6].upper()}",
                customer_id=f"CUST-{uuid.uuid4().hex[:6].upper()}",
                inquiry_type=template["type"],
                priority=template["priority"],
                order_id=order_id,
                created_at=self._state.current_step,
            )
            self._state.customer_inquiries.append(inquiry)

    def _generate_new_returns(self):
        """Generate new return requests based on return rate."""
        if self._task_config.return_rate <= 0:
            return

        # Only generate returns from completed orders
        if not self._state.orders_completed:
            return

        if self.rng.random() < self._task_config.return_rate:
            completed_order = self.rng.choice(self._state.orders_completed)
            if completed_order.items:
                item = self.rng.choice(completed_order.items)
                return_req = ReturnRequest(
                    return_id=f"RET-{uuid.uuid4().hex[:6].upper()}",
                    order_id=completed_order.order_id,
                    customer_id=completed_order.customer_id,
                    product_id=item.product_id,
                    quantity=min(item.quantity, 2),
                    reason=self.rng.choice(self.RETURN_REASONS),
                    created_at=self._state.current_step,
                )
                self._state.return_requests.append(return_req)

    def _check_inventory_alerts(self):
        """Check inventory levels and generate alerts."""
        for product_id, item in self._state.inventory.items():
            if item.is_out_of_stock:
                # Check if we don't already have an unresolved stockout alert
                has_alert = any(
                    a.alert_type == AlertType.STOCKOUT
                    and a.related_entity_id == product_id
                    and not a.resolved
                    for a in self._state.alerts
                )
                if not has_alert:
                    self._state.alerts.append(Alert(
                        alert_id=f"ALERT-{uuid.uuid4().hex[:6].upper()}",
                        alert_type=AlertType.STOCKOUT,
                        message=f"Out of stock: {item.product_name} ({product_id})",
                        severity="critical",
                        created_at=self._state.current_step,
                        related_entity_id=product_id,
                    ))
            elif item.is_low_stock:
                has_alert = any(
                    a.alert_type == AlertType.LOW_STOCK
                    and a.related_entity_id == product_id
                    and not a.resolved
                    for a in self._state.alerts
                )
                if not has_alert:
                    self._state.alerts.append(Alert(
                        alert_id=f"ALERT-{uuid.uuid4().hex[:6].upper()}",
                        alert_type=AlertType.LOW_STOCK,
                        message=f"Low stock: {item.product_name} ({item.current_stock} left)",
                        severity="warning",
                        created_at=self._state.current_step,
                        related_entity_id=product_id,
                    ))

    def _check_order_alerts(self):
        """Check for orders that are close to their promise date."""
        for order in self._state.orders_in_progress:
            steps_remaining = order.promised_by - self._state.current_step
            if steps_remaining <= 2 and order.status not in (OrderStatus.SHIPPED, OrderStatus.COMPLETED):
                has_alert = any(
                    a.alert_type == AlertType.SHIPPING_DELAY
                    and a.related_entity_id == order.order_id
                    and not a.resolved
                    for a in self._state.alerts
                )
                if not has_alert:
                    self._state.alerts.append(Alert(
                        alert_id=f"ALERT-{uuid.uuid4().hex[:6].upper()}",
                        alert_type=AlertType.SHIPPING_DELAY,
                        message=f"Order {order.order_id} due in {steps_remaining} steps",
                        severity="warning" if steps_remaining > 0 else "critical",
                        created_at=self._state.current_step,
                        related_entity_id=order.order_id,
                    ))

    def _process_order(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Process the next pending order."""
        reward = 0.0
        info: dict[str, Any] = {}

        if not self._state.orders_pending:
            info["result"] = "no_pending_orders"
            return reward - 0.1, info

        # Get the order to process (Optimized O(N) extraction)
        order = next((o for o in self._state.orders_pending if o.priority == "expedited"), self._state.orders_pending[0])

        # Check if we have inventory for all items
        can_fulfill = True
        missing_items = []

        for item in order.items:
            inv_item = self._state.inventory.get(item.product_id)
            if inv_item is None or inv_item.current_stock < item.quantity:
                can_fulfill = False
                missing_items.append(item.product_id)

        if not can_fulfill:
            # Move order to failed
            self._state.orders_pending.remove(order)
            order.status = OrderStatus.FAILED
            order.progress = 0.5  # Partial progress
            self._state.orders_failed.append(order)

            reward -= 0.5  # Failed order penalty
            info["result"] = "insufficient_inventory"
            info["missing_items"] = missing_items
            info["order_id"] = order.order_id

            # Stockout penalty
            reward -= 1.0
        else:
            # Deduct inventory
            for item in order.items:
                inv_item = self._state.inventory[item.product_id]
                inv_item.current_stock -= item.quantity

            # Move order through the pipeline
            self._state.orders_pending.remove(order)
            order.status = OrderStatus.COMPLETED
            order.progress = 1.0
            self._state.orders_completed.append(order)

            # Add revenue
            self._state.total_revenue += order.total_value

            reward += 1.0  # Completed order reward

            # Efficiency bonus for expedited orders
            if order.priority == "expedited":
                reward += 0.3
                # Resolve expedited alert
                for alert in self._state.alerts:
                    if (alert.alert_type == AlertType.EXPEDITED_ORDER
                            and alert.related_entity_id == order.order_id
                            and not alert.resolved):
                        alert.resolved = True
                        break

            info["result"] = "order_completed"
            info["order_id"] = order.order_id
            info["order_value"] = order.total_value

        return reward, info

    def _restock_item(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Queue a restock for an item."""
        reward = 0.0
        info: dict[str, Any] = {}

        target_id = action.target_id or action.parameters.get("product_id")

        if not target_id or target_id not in self._state.inventory:
            info["result"] = "invalid_product"
            return reward - 0.1, info

        item = self._state.inventory[target_id]

        if target_id in self._restock_queue:
            info["result"] = "already_restocking"
            return reward - 0.05, info

        if item.current_stock >= item.max_stock:
            info["result"] = "already_full"
            return reward - 0.05, info

        # Queue the restock
        self._restock_queue[target_id] = item.supplier_lead_time

        # Resolve low stock alert
        for alert in self._state.alerts:
            if (alert.related_entity_id == target_id
                    and alert.alert_type in (AlertType.LOW_STOCK, AlertType.STOCKOUT)
                    and not alert.resolved):
                alert.resolved = True
                break

        info["result"] = "restock_queued"
        info["product_id"] = target_id
        info["lead_time"] = item.supplier_lead_time

        return reward, info

    def _process_restock_queue(self):
        """Process the restock queue (advance by one step)."""
        completed = []
        for product_id, remaining in list(self._restock_queue.items()):
            # Edge case: 5% chance of supplier delay
            if self.rng.random() < 0.05:
                has_delay_alert = any(
                    a.alert_type == AlertType.SHIPPING_DELAY and a.related_entity_id == f"SUPPLIER-{product_id}" and not a.resolved
                    for a in self._state.alerts
                )
                if not has_delay_alert:
                    self._state.alerts.append(Alert(
                        alert_id=f"ALERT-{uuid.uuid4().hex[:6].upper()}",
                        alert_type=AlertType.SHIPPING_DELAY,
                        message=f"Supplier delay for {product_id}",
                        severity="warning",
                        created_at=self._state.current_step,
                        related_entity_id=f"SUPPLIER-{product_id}",
                    ))
            else:
                self._restock_queue[product_id] -= 1
                
            if self._restock_queue[product_id] <= 0:
                completed.append(product_id)

        for product_id in completed:
            item = self._state.inventory[product_id]
            item.current_stock = item.max_stock
            del self._restock_queue[product_id]
            # Resolve supplier delay alerts
            for alert in self._state.alerts:
                if alert.alert_type == AlertType.SHIPPING_DELAY and alert.related_entity_id == f"SUPPLIER-{product_id}":
                    alert.resolved = True

    def _handle_customer(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Handle a customer inquiry."""
        reward = 0.0
        info: dict[str, Any] = {}

        # Find open inquiries
        open_inquiries = [
            i for i in self._state.customer_inquiries
            if i.status == InquiryStatus.OPEN
        ]

        if not open_inquiries:
            info["result"] = "no_open_inquiries"
            return reward - 0.05, info

        # Get target inquiry
        target_id = action.target_id or action.parameters.get("inquiry_id")
        inquiry = None

        if target_id:
            inquiry = next((i for i in open_inquiries if i.inquiry_id == target_id), None)

        if inquiry is None:
            # Default to most urgent
            inquiry = max(open_inquiries, key=lambda i: (
                {"urgent": 4, "high": 3, "normal": 2, "low": 1}[i.priority.value],
                -i.created_at
            ))

        # Handle based on inquiry type
        response_quality = self.rng.random() * 0.5 + 0.5  # 0.5 to 1.0

        if inquiry.priority == InquiryPriority.URGENT:
            response_quality *= 0.8  # Harder to handle urgent inquiries well

        inquiry.response_quality = response_quality
        inquiry.status = InquiryStatus.RESOLVED
        inquiry.resolved = True

        if response_quality >= 0.7:
            reward += 0.5
            info["result"] = "customer_satisfied"
        else:
            reward -= 0.3
            inquiry.status = InquiryStatus.ESCALATED
            info["result"] = "customer_escalated"

        info["inquiry_id"] = inquiry.inquiry_id
        info["response_quality"] = response_quality

        return reward, info

    def _process_return(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Process a return request."""
        reward = 0.0
        info: dict[str, Any] = {}

        pending_returns = [
            r for r in self._state.return_requests
            if r.status == ReturnStatus.PENDING
        ]

        if not pending_returns:
            info["result"] = "no_pending_returns"
            return reward - 0.05, info

        target_id = action.target_id or action.parameters.get("return_id")
        return_req = None

        if target_id:
            return_req = next((r for r in pending_returns if r.return_id == target_id), None)

        if return_req is None:
            return_req = pending_returns[0]

        # Decide to approve or reject (simplified logic)
        approve = self.rng.random() < 0.7  # 70% approval rate

        if approve:
            return_req.status = ReturnStatus.APPROVED
            return_req.approved_at = self._state.current_step

            # Process refund
            refund_amount = return_req.order_value
            self._state.total_refunds += refund_amount
            self._state.total_revenue -= refund_amount

            # Restock the item
            if return_req.product_id in self._state.inventory:
                self._state.inventory[return_req.product_id].current_stock += return_req.quantity

            reward += 0.3
            info["result"] = "return_approved"
        else:
            return_req.status = ReturnStatus.REJECTED

            # Chance of dispute
            if self.rng.random() < 0.3:
                return_req.status = ReturnStatus.DISPUTED
                reward -= 0.4
                info["result"] = "return_disputed"
            else:
                reward -= 0.1
                info["result"] = "return_rejected"

        info["return_id"] = return_req.return_id
        return reward, info

    def _expedite_order(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Fast-track an order."""
        reward = 0.0
        info: dict[str, Any] = {}

        if not self._state.orders_pending:
            info["result"] = "no_pending_orders"
            return reward - 0.05, info

        # Move an order to the front of the queue
        target_id = action.target_id or action.parameters.get("order_id")
        order = None

        if target_id:
            order = next((o for o in self._state.orders_pending if o.order_id == target_id), None)

        if order is None:
            # Expedite the oldest order
            order = min(self._state.orders_pending, key=lambda o: o.created_at)

        order.priority = "expedited"
        order.promised_by = min(order.promised_by, self._state.current_step + 2)

        reward += 0.1  # Small bonus for proactive expediting
        info["result"] = "order_expedited"
        info["order_id"] = order.order_id

        return reward, info

    def _quality_check(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Perform quality check on orders (prepares for shipping)."""
        reward = 0.0
        info: dict[str, Any] = {}

        # In our simplified model, this validates orders in progress
        in_progress = [
            o for o in self._state.orders_in_progress
            if o.status == OrderStatus.PICKING
        ]

        if not in_progress:
            # Check pending orders to move to picking
            if self._state.orders_pending:
                order = self._state.orders_pending[0]
                self._state.orders_pending.remove(order)
                order.status = OrderStatus.PICKING
                order.progress = 0.3
                self._state.orders_in_progress.append(order)
                info["result"] = "started_picking"
                reward += 0.05
            else:
                info["result"] = "nothing_to_check"
                return reward - 0.05, info
        else:
            # Move picking orders to packed
            for order in in_progress:
                order.status = OrderStatus.PACKED
                order.progress = 0.6

            reward += 0.1
            info["result"] = "orders_packed"
            info["orders_affected"] = len(in_progress)

        return reward, info

    def _update_inventory(self, action: Action) -> tuple[float, dict[str, Any]]:
        """Update inventory records (audit adjustment)."""
        reward = 0.0
        info: dict[str, Any] = {}

        # Simulate inventory audit that finds discrepancies
        adjustments = 0
        for product_id, item in self._state.inventory.items():
            if self.rng.random() < 0.1:  # 10% chance of discrepancy per item
                adjustment = self.rng.randint(-2, 2)
                item.current_stock = max(0, item.current_stock + adjustment)
                adjustments += 1

        if adjustments > 0:
            reward += 0.1
            info["result"] = "inventory_updated"
            info["adjustments"] = adjustments
        else:
            info["result"] = "no_discrepancies"
            reward -= 0.05

        return reward, info

    def _check_termination(self) -> bool:
        """Check if the episode should terminate."""
        # Terminate if too many orders have failed
        total_orders = (
            len(self._state.orders_completed)
            + len(self._state.orders_failed)
        )
        if total_orders > 0:
            failure_rate = len(self._state.orders_failed) / total_orders
            if failure_rate > 0.5:
                return True

        # Terminate if inventory is completely depleted
        if self._state.inventory:
            all_empty = all(
                item.current_stock == 0
                for item in self._state.inventory.values()
            )
            if all_empty:
                return True

        return False

    def _update_customer_satisfaction(self):
        """Update the customer satisfaction metric."""
        # Based on completed orders, resolved inquiries, and processed returns
        scores = []

        # Order completion rate
        total_orders = len(self._state.orders_completed) + len(self._state.orders_failed)
        if total_orders > 0:
            scores.append(len(self._state.orders_completed) / total_orders)

        # Inquiry resolution
        inquiries = self._state.customer_inquiries
        resolved = [i for i in inquiries if i.resolved]
        if resolved:
            avg_quality = sum(i.response_quality or 0 for i in resolved) / len(resolved)
            scores.append(avg_quality)

        if scores:
            self._state.customer_satisfaction = sum(scores) / len(scores)

    def get_task_grader(self, task_id: str | None = None) -> "TaskGrader":
        """Get the grader for a specific task."""
        if task_id is None:
            task_id = self._state.task_id if self._state else "task_easy"
        return TaskGrader(task_id)

    def calculate_score(self) -> float:
        """Calculate the final score for the current episode."""
        grader = self.get_task_grader()
        return grader.grade(self._state)


class TaskGrader:
    """
    Task-specific grader that evaluates agent performance.

    Each task has its own grading criteria and thresholds.
    """

    def __init__(self, task_id: str):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self.config = TASK_CONFIGS[task_id]

    def grade(self, state: EnvironmentState) -> float:
        """
        Grade the episode and return a score from 0.0 to 1.0.

        Args:
            state: Final environment state

        Returns:
            Score between 0.0 and 1.0
        """
        if self.task_id == "task_easy":
            return self._grade_easy(state)
        elif self.task_id == "task_medium":
            return self._grade_medium(state)
        elif self.task_id == "task_hard":
            return self._grade_hard(state)
        return 0.0

    def _grade_easy(self, state: EnvironmentState) -> float:
        """
        Grade easy task: Simple Order Processing.

        Focus: Order completion rate and accuracy.
        """
        total_orders = len(state.orders_completed) + len(state.orders_failed)
        if total_orders == 0:
            return 0.0

        # Order completion rate (60% weight)
        completion_rate = len(state.orders_completed) / total_orders
        score = completion_rate * 0.6

        # Revenue efficiency (20% weight)
        max_possible_revenue = total_orders * 150  # Rough estimate
        revenue_efficiency = min(1.0, state.total_revenue / max_possible_revenue)
        score += revenue_efficiency * 0.2

        # Speed bonus (20% weight)
        if state.current_step > 0:
            speed = min(1.0, len(state.orders_completed) / state.current_step)
            score += speed * 0.2

        return min(1.0, max(0.0, score))

    def _grade_medium(self, state: EnvironmentState) -> float:
        """
        Grade medium task: Inventory Management.

        Focus: Order completion, inventory health, and customer service.
        """
        scores = []

        # Order completion rate (30% weight)
        total_orders = len(state.orders_completed) + len(state.orders_failed)
        if total_orders > 0:
            completion_rate = len(state.orders_completed) / total_orders
            scores.append(completion_rate * 0.3)
        else:
            scores.append(0.0)

        # Inventory health (30% weight)
        if state.inventory:
            avg_health = sum(
                item.stock_health for item in state.inventory.values()
            ) / len(state.inventory)
            scores.append(avg_health * 0.3)
        else:
            scores.append(0.0)

        # Stockout prevention (20% weight)
        stockouts = sum(
            1 for item in state.inventory.values()
            if item.is_out_of_stock
        )
        stockout_penalty = max(0, 1.0 - stockouts * 0.2)
        scores.append(stockout_penalty * 0.2)

        # Customer satisfaction (20% weight)
        scores.append(state.customer_satisfaction * 0.2)

        return min(1.0, max(0.0, sum(scores)))

    def _grade_hard(self, state: EnvironmentState) -> float:
        """
        Grade hard task: Full Operations Management.

        Focus: Balanced performance across all metrics.
        """
        scores = []

        # Order completion rate (25% weight)
        total_orders = len(state.orders_completed) + len(state.orders_failed)
        if total_orders > 0:
            completion_rate = len(state.orders_completed) / total_orders
            scores.append(completion_rate * 0.25)
        else:
            scores.append(0.0)

        # Inventory health (20% weight)
        if state.inventory:
            avg_health = sum(
                item.stock_health for item in state.inventory.values()
            ) / len(state.inventory)
            scores.append(avg_health * 0.2)
        else:
            scores.append(0.0)

        # Customer satisfaction (20% weight)
        scores.append(state.customer_satisfaction * 0.2)

        # Return management (15% weight)
        total_returns = len(state.return_requests)
        if total_returns > 0:
            processed_returns = sum(
                1 for r in state.return_requests
                if r.status in (ReturnStatus.REFUNDED, ReturnStatus.APPROVED)
            )
            return_rate = processed_returns / total_returns
            scores.append(return_rate * 0.15)
        else:
            scores.append(0.15)  # Bonus if no returns needed

        # Revenue efficiency (10% weight)
        net_revenue = state.total_revenue - state.total_refunds
        if total_orders > 0:
            max_possible = total_orders * 200
            revenue_score = min(1.0, max(0, net_revenue / max_possible))
            scores.append(revenue_score * 0.1)
        else:
            scores.append(0.0)

        # Operational efficiency (10% weight)
        if state.current_step > 0:
            efficiency = min(1.0, len(state.orders_completed) / (state.current_step * 0.5))
            scores.append(efficiency * 0.1)
        else:
            scores.append(0.0)

        return min(1.0, max(0.0, sum(scores)))