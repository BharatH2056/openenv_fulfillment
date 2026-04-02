"""
Typed Pydantic models for the E-commerce Fulfillment Environment.

This module defines all the data structures used in the environment,
including orders, inventory, customer inquiries, returns, and the
complete environment state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderStatus(str, Enum):
    """Status of an order in the fulfillment pipeline."""
    PENDING = "pending"
    PICKING = "picking"
    PACKED = "packed"
    QUALITY_CHECK = "quality_check"
    SHIPPED = "shipped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InquiryType(str, Enum):
    """Type of customer inquiry."""
    ORDER_STATUS = "order_status"
    PRODUCT_QUESTION = "product_question"
    SHIPPING_INQUIRY = "shipping_inquiry"
    COMPLAINT = "complaint"
    RETURN_REQUEST = "return_request"
    REFUND_REQUEST = "refund_request"


class InquiryPriority(str, Enum):
    """Priority level for customer inquiries."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class InquiryStatus(str, Enum):
    """Status of a customer inquiry."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class ReturnStatus(str, Enum):
    """Status of a return request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSING = "processing"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class AlertType(str, Enum):
    """Type of system alert."""
    LOW_STOCK = "low_stock"
    STOCKOUT = "stockout"
    EXPEDITED_ORDER = "expedited_order"
    QUALITY_ISSUE = "quality_issue"
    SHIPPING_DELAY = "shipping_delay"
    RETURN_SPIKE = "return_spike"


class ActionType(str, Enum):
    """Available actions for the agent."""
    PROCESS_ORDER = "process_order"
    RESTOCK_ITEM = "restock_item"
    HANDLE_CUSTOMER = "handle_customer"
    PROCESS_RETURN = "process_return"
    EXPEDITE_ORDER = "expedite_order"
    QUALITY_CHECK = "quality_check"
    UPDATE_INVENTORY = "update_inventory"
    IDLE = "idle"


class TaskDifficulty(str, Enum):
    """Difficulty level of a task."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class OrderItem:
    """An item within an order."""
    product_id: str
    product_name: str
    quantity: int
    unit_price: float

    def total_price(self) -> float:
        return self.quantity * self.unit_price


@dataclass
class Order:
    """Represents a customer order."""
    order_id: str
    customer_id: str
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    priority: str = "normal"  # "normal", "expedited"
    created_at: int = 0
    promised_by: int = 0
    progress: float = 0.0  # 0.0 to 1.0 completion
    quality_passed: bool | None = None

    @property
    def total_value(self) -> float:
        return sum(item.total_price() for item in self.items)

    @property
    def total_items(self) -> int:
        return sum(item.quantity for item in self.items)


@dataclass
class InventoryItem:
    """Represents an item in inventory."""
    product_id: str
    product_name: str
    category: str
    current_stock: int
    min_stock: int
    max_stock: int
    reorder_point: int
    unit_cost: float
    supplier_lead_time: int = 3  # steps to restock

    @property
    def is_low_stock(self) -> bool:
        return self.current_stock <= self.reorder_point

    @property
    def is_out_of_stock(self) -> bool:
        return self.current_stock <= 0

    @property
    def stock_health(self) -> float:
        """Returns a health metric from 0.0 (empty) to 1.0 (full)."""
        if self.max_stock == 0:
            return 0.0
        return min(1.0, self.current_stock / self.max_stock)


@dataclass
class CustomerInquiry:
    """Represents a customer service inquiry."""
    inquiry_id: str
    customer_id: str
    inquiry_type: InquiryType
    priority: InquiryPriority
    status: InquiryStatus = InquiryStatus.OPEN
    order_id: str | None = None
    created_at: int = 0
    response_quality: float | None = None  # 0.0 to 1.0
    resolved: bool = False

    @property
    def age(self) -> int:
        # Will be calculated relative to current step
        return 0


@dataclass
class ReturnRequest:
    """Represents a product return request."""
    return_id: str
    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    reason: str
    status: ReturnStatus = ReturnStatus.PENDING
    created_at: int = 0
    approved_at: int | None = None
    refunded_at: int | None = None

    @property
    def order_value(self) -> float:
        return self.quantity * 50.0  # Simplified


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    alert_type: AlertType
    message: str
    severity: str = "warning"  # "info", "warning", "critical"
    created_at: int = 0
    resolved: bool = False
    related_entity_id: str | None = None


@dataclass
class EnvironmentState:
    """Complete state of the environment."""
    # Orders
    orders_pending: list[Order] = field(default_factory=list)
    orders_in_progress: list[Order] = field(default_factory=list)
    orders_completed: list[Order] = field(default_factory=list)
    orders_failed: list[Order] = field(default_factory=list)

    # Inventory
    inventory: dict[str, InventoryItem] = field(default_factory=dict)

    # Customer service
    customer_inquiries: list[CustomerInquiry] = field(default_factory=list)

    # Returns
    return_requests: list[ReturnRequest] = field(default_factory=list)

    # Alerts
    alerts: list[Alert] = field(default_factory=list)

    # Metrics
    current_step: int = 0
    total_revenue: float = 0.0
    total_refunds: float = 0.0
    total_operating_cost: float = 0.0
    customer_satisfaction: float = 0.0

    @property
    def operating_profit(self) -> float:
        return self.total_revenue - self.total_refunds - self.total_operating_cost


    # Task info
    task_id: str = ""
    max_steps: int = 100

    def copy(self) -> EnvironmentState:
        """Create a deep copy of the state to prevent leakage between episodes."""
        import copy
        return copy.deepcopy(self)


@dataclass
class Action:
    """Represents an agent's action."""
    action_type: ActionType
    target_id: str | None = None  # Order ID, product ID, inquiry ID, etc.
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of taking a step in the environment."""
    state: EnvironmentState
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class Observation:
    """Observable state for the agent (simplified view of EnvironmentState)."""
    orders_pending_count: int
    orders_in_progress_count: int
    orders_completed_count: int
    orders_failed_count: int

    inventory_levels: dict[str, dict[str, Any]]
    low_stock_items: list[str]
    out_of_stock_items: list[str]

    customer_inquiries_count: int
    urgent_inquiries_count: int

    returns_pending_count: int
    returns_processing_count: int

    active_alerts: list[dict[str, Any]]

    current_step: int
    max_steps: int
    time_remaining: int

    total_revenue: float
    total_operating_cost: float
    operating_profit: float
    customer_satisfaction: float

    task_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to a dictionary for the agent."""
        return {
            "orders_pending": self.orders_pending_count,
            "orders_in_progress": self.orders_in_progress_count,
            "orders_completed": self.orders_completed_count,
            "orders_failed": self.orders_failed_count,
            "inventory_levels": {
                k: {"stock": v["stock"], "health": v["health"]}
                for k, v in self.inventory_levels.items()
            },
            "low_stock_items": self.low_stock_items,
            "out_of_stock_items": self.out_of_stock_items,
            "customer_queue": self.customer_inquiries_count,
            "urgent_inquiries": self.urgent_inquiries_count,
            "returns_pending": self.returns_pending_count,
            "returns_processing": self.returns_processing_count,
            "alerts": self.active_alerts,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "time_remaining": self.time_remaining,
            "total_revenue": self.total_revenue,
            "total_operating_cost": self.total_operating_cost,
            "operating_profit": self.operating_profit,
            "customer_satisfaction": self.customer_satisfaction,
            "task_id": self.task_id,
        }


@dataclass
class TaskConfig:
    """Configuration for a specific task."""
    task_id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    max_steps: int
    pass_threshold: float

    # Task-specific parameters
    initial_orders: int = 5
    order_arrival_rate: float = 0.3  # Probability per step
    initial_inventory_size: int = 10
    inquiry_arrival_rate: float = 0.1
    return_rate: float = 0.05


# Task configurations
TASK_CONFIGS: dict[str, TaskConfig] = {
    "task_easy": TaskConfig(
        task_id="task_easy",
        name="Simple Order Processing",
        description="Process incoming orders by picking items from inventory.",
        difficulty=TaskDifficulty.EASY,
        max_steps=50,
        pass_threshold=0.7,
        initial_orders=5,
        order_arrival_rate=0.2,
        initial_inventory_size=8,
        inquiry_arrival_rate=0.0,
        return_rate=0.0,
    ),
    "task_medium": TaskConfig(
        task_id="task_medium",
        name="Inventory Management",
        description="Manage inventory while processing orders.",
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=100,
        pass_threshold=0.6,
        initial_orders=8,
        order_arrival_rate=0.3,
        initial_inventory_size=12,
        inquiry_arrival_rate=0.05,
        return_rate=0.03,
    ),
    "task_hard": TaskConfig(
        task_id="task_hard",
        name="Full Operations Management",
        description="Handle complete e-commerce operations.",
        difficulty=TaskDifficulty.HARD,
        max_steps=200,
        pass_threshold=0.5,
        initial_orders=15,
        order_arrival_rate=0.4,
        initial_inventory_size=20,
        inquiry_arrival_rate=0.15,
        return_rate=0.08,
    ),
}