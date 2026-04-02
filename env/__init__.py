"""
E-commerce Fulfillment Environment Package

This package implements a real-world e-commerce order fulfillment environment
where AI agents learn to manage order processing, inventory management, and
customer service tasks.
"""

from env.models import (
    Order,
    InventoryItem,
    CustomerInquiry,
    ReturnRequest,
    EnvironmentState,
    Action,
    StepResult,
)
from env.environment import FulfillmentEnv

__all__ = [
    "Order",
    "InventoryItem",
    "CustomerInquiry",
    "ReturnRequest",
    "EnvironmentState",
    "Action",
    "StepResult",
    "FulfillmentEnv",
]

__version__ = "1.0.0"