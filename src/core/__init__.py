"""Core abstractions for CV Pipeline."""

from .task_registry import TaskType, TASK_CONFIGS, get_task_config
from .base_model import CVModel
from .factories import ModelFactory, TrainerFactory, EvaluatorFactory

__all__ = [
    "TaskType",
    "TASK_CONFIGS",
    "get_task_config",
    "CVModel",
    "ModelFactory",
    "TrainerFactory",
    "EvaluatorFactory",
]
