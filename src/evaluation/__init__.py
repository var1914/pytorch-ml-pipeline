"""Evaluation module for CV Pipeline."""

from .base_evaluator import BaseEvaluator
from .classification_evaluator import ClassificationEvaluator
from .detection_evaluator import DetectionEvaluator
from .segmentation_evaluator import SegmentationEvaluator

# Backward compatibility
from .evaluation import ModelEvaluator

__all__ = [
    "BaseEvaluator",
    "ClassificationEvaluator",
    "DetectionEvaluator",
    "SegmentationEvaluator",
    "ModelEvaluator",  # Deprecated, use ClassificationEvaluator
]
