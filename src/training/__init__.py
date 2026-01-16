"""Training module for CV Pipeline."""

from .base_trainer import BaseTrainer
from .classification_trainer import ClassificationTrainer
from .detection_trainer import DetectionTrainer
from .segmentation_trainer import SegmentationTrainer

# Backward compatibility
from .training import ModelTrainer

__all__ = [
    "BaseTrainer",
    "ClassificationTrainer",
    "DetectionTrainer",
    "SegmentationTrainer",
    "ModelTrainer",  # Deprecated, use ClassificationTrainer
]
