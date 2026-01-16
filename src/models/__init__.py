"""Model implementations for CV Pipeline."""

from .classification import ClassificationModel
from .detection import DetectionModel
from .segmentation import SegmentationModel
from .model import ResNetClassifier, ImageClassifier

__all__ = [
    # Primary models (recommended)
    "ClassificationModel",
    "DetectionModel",
    "SegmentationModel",
    # Simple classifier for quick prototyping
    "ResNetClassifier",
    "ImageClassifier",
]
