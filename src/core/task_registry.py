"""
Task Registry for CV Pipeline.

Defines task types (classification, detection, segmentation) and their
associated configurations including output formats, loss functions, and metrics.
"""

from enum import Enum
from typing import Any, Dict, List


class TaskType(Enum):
    """Supported CV task types."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

    @classmethod
    def from_string(cls, value: str) -> "TaskType":
        """Convert string to TaskType enum."""
        value_lower = value.lower()
        for task in cls:
            if task.value == value_lower:
                return task
        raise ValueError(f"Unknown task type: {value}. Valid types: {[t.value for t in cls]}")


# Task-specific configurations
TASK_CONFIGS: Dict[TaskType, Dict[str, Any]] = {
    TaskType.CLASSIFICATION: {
        "output_format": "(batch, num_classes)",
        "default_loss": "CrossEntropyLoss",
        "supported_losses": [
            "CrossEntropyLoss",
            "FocalLoss",
            "LabelSmoothingLoss",
            "BCEWithLogitsLoss"
        ],
        "metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "confusion_matrix"
        ],
        "data_format": "(image, label)",
        "supported_architectures": [
            "resnet18", "resnet34", "resnet50", "resnet101",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
            "vit_tiny", "vit_small", "vit_base",
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
            "convnext_tiny", "convnext_small", "convnext_base"
        ],
        "default_architecture": "resnet50"
    },
    TaskType.DETECTION: {
        "output_format": "(batch, num_boxes, 6)",  # x, y, w, h, conf, cls
        "default_loss": "YOLOLoss",
        "supported_losses": [
            "YOLOLoss",
            "FasterRCNNLoss",
            "RetinaNetLoss"
        ],
        "metrics": [
            "mAP",
            "mAP50",
            "mAP75",
            "precision",
            "recall",
            "f1"
        ],
        "data_format": "(image, boxes, labels)",
        "supported_architectures": [
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2",
            "retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2",
            "ssd300_vgg16", "ssdlite320_mobilenet_v3_large"
        ],
        "default_architecture": "yolov8s"
    },
    TaskType.SEGMENTATION: {
        "output_format": "(batch, num_classes, H, W)",
        "default_loss": "DiceLoss",
        "supported_losses": [
            "DiceLoss",
            "CrossEntropyLoss",
            "FocalLoss",
            "JaccardLoss",
            "TverskyLoss",
            "CombinedLoss"
        ],
        "metrics": [
            "iou",
            "miou",
            "dice",
            "pixel_accuracy",
            "class_accuracy"
        ],
        "data_format": "(image, mask)",
        "supported_architectures": [
            "unet", "unet++",
            "deeplabv3_resnet50", "deeplabv3_resnet101",
            "deeplabv3plus_resnet50", "deeplabv3plus_resnet101",
            "fpn_resnet50",
            "pspnet_resnet50",
            "manet_resnet50"
        ],
        "default_architecture": "unet"
    }
}


def get_task_config(task_type: TaskType) -> Dict[str, Any]:
    """
    Get configuration for a specific task type.

    Args:
        task_type: TaskType enum value

    Returns:
        Dictionary containing task configuration

    Example:
        config = get_task_config(TaskType.CLASSIFICATION)
        print(config["default_loss"])  # "CrossEntropyLoss"
    """
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"Unknown task type: {task_type}")
    return TASK_CONFIGS[task_type]


def get_supported_architectures(task_type: TaskType) -> List[str]:
    """Get list of supported architectures for a task type."""
    return get_task_config(task_type)["supported_architectures"]


def get_default_architecture(task_type: TaskType) -> str:
    """Get default architecture for a task type."""
    return get_task_config(task_type)["default_architecture"]


def get_supported_metrics(task_type: TaskType) -> List[str]:
    """Get list of supported metrics for a task type."""
    return get_task_config(task_type)["metrics"]


def validate_architecture(task_type: TaskType, architecture: str) -> bool:
    """
    Validate that an architecture is supported for a task type.

    Args:
        task_type: The task type
        architecture: The architecture name

    Returns:
        True if architecture is supported

    Raises:
        ValueError if architecture is not supported
    """
    supported = get_supported_architectures(task_type)
    if architecture not in supported:
        raise ValueError(
            f"Architecture '{architecture}' not supported for {task_type.value}. "
            f"Supported architectures: {supported}"
        )
    return True
