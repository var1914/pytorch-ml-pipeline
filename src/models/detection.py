"""
Detection Models for CV Pipeline.

Supports:
- YOLO family (YOLOv8) via ultralytics
- Faster R-CNN, RetinaNet, SSD via torchvision

Note: Detection models have different training/inference APIs.
YOLO uses ultralytics training, while torchvision models
use standard PyTorch training loops.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..core.base_model import CVModel
from ..core.task_registry import TaskType


class DetectionModel(CVModel):
    """
    Object detection model supporting multiple architectures.

    Supports:
    - YOLOv8 variants (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    - Faster R-CNN (fasterrcnn_resnet50_fpn)
    - RetinaNet (retinanet_resnet50_fpn)
    - SSD (ssd300_vgg16)

    Args:
        architecture: Model architecture name
        num_classes: Number of object classes (excluding background)
        pretrained: Whether to use pretrained weights
        conf_threshold: Confidence threshold for predictions (YOLO)
        iou_threshold: IoU threshold for NMS (YOLO)

    Example:
        model = DetectionModel(
            architecture='yolov8s',
            num_classes=80
        )
        detections = model(images)
    """

    YOLO_ARCHITECTURES = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    TORCHVISION_ARCHITECTURES = [
        'fasterrcnn_resnet50_fpn',
        'fasterrcnn_resnet50_fpn_v2',
        'retinanet_resnet50_fpn',
        'retinanet_resnet50_fpn_v2',
        'ssd300_vgg16',
        'ssdlite320_mobilenet_v3_large',
    ]

    def __init__(
        self,
        architecture: str = 'yolov8s',
        num_classes: int = 80,
        pretrained: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ):
        super().__init__()

        self._architecture = architecture
        self._num_classes = num_classes
        self._pretrained = pretrained
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._is_yolo = architecture in self.YOLO_ARCHITECTURES

        if self._is_yolo:
            self._build_yolo_model(architecture, num_classes, pretrained)
        else:
            self._build_torchvision_model(architecture, num_classes, pretrained, **kwargs)

    def _build_yolo_model(
        self,
        architecture: str,
        num_classes: int,
        pretrained: bool
    ) -> None:
        """Build YOLOv8 model using ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO models. "
                "Install with: pip install ultralytics"
            )

        # Load pretrained model or initialize new one
        if pretrained:
            self.model = YOLO(f'{architecture}.pt')
        else:
            # Load model architecture without weights
            self.model = YOLO(f'{architecture}.yaml')

        # Note: YOLO model training and num_classes adjustment
        # is handled during fine-tuning with model.train()

    def _build_torchvision_model(
        self,
        architecture: str,
        num_classes: int,
        pretrained: bool,
        **kwargs
    ) -> None:
        """Build torchvision detection model."""
        import torchvision
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            fasterrcnn_resnet50_fpn_v2,
            retinanet_resnet50_fpn,
            retinanet_resnet50_fpn_v2,
            ssd300_vgg16,
            ssdlite320_mobilenet_v3_large,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # Model constructor mapping
        model_constructors = {
            'fasterrcnn_resnet50_fpn': fasterrcnn_resnet50_fpn,
            'fasterrcnn_resnet50_fpn_v2': fasterrcnn_resnet50_fpn_v2,
            'retinanet_resnet50_fpn': retinanet_resnet50_fpn,
            'retinanet_resnet50_fpn_v2': retinanet_resnet50_fpn_v2,
            'ssd300_vgg16': ssd300_vgg16,
            'ssdlite320_mobilenet_v3_large': ssdlite320_mobilenet_v3_large,
        }

        if architecture not in model_constructors:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Load pretrained model
        weights = 'DEFAULT' if pretrained else None
        self.model = model_constructors[architecture](weights=weights)

        # Modify the classifier for custom number of classes
        if 'fasterrcnn' in architecture:
            # Get the number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes + 1  # +1 for background
            )

    @property
    def task_type(self) -> TaskType:
        return TaskType.DETECTION

    @property
    def output_spec(self) -> Dict[str, Any]:
        if self._is_yolo:
            return {
                "shape": "variable (num_detections, 6)",
                "format": "xyxy_conf_cls",
                "description": "YOLO output: [x1, y1, x2, y2, confidence, class]"
            }
        else:
            return {
                "shape": "List[Dict]",
                "format": "boxes_labels_scores",
                "description": "torchvision output: list of {boxes, labels, scores}"
            }

    def get_input_size(self) -> Tuple[int, int]:
        """Get recommended input size for detection."""
        if self._is_yolo:
            return (640, 640)  # YOLO default
        return (800, 800)  # Faster R-CNN typical size

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Union[List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Forward pass for detection.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            targets: Optional targets for training (torchvision models)

        Returns:
            For YOLO: Results object from ultralytics
            For torchvision: List of dicts with 'boxes', 'labels', 'scores'
        """
        if self._is_yolo:
            # YOLO inference
            results = self.model(x, conf=self._conf_threshold, iou=self._iou_threshold)
            return results
        else:
            # torchvision detection models
            if self.training and targets is not None:
                return self.model(x, targets)
            else:
                return self.model(x)

    def train_yolo(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        **kwargs
    ):
        """
        Train YOLO model using ultralytics training.

        Args:
            data_yaml: Path to data.yaml configuration file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            **kwargs: Additional ultralytics training arguments

        Returns:
            Training results
        """
        if not self._is_yolo:
            raise ValueError("train_yolo() only works with YOLO models")

        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            **kwargs
        )

    def predict(
        self,
        source: Union[str, torch.Tensor],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs
    ):
        """
        Run inference on images.

        Args:
            source: Image path, URL, tensor, or numpy array
            conf: Confidence threshold (overrides init value)
            iou: IoU threshold for NMS (overrides init value)

        Returns:
            Detection results
        """
        if self._is_yolo:
            return self.model.predict(
                source,
                conf=conf or self._conf_threshold,
                iou=iou or self._iou_threshold,
                **kwargs
            )
        else:
            # For torchvision models, use standard forward
            if isinstance(source, torch.Tensor):
                self.eval()
                with torch.no_grad():
                    return self.forward(source)
            else:
                raise ValueError(
                    "torchvision detection models require tensor input. "
                    "Load and preprocess image first."
                )

    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to different formats (YOLO only).

        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
        """
        if not self._is_yolo:
            raise ValueError("export() only works with YOLO models")
        return self.model.export(format=format, **kwargs)
