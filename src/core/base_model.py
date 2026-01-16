"""
Base Model Abstract Class for CV Pipeline.

Provides a common interface for all CV models (classification, detection, segmentation).
All model implementations should inherit from CVModel.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .task_registry import TaskType


class CVModel(nn.Module, ABC):
    """
    Abstract base class for all computer vision models.

    Provides a unified interface for classification, detection, and segmentation models.
    All concrete model implementations must inherit from this class and implement
    the abstract methods.

    Example:
        class MyClassificationModel(CVModel):
            def __init__(self, num_classes: int):
                super().__init__()
                self.backbone = ...

            @property
            def task_type(self) -> TaskType:
                return TaskType.CLASSIFICATION

            @property
            def output_spec(self) -> Dict[str, Any]:
                return {"shape": "(batch, num_classes)", "format": "logits"}

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.backbone(x)
    """

    def __init__(self):
        super().__init__()
        self._architecture: Optional[str] = None
        self._num_classes: Optional[int] = None

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """
        Return the task type for this model.

        Returns:
            TaskType enum value (CLASSIFICATION, DETECTION, or SEGMENTATION)
        """
        pass

    @property
    @abstractmethod
    def output_spec(self) -> Dict[str, Any]:
        """
        Return the output specification for this model.

        Returns:
            Dictionary describing output format:
            - shape: String describing output tensor shape
            - format: String describing data format (logits, boxes, masks, etc.)
            - additional keys as needed for task-specific info

        Example for classification:
            {"shape": "(batch, num_classes)", "format": "logits"}

        Example for detection:
            {"shape": "(batch, num_boxes, 6)", "format": "xyxy_conf_cls"}

        Example for segmentation:
            {"shape": "(batch, num_classes, H, W)", "format": "logits"}
        """
        pass

    @property
    def architecture(self) -> Optional[str]:
        """Return the model architecture name."""
        return self._architecture

    @property
    def num_classes(self) -> Optional[int]:
        """Return the number of output classes."""
        return self._num_classes

    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_param_summary(self) -> Dict[str, int]:
        """
        Get a summary of model parameters.

        Returns:
            Dictionary with total, trainable, and frozen parameter counts
        """
        total = self.get_num_params(trainable_only=False)
        trainable = self.get_num_params(trainable_only=True)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers (keep only head trainable).

        Override in subclasses for task-specific freezing logic.
        """
        # Default implementation freezes all parameters except those
        # in layers named 'head', 'fc', 'classifier'
        for name, param in self.named_parameters():
            if not any(layer_name in name for layer_name in ['head', 'fc', 'classifier']):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_input_size(self) -> Tuple[int, int]:
        """
        Get the expected input image size.

        Returns:
            Tuple of (height, width)

        Override in subclasses that have specific input size requirements.
        """
        return (224, 224)  # Default ImageNet size

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information for logging.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "task_type": self.task_type.value,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "params": self.get_param_summary(),
            "output_spec": self.output_spec,
            "input_size": self.get_input_size()
        }

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            For classification: Tensor of shape (batch, num_classes)
            For detection: Dict with 'boxes', 'scores', 'labels' or stacked tensor
            For segmentation: Tensor of shape (batch, num_classes, H, W)
        """
        pass


class ModelMixin:
    """
    Mixin class providing common functionality for CV models.

    Can be used alongside CVModel for shared utilities.
    """

    @staticmethod
    def load_pretrained_weights(
        model: nn.Module,
        checkpoint_path: str,
        strict: bool = True,
        map_location: Optional[str] = None
    ) -> nn.Module:
        """
        Load pretrained weights into a model.

        Args:
            model: PyTorch model
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce state_dict keys match
            map_location: Device mapping for loading

        Returns:
            Model with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)
        return model

    @staticmethod
    def init_weights(model: nn.Module, method: str = "kaiming") -> None:
        """
        Initialize model weights.

        Args:
            model: PyTorch model
            method: Initialization method ('kaiming', 'xavier', 'normal')
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "normal":
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
