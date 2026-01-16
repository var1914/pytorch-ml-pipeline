"""
Classification Models for CV Pipeline.

Supports various architectures through timm library:
- ResNet family (resnet18, resnet34, resnet50, resnet101)
- EfficientNet family (efficientnet_b0 through efficientnet_b7)
- Vision Transformers (vit_tiny, vit_small, vit_base)
- MobileNet family (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)
- ConvNeXt family (convnext_tiny, convnext_small, convnext_base)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..core.base_model import CVModel
from ..core.task_registry import TaskType


class ClassificationModel(CVModel):
    """
    Classification model supporting multiple architectures.

    Uses timm (PyTorch Image Models) library for architecture implementations.

    Args:
        architecture: Model architecture name (e.g., 'resnet50', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate before final classifier (0.0 to disable)
        freeze_backbone: Whether to freeze backbone layers

    Example:
        model = ClassificationModel(
            architecture='resnet50',
            num_classes=10,
            pretrained=True,
            dropout=0.2
        )
        output = model(images)  # (batch, num_classes)
    """

    # Architecture mapping for common naming variations
    ARCHITECTURE_ALIASES = {
        'resnet18': 'resnet18',
        'resnet34': 'resnet34',
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        'resnet152': 'resnet152',
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'efficientnet_b2': 'efficientnet_b2',
        'efficientnet_b3': 'efficientnet_b3',
        'efficientnet_b4': 'efficientnet_b4',
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'vit_base': 'vit_base_patch16_224',
        'vit_base_patch16_224': 'vit_base_patch16_224',
        'mobilenet_v2': 'mobilenetv2_100',
        'mobilenet_v3_small': 'mobilenetv3_small_100',
        'mobilenet_v3_large': 'mobilenetv3_large_100',
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'convnext_base': 'convnext_base',
    }

    # Default input sizes for different architectures
    INPUT_SIZES = {
        'efficientnet_b0': (224, 224),
        'efficientnet_b1': (240, 240),
        'efficientnet_b2': (260, 260),
        'efficientnet_b3': (300, 300),
        'efficientnet_b4': (380, 380),
        'vit': (224, 224),
        'default': (224, 224),
    }

    def __init__(
        self,
        architecture: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self._architecture = architecture
        self._num_classes = num_classes
        self._pretrained = pretrained
        self._dropout_rate = dropout

        # Resolve architecture alias
        timm_arch = self.ARCHITECTURE_ALIASES.get(architecture, architecture)

        # Create model using timm
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for classification models. "
                "Install with: pip install timm"
            )

        # Create the model
        self.model = timm.create_model(
            timm_arch,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    @property
    def output_spec(self) -> Dict[str, Any]:
        return {
            "shape": f"(batch, {self._num_classes})",
            "format": "logits",
            "activation": "softmax"
        }

    def get_input_size(self) -> Tuple[int, int]:
        """Get recommended input size for this architecture."""
        for key, size in self.INPUT_SIZES.items():
            if key in self._architecture.lower():
                return size
        return self.INPUT_SIZES['default']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classifier head."""
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head (architecture-specific)
        classifier_names = ['fc', 'classifier', 'head', 'head.fc']
        for name, param in self.model.named_parameters():
            if any(clf_name in name for clf_name in classifier_names):
                param.requires_grad = True

    def get_feature_extractor(self) -> nn.Module:
        """
        Get the backbone feature extractor (without classifier).

        Useful for transfer learning or feature extraction.
        """
        import timm
        return timm.create_model(
            self.ARCHITECTURE_ALIASES.get(self._architecture, self._architecture),
            pretrained=self._pretrained,
            num_classes=0,  # Remove classifier
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification head.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        return self.model.forward_features(x)


