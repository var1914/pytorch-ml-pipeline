"""
Segmentation Models for CV Pipeline.

Supports various architectures through segmentation-models-pytorch:
- UNet, UNet++
- DeepLabV3, DeepLabV3+
- FPN (Feature Pyramid Network)
- PSPNet
- MAnet
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..core.base_model import CVModel
from ..core.task_registry import TaskType


class SegmentationModel(CVModel):
    """
    Semantic segmentation model supporting multiple architectures.

    Uses segmentation-models-pytorch (smp) library for architecture implementations.

    Args:
        architecture: Model architecture ('unet', 'unet++', 'deeplabv3', etc.)
        num_classes: Number of segmentation classes
        encoder: Backbone encoder (e.g., 'resnet50', 'efficientnet-b0')
        pretrained: Whether to use pretrained encoder weights
        in_channels: Number of input channels (default: 3 for RGB)
        activation: Output activation (None for logits, 'softmax', 'sigmoid')

    Example:
        model = SegmentationModel(
            architecture='unet',
            num_classes=21,
            encoder='resnet50',
            pretrained=True
        )
        masks = model(images)  # (batch, num_classes, H, W)
    """

    # Supported architectures
    ARCHITECTURES = {
        'unet': 'Unet',
        'unet++': 'UnetPlusPlus',
        'unetplusplus': 'UnetPlusPlus',
        'deeplabv3': 'DeepLabV3',
        'deeplabv3plus': 'DeepLabV3Plus',
        'deeplabv3+': 'DeepLabV3Plus',
        'fpn': 'FPN',
        'pspnet': 'PSPNet',
        'pan': 'PAN',
        'linknet': 'Linknet',
        'manet': 'MAnet',
    }

    # Supported encoders (common ones)
    ENCODERS = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'mobilenet_v2',
        'vgg16', 'vgg19',
    ]

    def __init__(
        self,
        architecture: str = 'unet',
        num_classes: int = 2,
        encoder: str = 'resnet50',
        pretrained: bool = True,
        in_channels: int = 3,
        activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        self._architecture = architecture.lower()
        self._num_classes = num_classes
        self._encoder = encoder
        self._pretrained = pretrained
        self._in_channels = in_channels
        self._activation = activation

        # Validate architecture
        if self._architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Supported: {list(self.ARCHITECTURES.keys())}"
            )

        self._build_model(**kwargs)

    def _build_model(self, **kwargs) -> None:
        """Build the segmentation model using smp."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch is required for segmentation models. "
                "Install with: pip install segmentation-models-pytorch"
            )

        # Get the model class from smp
        model_class_name = self.ARCHITECTURES[self._architecture]
        model_class = getattr(smp, model_class_name)

        # Encoder weights
        encoder_weights = 'imagenet' if self._pretrained else None

        # Create the model
        self.model = model_class(
            encoder_name=self._encoder,
            encoder_weights=encoder_weights,
            in_channels=self._in_channels,
            classes=self._num_classes,
            activation=self._activation,
            **kwargs
        )

    @property
    def task_type(self) -> TaskType:
        return TaskType.SEGMENTATION

    @property
    def output_spec(self) -> Dict[str, Any]:
        activation = self._activation or 'logits'
        return {
            "shape": f"(batch, {self._num_classes}, H, W)",
            "format": activation,
            "description": f"Segmentation masks with {activation} output"
        }

    def get_input_size(self) -> Tuple[int, int]:
        """Get recommended input size for segmentation."""
        # Segmentation models are typically flexible with input size
        # but prefer sizes divisible by 32
        return (256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation masks of shape (batch, num_classes, H, W)
        """
        return self.model(x)

    def freeze_encoder(self) -> None:
        """Freeze the encoder (backbone) layers."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the encoder layers."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def get_preprocessing_fn(self):
        """
        Get preprocessing function for this model's encoder.

        Returns a function that normalizes images according to
        the encoder's pretrained weights.
        """
        import segmentation_models_pytorch as smp
        return smp.encoders.get_preprocessing_fn(
            self._encoder,
            pretrained='imagenet' if self._pretrained else None
        )

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Run inference and return predicted class indices.

        Args:
            x: Input tensor
            threshold: Threshold for binary segmentation

        Returns:
            Class indices tensor of shape (batch, H, W)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            if self._num_classes == 1:
                # Binary segmentation
                return (torch.sigmoid(output) > threshold).long().squeeze(1)
            else:
                # Multi-class segmentation
                return output.argmax(dim=1)

    def get_encoder_features(self, x: torch.Tensor) -> list:
        """
        Extract encoder features at different scales.

        Useful for feature pyramid analysis or transfer learning.

        Args:
            x: Input tensor

        Returns:
            List of feature tensors at different scales
        """
        return self.model.encoder(x)


class BinarySegmentationModel(SegmentationModel):
    """
    Convenience class for binary segmentation tasks.

    Automatically sets num_classes=1 and uses sigmoid activation.
    """

    def __init__(
        self,
        architecture: str = 'unet',
        encoder: str = 'resnet50',
        pretrained: bool = True,
        in_channels: int = 3,
        **kwargs
    ):
        super().__init__(
            architecture=architecture,
            num_classes=1,
            encoder=encoder,
            pretrained=pretrained,
            in_channels=in_channels,
            activation='sigmoid',
            **kwargs
        )


class MultiClassSegmentationModel(SegmentationModel):
    """
    Convenience class for multi-class segmentation tasks.

    Automatically uses softmax activation.
    """

    def __init__(
        self,
        architecture: str = 'unet',
        num_classes: int = 21,
        encoder: str = 'resnet50',
        pretrained: bool = True,
        in_channels: int = 3,
        **kwargs
    ):
        super().__init__(
            architecture=architecture,
            num_classes=num_classes,
            encoder=encoder,
            pretrained=pretrained,
            in_channels=in_channels,
            activation='softmax2d',
            **kwargs
        )
