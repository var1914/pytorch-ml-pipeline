"""
Simple ResNet Classifier for quick prototyping.

For production use, prefer the full-featured ClassificationModel:
    from src.models import ClassificationModel
    model = ClassificationModel(architecture='resnet50', num_classes=10)
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """
    ResNet50-based classifier for image classification tasks.

    A simple wrapper around torchvision's ResNet50 with a customizable
    classification head. Suitable for transfer learning and quick prototyping.

    For more architectures (EfficientNet, ViT, etc.), use ClassificationModel.

    Args:
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze ResNet50 layers for fine-tuning

    Example:
        model = ResNetClassifier(num_classes=10, pretrained=True)
        output = model(images)  # (batch, 10)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes

        # Use weights parameter (new API)
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Replace final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Optionally freeze backbone for fine-tuning
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all layers except the final classification layer."""
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the final layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.resnet(x)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Get the number of model parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Alias for convenience
ImageClassifier = ResNetClassifier
