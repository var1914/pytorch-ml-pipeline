"""
Segmentation Trainer for CV Pipeline.

Specialized trainer for semantic segmentation tasks with:
- Dice Loss, Cross-Entropy, or combined losses
- IoU, Dice coefficient metrics
- Support for both binary and multi-class segmentation
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config.settings import TrainingConfig, InfraConfig
from .base_trainer import BaseTrainer


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Works with both binary and multi-class segmentation.
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            inputs: Predictions (B, C, H, W) or (B, H, W) for binary
            targets: Ground truth (B, H, W) with class indices

        Returns:
            Dice loss value
        """
        # Get number of classes
        if inputs.dim() == 4:
            num_classes = inputs.size(1)
            # Apply softmax for multi-class
            inputs = F.softmax(inputs, dim=1)
            # One-hot encode targets
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        else:
            # Binary segmentation
            inputs = torch.sigmoid(inputs)
            targets_one_hot = targets.unsqueeze(1).float()

        # Flatten spatial dimensions
        inputs_flat = inputs.view(inputs.size(0), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), -1)

        # Compute Dice
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'mean':
            return 1.0 - dice.mean()
        elif self.reduction == 'sum':
            return 1.0 - dice.sum()
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation (Dice + CrossEntropy).
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.dice_weight * dice + self.ce_weight * ce


class SegmentationTrainer(BaseTrainer):
    """
    Trainer for semantic segmentation tasks.

    Supports:
    - Binary segmentation (1 class)
    - Multi-class segmentation (N classes)
    - Various loss functions (Dice, CrossEntropy, Combined)

    Metrics computed:
    - IoU (Intersection over Union)
    - Dice coefficient
    - Pixel accuracy

    Example:
        trainer = SegmentationTrainer(
            model=unet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainingConfig(batch_size=8),
            num_classes=21
        )
        history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        infra_config: Optional[InfraConfig] = None,
        num_classes: int = 2,
        loss_type: str = "combined",
        ignore_index: int = -100,
        **kwargs
    ):
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.ignore_index = ignore_index

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            infra_config=infra_config,
            **kwargs
        )

    def _get_default_criterion(self) -> nn.Module:
        """Get default loss function for segmentation."""
        if self.loss_type == "dice":
            return DiceLoss()
        elif self.loss_type == "ce" or self.loss_type == "crossentropy":
            return nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif self.loss_type == "combined":
            return CombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute segmentation loss."""
        return self.criterion(outputs, targets)

    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics.

        Args:
            outputs: Model outputs (B, C, H, W) logits
            targets: Ground truth masks (B, H, W)

        Returns:
            Dictionary with IoU, Dice, and pixel accuracy
        """
        with torch.no_grad():
            # Get predictions
            if outputs.size(1) == 1:
                # Binary segmentation
                preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                # Multi-class segmentation
                preds = outputs.argmax(dim=1)

            # Compute metrics
            iou = self._compute_iou(preds, targets)
            dice = self._compute_dice(preds, targets)
            pixel_acc = self._compute_pixel_accuracy(preds, targets)

        return {
            "iou": iou,
            "dice": dice,
            "pixel_acc": pixel_acc
        }

    def _compute_iou(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-6
    ) -> float:
        """Compute mean IoU."""
        ious = []

        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)

            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()

            if union > 0:
                ious.append((intersection / (union + eps)).item())

        return sum(ious) / len(ious) if ious else 0.0

    def _compute_dice(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-6
    ) -> float:
        """Compute mean Dice coefficient."""
        dices = []

        for cls in range(self.num_classes):
            pred_mask = (preds == cls).float()
            target_mask = (targets == cls).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()

            if union > 0:
                dice = (2.0 * intersection / (union + eps)).item()
                dices.append(dice)

        return sum(dices) / len(dices) if dices else 0.0

    def _compute_pixel_accuracy(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute pixel-wise accuracy."""
        correct = (preds == targets).sum().float()
        total = targets.numel()
        return (correct / total).item()

    def _prepare_batch(
        self,
        batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare segmentation batch."""
        inputs, masks = batch[0], batch[1]
        inputs = inputs.to(self.device)

        # Ensure masks are long tensors for CrossEntropyLoss
        if masks.dtype != torch.long:
            masks = masks.long()
        masks = masks.to(self.device)

        return inputs, masks

    def get_predictions(
        self,
        loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions for a dataset.

        Args:
            loader: DataLoader to get predictions for

        Returns:
            Tuple of (predictions, ground_truth_masks)
        """
        self.model.eval()
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for batch in loader:
                inputs, masks = self._prepare_batch(batch)
                outputs = self.model(inputs)

                if outputs.size(1) == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
                else:
                    preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())

        return torch.cat(all_preds), torch.cat(all_masks)


class BinarySegmentationTrainer(SegmentationTrainer):
    """
    Trainer specifically for binary segmentation.

    Uses BCE loss and computes binary-specific metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        infra_config: Optional[InfraConfig] = None,
        pos_weight: Optional[float] = None,
        **kwargs
    ):
        self.pos_weight = pos_weight
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            infra_config=infra_config,
            num_classes=2,
            **kwargs
        )

    def _get_default_criterion(self) -> nn.Module:
        """Get BCE loss for binary segmentation."""
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight])

        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary segmentation loss."""
        # Squeeze channel dimension if present
        if outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        return self.criterion(outputs, targets.float())
