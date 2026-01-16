"""
Classification Trainer for CV Pipeline.

Specialized trainer for image classification tasks with:
- CrossEntropyLoss (default) or custom loss functions
- Accuracy, precision, recall, F1 metrics
- Support for multi-class and binary classification
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config.settings import TrainingConfig, InfraConfig
from .base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """
    Trainer for image classification tasks.

    Supports:
    - Multi-class classification with CrossEntropyLoss
    - Binary classification with BCEWithLogitsLoss
    - Custom loss functions

    Metrics computed:
    - Accuracy
    - Per-batch loss

    Example:
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainingConfig(batch_size=32, num_epochs=50)
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
        criterion: Optional[nn.Module] = None,
        num_classes: int = 2,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            infra_config=infra_config,
            criterion=criterion,
            **kwargs
        )

    def _get_default_criterion(self) -> nn.Module:
        """Get default loss function for classification."""
        if self.num_classes == 2:
            # Binary classification - can use either CrossEntropy or BCE
            return nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )
        else:
            # Multi-class classification
            return nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss."""
        return self.criterion(outputs, targets)

    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            outputs: Model outputs (logits) of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch,)

        Returns:
            Dictionary with accuracy metric
        """
        with torch.no_grad():
            # Get predictions
            _, predicted = torch.max(outputs, dim=1)

            # Compute accuracy
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
            accuracy = correct / total

        return {"accuracy": accuracy}

    def get_predictions(
        self,
        loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions for a dataset.

        Args:
            loader: DataLoader to get predictions for

        Returns:
            Tuple of (predictions, probabilities, labels)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._prepare_batch(batch)
                outputs = self.model(inputs)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        return (
            torch.cat(all_preds),
            torch.cat(all_probs),
            torch.cat(all_labels)
        )


class BinaryClassificationTrainer(ClassificationTrainer):
    """
    Trainer specifically for binary classification.

    Uses BCEWithLogitsLoss for better numerical stability.
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
        """Get BCE loss for binary classification."""
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight])

        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary classification loss."""
        # For BCE loss, we need to squeeze outputs and convert targets to float
        if outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        return self.criterion(outputs, targets.float())

    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute binary classification metrics."""
        with torch.no_grad():
            # Get predictions
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()

            # Compute accuracy
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
            accuracy = correct / total

        return {"accuracy": accuracy}
