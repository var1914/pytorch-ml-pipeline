"""
Segmentation Evaluator for CV Pipeline.

Computes semantic segmentation metrics:
- IoU (Intersection over Union)
- mIoU (mean IoU)
- Dice coefficient
- Pixel accuracy
- Per-class metrics
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator


class SegmentationEvaluator(BaseEvaluator):
    """
    Evaluator for semantic segmentation tasks.

    Computes:
    - mIoU (mean Intersection over Union)
    - Per-class IoU
    - Dice coefficient
    - Pixel accuracy
    - Class-wise accuracy

    Example:
        evaluator = SegmentationEvaluator(
            model=unet_model,
            test_loader=test_loader,
            num_classes=21,
            class_names=['background', 'person', 'car', ...]
        )
        metrics = evaluator.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        num_classes: int = 2,
        ignore_index: int = -100,
    ):
        super().__init__(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            class_names=class_names
        )

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Auto-generate class names if not provided
        if self.class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]

        # Confusion matrix for computing metrics
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _process_outputs(
        self,
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process segmentation outputs."""
        if outputs.size(1) == 1:
            # Binary segmentation
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).squeeze(1).long()
        else:
            # Multi-class segmentation
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

        return preds, probs

    def _prepare_batch(
        self,
        batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare segmentation batch."""
        inputs, masks = batch[0], batch[1]
        inputs = inputs.to(self.device)

        if masks.dtype != torch.long:
            masks = masks.long()
        masks = masks.to(self.device)

        return inputs, masks

    def _collect_predictions(self) -> None:
        """Collect predictions and update confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, masks = self._prepare_batch(batch)
                outputs = self.model(inputs)

                preds, _ = self._process_outputs(outputs)

                # Update confusion matrix
                self._update_confusion_matrix(preds, masks)

    def _update_confusion_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """Update confusion matrix with batch predictions."""
        preds_np = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        # Filter out ignored indices
        mask = targets_np != self.ignore_index
        preds_np = preds_np[mask]
        targets_np = targets_np[mask]

        # Update confusion matrix
        for pred, target in zip(preds_np, targets_np):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1

    def evaluate(self) -> Dict[str, float]:
        """
        Run segmentation evaluation.

        Returns:
            Dictionary with mIoU, Dice, pixel accuracy, and per-class metrics
        """
        self.logger.info("Running segmentation evaluation...")

        # Collect predictions and build confusion matrix
        self._collect_predictions()

        # Compute metrics from confusion matrix
        self.metrics = {}

        # Per-class IoU
        iou_per_class = self._compute_iou_per_class()
        for i, iou in enumerate(iou_per_class):
            self.metrics[f'iou_{self.class_names[i]}'] = iou

        # Mean IoU
        valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
        self.metrics['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0

        # Per-class Dice
        dice_per_class = self._compute_dice_per_class()
        for i, dice in enumerate(dice_per_class):
            self.metrics[f'dice_{self.class_names[i]}'] = dice

        # Mean Dice
        valid_dices = [dice for dice in dice_per_class if not np.isnan(dice)]
        self.metrics['mean_dice'] = np.mean(valid_dices) if valid_dices else 0.0

        # Pixel accuracy
        self.metrics['pixel_accuracy'] = self._compute_pixel_accuracy()

        # Class-wise accuracy
        class_accuracy = self._compute_class_accuracy()
        for i, acc in enumerate(class_accuracy):
            self.metrics[f'accuracy_{self.class_names[i]}'] = acc

        # Mean class accuracy
        valid_accs = [acc for acc in class_accuracy if not np.isnan(acc)]
        self.metrics['mean_class_accuracy'] = np.mean(valid_accs) if valid_accs else 0.0

        # Frequency-weighted IoU
        self.metrics['fwIoU'] = self._compute_frequency_weighted_iou()

        self.logger.info("Evaluation complete")
        self.print_summary()

        return self.metrics

    def _compute_iou_per_class(self) -> List[float]:
        """Compute IoU for each class."""
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            if tp + fp + fn == 0:
                ious.append(np.nan)
            else:
                ious.append(tp / (tp + fp + fn))

        return ious

    def _compute_dice_per_class(self) -> List[float]:
        """Compute Dice coefficient for each class."""
        dices = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            if 2 * tp + fp + fn == 0:
                dices.append(np.nan)
            else:
                dices.append(2 * tp / (2 * tp + fp + fn))

        return dices

    def _compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()

        if total == 0:
            return 0.0
        return correct / total

    def _compute_class_accuracy(self) -> List[float]:
        """Compute per-class accuracy."""
        accuracies = []
        for i in range(self.num_classes):
            total = self.confusion_matrix[i, :].sum()
            if total == 0:
                accuracies.append(np.nan)
            else:
                accuracies.append(self.confusion_matrix[i, i] / total)

        return accuracies

    def _compute_frequency_weighted_iou(self) -> float:
        """Compute frequency-weighted IoU."""
        class_freqs = self.confusion_matrix.sum(axis=1)
        total = class_freqs.sum()

        if total == 0:
            return 0.0

        ious = self._compute_iou_per_class()
        fw_iou = 0.0

        for i, (freq, iou) in enumerate(zip(class_freqs, ious)):
            if not np.isnan(iou):
                fw_iou += (freq / total) * iou

        return fw_iou

    def generate_visualizations(
        self,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """Generate segmentation visualizations."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path("./evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)

        self.visualization_paths = []

        # 1. Confusion Matrix
        cm_path = output_dir / "segmentation_confusion_matrix.png"
        self._plot_confusion_matrix(cm_path)
        self.visualization_paths.append(str(cm_path))

        # 2. Per-class IoU bar chart
        iou_path = output_dir / "per_class_iou.png"
        self._plot_per_class_metrics(iou_path, 'IoU')
        self.visualization_paths.append(str(iou_path))

        # 3. Per-class Dice bar chart
        dice_path = output_dir / "per_class_dice.png"
        self._plot_per_class_metrics(dice_path, 'Dice')
        self.visualization_paths.append(str(dice_path))

        # 4. Save metrics report
        report_path = output_dir / "segmentation_report.txt"
        self._save_report(report_path)
        self.visualization_paths.append(str(report_path))

        self.logger.info(f"Generated {len(self.visualization_paths)} visualizations")
        return self.visualization_paths

    def _plot_confusion_matrix(self, save_path: Path) -> None:
        """Plot normalized confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Normalize confusion matrix
        cm_normalized = self.confusion_matrix.astype('float') / (
            self.confusion_matrix.sum(axis=1, keepdims=True) + 1e-6
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True if self.num_classes <= 10 else False,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_per_class_metrics(self, save_path: Path, metric_name: str) -> None:
        """Plot per-class metrics bar chart."""
        import matplotlib.pyplot as plt

        if metric_name == 'IoU':
            values = self._compute_iou_per_class()
        else:
            values = self._compute_dice_per_class()

        # Filter out NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        valid_names = [self.class_names[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(valid_values)), valid_values, color='steelblue')
        plt.xticks(range(len(valid_values)), valid_names, rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel(metric_name)
        plt.title(f'Per-Class {metric_name}')
        plt.axhline(y=np.mean(valid_values), color='r', linestyle='--', label=f'Mean: {np.mean(valid_values):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _save_report(self, save_path: Path) -> None:
        """Save detailed evaluation report."""
        with open(save_path, 'w') as f:
            f.write("Segmentation Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"mIoU: {self.metrics.get('mIoU', 0):.4f}\n")
            f.write(f"Mean Dice: {self.metrics.get('mean_dice', 0):.4f}\n")
            f.write(f"Pixel Accuracy: {self.metrics.get('pixel_accuracy', 0):.4f}\n")
            f.write(f"Mean Class Accuracy: {self.metrics.get('mean_class_accuracy', 0):.4f}\n")
            f.write(f"Frequency-Weighted IoU: {self.metrics.get('fwIoU', 0):.4f}\n\n")

            f.write("Per-Class Metrics:\n")
            f.write("-" * 30 + "\n")
            for i, name in enumerate(self.class_names):
                iou = self.metrics.get(f'iou_{name}', np.nan)
                dice = self.metrics.get(f'dice_{name}', np.nan)
                acc = self.metrics.get(f'accuracy_{name}', np.nan)
                f.write(f"{name}: IoU={iou:.4f}, Dice={dice:.4f}, Acc={acc:.4f}\n")
