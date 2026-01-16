"""
Classification Evaluator for CV Pipeline.

Computes classification-specific metrics and visualizations:
- Accuracy, Precision, Recall, F1
- ROC curves and AUC
- Confusion matrix
- Classification report
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for image classification tasks.

    Computes:
    - Accuracy, Precision, Recall, F1 (weighted and macro)
    - AUC-ROC (for binary classification)
    - Confusion matrix
    - Per-class metrics

    Generates:
    - Confusion matrix heatmap
    - ROC curve (binary classification)
    - Precision-Recall curve

    Example:
        evaluator = ClassificationEvaluator(
            model=model,
            test_loader=test_loader,
            class_names=['benign', 'malignant']
        )
        metrics = evaluator.evaluate()
        evaluator.generate_visualizations(output_dir='./results')
        evaluator.log_to_mlflow()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        num_classes: int = 2,
    ):
        super().__init__(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            class_names=class_names
        )
        self.num_classes = num_classes

        # Auto-generate class names if not provided
        if self.class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]

    def _process_outputs(
        self,
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process classification outputs."""
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)
        return preds, probs

    def evaluate(self) -> Dict[str, float]:
        """
        Run classification evaluation.

        Returns:
            Dictionary with accuracy, precision, recall, f1, and optionally AUC
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            classification_report
        )

        self.logger.info("Running classification evaluation...")

        # Collect predictions
        self._collect_predictions()

        preds_np = self.predictions.numpy()
        labels_np = self.ground_truth.numpy()

        # Basic metrics
        self.metrics = {
            'accuracy': accuracy_score(labels_np, preds_np),
            'precision_weighted': precision_score(labels_np, preds_np, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels_np, preds_np, average='weighted', zero_division=0),
            'f1_weighted': f1_score(labels_np, preds_np, average='weighted', zero_division=0),
            'precision_macro': precision_score(labels_np, preds_np, average='macro', zero_division=0),
            'recall_macro': recall_score(labels_np, preds_np, average='macro', zero_division=0),
            'f1_macro': f1_score(labels_np, preds_np, average='macro', zero_division=0),
        }

        # AUC-ROC for binary classification
        if self.num_classes == 2 and self.probabilities is not None:
            try:
                probs_np = self.probabilities[:, 1].numpy()
                self.metrics['auc_roc'] = roc_auc_score(labels_np, probs_np)
            except Exception as e:
                self.logger.warning(f"Could not compute AUC-ROC: {e}")

        # Store classification report
        self.classification_report = classification_report(
            labels_np, preds_np,
            target_names=self.class_names,
            zero_division=0
        )

        self.logger.info("Evaluation complete")
        self.print_summary()

        return self.metrics

    def generate_visualizations(
        self,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate classification visualizations.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            List of paths to generated files
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path("./evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)

        self.visualization_paths = []

        # Ensure predictions are collected
        if self.predictions is None:
            self._collect_predictions()

        preds_np = self.predictions.numpy()
        labels_np = self.ground_truth.numpy()

        # 1. Confusion Matrix
        cm_path = output_dir / "confusion_matrix.png"
        self._plot_confusion_matrix(preds_np, labels_np, cm_path)
        self.visualization_paths.append(str(cm_path))

        # 2. ROC Curve (binary classification)
        if self.num_classes == 2 and self.probabilities is not None:
            roc_path = output_dir / "roc_curve.png"
            self._plot_roc_curve(labels_np, roc_path)
            self.visualization_paths.append(str(roc_path))

        # 3. Precision-Recall Curve (binary classification)
        if self.num_classes == 2 and self.probabilities is not None:
            pr_path = output_dir / "precision_recall_curve.png"
            self._plot_precision_recall_curve(labels_np, pr_path)
            self.visualization_paths.append(str(pr_path))

        # 4. Save classification report
        report_path = output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.classification_report)
        self.visualization_paths.append(str(report_path))

        self.logger.info(f"Generated {len(self.visualization_paths)} visualizations")
        return self.visualization_paths

    def _plot_confusion_matrix(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        save_path: Path
    ) -> None:
        """Plot and save confusion matrix."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_roc_curve(
        self,
        labels: np.ndarray,
        save_path: Path
    ) -> None:
        """Plot and save ROC curve."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        probs = self.probabilities[:, 1].numpy()
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_precision_recall_curve(
        self,
        labels: np.ndarray,
        save_path: Path
    ) -> None:
        """Plot and save Precision-Recall curve."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        probs = self.probabilities[:, 1].numpy()
        precision, recall, _ = precision_recall_curve(labels, probs)
        avg_precision = average_precision_score(labels, probs)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def get_misclassified_samples(
        self,
        max_samples: int = 10
    ) -> List[Tuple[int, int, int]]:
        """
        Get indices of misclassified samples.

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            List of tuples (index, predicted_class, true_class)
        """
        if self.predictions is None:
            self._collect_predictions()

        misclassified = []
        for i, (pred, true) in enumerate(zip(self.predictions, self.ground_truth)):
            if pred != true:
                misclassified.append((i, pred.item(), true.item()))
                if len(misclassified) >= max_samples:
                    break

        return misclassified
