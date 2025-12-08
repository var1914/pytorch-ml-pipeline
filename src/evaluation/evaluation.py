from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from io import BytesIO
import json

from ..minio.minio_init import MinIO


class ModelEvaluator:
    """
    Comprehensive model evaluation for binary/multi-class classification.

    Provides:
    - Standard metrics (accuracy, precision, recall, F1, AUC)
    - Confusion matrix visualization
    - ROC curve plotting
    - MLflow integration for metric logging
    - MinIO storage for evaluation artifacts
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        minio_config: Optional[Dict] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "model_evaluation"
    ):
        """
        Initialize the ModelEvaluator.

        Args:
            model: Trained PyTorch model.
            device: Device to run evaluation on. Auto-detects if None.
            minio_config: MinIO configuration for storing artifacts.
            mlflow_tracking_uri: MLflow tracking server URI.
            mlflow_experiment_name: Name of the MLflow experiment.
        """
        self.model = model
        self.logger = self._setup_logger()
        self.device = self._setup_device(device)
        self.model.to(self.device)
        self.model.eval()

        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_experiment_name = mlflow_experiment_name

        # Setup MinIO
        self.minio = MinIO(minio_config) if minio_config else None

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for the evaluator."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Auto-detect or setup specified device."""
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        log_to_mlflow: bool = True,
        save_visualizations: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.

        Args:
            test_loader: DataLoader for test data.
            log_to_mlflow: Whether to log metrics to MLflow.
            save_visualizations: Whether to generate and save visualizations.

        Returns:
            Dictionary containing evaluation metrics.
        """
        self.logger.info("Starting model evaluation...")

        # Get predictions
        y_true, y_pred, y_proba = self._get_predictions(test_loader)

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)

        # Log to MLflow
        if log_to_mlflow:
            self._log_to_mlflow(metrics, y_true, y_pred, y_proba, save_visualizations)

        # Print summary
        self._print_summary(metrics)

        return metrics

    def _get_predictions(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from the model.

        Returns:
            Tuple of (true_labels, predicted_labels, predicted_probabilities).
        """
        y_true = []
        y_pred = []
        y_proba = []

        test_bar = tqdm(test_loader, desc="Evaluating", leave=False)

        for inputs, labels in test_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_proba.extend(probabilities.cpu().numpy())

        return np.array(y_true), np.array(y_pred), np.array(y_proba)

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Add AUC for binary classification
        if y_proba.shape[1] == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])

        return metrics

    def _print_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
        self.logger.info("=" * 60)

    def _log_to_mlflow(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        save_visualizations: bool
    ) -> None:
        """Log metrics and artifacts to MLflow."""
        try:
            with mlflow.start_run():
                # Log metrics
                mlflow.log_metrics(metrics)

                # Log classification report
                report = classification_report(y_true, y_pred)
                mlflow.log_text(report, "classification_report.txt")

                # Generate and log visualizations
                if save_visualizations:
                    self._save_confusion_matrix(y_true, y_pred)
                    if y_proba.shape[1] == 2:
                        self._save_roc_curve(y_true, y_proba[:, 1])

                self.logger.info("Metrics logged to MLflow successfully")

        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {str(e)}")

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Generate and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save to MLflow
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()

        self.logger.info("Confusion matrix saved")

    def _save_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """Generate and save ROC curve (binary classification only)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # Save to MLflow
        mlflow.log_figure(plt.gcf(), "roc_curve.png")
        plt.close()

        self.logger.info("ROC curve saved")

    def get_classification_report(self, test_loader: DataLoader) -> str:
        """
        Get detailed classification report.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Classification report as string.
        """
        y_true, y_pred, _ = self._get_predictions(test_loader)
        return classification_report(y_true, y_pred)

    def get_confusion_matrix(self, test_loader: DataLoader) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Confusion matrix as numpy array.
        """
        y_true, y_pred, _ = self._get_predictions(test_loader)
        return confusion_matrix(y_true, y_pred)
