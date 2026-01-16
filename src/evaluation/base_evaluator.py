"""
Base Evaluator Abstract Class for CV Pipeline.

Provides common evaluation functionality shared across all task types.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseEvaluator(ABC):
    """
    Abstract base evaluator for all CV tasks.

    Provides:
    - Device management
    - MLflow integration for logging metrics and artifacts
    - Common evaluation workflow

    Subclasses must implement:
    - evaluate(): Run evaluation and return metrics
    - generate_visualizations(): Create task-specific visualizations

    Example:
        evaluator = ClassificationEvaluator(
            model=model,
            test_loader=test_loader,
            class_names=['cat', 'dog']
        )
        metrics = evaluator.evaluate()
        evaluator.log_to_mlflow()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.class_names = class_names

        # Setup device
        self.device = self._setup_device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Results storage
        self.metrics: Dict[str, float] = {}
        self.predictions: Optional[torch.Tensor] = None
        self.ground_truth: Optional[torch.Tensor] = None
        self.probabilities: Optional[torch.Tensor] = None

        # Visualization paths
        self.visualization_paths: List[str] = []

        # Logger
        self.logger = self._setup_logger()

    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """Setup evaluation device."""
        if device:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_logger(self) -> logging.Logger:
        """Setup evaluator logger."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
        return logger

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation and compute metrics.

        Returns:
            Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def generate_visualizations(
        self,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate task-specific visualizations.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            List of paths to generated visualization files
        """
        pass

    def get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all predictions and ground truth labels.

        Returns:
            Tuple of (predictions, ground_truth)
        """
        if self.predictions is None or self.ground_truth is None:
            self._collect_predictions()
        return self.predictions, self.ground_truth

    def _collect_predictions(self) -> None:
        """Collect all predictions from the test loader."""
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, labels = self._prepare_batch(batch)
                outputs = self.model(inputs)

                preds, probs = self._process_outputs(outputs)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                if probs is not None:
                    all_probs.append(probs.cpu())

        self.predictions = torch.cat(all_preds)
        self.ground_truth = torch.cat(all_labels)
        if all_probs:
            self.probabilities = torch.cat(all_probs)

    def _prepare_batch(
        self,
        batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for evaluation."""
        inputs, labels = batch[0], batch[1]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    @abstractmethod
    def _process_outputs(
        self,
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process model outputs to get predictions and probabilities.

        Args:
            outputs: Raw model outputs

        Returns:
            Tuple of (predictions, probabilities)
        """
        pass

    def log_to_mlflow(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "evaluation",
        run_name: Optional[str] = None
    ) -> None:
        """
        Log metrics and visualizations to MLflow.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            run_name: Optional run name
        """
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=run_name):
                # Log metrics
                mlflow.log_metrics(self.metrics)

                # Log visualizations as artifacts
                for viz_path in self.visualization_paths:
                    if Path(viz_path).exists():
                        mlflow.log_artifact(viz_path)

                self.logger.info(f"Logged metrics and {len(self.visualization_paths)} artifacts to MLflow")

        except ImportError:
            self.logger.warning("MLflow not installed. Skipping MLflow logging.")
        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {e}")

    def print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        print("=" * 50 + "\n")

    def save_metrics(self, path: str) -> None:
        """Save metrics to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {path}")
