"""
Detection Trainer for CV Pipeline.

Specialized trainer for object detection tasks with:
- Support for YOLO models (via ultralytics)
- Support for torchvision detection models (Faster R-CNN, RetinaNet, SSD)
- mAP, precision, recall metrics
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config.settings import TrainingConfig, InfraConfig
from .base_trainer import BaseTrainer


class DetectionTrainer(BaseTrainer):
    """
    Trainer for object detection tasks.

    Note: Detection training differs significantly from classification:
    - YOLO models use their own training loop (ultralytics)
    - torchvision models use custom collate functions and loss computation

    For YOLO models, use the train_yolo() method instead of train().

    Example (torchvision):
        trainer = DetectionTrainer(
            model=faster_rcnn_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainingConfig(batch_size=8)
        )
        history = trainer.train()

    Example (YOLO):
        trainer = DetectionTrainer(
            model=yolo_model,
            train_loader=None,  # Not used for YOLO
            val_loader=None,
            config=TrainingConfig()
        )
        results = trainer.train_yolo(data_yaml="data.yaml", epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        infra_config: Optional[InfraConfig] = None,
        **kwargs
    ):
        # Check if model is YOLO
        self._is_yolo = self._check_is_yolo(model)

        if self._is_yolo:
            # For YOLO, we don't need the standard training setup
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.config = config
            self.infra_config = infra_config
            self.device = self._setup_device_yolo()
            self.logger = self._setup_logger()
            self.history = {}
        else:
            # Standard PyTorch training setup for torchvision models
            super().__init__(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                infra_config=infra_config,
                **kwargs
            )

    def _check_is_yolo(self, model: nn.Module) -> bool:
        """Check if the model is a YOLO model."""
        # Check for ultralytics YOLO
        model_class = model.__class__.__name__
        if 'YOLO' in model_class or hasattr(model, 'model'):
            # Check if it has ultralytics YOLO attributes
            if hasattr(model, 'train') and hasattr(model, 'predict'):
                return True
        return False

    def _setup_device_yolo(self) -> torch.device:
        """Setup device for YOLO (simplified)."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _setup_logger(self):
        """Setup logger for detection trainer."""
        import logging
        logger = logging.getLogger("DetectionTrainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
        return logger

    def _get_default_criterion(self) -> Optional[nn.Module]:
        """
        Detection models typically have built-in loss computation.
        Return None for detection.
        """
        return None

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute detection loss.

        For torchvision models, the model returns a loss dict during training.
        """
        if isinstance(outputs, dict):
            # torchvision detection models return dict of losses during training
            return sum(loss for loss in outputs.values())
        return torch.tensor(0.0)

    def _compute_metrics(
        self,
        outputs: Any,
        targets: Any
    ) -> Dict[str, float]:
        """
        Compute detection metrics (simplified for training loop).

        Full metrics (mAP) are computed during evaluation.
        """
        # During training, we just track loss
        return {}

    def _prepare_batch(
        self,
        batch: Tuple
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Prepare detection batch.

        Detection batches have variable number of boxes per image,
        so they're handled as lists rather than stacked tensors.
        """
        images, targets = batch

        # Move images to device
        if isinstance(images, torch.Tensor):
            images = [img.to(self.device) for img in images]
        else:
            images = [img.to(self.device) for img in images]

        # Move targets to device
        targets = [
            {k: v.to(self.device) for k, v in t.items()}
            for t in targets
        ]

        return images, targets

    def _train_step(
        self,
        batch: Tuple
    ) -> Tuple[float, Dict[str, float]]:
        """Single training step for torchvision detection models."""
        images, targets = self._prepare_batch(batch)

        self.optimizer.zero_grad()

        # torchvision detection models return losses during training
        self.model.train()
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()

        if self.config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

        self.optimizer.step()

        return losses.item(), {}

    def _val_step(
        self,
        batch: Tuple
    ) -> Tuple[float, Dict[str, float]]:
        """Single validation step for torchvision detection models."""
        images, targets = self._prepare_batch(batch)

        # For validation, we need to get both losses and predictions
        self.model.train()  # Keep in train mode to get losses
        with torch.no_grad():
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        return losses.item(), {}

    def train(self) -> Dict[str, List[float]]:
        """
        Train the detection model.

        For YOLO models, this redirects to train_yolo().
        For torchvision models, uses standard training loop.
        """
        if self._is_yolo:
            raise ValueError(
                "YOLO models should use train_yolo() method. "
                "Call trainer.train_yolo(data_yaml='data.yaml', epochs=100)"
            )

        return super().train()

    def train_yolo(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        imgsz: int = 640,
        batch: Optional[int] = None,
        project: str = "runs/detect",
        name: str = "train",
        **kwargs
    ) -> Any:
        """
        Train YOLO model using ultralytics training.

        Args:
            data_yaml: Path to data.yaml configuration file
            epochs: Number of training epochs (defaults to config)
            imgsz: Input image size
            batch: Batch size (defaults to config)
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional ultralytics training arguments

        Returns:
            Training results from ultralytics
        """
        if not self._is_yolo:
            raise ValueError(
                "train_yolo() only works with YOLO models. "
                "Use train() for torchvision models."
            )

        epochs = epochs or self.config.num_epochs
        batch = batch or self.config.batch_size

        self.logger.info(f"Starting YOLO training for {epochs} epochs")
        self.logger.info(f"Data config: {data_yaml}")

        # Setup MLflow tracking
        self._setup_mlflow()

        try:
            # Train using ultralytics
            results = self.model.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=name,
                **kwargs
            )

            # Log results to MLflow
            if self._mlflow_run:
                try:
                    import mlflow
                    # Log final metrics
                    if hasattr(results, 'results_dict'):
                        mlflow.log_metrics(results.results_dict)
                except Exception as e:
                    self.logger.warning(f"Failed to log YOLO results to MLflow: {e}")

            self.history = {'yolo_results': results}
            return results

        finally:
            self._finish_mlflow()

    def _setup_mlflow(self) -> None:
        """Setup MLflow for detection training."""
        if not self.infra_config:
            return

        try:
            import mlflow
            mlflow.set_tracking_uri(self.infra_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name if hasattr(self, 'experiment_name') else "detection")
            self._mlflow_run = mlflow.start_run()
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow: {e}")
            self._mlflow_run = None

    def _finish_mlflow(self) -> None:
        """Finish MLflow run."""
        if hasattr(self, '_mlflow_run') and self._mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
