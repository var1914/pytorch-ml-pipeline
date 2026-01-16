"""
Base Trainer Abstract Class for CV Pipeline.

Provides common training functionality shared across all task types.
Task-specific trainers (classification, detection, segmentation) inherit from this.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.settings import TrainingConfig, InfraConfig


class BaseTrainer(ABC):
    """
    Abstract base trainer for all CV tasks.

    Provides:
    - Device management (auto-detect GPU/MPS/CPU)
    - Training loop structure
    - MLflow integration
    - MinIO checkpoint storage
    - Early stopping
    - Learning rate scheduling

    Subclasses must implement:
    - _compute_loss(): Task-specific loss computation
    - _compute_metrics(): Task-specific metric computation
    - _train_step(): Single training step
    - _val_step(): Single validation step

    Example:
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            infra_config=infra_config
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
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        experiment_name: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.infra_config = infra_config

        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # Setup training components
        self.criterion = criterion or self._get_default_criterion()
        self.optimizer = optimizer or self._setup_optimizer()
        self.scheduler = scheduler or self._setup_scheduler()

        # MLflow setup
        self.experiment_name = experiment_name or "cv_pipeline"
        self._mlflow_run = None

        # MinIO client (lazy initialization)
        self._minio_client = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
        }

        # Setup logging
        self.logger = self._setup_logger()

        # Mixed precision
        self.scaler = None
        if config.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')

    def _setup_device(self) -> torch.device:
        """Setup and return the training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.info("Using Apple MPS")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"Using device: {device}")
        return device

    def _setup_logger(self) -> logging.Logger:
        """Setup training logger."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
        return logger

    def _setup_optimizer(self) -> Optimizer:
        """Setup optimizer based on config."""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> Optional[_LRScheduler]:
        """Setup learning rate scheduler based on config."""
        if not self.config.scheduler:
            return None

        scheduler_name = self.config.scheduler.lower()

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
        else:
            return None

    @abstractmethod
    def _get_default_criterion(self) -> nn.Module:
        """Return the default loss function for this task type."""
        pass

    @abstractmethod
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        pass

    @abstractmethod
    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass

    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            loss, metrics = self._train_step(batch)
            total_loss += loss

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                **{k: f'{v:.4f}' for k, v in metrics.items()}
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def _train_step(self, batch: Tuple) -> Tuple[float, Dict[str, float]]:
        """Single training step."""
        inputs, targets = self._prepare_batch(batch)

        self.optimizer.zero_grad()

        if self.scaler:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.config.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, targets)
            loss.backward()

            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

        metrics = self._compute_metrics(outputs.detach(), targets)
        return loss.item(), metrics

    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

            for batch in pbar:
                loss, metrics = self._val_step(batch)
                total_loss += loss

                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    **{k: f'{v:.4f}' for k, v in metrics.items()}
                })

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def _val_step(self, batch: Tuple) -> Tuple[float, Dict[str, float]]:
        """Single validation step."""
        inputs, targets = self._prepare_batch(batch)

        if self.scaler:
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, targets)

        metrics = self._compute_metrics(outputs, targets)
        return loss.item(), metrics

    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training/validation."""
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # Initialize MLflow
        self._setup_mlflow()

        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_metrics = self._train_epoch()
                self.history['train_loss'].append(train_loss)

                # Validation
                val_loss, val_metrics = self._validate_epoch()
                self.history['val_loss'].append(val_loss)

                # Store metrics in history
                for key, value in train_metrics.items():
                    if f'train_{key}' not in self.history:
                        self.history[f'train_{key}'] = []
                    self.history[f'train_{key}'].append(value)

                for key, value in val_metrics.items():
                    if f'val_{key}' not in self.history:
                        self.history[f'val_{key}'] = []
                    self.history[f'val_{key}'].append(value)

                # Log to MLflow
                self._log_epoch_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics)

                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1

                # Periodic checkpoint
                if self.infra_config and self.infra_config.save_every_n_epochs > 0:
                    if (epoch + 1) % self.infra_config.save_every_n_epochs == 0:
                        self._save_checkpoint(is_best=False)

                # Early stopping
                if self.config.early_stopping_patience > 0:
                    if self.patience_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                # Log epoch summary
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        finally:
            self._finish_mlflow()

        return self.history

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if not self.infra_config:
            return

        try:
            import mlflow
            mlflow.set_tracking_uri(self.infra_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_run = mlflow.start_run()

            # Log config as params
            mlflow.log_params({
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "num_epochs": self.config.num_epochs,
            })
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow: {e}")

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics to MLflow."""
        if not self._mlflow_run:
            return

        try:
            import mlflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }, step=epoch)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to MLflow: {e}")

    def _finish_mlflow(self) -> None:
        """Finish MLflow run."""
        if self._mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                self.logger.warning(f"Failed to end MLflow run: {e}")

    def _save_checkpoint(self, is_best: bool = False) -> str:
        """Save model checkpoint."""
        if not self.infra_config:
            return ""

        checkpoint_dir = Path(self.infra_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss if is_best else self.history['val_loss'][-1],
            'history': self.history,
            'config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
            }
        }

        if is_best:
            path = checkpoint_dir / "best_model.pt"
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

        return str(path)

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        self.logger.info(f"Loaded checkpoint from {path}")
