from typing import Any, Callable, Dict, Optional, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datetime
import pickle
from io import BytesIO
import json
import logging

from minio import Minio
from minio.error import S3Error
import mlflow
import mlflow.pytorch

class ModelTrainer():
    """
    Production-grade trainer for PyTorch models with MLflow tracking and MinIO storage.
    
    Handles complete training workflow including:
    - Training and validation loops
    - Metric tracking and logging
    - Model checkpointing to MinIO
    - Experiment tracking with MLflow
    - Early stopping and best model selection
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[str] = None,
        lr: float = 0.001,
        minio_config: Optional[Dict[str, Any]] = None,
        mlflow_experiment_name: str = "model_training",
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            criterion: Loss function (default: CrossEntropyLoss).
            optimizer: Optimizer instance (default: Adam with specified lr).
            device: Device to train on. Auto-detects if None.
            lr: Learning rate for optimizer (ignored if optimizer provided).
            checkpoint_dir: Directory to save model checkpoints.
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Setup logging
        self.logger = self._setup_logger()

        # Auto-detect device
        self.device = self._setup_device(device)
        self.logger.info(f"Using device: {self.device}")
        print(f"Using device: {self.device}")

        # Initialize criterion and optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(), 
            lr=self.lr
        )

        # Move model to device
        self.model.to(self.device)

        # Setup MLflow
        self.mlflow_experiment_name = mlflow_experiment_name
        mlflow.set_experiment(mlflow_experiment_name)
        
        # Setup MinIO
        self.minio_config = minio_config or self._default_minio_config()
        self.minio_client = self._setup_minio_client()
        self._ensure_bucket_exists()

        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for the trainer."""
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
    
    @staticmethod
    def _default_minio_config() -> Dict[str, Any]:
        """Default MinIO configuration."""
        return {
            'endpoint': 'minio:9000',
            'access_key': 'admin',
            'secret_key': 'admin123',
            'secure': False,
            'bucket_name': 'ml-models'
        }
    
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
    
    def _setup_minio_client(self) -> Minio:
        """Initialize MinIO client with error handling."""
        try:
            client = Minio(
                self.minio_config['endpoint'],
                access_key=self.minio_config['access_key'],
                secret_key=self.minio_config['secret_key'],
                secure=self.minio_config['secure']
            )
            self.logger.info("MinIO client initialized successfully")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize MinIO client: {str(e)}")
            raise RuntimeError(f"MinIO initialization failed: {str(e)}") from e
        
    def _ensure_bucket_exists(self) -> None:
        """Ensure MinIO bucket exists, create if not."""
        bucket_name = self.minio_config['bucket_name']
        try:
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                self.logger.info(f"Created MinIO bucket: {bucket_name}")
            else:
                self.logger.info(f"Using existing MinIO bucket: {bucket_name}")
        except S3Error as e:
            self.logger.error(f"Failed to setup bucket: {str(e)}")
            raise

    def _train_epoch(self) -> Tuple[float, float]:
        """
        Execute one training epoch.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(self.train_loader, desc="Training", leave=False)
        

            
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)


            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = running_loss / len(self.train_loader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Execute one validation epoch.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        val_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            val_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = running_loss / len(self.val_loader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best_only: bool = True,
        log_to_mlflow: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model with MLflow tracking.
        
        Args:
            num_epochs: Number of epochs to train.
            early_stopping_patience: Stop if validation loss doesn't improve
                                    for this many epochs. None disables.
            save_best_only: Only save checkpoints when validation improves.
            log_to_mlflow: Enable MLflow experiment tracking.
            
        Returns:
            Dictionary containing training history with keys:
            'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0

        # Start MLflow run
        if log_to_mlflow:
            mlflow.start_run()
            
            # Log hyperparameters
            mlflow.log_params({
                'lr': self.lr,
                'optimizer': type(self.optimizer).__name__,
                'criterion': type(self.criterion).__name__,
                'num_epochs': num_epochs,
                'device': str(self.device),
                'early_stopping_patience': early_stopping_patience
            })
        try:
            print(f"\nStarting training for {num_epochs} epochs...")
            print("=" * 60)
            
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 60)
                
                # Training phase
                train_loss, train_acc = self._train_epoch()
                
                # Validation phase
                val_loss, val_acc = self._validate_epoch()
                
                # Store metrics
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Print epoch summary
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
                
                # Save checkpoint
                if self.checkpoint_dir:
                    is_best = val_loss < best_val_loss
                    
                    if is_best:
                        best_val_loss = val_loss
                        patience_counter = 0
                        print(f"✓ New best validation loss: {val_loss:.4f}")
                    
                    if not save_best_only or is_best:
                        self._save_checkpoint(epoch, val_loss, is_best)
                
                # Early stopping check
                if early_stopping_patience:
                    if val_loss >= best_val_loss:
                        patience_counter += 1
                        print(f"No improvement for {patience_counter} epoch(s)")
                        
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                            break
            
            print("\n" + "=" * 60)
            print("Training completed!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            
            # Log final best model to MLflow
            if log_to_mlflow:
                mlflow.log_metric('best_val_loss', best_val_loss)
        
        finally:
            if log_to_mlflow:
                mlflow.end_run()
        
        return history
    
    def _save_model_artifacts(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Tuple[str, str]:
        """
        Save model to both local checkpoint (optional), MinIO, and MLflow.
        
        Args:
            epoch: Current epoch number.
            metrics: Dictionary of training metrics.
            is_best: Whether this is the best model so far.
            
        Returns:
            Tuple of (minio_model_path, minio_metadata_path).
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save local checkpoint (optional)
        if self.checkpoint_dir:
            self._save_local_checkpoint(epoch, metrics['val_loss'], is_best)
        
        # 2. Save model to MinIO
        model_path = self._save_model_to_minio(timestamp)
        
        # 3. Save metadata to MinIO
        metadata_path = self._save_metadata_to_minio(
            timestamp, model_path, metrics
        )
        
        # 4. Log model to MLflow
        try:
            mlflow.pytorch.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=f"{self.mlflow_experiment_name}_model"
            )
            
            # Log additional artifacts
            mlflow.log_artifact(metadata_path) if self.checkpoint_dir else None
            
            self.logger.info(
                f"Model artifacts saved successfully (MinIO: {model_path})"
            )
        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {str(e)}")
        
        return model_path, metadata_path
    
    def _save_model_to_minio(self, timestamp: str) -> str:
        """Save model state dict to MinIO."""
        bucket_name = self.minio_config['bucket_name']
        model_path = f"models/{timestamp}/model.pt"
        
        try:
            # Serialize model
            model_buffer = BytesIO()
            torch.save(self.model.state_dict(), model_buffer)
            model_buffer.seek(0)
            
            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=bucket_name,
                object_name=model_path,
                data=model_buffer,
                length=len(model_buffer.getvalue()),
                content_type='application/octet-stream'
            )
            
            self.logger.info(f"Model saved to MinIO: {model_path}")
            return model_path
            
        except S3Error as e:
            self.logger.error(f"Failed to save model to MinIO: {str(e)}")
            raise
    
    def _save_metadata_to_minio(
        self,
        timestamp: str,
        model_path: str,
        metrics: Dict[str, float]
    ) -> str:
        """Save training metadata to MinIO."""
        bucket_name = self.minio_config['bucket_name']
        metadata_path = f"models/{timestamp}/metadata.json"
        
        metadata = {
            'timestamp': timestamp,
            'model_path': model_path,
            'metrics': metrics,
            'hyperparameters': {
                'lr': self.lr,
                'optimizer': type(self.optimizer).__name__,
                'criterion': type(self.criterion).__name__
            },
            'device': str(self.device),
            'mlflow_experiment': self.mlflow_experiment_name
        }
        
        try:
            # Serialize metadata
            metadata_buffer = BytesIO(
                json.dumps(metadata, indent=2).encode('utf-8')
            )
            metadata_buffer.seek(0)
            
            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=bucket_name,
                object_name=metadata_path,
                data=metadata_buffer,
                length=len(metadata_buffer.getvalue()),
                content_type='application/json'
            )
            
            self.logger.info(f"Metadata saved to MinIO: {metadata_path}")
            return metadata_path
            
        except S3Error as e:
            self.logger.error(f"Failed to save metadata to MinIO: {str(e)}")
            raise
    
    def _save_local_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ) -> None:
        """Save local checkpoint file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved locally: {best_path}")
    
    def load_from_minio(self, model_path: str) -> None:
        """
        Load model from MinIO.
        
        Args:
            model_path: Path to model in MinIO (e.g., 'models/20241204_120000/model.pt').
        """
        bucket_name = self.minio_config['bucket_name']
        
        try:
            # Download model from MinIO
            response = self.minio_client.get_object(bucket_name, model_path)
            model_buffer = BytesIO(response.read())
            model_buffer.seek(0)
            
            # Load model state dict
            state_dict = torch.load(model_buffer, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.logger.info(f"Model loaded from MinIO: {model_path}")
            
        except S3Error as e:
            self.logger.error(f"Failed to load model from MinIO: {str(e)}")
            raise
        finally:
            response.close()
            response.release_conn()
    
    def load_local_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Load model from local checkpoint.
        
        Args:
            checkpoint_path: Path to local checkpoint file.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        self.logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint