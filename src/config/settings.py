"""
Pydantic-based configuration system for CV Pipeline.

Supports:
- YAML config files with environment variable interpolation
- Hierarchical configuration (data, model, training, infra)
- Runtime validation and type checking
- Default values with easy overrides

Usage:
    from src.config import load_config, CVPipelineConfig

    # Load from YAML
    config = load_config("configs/templates/medical_imaging.yaml")

    # Or create programmatically
    config = CVPipelineConfig(
        data=DataConfig(dataset_name="pcam", data_root="./data"),
        model=ModelConfig(task_type="classification", architecture="resnet50", num_classes=2),
        training=TrainingConfig(batch_size=32, num_epochs=50),
        infra=InfraConfig(minio_endpoint="localhost:9000", mlflow_tracking_uri="http://localhost:5000")
    )
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""

    dataset_name: str = Field(
        default="custom",
        description="Dataset name (pcam, cifar10, imagenet, mvtec, custom, etc.)"
    )
    data_root: str = Field(
        default="./data",
        description="Root directory for dataset storage"
    )
    image_size: Tuple[int, int] = Field(
        default=(224, 224),
        description="Input image size (height, width)"
    )
    normalization_mean: List[float] = Field(
        default=[0.485, 0.456, 0.406],
        description="Normalization mean (default: ImageNet)"
    )
    normalization_std: List[float] = Field(
        default=[0.229, 0.224, 0.225],
        description="Normalization std (default: ImageNet)"
    )
    augmentation_level: str = Field(
        default="medium",
        description="Augmentation intensity: light, medium, heavy"
    )
    num_workers: int = Field(
        default=4,
        description="Number of data loading workers"
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer"
    )

    @field_validator("augmentation_level")
    @classmethod
    def validate_augmentation_level(cls, v: str) -> str:
        valid_levels = ["light", "medium", "heavy", "none"]
        if v.lower() not in valid_levels:
            raise ValueError(f"augmentation_level must be one of {valid_levels}")
        return v.lower()


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    task_type: str = Field(
        default="classification",
        description="Task type: classification, detection, segmentation"
    )
    architecture: str = Field(
        default="resnet50",
        description="Model architecture (resnet50, efficientnet_b0, yolov8s, unet, etc.)"
    )
    num_classes: int = Field(
        default=2,
        description="Number of output classes"
    )
    pretrained: bool = Field(
        default=True,
        description="Use pretrained weights"
    )
    freeze_backbone: bool = Field(
        default=False,
        description="Freeze backbone layers for fine-tuning"
    )
    dropout: float = Field(
        default=0.0,
        description="Dropout rate (0.0 to disable)"
    )

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        valid_types = ["classification", "detection", "segmentation"]
        if v.lower() not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}")
        return v.lower()


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""

    batch_size: int = Field(
        default=32,
        description="Training batch size"
    )
    num_epochs: int = Field(
        default=50,
        description="Maximum number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001,
        description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=0.0001,
        description="L2 regularization weight decay"
    )
    optimizer: str = Field(
        default="adam",
        description="Optimizer: adam, adamw, sgd"
    )
    scheduler: Optional[str] = Field(
        default=None,
        description="LR scheduler: cosine, step, plateau, none"
    )
    early_stopping_patience: int = Field(
        default=5,
        description="Early stopping patience (0 to disable)"
    )
    device: str = Field(
        default="auto",
        description="Device: auto, cuda, mps, cpu"
    )
    mixed_precision: bool = Field(
        default=False,
        description="Use automatic mixed precision (AMP)"
    )
    gradient_clip: Optional[float] = Field(
        default=None,
        description="Gradient clipping max norm (None to disable)"
    )

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        valid_optimizers = ["adam", "adamw", "sgd"]
        if v.lower() not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
        return v.lower()


class InfraConfig(BaseModel):
    """Infrastructure and MLOps configuration."""

    # MinIO configuration
    minio_endpoint: str = Field(
        default="localhost:9000",
        description="MinIO server endpoint"
    )
    minio_access_key: str = Field(
        default="admin",
        description="MinIO access key"
    )
    minio_secret_key: str = Field(
        default="admin123",
        description="MinIO secret key"
    )
    minio_bucket: str = Field(
        default="ml-models",
        description="MinIO bucket for model storage"
    )
    minio_secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO"
    )

    # MLflow configuration
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: Optional[str] = Field(
        default=None,
        description="MLflow experiment name (auto-generated if None)"
    )

    # Checkpoint configuration
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Local checkpoint directory"
    )
    save_every_n_epochs: int = Field(
        default=5,
        description="Save checkpoint every N epochs (0 to disable)"
    )


class CVPipelineConfig(BaseSettings):
    """
    Main configuration class for the CV Pipeline.

    Combines all configuration sections and supports:
    - Loading from YAML files
    - Environment variable overrides
    - Programmatic configuration
    """

    model_config = SettingsConfigDict(
        env_prefix="CV_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    # Configuration sections
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)

    # Pipeline metadata
    name: str = Field(
        default="cv_pipeline",
        description="Pipeline name for logging and tracking"
    )
    version: str = Field(
        default="1.0.0",
        description="Pipeline version"
    )
    description: Optional[str] = Field(
        default=None,
        description="Pipeline description"
    )

    def get_experiment_name(self) -> str:
        """Generate MLflow experiment name from config."""
        if self.infra.mlflow_experiment_name:
            return self.infra.mlflow_experiment_name
        return f"{self.data.dataset_name}_{self.model.architecture}_{self.model.task_type}"

    def get_model_name(self) -> str:
        """Generate model registry name."""
        return f"cv_{self.model.architecture}_{self.model.task_type}_{self.data.dataset_name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "name": self.name,
            "version": self.version,
            "data": self.data.model_dump(),
            "model": self.model.model_dump(),
            "training": self.training.model_dump(),
            "infra": {
                k: v for k, v in self.infra.model_dump().items()
                if "key" not in k.lower()  # Exclude secrets
            }
        }


def _interpolate_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interpolate environment variables in config values.

    Supports formats:
    - ${VAR_NAME} - required variable
    - ${VAR_NAME:-default} - variable with default
    """
    import re

    def interpolate_value(value: Any) -> Any:
        if isinstance(value, str):
            # Match ${VAR} or ${VAR:-default}
            pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

            def replace(match):
                var_name = match.group(1)
                default = match.group(2)
                env_value = os.environ.get(var_name)

                if env_value is not None:
                    return env_value
                elif default is not None:
                    return default
                else:
                    raise ValueError(f"Environment variable {var_name} not set and no default provided")

            return re.sub(pattern, replace, value)
        elif isinstance(value, dict):
            return {k: interpolate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [interpolate_value(item) for item in value]
        return value

    return interpolate_value(config_dict)


def load_config(config_path: Union[str, Path]) -> CVPipelineConfig:
    """
    Load configuration from YAML file.

    Supports:
    - Environment variable interpolation (${VAR} or ${VAR:-default})
    - Base config inheritance via _base_ key
    - Nested configuration merging

    Args:
        config_path: Path to YAML config file

    Returns:
        CVPipelineConfig instance

    Example:
        config = load_config("configs/templates/medical_imaging.yaml")
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("omegaconf is required for YAML config loading. Install with: pip install omegaconf")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML with OmegaConf
    config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    # Handle base config inheritance
    if "_base_" in config_dict:
        base_path = config_path.parent / config_dict.pop("_base_")
        base_config = OmegaConf.to_container(OmegaConf.load(base_path), resolve=True)
        config_dict = _deep_merge(base_config, config_dict)

    # Interpolate environment variables
    config_dict = _interpolate_env_vars(config_dict)

    return CVPipelineConfig(**config_dict)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_config(**kwargs) -> CVPipelineConfig:
    """
    Create configuration programmatically.

    Args:
        **kwargs: Configuration overrides

    Returns:
        CVPipelineConfig instance

    Example:
        config = get_config(
            data={"dataset_name": "pcam", "image_size": (96, 96)},
            model={"architecture": "resnet50", "num_classes": 2},
            training={"batch_size": 64}
        )
    """
    # Build nested config objects
    data = DataConfig(**kwargs.get("data", {}))
    model = ModelConfig(**kwargs.get("model", {}))
    training = TrainingConfig(**kwargs.get("training", {}))
    infra = InfraConfig(**kwargs.get("infra", {}))

    return CVPipelineConfig(
        data=data,
        model=model,
        training=training,
        infra=infra,
        name=kwargs.get("name", "cv_pipeline"),
        version=kwargs.get("version", "1.0.0"),
        description=kwargs.get("description")
    )
