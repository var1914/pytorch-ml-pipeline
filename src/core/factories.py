"""
Factory classes for creating models, trainers, and evaluators.

Provides a unified interface for instantiating task-specific components
based on configuration.
"""

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from .task_registry import TaskType, validate_architecture

if TYPE_CHECKING:
    from .base_model import CVModel
    from ..training.base_trainer import BaseTrainer
    from ..evaluation.base_evaluator import BaseEvaluator


class ModelFactory:
    """
    Factory for creating CV models based on task type and architecture.

    Example:
        model = ModelFactory.create(
            task_type="classification",
            architecture="resnet50",
            num_classes=10,
            pretrained=True
        )
    """

    # Registry of model constructors by task type
    _classification_models: Dict[str, Callable] = {}
    _detection_models: Dict[str, Callable] = {}
    _segmentation_models: Dict[str, Callable] = {}

    @classmethod
    def register_classification_model(cls, name: str, constructor: Callable) -> None:
        """Register a classification model constructor."""
        cls._classification_models[name] = constructor

    @classmethod
    def register_detection_model(cls, name: str, constructor: Callable) -> None:
        """Register a detection model constructor."""
        cls._detection_models[name] = constructor

    @classmethod
    def register_segmentation_model(cls, name: str, constructor: Callable) -> None:
        """Register a segmentation model constructor."""
        cls._segmentation_models[name] = constructor

    @classmethod
    def create(
        cls,
        task_type: str,
        architecture: str,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> "CVModel":
        """
        Create a model instance.

        Args:
            task_type: Task type string (classification, detection, segmentation)
            architecture: Model architecture name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific arguments

        Returns:
            CVModel instance

        Raises:
            ValueError: If task type or architecture is not supported
        """
        task = TaskType.from_string(task_type)

        if task == TaskType.CLASSIFICATION:
            return cls._create_classification_model(architecture, num_classes, pretrained, **kwargs)
        elif task == TaskType.DETECTION:
            return cls._create_detection_model(architecture, num_classes, pretrained, **kwargs)
        elif task == TaskType.SEGMENTATION:
            return cls._create_segmentation_model(architecture, num_classes, pretrained, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @classmethod
    def _create_classification_model(
        cls,
        architecture: str,
        num_classes: int,
        pretrained: bool,
        **kwargs
    ) -> "CVModel":
        """Create a classification model."""
        # Check if custom model is registered
        if architecture in cls._classification_models:
            return cls._classification_models[architecture](num_classes, pretrained, **kwargs)

        # Otherwise, use the built-in ClassificationModel
        from ..models.classification import ClassificationModel
        return ClassificationModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )

    @classmethod
    def _create_detection_model(
        cls,
        architecture: str,
        num_classes: int,
        pretrained: bool,
        **kwargs
    ) -> "CVModel":
        """Create a detection model."""
        if architecture in cls._detection_models:
            return cls._detection_models[architecture](num_classes, pretrained, **kwargs)

        from ..models.detection import DetectionModel
        return DetectionModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )

    @classmethod
    def _create_segmentation_model(
        cls,
        architecture: str,
        num_classes: int,
        pretrained: bool,
        **kwargs
    ) -> "CVModel":
        """Create a segmentation model."""
        if architecture in cls._segmentation_models:
            return cls._segmentation_models[architecture](num_classes, pretrained, **kwargs)

        from ..models.segmentation import SegmentationModel
        return SegmentationModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )

    @classmethod
    def list_models(cls, task_type: Optional[str] = None) -> Dict[str, list]:
        """
        List available models.

        Args:
            task_type: Optional task type to filter by

        Returns:
            Dictionary mapping task types to lists of available architectures
        """
        from .task_registry import get_supported_architectures

        if task_type:
            task = TaskType.from_string(task_type)
            return {task_type: get_supported_architectures(task)}

        return {
            "classification": get_supported_architectures(TaskType.CLASSIFICATION),
            "detection": get_supported_architectures(TaskType.DETECTION),
            "segmentation": get_supported_architectures(TaskType.SEGMENTATION),
        }


class TrainerFactory:
    """
    Factory for creating task-specific trainers.

    Example:
        trainer = TrainerFactory.create(
            task_type="classification",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )
    """

    @staticmethod
    def create(
        task_type: str,
        model: "CVModel",
        train_loader,
        val_loader,
        config,
        infra_config=None,
        **kwargs
    ) -> "BaseTrainer":
        """
        Create a trainer instance.

        Args:
            task_type: Task type string
            model: CVModel instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: TrainingConfig instance
            infra_config: Optional InfraConfig instance
            **kwargs: Additional trainer arguments

        Returns:
            BaseTrainer subclass instance
        """
        task = TaskType.from_string(task_type)

        if task == TaskType.CLASSIFICATION:
            from ..training.classification_trainer import ClassificationTrainer
            return ClassificationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                infra_config=infra_config,
                **kwargs
            )
        elif task == TaskType.DETECTION:
            from ..training.detection_trainer import DetectionTrainer
            return DetectionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                infra_config=infra_config,
                **kwargs
            )
        elif task == TaskType.SEGMENTATION:
            from ..training.segmentation_trainer import SegmentationTrainer
            return SegmentationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                infra_config=infra_config,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class EvaluatorFactory:
    """
    Factory for creating task-specific evaluators.

    Example:
        evaluator = EvaluatorFactory.create(
            task_type="classification",
            model=model,
            test_loader=test_loader
        )
        metrics = evaluator.evaluate()
    """

    @staticmethod
    def create(
        task_type: str,
        model: "CVModel",
        test_loader,
        config=None,
        **kwargs
    ) -> "BaseEvaluator":
        """
        Create an evaluator instance.

        Args:
            task_type: Task type string
            model: CVModel instance
            test_loader: Test DataLoader
            config: Optional configuration
            **kwargs: Additional evaluator arguments

        Returns:
            BaseEvaluator subclass instance
        """
        task = TaskType.from_string(task_type)

        if task == TaskType.CLASSIFICATION:
            from ..evaluation.classification_evaluator import ClassificationEvaluator
            return ClassificationEvaluator(
                model=model,
                test_loader=test_loader,
                config=config,
                **kwargs
            )
        elif task == TaskType.DETECTION:
            from ..evaluation.detection_evaluator import DetectionEvaluator
            return DetectionEvaluator(
                model=model,
                test_loader=test_loader,
                config=config,
                **kwargs
            )
        elif task == TaskType.SEGMENTATION:
            from ..evaluation.segmentation_evaluator import SegmentationEvaluator
            return SegmentationEvaluator(
                model=model,
                test_loader=test_loader,
                config=config,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class LossFunctionFactory:
    """
    Factory for creating task-specific loss functions.

    Example:
        criterion = LossFunctionFactory.create(
            task_type="classification",
            loss_name="CrossEntropyLoss"
        )
    """

    @staticmethod
    def create(
        task_type: str,
        loss_name: Optional[str] = None,
        **kwargs
    ):
        """
        Create a loss function.

        Args:
            task_type: Task type string
            loss_name: Optional loss function name (uses default if None)
            **kwargs: Loss function arguments

        Returns:
            Loss function (nn.Module)
        """
        import torch.nn as nn

        task = TaskType.from_string(task_type)

        # Use default loss if not specified
        if loss_name is None:
            from .task_registry import get_task_config
            loss_name = get_task_config(task)["default_loss"]

        # Classification losses
        if task == TaskType.CLASSIFICATION:
            if loss_name == "CrossEntropyLoss":
                return nn.CrossEntropyLoss(**kwargs)
            elif loss_name == "BCEWithLogitsLoss":
                return nn.BCEWithLogitsLoss(**kwargs)
            elif loss_name == "FocalLoss":
                from ..losses.focal import FocalLoss
                return FocalLoss(**kwargs)
            elif loss_name == "LabelSmoothingLoss":
                return nn.CrossEntropyLoss(label_smoothing=kwargs.get("smoothing", 0.1))

        # Detection losses
        elif task == TaskType.DETECTION:
            # Detection models typically have built-in losses
            # Return None to use model's default loss
            return None

        # Segmentation losses
        elif task == TaskType.SEGMENTATION:
            if loss_name == "CrossEntropyLoss":
                return nn.CrossEntropyLoss(**kwargs)
            elif loss_name == "DiceLoss":
                from ..losses.dice import DiceLoss
                return DiceLoss(**kwargs)
            elif loss_name == "JaccardLoss":
                from ..losses.jaccard import JaccardLoss
                return JaccardLoss(**kwargs)

        raise ValueError(f"Unknown loss function: {loss_name} for task {task_type}")
