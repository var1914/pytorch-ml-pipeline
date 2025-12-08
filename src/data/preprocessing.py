from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import logging


class DataPreprocessor:
    """
    Data preprocessing utilities for PyTorch datasets.

    Provides:
    - Data validation
    - Train/Val/Test splitting with stratification support
    - Advanced data augmentation
    - Data quality checks
    """

    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.logger = self._setup_logger()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for the preprocessor."""
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

    def get_data_augmentation_transforms(
        self,
        image_size: int = 96,
        augmentation_level: str = 'medium'
    ) -> transforms.Compose:
        """
        Get data augmentation transforms for training.

        Args:
            image_size: Target image size.
            augmentation_level: Level of augmentation ('light', 'medium', 'heavy').

        Returns:
            Composed transforms.
        """
        if augmentation_level == 'light':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        elif augmentation_level == 'medium':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        elif augmentation_level == 'heavy':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10
                ),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.15
                ),
                transforms.RandomErasing(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:
            raise ValueError(f"Invalid augmentation_level: {augmentation_level}")

        self.logger.info(f"Created {augmentation_level} augmentation transforms")
        return transform

    def get_validation_transforms(self, image_size: int = 96) -> transforms.Compose:
        """
        Get transforms for validation/test data (no augmentation).

        Args:
            image_size: Target image size.

        Returns:
            Composed transforms.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform

    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            dataset: PyTorch dataset to split.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        self.logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

        return train_dataset, val_dataset, test_dataset

    def validate_dataset(self, dataset: Dataset) -> Dict[str, any]:
        """
        Validate dataset quality and report statistics.

        Args:
            dataset: PyTorch dataset to validate.

        Returns:
            Dictionary containing validation results.
        """
        self.logger.info("Validating dataset...")

        validation_results = {
            'total_samples': len(dataset),
            'valid_samples': 0,
            'corrupted_samples': [],
            'label_distribution': {},
            'image_shapes': []
        }

        for idx in range(len(dataset)):
            try:
                image, label = dataset[idx]

                # Check if image is valid
                if isinstance(image, torch.Tensor):
                    validation_results['image_shapes'].append(image.shape)
                    validation_results['valid_samples'] += 1

                    # Track label distribution
                    label_key = int(label) if isinstance(label, (int, torch.Tensor)) else str(label)
                    validation_results['label_distribution'][label_key] = \
                        validation_results['label_distribution'].get(label_key, 0) + 1

            except Exception as e:
                validation_results['corrupted_samples'].append((idx, str(e)))
                self.logger.warning(f"Corrupted sample at index {idx}: {str(e)}")

        # Calculate validation summary
        validation_results['corruption_rate'] = \
            len(validation_results['corrupted_samples']) / validation_results['total_samples']

        self.logger.info(f"Validation complete: {validation_results['valid_samples']}/{validation_results['total_samples']} valid samples")
        self.logger.info(f"Label distribution: {validation_results['label_distribution']}")

        if validation_results['corrupted_samples']:
            self.logger.warning(f"Found {len(validation_results['corrupted_samples'])} corrupted samples")

        return validation_results

    def create_weighted_sampler(self, dataset: Dataset) -> torch.utils.data.WeightedRandomSampler:
        """
        Create a weighted sampler for imbalanced datasets.

        Args:
            dataset: PyTorch dataset.

        Returns:
            WeightedRandomSampler instance.
        """
        # Get all labels
        labels = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            labels.append(int(label))

        # Calculate class weights
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        # Create sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        self.logger.info(f"Created weighted sampler with class weights: {class_weights}")

        return sampler

    def check_class_balance(self, dataset: Dataset) -> Dict[int, float]:
        """
        Check class balance in dataset.

        Args:
            dataset: PyTorch dataset.

        Returns:
            Dictionary mapping class labels to their proportions.
        """
        labels = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            labels.append(int(label))

        unique, counts = np.unique(labels, return_counts=True)
        proportions = counts / len(labels)

        balance = dict(zip(unique, proportions))

        self.logger.info("Class balance:")
        for class_id, proportion in balance.items():
            self.logger.info(f"  Class {class_id}: {proportion:.2%} ({counts[class_id]} samples)")

        return balance
