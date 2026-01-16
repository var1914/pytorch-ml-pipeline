"""
Data Loading Module for CV Pipeline.

Provides a unified interface for loading any PyTorch dataset with:
- Configurable transforms
- MinIO integration for data storage
- Parquet conversion for batch processing
- Statistics and visualization utilities
"""

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from minio.error import S3Error

from ..config.settings import DataConfig, InfraConfig
from ..minio.minio_init import MinIO


class DataDownloader:
    """
    A generic utility class for loading, analyzing, and visualizing PyTorch datasets.

    Works with any PyTorch dataset including torchvision.datasets (CIFAR-10, ImageNet,
    MNIST, PCAM, etc.) and custom datasets implementing the Dataset interface.

    This class provides a unified interface for:
    - Loading data from any PyTorch dataset
    - Computing statistics and label distributions
    - Visualizing sample images
    - Converting batches to parquet format for storage
    - Uploading data to MinIO object storage

    Example with config:
        from src.config import DataConfig, InfraConfig

        data_config = DataConfig(dataset_name="pcam", data_root="./data")
        infra_config = InfraConfig(minio_endpoint="localhost:9000")

        downloader = DataDownloader(data_config=data_config, infra_config=infra_config)
        dataset = downloader.load_data(datasets.PCAM, split="train")

    Example without config (legacy):
        downloader = DataDownloader()
        dataset = downloader.load_data(datasets.PCAM, root_path="./data", split="train")
    """

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        infra_config: Optional[InfraConfig] = None,
        default_transform: Optional[Callable] = None
    ):
        """
        Initialize the DataDownloader.

        Args:
            data_config: Data configuration with dataset settings
            infra_config: Infrastructure configuration with MinIO settings
            default_transform: Default transform to apply if none is specified
        """
        self.data_config = data_config
        self.infra_config = infra_config
        self.dataset: Optional[Dataset] = None
        self.default_transform = default_transform
        self.logger = self._setup_logger()

        # Setup MinIO if config provided
        self._minio_client: Optional[MinIO] = None
        if infra_config:
            self._setup_minio()
        else:
            # Legacy default config
            self.minio_config = {
                'endpoint': 'minio:9000',
                'access_key': 'admin',
                'secret_key': 'admin123',
                'secure': False,
                'bucket_name': 'dataset'
            }
            try:
                self._minio_client = MinIO(self.minio_config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize MinIO: {e}")

    def _setup_minio(self) -> None:
        """Setup MinIO client from infra config."""
        if not self.infra_config:
            return

        self.minio_config = {
            'endpoint': self.infra_config.minio_endpoint,
            'access_key': self.infra_config.minio_access_key,
            'secret_key': self.infra_config.minio_secret_key,
            'secure': self.infra_config.minio_secure,
            'bucket_name': self.infra_config.minio_bucket
        }
        try:
            self._minio_client = MinIO(self.minio_config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize MinIO: {e}")

    @property
    def minio(self) -> Optional[MinIO]:
        """Get MinIO client (lazy initialization for backward compatibility)."""
        return self._minio_client

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for the data downloader."""
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

    def load_data(
        self,
        dataset_class: Type[Dataset],
        root_path: Optional[str] = None,
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        **kwargs: Any
    ) -> Dataset:
        """
        Load a PyTorch dataset with the specified configuration.

        Args:
            dataset_class: The PyTorch dataset class (e.g., MNIST, CIFAR10)
            root_path: Root directory for dataset storage (uses config if None)
            split: Dataset split to load ('train', 'test', 'val')
            download: Whether to download the dataset if not found
            transform: Transform to apply to the data
            **kwargs: Additional arguments passed to the dataset constructor

        Returns:
            The loaded PyTorch dataset instance

        Raises:
            ValueError: If the dataset cannot be loaded with the given parameters
        """
        # Use config values if available
        if root_path is None:
            root_path = self.data_config.data_root if self.data_config else "./data"

        # Use configured transform or default
        if transform is None:
            transform = self._get_default_transform()

        try:
            self.dataset = dataset_class(
                root=root_path,
                split=split,
                download=download,
                transform=transform,
                **kwargs
            )
            self.logger.info(
                f"Loaded {dataset_class.__name__} ({split}) with {len(self.dataset)} samples"
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}") from e

        return self.dataset

    def _get_default_transform(self) -> Callable:
        """Get default transform based on config."""
        from torchvision import transforms

        if self.default_transform:
            return self.default_transform

        if self.data_config:
            return transforms.Compose([
                transforms.Resize(self.data_config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.data_config.normalization_mean,
                    std=self.data_config.normalization_std
                )
            ])

        return transforms.ToTensor()

    def create_dataloader(
        self,
        dataset: Optional[Dataset] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for the dataset.

        Args:
            dataset: Dataset to create loader for (uses self.dataset if None)
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader instance
        """
        dataset = dataset or self.dataset
        if dataset is None:
            raise RuntimeError("No dataset available. Call load_data() first.")

        # Use config values or defaults
        if self.data_config:
            batch_size = batch_size or 32
            num_workers = num_workers if num_workers is not None else self.data_config.num_workers
            pin_memory = pin_memory if pin_memory is not None else self.data_config.pin_memory
        else:
            batch_size = batch_size or 32
            num_workers = num_workers if num_workers is not None else 4
            pin_memory = pin_memory if pin_memory is not None else True

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

    def convert_to_parquet_batches(
        self,
        dataloader: DataLoader,
        output_dir: str,
        bucket_name: Optional[str] = None,
        upload_to_minio: bool = True
    ) -> int:
        """
        Convert DataLoader batches to Parquet files and optionally upload to MinIO.

        Args:
            dataloader: DataLoader to convert
            output_dir: Directory to save Parquet files
            bucket_name: MinIO bucket name (uses config default if None)
            upload_to_minio: Whether to upload files to MinIO

        Returns:
            Number of batches converted
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Saving batches to: {output_dir}")

        bucket_name = bucket_name or self.minio_config.get('bucket_name', 'dataset')
        batch_count = 0

        try:
            for i, batch in enumerate(dataloader):
                images, labels = batch

                # Convert to dictionary for pandas
                batch_data = {
                    'image': list(images.numpy()),
                    'label': labels.numpy()
                }

                df = pd.DataFrame(batch_data)
                filename = os.path.join(output_dir, f'batch_{i+1:05d}.parquet')
                df.to_parquet(filename, engine='pyarrow', index=False)

                # Upload to MinIO
                if upload_to_minio and self._minio_client:
                    try:
                        self._minio_client.client.fput_object(
                            bucket_name,
                            os.path.basename(filename),
                            filename
                        )
                    except S3Error as e:
                        self.logger.error(f"Error uploading {filename}: {e}")

                batch_count = i + 1
                if batch_count % 100 == 0:
                    self.logger.info(f"Saved {batch_count} batches...")

        except Exception as e:
            self.logger.error(f"Error converting to Parquet: {str(e)}")

        self.logger.info(f"Conversion complete. Total batches: {batch_count}")
        return batch_count

    def get_data_stats(self) -> Tuple[int, Dict[int, int]]:
        """
        Compute statistics for the loaded dataset.

        Returns:
            Tuple of (dataset_size, label_distribution)

        Raises:
            RuntimeError: If no dataset has been loaded
        """
        if self.dataset is None:
            raise RuntimeError("No dataset loaded. Call load_data() first.")

        dataset_size = len(self.dataset)
        dataset_labels = [self.dataset[i][1] for i in range(dataset_size)]
        label_distribution = dict(Counter(dataset_labels))

        return dataset_size, label_distribution

    def visualize_samples(
        self,
        num_samples: int = 16,
        nrows: int = 4,
        ncols: int = 4,
        figsize: Tuple[int, int] = (12, 8),
        cmap: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize a grid of sample images from the dataset.

        Args:
            num_samples: Number of samples to visualize
            nrows: Number of rows in the grid
            ncols: Number of columns in the grid
            figsize: Figure size as (width, height)
            cmap: Colormap for grayscale images
            save_path: Optional path to save the figure

        Raises:
            RuntimeError: If no dataset has been loaded
        """
        import matplotlib.pyplot as plt

        if self.dataset is None:
            raise RuntimeError("No dataset loaded. Call load_data() first.")

        if num_samples > nrows * ncols:
            num_samples = nrows * ncols
            self.logger.warning(f"Reduced num_samples to {num_samples}")

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel() if nrows * ncols > 1 else [axes]

        for idx in range(num_samples):
            image, label = self.dataset[idx]

            # Convert tensor to numpy
            if hasattr(image, 'numpy'):
                image = image.numpy()
                # Handle channel-first format (C, H, W) -> (H, W, C)
                if image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[-1] == 1:
                    image = image.squeeze(-1)

            axes[idx].imshow(image, cmap=cmap)
            axes[idx].set_title(f"Label: {label}", fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            self.logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Split the dataset into train/val/test subsets.

        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_subset, val_subset, test_subset)
        """
        if self.dataset is None:
            raise RuntimeError("No dataset loaded. Call load_data() first.")

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n_samples = len(self.dataset)

        # Generate indices
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_samples, generator=generator).tolist()

        # Compute split points
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        # Create subsets
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        test_subset = Subset(self.dataset, test_indices)

        self.logger.info(
            f"Split dataset: train={len(train_subset)}, "
            f"val={len(val_subset)}, test={len(test_subset)}"
        )

        return train_subset, val_subset, test_subset


# Alias for backward compatibility
DatasetLoader = DataDownloader
