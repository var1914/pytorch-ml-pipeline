from typing import Any, Callable, Dict, Optional, Tuple, Type
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim

import h5py
import numpy as np
import matplotlib.pyplot as plt # For visualization
from collections import Counter
import logging
import os
import sys
import pandas as pd

from ..minio.minio_init import MinIO
from minio.error import S3Error

class DataDownloader():
    
    """
    A utility class for loading, analyzing, and visualizing PyTorch datasets.
    
    This class provides a unified interface for working with PyTorch datasets,
    including loading data, computing statistics, and visualizing samples.
    """

    def __init__(self, default_transform: Optional[Callable] = None):
        """
        Initialize the DatasetLoader.
        
        Args:
            default_transform: Default transform to apply if none is specified.
                             Defaults to transforms.ToTensor() if None.
        """
        self.dataset: Optional[Dataset] = None
        self.default_transform = default_transform or transforms.ToTensor()
        self.logger = self._setup_logger()

        # Setup MinIO
        self.minio_config = {
            'endpoint': 'minio:9000',
            'access_key': 'admin',
            'secret_key': 'admin123',
            'secure': False,
            'bucket_name': 'dataset'
        }
        self.minio = MinIO(self.minio_config)

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
            root_path: str, 
            split: str, 
            download: bool = True, 
            transform: Optional[Callable] = None, 
            **kwargs: Any
        ) -> Dataset:

        """
        Load a PyTorch dataset with the specified configuration.

        Args:
            dataset_class: The PyTorch dataset class (e.g., MNIST, CIFAR10).
            root_path: Root directory where the dataset will be stored.
            split: Dataset split to load (e.g., 'train', 'test', 'val').
            download: Whether to download the dataset if not found locally.
            transform: Transform to apply to the data. Uses default if None.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            The loaded PyTorch dataset instance.
            
        Raises:
            ValueError: If the dataset cannot be loaded with the given parameters.
        """
        
        transform = transform or self.default_transform
        try:
            self.dataset = dataset_class(
                root=root_path,
                split=split,
                download=download,
                transform=transform,
                **kwargs
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}") from e
        
        return self.dataset

    def convert_to_parquet_batches(self, 
                                   dataloader: Type[DataLoader], 
                                   output_dir: str,
                                   bucket_name: str
                                   ):
        """
        Iterates through a PyTorch DataLoader and saves each batch as a separate Parquet file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created directory: {output_dir}")
        try:
            for i, batch in enumerate(dataloader):
                # PCAM typically returns (images, labels)
                images, labels = batch
            
                # Convert the batch to a dictionary for pandas
                batch_data = {
                    'image': list(images.numpy()),
                    'label': labels.numpy()
                }

                df = pd.DataFrame(batch_data)
        
                # Define the output filename for the current batch
                filename = os.path.join(output_dir, f'batch_{i+1:05d}.parquet')
        
                # Save the pandas DataFrame to a Parquet file using the 'pyarrow' engine
                df.to_parquet(filename, engine='pyarrow', index=False)

                try:
                    bucket_name = self.minio_config['bucket_name']
                    self.minio.client.fput_object(
                        bucket_name,
                        os.path.basename(filename),
                        filename,
                        # You can add metadata, content_type, etc. here
                    )
                    self.logger.info(f"Uploaded {filename}")
                except S3Error as e:
                    self.logger.error(f"Error uploading {filename}: {e}")

                if (i + 1) % 100 == 0:
                    self.logger.info(f"Saved {i+1} batches to disk...")

        except Exception as e:
                self.logger.error(f"Error Converting to Parquet {str(e)}")

        self.logger.info(f"\nConversion complete. Total batches saved: {i+1} files in '{output_dir}'.")

    def get_data_stats(self):
        """
        Compute statistics for the loaded dataset.

        Returns:
            A tuple containing:
                - dataset_size: Total number of samples in the dataset.
                - label_distribution: Dictionary mapping labels to their counts.
                
        Raises:
            RuntimeError: If no dataset has been loaded.
        """

        if self.dataset is None:
            raise RuntimeError("No Dataset loaded, call load_data() first")
        
        dataset_size = len(self.dataset)
        dataset_labels = [self.dataset[i][1] for i in range(dataset_size)]
        label_distribution = dict(Counter(dataset_labels))

        return dataset_size, label_distribution


    def visualize_samples(
            self, 
            num_samples: int, 
            nrows: int, 
            ncols: int, 
            figsize: Tuple[int, int] = (12,8),
            cmap: Optional[str] = None
        ) -> None:
        """
        Visualize a grid of sample images from the dataset.

        Args:
            num_samples: Number of samples to visualize.
            nrows: Number of rows in the grid.
            ncols: Number of columns in the grid.
            figsize: Figure size as (width, height) in inches.
            cmap: Colormap for grayscale images (e.g., 'gray').
            
        Raises:
            RuntimeError: If no dataset has been loaded.
            ValueError: If num_samples exceeds nrows * ncols.
        """
        if self.dataset is None:
            raise RuntimeError("No Dataset loaded, call load_data() first")
        
        if num_samples > nrows * ncols:
            raise ValueError(
                f"num_samples ({num_samples}) exceeds grid capacity "
                f"({nrows * ncols})"
            )
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel() if nrows * ncols > 1 else [axes]

        for idx in range(num_samples):
            image, label = self.dataset[idx]


            # Convert tensor to numpy array if needed
            if hasattr(image, 'numpy'):
                image = image.numpy()
                # Handle channel-first format (C, H, W) -> (H, W, C)
                if image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
                # Squeeze single channel
                if image.shape[-1] == 1:
                    image = image.squeeze(-1)
            
            axes[idx].imshow(image)
            axes[idx].set_title(f"Label: {label}", fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
