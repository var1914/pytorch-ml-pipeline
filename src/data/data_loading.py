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
        dataset_labels = [self.dataset[i][1] for i in range(len(dataset_size))]
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
            image, label = self.train_dataset[idx]


            axes[idx].imshow(np.array(image))
            axes[idx].set_title(f"Label: {label}", fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
