"""
Pytest fixtures for CV Pipeline tests.

Creates temporary directories with sample images for testing.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def sample_image_folder(tmp_path_factory):
    """
    Create a temporary folder with sample images organized by class.

    Structure:
        temp_dir/
        ├── class_0/
        │   ├── img_0.jpg
        │   ├── img_1.jpg
        │   └── img_2.jpg
        └── class_1/
            ├── img_0.jpg
            ├── img_1.jpg
            └── img_2.jpg
    """
    base_dir = tmp_path_factory.mktemp("sample_images")

    # Create 2 classes with 5 images each
    for class_idx in range(2):
        class_dir = base_dir / f"class_{class_idx}"
        class_dir.mkdir()

        for img_idx in range(5):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"img_{img_idx}.jpg")

    yield str(base_dir)


@pytest.fixture(scope="session")
def sample_image_folder_imbalanced(tmp_path_factory):
    """Create a folder with imbalanced classes for testing."""
    base_dir = tmp_path_factory.mktemp("imbalanced_images")

    # Class 0: 10 images, Class 1: 2 images (5:1 ratio)
    class_sizes = [10, 2]

    for class_idx, num_images in enumerate(class_sizes):
        class_dir = base_dir / f"class_{class_idx}"
        class_dir.mkdir()

        for img_idx in range(num_images):
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"img_{img_idx}.jpg")

    yield str(base_dir)


@pytest.fixture(scope="session")
def sample_image_folder_multiclass(tmp_path_factory):
    """Create a folder with multiple classes."""
    base_dir = tmp_path_factory.mktemp("multiclass_images")

    # 5 classes with 3 images each
    for class_idx in range(5):
        class_dir = base_dir / f"category_{class_idx}"
        class_dir.mkdir()

        for img_idx in range(3):
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"sample_{img_idx}.png")

    yield str(base_dir)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    yield str(tmp_path)


@pytest.fixture(scope="session")
def device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
