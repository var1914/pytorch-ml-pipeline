"""
CV Pipeline - A Practical Toolkit for Computer Vision Research

Quick Start:
    # Train a classifier on your images
    cv-pipeline train --data ./my_images --model resnet50

    # Analyze your dataset
    from cv_pipeline import analyze_dataset
    analyze_dataset("./my_images")

    # Compare models
    from cv_pipeline import compare_models
    compare_models(["resnet50", "efficientnet_b0"], test_loader)

    # Generate experiment notebook
    from cv_pipeline import generate_notebook
    generate_notebook("classification", "my_experiment.ipynb")

Installation:
    pip install -e .
"""

from .utils import (
    analyze_dataset,
    compare_models,
    export_model,
    plot_results,
    load_image_folder,
    quick_train,
    get_model,
)
from .notebook_generator import generate_notebook

__version__ = "1.0.0"

__all__ = [
    # Data utilities
    "analyze_dataset",
    "load_image_folder",
    # Model utilities
    "get_model",
    "compare_models",
    "export_model",
    # Training
    "quick_train",
    "plot_results",
    # Notebook
    "generate_notebook",
]
