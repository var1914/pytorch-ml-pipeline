"""
Setup script for CV Pipeline.

Install with:
    pip install -e .

Then use:
    cv-pipeline train --data ./my_images --model resnet50
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="cv-pipeline",
    version="1.0.0",
    description="A practical toolkit for computer vision research and production",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ML Pipeline Team",
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "albumentations>=1.3.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "detection": [
            "ultralytics>=8.0.0",
            "pycocotools>=2.0.0",
        ],
        "segmentation": [
            "segmentation-models-pytorch>=0.3.0",
            "opencv-python>=4.8.0",
        ],
        "mlops": [
            "mlflow>=2.0.0",
            "minio>=7.0.0",
        ],
        "full": [
            "ultralytics>=8.0.0",
            "pycocotools>=2.0.0",
            "segmentation-models-pytorch>=0.3.0",
            "opencv-python>=4.8.0",
            "mlflow>=2.0.0",
            "minio>=7.0.0",
            "pydantic>=2.0.0",
            "omegaconf>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-pipeline=cv_pipeline.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="deep-learning computer-vision pytorch classification detection segmentation",
)
