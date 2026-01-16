#!/usr/bin/env python3
"""
Medical Imaging Training Script

Train classification models for medical imaging tasks:
- Cancer detection (PCAM histopathology)
- Chest X-ray classification
- Skin lesion analysis (ISIC)

Usage:
    python train.py --config configs/templates/medical_imaging.yaml
    python train.py --dataset pcam --epochs 30 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torchvision import datasets, transforms

from src.config import load_config, get_config, CVPipelineConfig
from src.core import ModelFactory, TrainerFactory, EvaluatorFactory
from src.data import DataDownloader


def get_dataset_class(dataset_name: str):
    """Get the appropriate dataset class."""
    dataset_map = {
        'pcam': datasets.PCAM,
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
    }
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    return dataset_map[dataset_name.lower()]


def main():
    parser = argparse.ArgumentParser(description="Medical Imaging Training")
    parser.add_argument('--config', type=str, help="Path to config YAML file")
    parser.add_argument('--dataset', type=str, default='pcam', help="Dataset name")
    parser.add_argument('--data-root', type=str, default='./data/medical', help="Data root directory")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--model', type=str, default='resnet50', help="Model architecture")
    parser.add_argument('--num-classes', type=int, default=2, help="Number of classes")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config(
            name="medical_imaging_training",
            data={
                "dataset_name": args.dataset,
                "data_root": args.data_root,
                "image_size": (96, 96) if args.dataset == 'pcam' else (224, 224),
                "augmentation_level": "medium"
            },
            model={
                "task_type": "classification",
                "architecture": args.model,
                "num_classes": args.num_classes,
                "pretrained": True
            },
            training={
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "learning_rate": args.lr,
                "early_stopping_patience": 7
            }
        )

    print(f"Training medical imaging model: {config.model.architecture}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Epochs: {config.training.num_epochs}")

    # Setup data
    data_downloader = DataDownloader(
        data_config=config.data,
        infra_config=config.infra
    )

    # Custom transforms for medical imaging
    train_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.data.normalization_mean,
            std=config.data.normalization_std
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.data.normalization_mean,
            std=config.data.normalization_std
        )
    ])

    # Load dataset
    dataset_class = get_dataset_class(config.data.dataset_name)

    train_dataset = data_downloader.load_data(
        dataset_class,
        root_path=config.data.data_root,
        split='train',
        transform=train_transform
    )

    val_dataset = dataset_class(
        root=config.data.data_root,
        split='val' if config.data.dataset_name == 'pcam' else 'test',
        download=True,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = data_downloader.create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )

    val_loader = data_downloader.create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = ModelFactory.create(
        task_type=config.model.task_type,
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout
    )

    print(f"Model parameters: {model.get_num_params():,}")

    # Create trainer
    trainer = TrainerFactory.create(
        task_type=config.model.task_type,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        infra_config=config.infra,
        num_classes=config.model.num_classes,
        experiment_name=config.get_experiment_name()
    )

    # Train
    print("\nStarting training...")
    history = trainer.train()

    # Evaluate
    print("\nRunning final evaluation...")
    evaluator = EvaluatorFactory.create(
        task_type=config.model.task_type,
        model=model,
        test_loader=val_loader,
        num_classes=config.model.num_classes,
        class_names=['Normal', 'Tumor'] if config.model.num_classes == 2 else None
    )

    metrics = evaluator.evaluate()
    evaluator.generate_visualizations(output_dir='./results/medical_imaging')

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
