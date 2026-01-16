"""
CV Pipeline Utilities - Tools ML Researchers Actually Need

Quick, practical utilities for common CV tasks:
- Dataset analysis
- Model comparison
- Quick training
- Export for deployment
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Dataset Utilities
# =============================================================================

def analyze_dataset(
    data_path: str,
    show_samples: bool = True,
    save_report: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze an image dataset and get useful statistics.

    Works with folder structure:
        data_path/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg

    Args:
        data_path: Path to image folder
        show_samples: Whether to display sample images
        save_report: Optional path to save JSON report

    Returns:
        Dictionary with dataset statistics

    Example:
        stats = analyze_dataset("./my_images")
        print(f"Classes: {stats['classes']}")
        print(f"Total images: {stats['total_images']}")
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Path not found: {data_path}")

    # Collect statistics
    stats = {
        "path": str(data_path),
        "classes": [],
        "class_distribution": {},  # Also accessible as class_counts
        "total_images": 0,
        "image_sizes": [],
        "formats": {},  # File format counts
        "issues": []
    }

    # Analyze each class folder
    class_folders = [f for f in data_path.iterdir() if f.is_dir()]

    if not class_folders:
        # Check if images are directly in the folder (no class structure)
        image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
        if image_files:
            stats["issues"].append("No class folders found. Images are in root directory.")
            stats["total_images"] = len(image_files)
            return stats
        else:
            raise ValueError(f"No class folders or images found in {data_path}")

    stats["classes"] = sorted([f.name for f in class_folders])
    stats["num_classes"] = len(stats["classes"])

    print(f"\n📊 Analyzing dataset: {data_path}")
    print(f"   Found {len(class_folders)} classes\n")

    sizes_sample = []
    format_counter = Counter()

    for class_folder in tqdm(class_folders, desc="Scanning classes"):
        class_name = class_folder.name
        images = list(class_folder.glob("*"))
        image_files = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']]

        stats["class_distribution"][class_name] = len(image_files)
        stats["total_images"] += len(image_files)

        # Sample image sizes (first 10 per class)
        for img_path in image_files[:10]:
            try:
                with Image.open(img_path) as img:
                    sizes_sample.append(img.size)
                    format_counter[img_path.suffix.lower()] += 1
            except Exception as e:
                stats["issues"].append(f"Cannot read: {img_path} - {e}")

    # Convert Counter to dict for JSON serialization
    stats["formats"] = dict(format_counter)

    # Analyze image sizes
    if sizes_sample:
        widths = [s[0] for s in sizes_sample]
        heights = [s[1] for s in sizes_sample]
        stats["image_sizes"] = {
            "min_width": min(widths),
            "max_width": max(widths),
            "avg_width": int(np.mean(widths)),
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_height": int(np.mean(heights)),
        }

    # Check for class imbalance
    counts = list(stats["class_distribution"].values())
    if counts:
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        stats["imbalance_ratio"] = round(imbalance_ratio, 2)
        if imbalance_ratio > 10:
            stats["issues"].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}x)")
        elif imbalance_ratio > 3:
            stats["issues"].append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}x)")

    # Print summary
    _print_dataset_summary(stats)

    # Show sample images
    if show_samples:
        _show_sample_images(data_path, stats["classes"][:4])

    # Save report
    if save_report:
        with open(save_report, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {save_report}")

    return stats


def _print_dataset_summary(stats: Dict[str, Any]) -> None:
    """Print formatted dataset summary."""
    print("\n" + "=" * 50)
    print("📊 DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total images:  {stats['total_images']:,}")
    print(f"  Classes:       {stats.get('num_classes', len(stats['classes']))}")

    if stats.get("image_sizes"):
        sizes = stats["image_sizes"]
        print(f"  Image sizes:   {sizes['avg_width']}x{sizes['avg_height']} (avg)")

    class_dist = stats.get("class_distribution", {})
    if class_dist:
        print(f"\n  Class distribution:")
        max_count = max(class_dist.values()) if class_dist.values() else 1
        for cls, count in sorted(class_dist.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * min(int(count / max_count * 20), 20)
            print(f"    {cls:<20} {count:>6}  {bar}")

        if len(class_dist) > 10:
            print(f"    ... and {len(class_dist) - 10} more classes")

    if stats.get("issues"):
        print(f"\n  ⚠️  Issues found:")
        for issue in stats["issues"]:
            print(f"    - {issue}")

    print("=" * 50 + "\n")


def _show_sample_images(data_path: Path, classes: List[str], n_per_class: int = 2) -> None:
    """Display sample images from each class."""
    try:
        import matplotlib.pyplot as plt

        n_classes = min(len(classes), 4)
        fig, axes = plt.subplots(n_classes, n_per_class, figsize=(8, 2 * n_classes))

        if n_classes == 1:
            axes = [axes]

        for i, cls in enumerate(classes[:n_classes]):
            class_path = data_path / cls
            images = list(class_path.glob("*"))[:n_per_class]

            for j, img_path in enumerate(images):
                ax = axes[i][j] if n_per_class > 1 else axes[i]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(f"{cls}", fontsize=10)
                    ax.axis('off')
                except:
                    ax.text(0.5, 0.5, "Error", ha='center')
                    ax.axis('off')

        plt.suptitle("Sample Images", fontsize=12)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Install matplotlib to view sample images: pip install matplotlib")


def load_image_folder(
    data_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Load images from a folder structure into train/val DataLoaders.

    Folder structure:
        data_path/
            class_a/
            class_b/

    Args:
        data_path: Path to image folder
        image_size: Resize images to this size
        batch_size: Batch size for DataLoader
        augment: Apply data augmentation to training set
        split: Validation split ratio
        seed: Random seed for reproducibility

    Returns:
        (train_loader, val_loader, class_names)

    Example:
        train_loader, val_loader, classes = load_image_folder("./my_images")
        print(f"Classes: {classes}")
    """
    # Define transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
    class_names = full_dataset.classes

    # Split into train/val
    n_samples = len(full_dataset)
    n_val = int(n_samples * split)
    n_train = n_samples - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Update val transform (no augmentation)
    val_dataset.dataset = datasets.ImageFolder(data_path, transform=val_transform)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"✓ Loaded {n_train} training, {n_val} validation images")
    print(f"  Classes: {class_names}")

    return train_loader, val_loader, class_names


# =============================================================================
# Model Utilities
# =============================================================================

def get_model(
    name: str = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Get a model by name - simple one-liner.

    Args:
        name: Model architecture (resnet50, efficientnet_b0, vit_base, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights

    Returns:
        PyTorch model

    Example:
        model = get_model("resnet50", num_classes=10)
        model = get_model("efficientnet_b3", num_classes=100)
    """
    try:
        import timm
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        print(f"✓ Loaded {name} with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    except ImportError:
        raise ImportError("Install timm: pip install timm")


def compare_models(
    models: Union[List[str], List[nn.Module]],
    test_loader: DataLoader,
    num_classes: int = None,
    device: str = "auto"
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test set.

    Args:
        models: List of model names (str) or model instances
        test_loader: Test DataLoader
        num_classes: Number of classes (required if passing model names)
        device: Device to use (auto, cuda, cpu)

    Returns:
        Dictionary mapping model names to metrics

    Example:
        results = compare_models(
            ["resnet50", "efficientnet_b0", "vit_base"],
            test_loader,
            num_classes=10
        )
        print(results)
    """
    from sklearn.metrics import accuracy_score, f1_score

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    results = {}

    for model_item in models:
        # Load model if string
        if isinstance(model_item, str):
            if num_classes is None:
                raise ValueError("num_classes required when passing model names")
            model_name = model_item
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
        else:
            model = model_item
            model_name = model.__class__.__name__

        model = model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        print(f"\n🔍 Evaluating {model_name}...")

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        num_params = sum(p.numel() for p in model.parameters())

        results[model_name] = {
            "accuracy": accuracy,  # 0-1 range
            "f1_score": f1,        # 0-1 range
            "params": num_params,  # For compatibility
            "num_params": num_params
        }

        print(f"   Accuracy: {accuracy*100:.2f}%  |  F1: {f1*100:.2f}%")

    # Print comparison table
    _print_comparison_table(results)

    return results


def _print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'Params':>12}")
    print("-" * 60)

    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["accuracy"])

    for model_name, metrics in sorted_results:
        params_str = f"{metrics['params']/1e6:.1f}M"
        acc_pct = metrics['accuracy'] * 100
        f1_pct = metrics['f1_score'] * 100
        print(f"{model_name:<25} {acc_pct:>9.2f}% {f1_pct:>9.2f}% {params_str:>12}")

    print("=" * 60)
    best_model = sorted_results[0][0]
    print(f"🏆 Best: {best_model}")
    print()


def export_model(
    model: nn.Module,
    output_path: str,
    format: str = "torchscript",
    input_size: Tuple[int, int] = (224, 224)
) -> str:
    """
    Export model for deployment.

    Args:
        model: PyTorch model
        output_path: Where to save the exported model
        format: Export format (torchscript, onnx, state_dict)
        input_size: Input image size for tracing

    Returns:
        Path to exported model

    Example:
        export_model(model, "./model.pt", format="torchscript")
        export_model(model, "./model.onnx", format="onnx")
    """
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "torchscript":
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(output_path))
        print(f"✓ Saved TorchScript model to {output_path}")

    elif format == "onnx":
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"✓ Saved ONNX model to {output_path}")

    elif format == "state_dict":
        torch.save(model.state_dict(), output_path)
        print(f"✓ Saved state dict to {output_path}")

    else:
        raise ValueError(f"Unknown format: {format}. Use torchscript, onnx, or state_dict")

    return str(output_path)


# =============================================================================
# Training Utilities
# =============================================================================

def quick_train(
    data_path: str,
    model: Union[str, nn.Module] = "resnet50",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    image_size: Tuple[int, int] = (224, 224),
    device: str = "auto",
    save_path: Optional[str] = None,
    pretrained: bool = True,
    augment: bool = True,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Quick training on an image folder - one function does everything.

    Args:
        data_path: Path to image folder (with class subfolders)
        model: Model name (str) or model instance
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        image_size: Input image size
        device: Device (auto, cuda, cpu)
        save_path: Optional path to save best model
        pretrained: Use pretrained weights
        augment: Apply data augmentation
        verbose: Print training progress

    Returns:
        (trained_model, training_history)

    Example:
        model, history = quick_train(
            "./my_images",
            model="resnet50",
            epochs=10
        )
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    if verbose:
        print(f"🖥️  Using device: {device}")

    # Load data
    if verbose:
        print(f"\n📂 Loading data from {data_path}")
    train_loader, val_loader, class_names = load_image_folder(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        augment=augment
    )
    num_classes = len(class_names)

    # Get model
    if isinstance(model, str):
        model = get_model(model, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    if verbose:
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print("-" * 50)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}: Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_names': class_names,
                    'val_acc': val_acc
                }, save_path)

    if verbose:
        print("-" * 50)
        print(f"✅ Training complete! Best Val Acc: {best_val_acc*100:.2f}%")

        if save_path:
            print(f"💾 Model saved to {save_path}")

    return model, history


def plot_results(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    style: str = "default"
) -> None:
    """
    Plot training results.

    Args:
        history: Training history dict with train_loss, val_loss (and optionally train_acc, val_acc)
        save_path: Optional path to save figure
        style: Plot style (default, paper, dark)

    Example:
        model, history = quick_train("./data", epochs=10)
        plot_results(history, save_path="results.png")
    """
    import matplotlib.pyplot as plt

    if style == "paper":
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            pass
    elif style == "dark":
        try:
            plt.style.use('dark_background')
        except:
            pass

    # Determine what to plot
    has_loss = "train_loss" in history and "val_loss" in history
    has_acc = "train_acc" in history or "val_acc" in history

    if has_loss and has_acc:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = None

    # Get epochs from available data
    if "train_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
    elif "val_loss" in history:
        epochs = range(1, len(history["val_loss"]) + 1)
    else:
        epochs = range(1, len(list(history.values())[0]) + 1)

    # Loss plot
    if has_loss:
        ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc and ax2 is not None:
        if "train_acc" in history:
            ax2.plot(epochs, [a*100 for a in history["train_acc"]], 'b-', label='Train Acc', linewidth=2)
        if "val_acc" in history:
            ax2.plot(epochs, [a*100 for a in history["val_acc"]], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")

    plt.close(fig)  # Close to avoid display in non-interactive environments
