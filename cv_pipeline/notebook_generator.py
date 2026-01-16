"""
Notebook Generator - Create Jupyter notebooks for CV experiments.

Generates ready-to-run notebooks with best practices built in.
"""

import json
from pathlib import Path
from typing import Optional


def _create_cell(cell_type: str, source: str, metadata: dict = None) -> dict:
    """Create a notebook cell."""
    return {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source.split("\n"),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def generate_classification_notebook(
    data_path: Optional[str] = None,
    model: str = "resnet50",
) -> list:
    """Generate cells for a classification notebook."""
    data_path = data_path or "./data/images"

    cells = [
        _create_cell("markdown", f"""# Image Classification Experiment

This notebook provides a complete workflow for training and evaluating image classifiers.

**Quick links:**
- [1. Setup](#1.-Setup)
- [2. Data Analysis](#2.-Data-Analysis)
- [3. Model Training](#3.-Model-Training)
- [4. Evaluation](#4.-Evaluation)
- [5. Export](#5.-Export)
"""),

        _create_cell("markdown", "## 1. Setup"),

        _create_cell("code", """# Install dependencies (if needed)
# !pip install torch torchvision timm albumentations matplotlib

import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import cv_pipeline utilities
from cv_pipeline import (
    analyze_dataset,
    load_image_folder,
    get_model,
    quick_train,
    compare_models,
    export_model,
    plot_results,
)

# Check device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
"""),

        _create_cell("markdown", "## 2. Data Analysis\n\nLet's understand our dataset before training."),

        _create_cell("code", f"""# Set your data path
DATA_PATH = "{data_path}"

# Analyze the dataset
stats = analyze_dataset(DATA_PATH, show_samples=True)

print(f"Total images: {{stats['total_images']}}")
print(f"Number of classes: {{stats['num_classes']}}")
print(f"\\nClass distribution:")
for cls, count in stats.get('class_distribution', {{}}).items():
    print(f"  {{cls}}: {{count}}")
"""),

        _create_cell("code", """# Load data into DataLoaders
train_loader, val_loader, class_names = load_image_folder(
    DATA_PATH,
    image_size=(224, 224),
    batch_size=32,
    augment=True,  # Enable augmentation for training
    split=0.2,     # 80% train, 20% validation
)

print(f"Classes: {class_names}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
"""),

        _create_cell("markdown", "## 3. Model Training"),

        _create_cell("code", f"""# Option 1: Quick training (one-liner)
model, history = quick_train(
    DATA_PATH,
    model="{model}",
    epochs=10,
    batch_size=32,
    lr=0.001,
    device=device,
    verbose=True,
)

# Plot training curves
plot_results(history)
"""),

        _create_cell("code", """# Option 2: Manual training loop (more control)
# Uncomment to use this approach instead

# from src.models import ClassificationModel
# from src.training import ClassificationTrainer
# from src.config import TrainingConfig
#
# model = ClassificationModel(
#     architecture="resnet50",
#     num_classes=len(class_names),
#     pretrained=True,
# )
#
# config = TrainingConfig(
#     batch_size=32,
#     num_epochs=20,
#     learning_rate=0.001,
# )
#
# trainer = ClassificationTrainer(model, train_loader, val_loader, config)
# history = trainer.train()
"""),

        _create_cell("markdown", "## 4. Evaluation"),

        _create_cell("code", """# Evaluate on validation set
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
"""),

        _create_cell("code", """# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    all_labels, all_preds,
    display_labels=class_names,
    ax=ax,
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
"""),

        _create_cell("code", """# Compare with other architectures (optional)
results = compare_models(
    models=["resnet50", "efficientnet_b0"],
    test_loader=val_loader,
    num_classes=len(class_names),
    device=device,
)

print("\\nModel Comparison:")
for name, metrics in results.items():
    print(f"  {name}: {metrics['accuracy']*100:.2f}% accuracy")
"""),

        _create_cell("markdown", "## 5. Export"),

        _create_cell("code", """# Save the trained model
torch.save(model.state_dict(), "best_model.pth")
print("Model weights saved to: best_model.pth")

# Export for deployment
export_path = export_model(model, "model_deployed.pt", format="torchscript")
print(f"Deployed model saved to: {export_path}")

# Export to ONNX (for cross-platform deployment)
# export_model(model, "model.onnx", format="onnx")
"""),

        _create_cell("markdown", """## Next Steps

1. **Improve accuracy**: Try different architectures, learning rates, or augmentations
2. **Handle class imbalance**: Use weighted loss or oversampling
3. **Deploy**: Use the exported model in production
4. **Track experiments**: Enable MLflow tracking for systematic experimentation

```python
# Enable MLflow tracking
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-classification-experiment")

with mlflow.start_run():
    # Your training code here
    mlflow.log_params({"model": "resnet50", "epochs": 10})
    mlflow.log_metric("accuracy", final_accuracy)
    mlflow.pytorch.log_model(model, "model")
```
"""),
    ]

    return cells


def generate_detection_notebook(
    data_path: Optional[str] = None,
    model: str = "yolov8n",
) -> list:
    """Generate cells for a detection notebook."""
    data_path = data_path or "./data/detection"

    cells = [
        _create_cell("markdown", """# Object Detection Experiment

Train and evaluate object detection models using YOLOv8.

**Requirements:**
- Data in YOLO format (images/ and labels/ folders)
- Each label file contains: `class_id x_center y_center width height` (normalized)
"""),

        _create_cell("code", """# Setup
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
"""),

        _create_cell("code", f"""# Load a pretrained model
model = YOLO("{model}.pt")

# Train on your data
DATA_YAML = "{data_path}/data.yaml"  # Path to your data config

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    device=device,
)
"""),

        _create_cell("code", """# Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
"""),

        _create_cell("code", """# Inference on new images
results = model.predict("path/to/image.jpg", save=True)

# Display results
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    r.show()
"""),

        _create_cell("code", """# Export for deployment
model.export(format="onnx")  # Or "torchscript", "tflite", etc.
"""),
    ]

    return cells


def generate_segmentation_notebook(
    data_path: Optional[str] = None,
    model: str = "unet",
) -> list:
    """Generate cells for a segmentation notebook."""
    data_path = data_path or "./data/segmentation"

    cells = [
        _create_cell("markdown", """# Image Segmentation Experiment

Train and evaluate segmentation models (UNet, DeepLabV3).

**Data format:**
- images/: Input images
- masks/: Segmentation masks (same filename, pixel values = class IDs)
"""),

        _create_cell("code", """# Setup
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
"""),

        _create_cell("code", f"""# Create model
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,  # Number of classes
)
model = model.to(device)

print(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")
"""),

        _create_cell("code", f"""# Data loading
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.images = sorted(self.images_dir.glob("*.png")) + sorted(self.images_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.name

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.long()

# Transforms
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

DATA_PATH = "{data_path}"
dataset = SegmentationDataset(f"{{DATA_PATH}}/images", f"{{DATA_PATH}}/masks", transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""),

        _create_cell("code", """# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = smp.losses.DiceLoss(mode='multiclass')

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
"""),

        _create_cell("code", """# Evaluation - IoU metric
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.argmax(dim=1)
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious)

model.eval()
with torch.no_grad():
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        iou = compute_iou(outputs, masks, num_classes=2)
        print(f"IoU: {iou:.4f}")
        break
"""),
    ]

    return cells


def generate_notebook(
    task: str = "classification",
    output_path: str = "experiment.ipynb",
    data_path: Optional[str] = None,
    model: str = "resnet50",
) -> str:
    """
    Generate a Jupyter notebook for the specified task.

    Args:
        task: Task type ("classification", "detection", "segmentation")
        output_path: Where to save the notebook
        data_path: Path to data (included in notebook)
        model: Model architecture to use

    Returns:
        Path to generated notebook
    """
    generators = {
        "classification": generate_classification_notebook,
        "detection": generate_detection_notebook,
        "segmentation": generate_segmentation_notebook,
    }

    if task not in generators:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(generators.keys())}")

    cells = generators[task](data_path=data_path, model=model)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells,
    }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    return str(output_path)


if __name__ == "__main__":
    # Generate example notebooks
    generate_notebook("classification", "classification_experiment.ipynb")
    generate_notebook("detection", "detection_experiment.ipynb")
    generate_notebook("segmentation", "segmentation_experiment.ipynb")
    print("Notebooks generated!")
