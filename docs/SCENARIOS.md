# CV Pipeline - Use Cases & Scenarios Guide

A comprehensive guide for data scientists and ML researchers on how to use this toolkit for real-world computer vision problems.

## Table of Contents

1. [Quick Prototyping](#1-quick-prototyping)
2. [Dataset Analysis & Quality Assessment](#2-dataset-analysis--quality-assessment)
3. [Model Architecture Selection](#3-model-architecture-selection)
4. [Transfer Learning](#4-transfer-learning)
5. [Production Deployment](#5-production-deployment)
6. [Notebook-Based Research](#6-notebook-based-research)
7. [Industry-Specific Solutions](#7-industry-specific-solutions)
8. [End-to-End Workflows](#8-end-to-end-workflows)
9. [Common Problems & Solutions](#9-common-problems--solutions)

---

## 1. Quick Prototyping

### Scenario: "I have a new dataset and need a baseline model quickly"

**Use Case:** You received a new image dataset and need to validate if a classification approach is viable before investing more time.

#### Option A: Python API (Recommended for notebooks)

```python
from cv_pipeline import quick_train, plot_results
import torch

# Train a classifier in one function call
model, history = quick_train(
    data_path="./my_dataset",
    model="resnet50",
    epochs=10,
    batch_size=32,
    lr=0.001,
    verbose=True
)

# Visualize results
plot_results(history, save_path="baseline_results.png")

# Save the model
torch.save(model.state_dict(), "baseline_model.pth")
```

#### Option B: CLI (Recommended for quick experiments)

```bash
# Train directly from command line
cv-pipeline train --data ./my_dataset --model resnet50 --epochs 10 --output baseline.pth

# Results are automatically saved:
# - baseline.pth (model weights)
# - baseline_training.png (loss/accuracy curves)
```

#### When to use:
- Hackathons or time-constrained projects
- Initial feasibility check
- Quick demos to stakeholders
- Validating data quality through training

---

## 2. Dataset Analysis & Quality Assessment

### Scenario: "I received a dataset from a client and need to understand it"

**Use Case:** Before any modeling, you need to understand class distribution, image quality, and potential issues.

#### Comprehensive Analysis

```python
from cv_pipeline import analyze_dataset

# Analyze and get detailed statistics
stats = analyze_dataset(
    "./client_data",
    show_samples=True,      # Display sample images
    save_report="analysis_report.json"  # Save for documentation
)

# Access statistics programmatically
print(f"Total images: {stats['total_images']}")
print(f"Number of classes: {stats['num_classes']}")
print(f"Class distribution: {stats['class_distribution']}")
print(f"Image sizes: {stats['image_sizes']}")
print(f"Imbalance ratio: {stats.get('imbalance_ratio', 'N/A')}")

# Check for issues
if stats['issues']:
    print("⚠️ Issues found:")
    for issue in stats['issues']:
        print(f"  - {issue}")
```

#### CLI Quick Check

```bash
# Quick analysis from command line
cv-pipeline analyze --data ./client_data --save report.json

# Without sample visualization (faster)
cv-pipeline analyze --data ./client_data --no-samples
```

#### What you'll learn:
- **Class distribution:** Are classes balanced? (important for loss function choice)
- **Image sizes:** Do images need resizing? Are they consistent?
- **File formats:** Mixed formats? Potential compatibility issues?
- **Data quality issues:** Corrupt files, unreadable images

#### Recommended actions based on analysis:

| Finding | Recommended Action |
|---------|-------------------|
| Imbalance ratio > 3 | Use weighted loss or oversampling |
| Imbalance ratio > 10 | Consider data augmentation + class weights |
| Mixed image sizes | Standardize in preprocessing |
| < 100 images/class | Use transfer learning, heavy augmentation |
| Corrupt files found | Clean dataset before training |

---

## 3. Model Architecture Selection

### Scenario: "Which model architecture should I use?"

**Use Case:** You need to find the best trade-off between accuracy, model size, and inference speed for your deployment constraints.

#### Compare Multiple Architectures

```python
from cv_pipeline import load_image_folder, compare_models

# Load your test data
train_loader, val_loader, class_names = load_image_folder(
    "./my_dataset",
    batch_size=32,
    split=0.2
)

# Compare different model families
results = compare_models(
    models=[
        # Lightweight (mobile/edge)
        "mobilenetv2_100",
        "efficientnet_b0",

        # Medium (server)
        "resnet50",
        "efficientnet_b2",

        # Heavy (best accuracy)
        "efficientnet_b4",
        "convnext_base",

        # Transformer
        "vit_small_patch16_224",
    ],
    test_loader=train_loader,
    num_classes=len(class_names),
    device="auto"
)

# Results include accuracy, F1 score, and parameter count
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Parameters: {metrics['params']/1e6:.1f}M")
```

#### CLI Comparison

```bash
# Compare models from command line
cv-pipeline compare \
    --models resnet50,efficientnet_b0,mobilenetv2_100 \
    --data ./test_images \
    --batch-size 32
```

#### Model Selection Guide

| Deployment Target | Recommended Models | Typical Size |
|-------------------|-------------------|--------------|
| Mobile/Edge | mobilenetv2_100, efficientnet_b0 | 3-5M params |
| Web Browser (ONNX) | efficientnet_b0, resnet18 | 5-11M params |
| Server (GPU) | resnet50, efficientnet_b2 | 25-30M params |
| Best Accuracy | efficientnet_b4, convnext_base | 50-100M params |
| Transformers | vit_small_patch16_224 | 22M params |

#### Accuracy vs. Speed Trade-off

```python
# For real-time applications (>30 FPS)
model = get_model("mobilenetv2_100", num_classes=10)  # ~3.4M params

# For near real-time (10-30 FPS)
model = get_model("efficientnet_b0", num_classes=10)  # ~5.3M params

# For batch processing (accuracy priority)
model = get_model("efficientnet_b4", num_classes=10)  # ~19M params
```

---

## 4. Transfer Learning

### Scenario: "I have limited data (100-1000 images)"

**Use Case:** Small datasets require careful transfer learning to avoid overfitting.

#### Two-Phase Transfer Learning

```python
from cv_pipeline import get_model, load_image_folder
import torch
import torch.nn as nn

# Load data with augmentation
train_loader, val_loader, class_names = load_image_folder(
    "./small_dataset",
    batch_size=16,
    augment=True,  # Important for small datasets
    split=0.2
)

# Get pretrained model
model = get_model("resnet50", num_classes=len(class_names), pretrained=True)

# PHASE 1: Freeze backbone, train classifier only
print("Phase 1: Training classifier head...")
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier (last layer)
for param in model.fc.parameters():  # ResNet
    param.requires_grad = True
# For EfficientNet: model.classifier.parameters()
# For ViT: model.head.parameters()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Train for a few epochs...

# PHASE 2: Unfreeze all, fine-tune with lower LR
print("Phase 2: Fine-tuning entire model...")
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-4},      # Classifier
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Late layers
    {'params': model.layer3.parameters(), 'lr': 1e-6},  # Earlier layers
], weight_decay=0.01)

# Train with lower learning rate...
```

#### Using the Transfer Learning Example

```bash
# Run the pre-built transfer learning script
python examples/transfer_learning.py \
    --data ./small_dataset \
    --model resnet50 \
    --phase1-epochs 5 \
    --phase2-epochs 15
```

#### Tips for Small Datasets

| Dataset Size | Recommended Approach |
|--------------|---------------------|
| < 100 images | Heavy augmentation, freeze most layers, small model |
| 100-500 images | Two-phase training, moderate augmentation |
| 500-1000 images | Fine-tune all layers with low LR |
| > 1000 images | Can train from scratch with augmentation |

---

## 5. Production Deployment

### Scenario: "I need to deploy my model"

**Use Case:** Export trained model for various deployment targets.

#### Export Options

```python
from cv_pipeline import get_model, export_model, quick_train

# Train or load your model
model, _ = quick_train("./data", model="efficientnet_b0", epochs=20)

# Export for PyTorch serving (TorchServe, custom server)
export_model(model, "model.pt", format="torchscript")

# Export for cross-platform (TensorRT, OpenVINO, ONNX Runtime)
export_model(model, "model.onnx", format="onnx")

# Export weights only (for continued training)
export_model(model, "model_weights.pth", format="state_dict")
```

#### CLI Export

```bash
# Export trained model
cv-pipeline export --model trained.pth --format torchscript --output model.pt
cv-pipeline export --model trained.pth --format onnx --output model.onnx
```

#### Deployment Target Guide

| Target | Format | Tools |
|--------|--------|-------|
| PyTorch Server | TorchScript (.pt) | TorchServe, FastAPI |
| NVIDIA GPU | ONNX → TensorRT | Triton, TensorRT |
| Intel CPU | ONNX → OpenVINO | OpenVINO Runtime |
| Browser | ONNX | ONNX.js, Transformers.js |
| Mobile (iOS) | TorchScript → CoreML | coremltools |
| Mobile (Android) | TorchScript | PyTorch Mobile |
| Edge (Jetson) | ONNX → TensorRT | Jetson Inference |

#### Loading Exported Models

```python
# TorchScript
model = torch.jit.load("model.pt")
output = model(input_tensor)

# ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {"input": input_array})
```

---

## 6. Notebook-Based Research

### Scenario: "I'm starting a new research project"

**Use Case:** Generate a complete Jupyter notebook template with best practices.

#### Generate Experiment Notebooks

```bash
# Classification experiment
cv-pipeline notebook --task classification --output classification_exp.ipynb

# Object detection with YOLO
cv-pipeline notebook --task detection --output detection_exp.ipynb

# Semantic segmentation
cv-pipeline notebook --task segmentation --output segmentation_exp.ipynb

# With specific data path and model
cv-pipeline notebook \
    --task classification \
    --data ./my_project/data \
    --model efficientnet_b0 \
    --output my_experiment.ipynb
```

#### Python API

```python
from cv_pipeline import generate_notebook

# Generate with custom settings
generate_notebook(
    task="classification",
    output_path="experiment.ipynb",
    data_path="./my_data",
    model="resnet50"
)
```

#### What's Included in Generated Notebooks

**Classification Notebook:**
- Setup and imports
- Dataset analysis
- Data loading with augmentation
- Model training (quick_train + manual loop)
- Evaluation metrics (classification report, confusion matrix)
- Model comparison
- Export for deployment

**Detection Notebook:**
- YOLOv8 setup
- Data format requirements (YOLO format)
- Training configuration
- Evaluation (mAP metrics)
- Inference examples

**Segmentation Notebook:**
- UNet/DeepLab setup
- Custom dataset class
- Training loop with Dice loss
- IoU evaluation
- Visualization of predictions

---

## 7. Industry-Specific Solutions

### Scenario: "I'm working on a domain-specific problem"

**Use Case:** Pre-configured templates for common industry applications.

#### Medical Imaging

```bash
# Train on medical imaging datasets
python templates/medical_imaging/train.py \
    --dataset pcam \
    --model resnet50 \
    --epochs 30 \
    --batch-size 64
```

**Supported tasks:**
- Cancer detection (histopathology)
- Chest X-ray classification
- Skin lesion analysis
- Retinal disease detection

**Key considerations:**
- Class imbalance handling
- High-resolution image support
- Interpretability requirements

#### Manufacturing Quality Control

```bash
# Defect detection training
python templates/manufacturing_qc/train.py \
    --data ./defect_images \
    --task detection
```

**Supported tasks:**
- Defect detection
- Surface inspection
- Assembly verification
- Anomaly detection

**Key considerations:**
- Real-time inference requirements
- High precision needed
- Few defect samples (imbalanced)

#### Retail & E-commerce

```bash
# Product classification
python templates/retail/train.py \
    --data ./product_images \
    --num-classes 1000
```

**Supported tasks:**
- Product categorization
- Visual search
- Inventory management
- Price tag recognition

#### Document Processing

```bash
# Document classification
python templates/document_processing/train.py \
    --data ./documents \
    --task classification
```

**Supported tasks:**
- Document type classification
- Form recognition
- Invoice processing
- ID document verification

---

## 8. End-to-End Workflows

### Workflow A: New Project from Scratch

```bash
# Step 1: Analyze the data
cv-pipeline analyze --data ./raw_data --save analysis.json

# Step 2: Quick baseline
cv-pipeline train --data ./raw_data --model resnet18 --epochs 5 --output baseline.pth

# Step 3: Compare architectures
cv-pipeline compare --models resnet50,efficientnet_b0,mobilenetv2_100 --data ./raw_data

# Step 4: Full training with best model
cv-pipeline train --data ./raw_data --model efficientnet_b0 --epochs 50 --output best_model.pth

# Step 5: Export for deployment
cv-pipeline export --model best_model.pth --format onnx --output production_model.onnx
```

### Workflow B: Research Experiment

```python
from cv_pipeline import (
    analyze_dataset,
    load_image_folder,
    get_model,
    compare_models,
    quick_train,
    export_model,
    plot_results
)

# 1. Understand the data
stats = analyze_dataset("./research_data", save_report="data_analysis.json")

# 2. Load data
train_loader, val_loader, classes = load_image_folder(
    "./research_data",
    batch_size=32,
    augment=True
)

# 3. Experiment with architectures
architectures = ["resnet50", "efficientnet_b0", "vit_small_patch16_224"]
results = compare_models(architectures, val_loader, num_classes=len(classes))

# 4. Train best model
best_arch = max(results, key=lambda x: results[x]['accuracy'])
model, history = quick_train(
    "./research_data",
    model=best_arch,
    epochs=50,
    save_path="best_model.pth"
)

# 5. Visualize and export
plot_results(history, save_path="training_curves.png")
export_model(model, "model_final.onnx", format="onnx")
```

### Workflow C: MLOps Pipeline (with infrastructure)

```python
from src.config import load_config
from src.core import ModelFactory, TrainerFactory, EvaluatorFactory

# Load configuration
config = load_config("configs/production.yaml")

# Create components using factories
model = ModelFactory.create(
    task_type=config.model.task_type,
    architecture=config.model.architecture,
    num_classes=config.model.num_classes
)

trainer = TrainerFactory.create(
    task_type=config.model.task_type,
    model=model,
    config=config.training,
    infra_config=config.infra  # MLflow, MinIO settings
)

# Train with full MLOps integration
history = trainer.train()

# Evaluate
evaluator = EvaluatorFactory.create(
    task_type=config.model.task_type,
    model=model,
    test_loader=test_loader
)
metrics = evaluator.evaluate()
```

---

## 9. Common Problems & Solutions

### Problem: Class Imbalance

```python
from cv_pipeline import analyze_dataset
import torch
import torch.nn as nn

# Detect imbalance
stats = analyze_dataset("./data")
if stats.get('imbalance_ratio', 1) > 3:
    print(f"⚠️ Class imbalance detected: {stats['imbalance_ratio']}x")

# Solution 1: Weighted loss
class_counts = list(stats['class_distribution'].values())
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights)

# Solution 2: Oversampling (in data loader)
from torch.utils.data import WeightedRandomSampler

sample_weights = [weights[label] for _, label in dataset]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Problem: Overfitting on Small Dataset

```python
# Solution: Heavy augmentation + dropout + early stopping
from cv_pipeline import load_image_folder, get_model

# Enable augmentation
train_loader, val_loader, classes = load_image_folder(
    "./small_data",
    augment=True,  # Enables RandomFlip, Rotation, ColorJitter
    batch_size=16   # Smaller batch size
)

# Use model with dropout
import timm
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=len(classes),
    drop_rate=0.3,      # Dropout
    drop_path_rate=0.2  # Stochastic depth
)
```

### Problem: Model Too Slow for Deployment

```python
from cv_pipeline import compare_models, export_model

# Find faster model with acceptable accuracy
results = compare_models(
    ["mobilenetv2_100", "efficientnet_b0", "resnet18"],
    test_loader, num_classes=10
)

# Choose based on accuracy/speed trade-off
# mobilenetv2_100: ~3.4M params, fastest
# efficientnet_b0: ~5.3M params, good balance
# resnet18: ~11M params, baseline

# Optimize with quantization (post-training)
import torch
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Problem: Out of Memory (OOM)

```python
# Solution 1: Reduce batch size
train_loader, _, _ = load_image_folder("./data", batch_size=8)  # Instead of 32

# Solution 2: Use gradient accumulation
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    loss = criterion(model(images), labels)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Use smaller model
model = get_model("mobilenetv2_100", num_classes=10)  # Instead of resnet50

# Solution 4: Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(images)
    loss = criterion(output, labels)
```

### Problem: Poor Generalization

```python
# Solution: Data augmentation + regularization
from torchvision import transforms

# Strong augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

# Weight decay (L2 regularization)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## Quick Reference Card

| Task | Command/Function |
|------|-----------------|
| Analyze dataset | `cv-pipeline analyze --data ./path` |
| Train model | `cv-pipeline train --data ./path --model resnet50` |
| Compare models | `cv-pipeline compare --models a,b,c --data ./path` |
| Export model | `cv-pipeline export --model m.pth --format onnx` |
| Generate notebook | `cv-pipeline notebook --task classification` |
| Quick train (Python) | `quick_train("./data", model="resnet50", epochs=10)` |
| Load data | `load_image_folder("./data", batch_size=32)` |
| Get model | `get_model("efficientnet_b0", num_classes=10)` |

---

## Additional Resources

- **Examples:** See `examples/` directory for runnable scripts
- **Templates:** See `templates/` for industry-specific solutions
- **Tests:** See `tests/` for usage examples
- **Config:** See `configs/` for YAML configuration examples

For issues or feature requests, please open a GitHub issue.
