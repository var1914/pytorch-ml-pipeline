# CV Pipeline - Computer Vision Training Toolkit

A practical toolkit for ML researchers and data scientists. Analyze datasets, train models, compare architectures, and export for deployment - all in a few lines of code.

```python
from cv_pipeline import quick_train, export_model

# Train a classifier in one line
model, history = quick_train("./my_images", model="resnet50", epochs=10)

# Export for deployment
export_model(model, "model.onnx", format="onnx")
```

## Why This Toolkit?

**For Researchers**: Focus on experiments, not boilerplate. Quick iteration with pretrained models.

**For Data Scientists**: Dataset analysis, model comparison, and production-ready exports.

**For Teams**: Standardized workflows across classification, detection, and segmentation tasks.

> **New to this toolkit?** Check out the [Scenarios & Use Cases Guide](docs/SCENARIOS.md) for detailed examples.

## What This Toolkit Does (and Doesn't Do)

| Included | Not Included |
|----------|--------------|
| Dataset analysis & validation | Model serving API |
| Data loading & augmentation | Docker/Kubernetes deployment |
| 50+ model architectures (via timm) | Inference server |
| Training loops with progress tracking | Production monitoring |
| Model comparison & benchmarking | A/B testing |
| Export to TorchScript, ONNX, state_dict | Cloud deployment |

**This is a training toolkit.** It outputs deployment-ready model files (`.pt`, `.onnx`) that you deploy with your own infrastructure (FastAPI, TorchServe, Triton, etc.).

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/ml-pipeline-cv.git
cd ml-pipeline-cv
pip install -e .
```

### Option 1: CLI (Fastest)

```bash
# Analyze your dataset
cv-pipeline analyze --data ./my_images

# Train a model
cv-pipeline train --data ./my_images --model resnet50 --epochs 10

# Compare architectures
cv-pipeline compare --models resnet50,efficientnet_b0 --data ./test_images

# Export for deployment
cv-pipeline export --model trained.pth --format onnx
```

### Option 2: Python API

```python
from cv_pipeline import (
    analyze_dataset,
    quick_train,
    compare_models,
    export_model,
)

# 1. Understand your data
stats = analyze_dataset("./my_images")
print(f"Found {stats['num_classes']} classes, {stats['total_images']} images")

# 2. Train a model
model, history = quick_train(
    "./my_images",
    model="efficientnet_b0",
    epochs=10,
    batch_size=32,
)

# 3. Export (TorchScript for PyTorch serving, ONNX for cross-platform)
export_model(model, "model.pt", format="torchscript")
export_model(model, "model.onnx", format="onnx")
```

### Option 3: Generate Notebook

```bash
cv-pipeline notebook --task classification --output my_experiment.ipynb
```

## Features

| Feature | CLI Command | Python Function |
|---------|-------------|-----------------|
| Dataset analysis | `cv-pipeline analyze` | `analyze_dataset()` |
| Quick training | `cv-pipeline train` | `quick_train()` |
| Model comparison | `cv-pipeline compare` | `compare_models()` |
| Model export | `cv-pipeline export` | `export_model()` |
| Notebook generation | `cv-pipeline notebook` | `generate_notebook()` |

## Supported Architectures

All architectures available via [timm](https://github.com/huggingface/pytorch-image-models):

### Classification
- **ResNet**: resnet18, resnet34, resnet50, resnet101
- **EfficientNet**: efficientnet_b0 through efficientnet_b4
- **Vision Transformer**: vit_tiny, vit_small, vit_base
- **MobileNet**: mobilenetv2_100, mobilenetv3_small, mobilenetv3_large
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base

### Detection (templates)
- YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l
- Faster R-CNN: fasterrcnn_resnet50_fpn

### Segmentation (templates)
- UNet, UNet++
- DeepLabV3, DeepLabV3+

## Examples

### Compare Models on Your Data

```python
from cv_pipeline import load_image_folder, compare_models

# Load your test set
_, test_loader, classes = load_image_folder("./test_images", split=0.0)

# Compare multiple architectures
results = compare_models(
    models=["resnet50", "efficientnet_b0", "mobilenetv2_100"],
    test_loader=test_loader,
    num_classes=len(classes),
)

for model, metrics in results.items():
    print(f"{model}: {metrics['accuracy']*100:.1f}% ({metrics['params']/1e6:.1f}M params)")
```

### Transfer Learning

```python
from cv_pipeline import get_model
import torch

# Load pretrained model
model = get_model("resnet50", num_classes=10, pretrained=True)

# Freeze backbone, train head only
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Your training loop here...
```

### Full Workflow

```python
from cv_pipeline import analyze_dataset, quick_train, export_model, plot_results

# 1. Analyze
stats = analyze_dataset("./my_images")
if stats.get("imbalance_ratio", 1) > 5:
    print("Warning: Class imbalance detected")

# 2. Train
model, history = quick_train(
    "./my_images",
    model="efficientnet_b0",
    epochs=20,
    save_path="best_model.pth"
)

# 3. Visualize
plot_results(history, save_path="training_curves.png")

# 4. Export
export_model(model, "model.onnx", format="onnx")
```

## Project Structure

```
ml-pipeline-cv/
├── cv_pipeline/                    # Main toolkit
│   ├── __init__.py                 # Public API exports
│   ├── cli.py                      # Command-line interface
│   ├── utils.py                    # Core utilities
│   └── notebook_generator.py       # Jupyter notebook templates
├── src/
│   ├── config/                     # Pydantic configuration
│   ├── core/                       # Base classes & factories
│   ├── models/                     # Model implementations
│   ├── training/                   # Training loops
│   ├── evaluation/                 # Metrics & visualization
│   └── data/                       # Data loading
├── tests/                          # Test suite
├── scripts/
│   └── test_scenarios.py           # Lightweight validation script
├── docs/
│   └── SCENARIOS.md                # Use cases guide
├── examples/                       # Ready-to-run examples
└── configs/                        # YAML configurations
```

## Export Formats

| Format | Use Case | Command |
|--------|----------|---------|
| `torchscript` | PyTorch serving, TorchServe | `export_model(model, "model.pt", format="torchscript")` |
| `onnx` | Cross-platform (ONNX Runtime, TensorRT, OpenVINO) | `export_model(model, "model.onnx", format="onnx")` |
| `state_dict` | Resume training, fine-tuning | `export_model(model, "weights.pth", format="state_dict")` |

## After Export: Deployment Options

The toolkit exports models. For deployment, use:

- **[TorchServe](https://pytorch.org/serve/)** - PyTorch's official serving solution
- **[ONNX Runtime](https://onnxruntime.ai/)** - Cross-platform inference
- **[Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)** - High-performance GPU serving
- **[FastAPI](https://fastapi.tiangolo.com/)** + custom code - Lightweight REST API
- **[BentoML](https://www.bentoml.com/)** - ML model packaging and deployment

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run lightweight scenario validation
python scripts/test_scenarios.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+

```bash
pip install -e .
```

## Documentation

| Document | Description |
|----------|-------------|
| [Scenarios Guide](docs/SCENARIOS.md) | Comprehensive use cases for data scientists |
| [examples/](examples/) | Ready-to-run Python scripts |
| [configs/](configs/) | YAML configuration examples |

## Contributing

Contributions welcome:
- New model architectures
- Documentation improvements
- Bug fixes

## License

MIT License

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/)
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [MLflow](https://mlflow.org/) - Experiment tracking (optional)
