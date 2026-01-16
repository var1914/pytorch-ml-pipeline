# CV Pipeline - Computer Vision Research Toolkit

A practical toolkit for ML researchers and data scientists. Train models, compare architectures, and deploy to production - all in a few lines of code.

```python
# Train a classifier in one line
from cv_pipeline import quick_train
model, history = quick_train("./my_images", model="resnet50", epochs=10)
```

## Why This Toolkit?

**For Researchers**: Focus on experiments, not boilerplate. Quick iteration with pretrained models.

**For Data Scientists**: Production-ready code that scales. YAML configs, MLflow tracking, MinIO storage.

**For Teams**: Standardized workflows across classification, detection, and segmentation tasks.

> **New to this toolkit?** Check out the [Scenarios & Use Cases Guide](docs/SCENARIOS.md) for detailed examples covering dataset analysis, model selection, transfer learning, deployment, and industry-specific solutions.

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
    generate_notebook,
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

# 3. Export for production
export_model(model, "model.onnx", format="onnx")
```

### Option 3: Generate Notebook

```bash
cv-pipeline notebook --task classification --output my_experiment.ipynb
```

Or in Python:
```python
from cv_pipeline import generate_notebook
generate_notebook("classification", "my_experiment.ipynb", data_path="./my_images")
```

## Features at a Glance

| Feature | CLI Command | Python Function |
|---------|-------------|-----------------|
| Dataset analysis | `cv-pipeline analyze` | `analyze_dataset()` |
| Quick training | `cv-pipeline train` | `quick_train()` |
| Model comparison | `cv-pipeline compare` | `compare_models()` |
| Model export | `cv-pipeline export` | `export_model()` |
| Notebook generation | `cv-pipeline notebook` | `generate_notebook()` |

## Supported Tasks & Architectures

### Classification
- **ResNet**: resnet18, resnet34, resnet50, resnet101
- **EfficientNet**: efficientnet_b0 through efficientnet_b4
- **Vision Transformer**: vit_tiny, vit_small, vit_base
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base

### Detection (via ultralytics)
- YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l
- Faster R-CNN: fasterrcnn_resnet50_fpn

### Segmentation (via segmentation-models-pytorch)
- UNet, UNet++
- DeepLabV3, DeepLabV3+
- FPN, PSPNet

## Examples

### Compare Models on Your Data

```python
from cv_pipeline import load_image_folder, compare_models

# Load your test set
_, test_loader, classes = load_image_folder("./test_images")

# Compare multiple architectures
results = compare_models(
    models=["resnet50", "efficientnet_b0", "mobilenet_v2"],
    test_loader=test_loader,
    num_classes=len(classes),
)

for model, metrics in results.items():
    print(f"{model}: {metrics['accuracy']*100:.1f}% ({metrics['params']/1e6:.1f}M params)")
```

### Transfer Learning

```python
from cv_pipeline import get_model, load_image_folder
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

### Export to ONNX

```python
from cv_pipeline import export_model, get_model

model = get_model("efficientnet_b0", num_classes=10)
# ... train your model ...

export_model(model, "model.onnx", format="onnx", input_size=(224, 224))
# Now deploy with ONNX Runtime, TensorRT, or OpenVINO
```

## Project Structure

```
ml-pipeline-cv/
├── cv_pipeline/                    # Researcher toolkit (NEW)
│   ├── cli.py                      # Command-line interface
│   ├── utils.py                    # Core utilities
│   └── notebook_generator.py       # Jupyter notebook templates
├── src/
│   ├── config/                     # Pydantic configuration
│   ├── core/                       # Base classes & factories
│   ├── models/                     # Model implementations
│   │   ├── classification.py       # timm-based classifiers
│   │   ├── detection.py            # YOLO, Faster R-CNN
│   │   └── segmentation.py         # UNet, DeepLab
│   ├── training/                   # Training loops
│   ├── evaluation/                 # Metrics & visualization
│   ├── data/                       # Data loading
│   └── airflow/                    # Production DAGs
├── examples/                       # Ready-to-run examples
│   ├── quickstart.py
│   ├── model_comparison.py
│   ├── transfer_learning.py
│   └── deploy_model.py
├── templates/                      # Business use case templates
│   ├── medical_imaging/
│   ├── manufacturing_qc/
│   ├── retail/
│   └── document_processing/
└── configs/                        # YAML configurations
```

## Advanced: Production Pipeline

For production deployments, the framework includes:

- **MLflow Integration**: Experiment tracking & model registry
- **MinIO Storage**: S3-compatible artifact storage
- **Airflow DAGs**: Automated training pipelines
- **YAML Configs**: Reproducible experiments

### Using Config Files

```yaml
# configs/my_experiment.yaml
data:
  dataset_name: custom
  data_root: ./data/my_images
  image_size: [224, 224]

model:
  task_type: classification
  architecture: efficientnet_b0
  num_classes: 10
  pretrained: true

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

```python
from src.config import load_config
from src.core import ModelFactory, TrainerFactory

config = load_config("configs/my_experiment.yaml")
model = ModelFactory.create(**config.model.model_dump())
trainer = TrainerFactory.create(model=model, config=config.training, ...)
```

### Business Templates

Pre-configured templates for common use cases:

| Template | Use Case | Dataset Examples |
|----------|----------|------------------|
| `medical_imaging/` | Cancer detection, X-ray classification | PCAM, ChestX-ray14 |
| `manufacturing_qc/` | Defect detection, quality control | MVTec AD |
| `retail/` | Product classification, visual search | Products-10K |
| `document_processing/` | Document classification, form detection | RVL-CDIP |

```bash
# Train using a template
python templates/medical_imaging/train.py --dataset pcam --epochs 30
```

## Infrastructure Setup (Optional)

For full MLOps capabilities:

```bash
# MinIO (object storage)
docker run -d -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=admin" \
  -e "MINIO_ROOT_PASSWORD=admin123" \
  minio/minio server /data --console-address ":9001"

# MLflow (experiment tracking)
mlflow server --host 0.0.0.0 --port 5000

# Airflow (orchestration) - see docs for full setup
```

## Requirements

Core:
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+

Install options:
```bash
pip install -e .                    # Core (classification)
pip install -e ".[detection]"       # + YOLOv8
pip install -e ".[segmentation]"    # + UNet, DeepLab
pip install -e ".[full]"            # Everything
```

## Common Workflows

### 1. New Dataset Evaluation

```bash
# Analyze first
cv-pipeline analyze --data ./new_dataset --save report.json

# Quick baseline
cv-pipeline train --data ./new_dataset --model resnet50 --epochs 5

# Compare architectures
cv-pipeline compare --models resnet50,efficientnet_b0,mobilenet_v2 --data ./test
```

### 2. Model Selection

```python
# Find the best accuracy/size trade-off
results = compare_models(
    ["mobilenet_v2", "efficientnet_b0", "resnet50"],
    test_loader, num_classes=10
)

# Pick based on your deployment constraints
# - Mobile: mobilenet_v2 (3.4M params)
# - Server: efficientnet_b0 (5.3M params)
# - Best accuracy: resnet50 (25.6M params)
```

### 3. Production Deployment

```python
# Train
model, _ = quick_train("./data", model="efficientnet_b0", epochs=20)

# Export
export_model(model, "model.pt", format="torchscript")  # PyTorch serving
export_model(model, "model.onnx", format="onnx")       # Cross-platform
```

## Documentation

| Document | Description |
|----------|-------------|
| [Scenarios Guide](docs/SCENARIOS.md) | Comprehensive use cases for data scientists |
| [examples/](examples/) | Ready-to-run Python scripts |
| [templates/](templates/) | Industry-specific solutions |
| [configs/](configs/) | YAML configuration examples |

## Contributing

Contributions welcome! Areas of interest:
- New model architectures
- Additional business templates
- Documentation improvements
- Bug fixes

## License

MIT License

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/)
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [MLflow](https://mlflow.org/) - Experiment tracking
- [MinIO](https://min.io/) - Object storage
