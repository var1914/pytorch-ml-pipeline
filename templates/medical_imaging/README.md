# Medical Imaging Template

A ready-to-use template for medical image classification tasks.

## Supported Use Cases

- **Cancer Detection**: Histopathology patch classification (PCAM)
- **Chest X-ray Classification**: Pneumonia, COVID-19 detection
- **Skin Lesion Analysis**: Dermatological condition classification (ISIC)

## Quick Start

```bash
# Using config file
python train.py --config configs/templates/medical_imaging.yaml

# Using command line arguments
python train.py --dataset pcam --epochs 30 --batch-size 64 --model resnet50
```

## Supported Datasets

| Dataset | Description | Classes | Image Size |
|---------|-------------|---------|------------|
| PCAM | Histopathology patches | 2 (tumor/normal) | 96x96 |
| ChestX-ray | Chest radiographs | 2-14 | 224x224 |
| ISIC | Dermoscopy images | 7+ | 224x224 |

## Model Architectures

Recommended architectures for medical imaging:

- **ResNet-50**: Good baseline, well-studied
- **EfficientNet-B3**: Better accuracy/compute tradeoff
- **DenseNet-121**: Strong for chest X-ray tasks

## Configuration

See `configs/templates/medical_imaging.yaml` for full configuration options.

Key hyperparameters:
- Lower learning rate (0.0001) - medical images benefit from fine-tuning
- Dropout (0.3) - helps with smaller medical datasets
- Medium augmentation - rotation, flips, color jitter

## Output

Results are saved to `./results/medical_imaging/`:
- `confusion_matrix.png` - Classification performance
- `roc_curve.png` - ROC curve (binary classification)
- `classification_report.txt` - Detailed metrics

## MLflow Integration

Experiments are tracked in MLflow under the experiment name:
`medical_imaging_experiments`

Access the UI at: http://localhost:5000
