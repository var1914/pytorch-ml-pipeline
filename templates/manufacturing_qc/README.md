# Manufacturing QC Template

A template for manufacturing quality control and defect detection.

## Supported Use Cases

- **Defect Detection**: Assembly line quality inspection
- **Surface Inspection**: Scratch, dent, contamination detection
- **Anomaly Detection**: Out-of-distribution sample identification

## Quick Start

```bash
# Using config file
python train.py --config configs/templates/manufacturing_qc.yaml

# For defect detection (YOLO)
python train.py --task detection --model yolov8s --data-yaml data/defects.yaml
```

## Supported Tasks

| Task | Models | Output |
|------|--------|--------|
| Classification | ResNet, EfficientNet | Defect type |
| Detection | YOLOv8, Faster R-CNN | Bounding boxes |
| Segmentation | UNet, DeepLabV3 | Defect masks |

## Configuration

See `configs/templates/manufacturing_qc.yaml` for full options.

Key settings:
- Image size: 256x256 (or higher for small defects)
- Heavy augmentation for limited data
- Detection models for localization
