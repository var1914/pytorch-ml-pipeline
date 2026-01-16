# Retail/E-commerce Template

A template for product classification and visual search.

## Supported Use Cases

- **Product Classification**: Categorize products by image
- **Visual Search**: Find similar products
- **Inventory Management**: Automated stock counting

## Quick Start

```bash
# Using config file
python train.py --config configs/templates/retail.yaml

# Command line
python train.py --dataset custom --model efficientnet_b3 --num-classes 1000
```

## Recommended Models

| Use Case | Model | Why |
|----------|-------|-----|
| Product Classification | EfficientNet-B3 | Good accuracy/speed balance |
| Visual Search | ViT-Base | Strong feature representations |
| Fine-grained | EfficientNet-B4+ | Higher resolution |

## Configuration

See `configs/templates/retail.yaml` for full options.

Key settings:
- Image size: 224x224 (standard)
- Medium augmentation
- AdamW optimizer with cosine schedule
