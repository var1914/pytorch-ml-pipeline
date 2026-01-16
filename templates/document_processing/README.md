# Document Processing Template

A template for document classification and analysis.

## Supported Use Cases

- **Document Classification**: Categorize documents (invoices, contracts, forms)
- **OCR Preprocessing**: Image quality assessment
- **Form Recognition**: Structured document processing

## Quick Start

```bash
# Using config file
python train.py --config configs/templates/document_processing.yaml

# Command line
python train.py --dataset rvl_cdip --model vit_base --num-classes 16
```

## Recommended Models

| Use Case | Model | Why |
|----------|-------|-----|
| Document Classification | ViT-Base | Best for document structure |
| Form Processing | ResNet-50 | Efficient, reliable |
| OCR Enhancement | EfficientNet | Good for quality assessment |

## Configuration

See `configs/templates/document_processing.yaml` for full options.

Key settings:
- Vision Transformer preferred
- Light augmentation (preserve text)
- Lower learning rate (3e-5) for ViT
