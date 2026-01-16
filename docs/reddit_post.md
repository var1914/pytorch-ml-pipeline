# Reddit Post for r/machinelearning

---

**Title:** [P] cv-pipeline: A minimal PyTorch toolkit for CV researchers who hate boilerplate

---

**Body:**

I got tired of copy-pasting the same data loading, training loops, and export code for every CV project. So I built a toolkit that handles the boring stuff.

**What it does:**

```python
from cv_pipeline import quick_train, analyze_dataset, export_model

# Analyze your dataset
analyze_dataset("./my_images")

# Train (one line)
model, history = quick_train("./my_images", model="efficientnet_b0", epochs=10)

# Export for deployment
export_model(model, "model.onnx", format="onnx")
```

**Key features:**

- **Data loading** - Point to a folder, get DataLoaders. Handles splits, augmentation, normalization.
- **50+ architectures** - ResNet, EfficientNet, ViT, MobileNet via timm. One-line model loading.
- **Dataset analysis** - Class distribution, imbalance detection, image stats.
- **Model comparison** - Benchmark multiple architectures on your data.
- **Export** - TorchScript, ONNX, state_dict.
- **CLI** - `cv-pipeline train --data ./images --model resnet50 --epochs 20`
- **Notebook generator** - Auto-generate starter notebooks for classification/detection/segmentation.

**CLI example:**

```bash
# Analyze dataset
cv-pipeline analyze --data ./images

# Train
cv-pipeline train --data ./images --model efficientnet_b0 --epochs 20

# Compare models
cv-pipeline compare --models resnet50,efficientnet_b0,vit_base --data ./images
```

**Not a framework** - just utilities. Use with your existing PyTorch code. No lock-in.

Built for rapid prototyping and experiment iteration. Includes configs for medical imaging, manufacturing QC, retail, and document processing use cases.

GitHub: [link]

Feedback welcome. What utilities would you add?

---

# Alternative shorter version (if above is too long)

---

**Title:** [P] cv-pipeline: One-line training for PyTorch CV projects

---

**Body:**

Built a minimal toolkit to skip the boilerplate in CV projects:

```python
from cv_pipeline import quick_train, export_model

# Train on image folder
model, history = quick_train("./my_images", model="efficientnet_b0", epochs=10)

# Export
export_model(model, "model.onnx", format="onnx")
```

**Features:**
- 50+ architectures (timm)
- Dataset analysis & class imbalance detection
- Model comparison benchmarks
- CLI: `cv-pipeline train --data ./images --model resnet50`
- Export: TorchScript, ONNX
- Notebook generator for classification/detection/segmentation

Not a framework - just utilities that work with existing PyTorch code.

GitHub: [link]

What's missing? Happy to add features.
