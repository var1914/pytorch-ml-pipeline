#!/usr/bin/env python3
"""
Quickstart Example - Train a classifier in under 10 lines of code.

This demonstrates the simplest possible workflow with cv_pipeline.

Usage:
    python examples/quickstart.py --data ./my_images
"""

from cv_pipeline import quick_train, plot_results
import torch

# Train a model (that's it!)
model, history = quick_train(
    data_path="./data/sample_images",  # Folder with class subfolders
    model="resnet50",
    epochs=5,
    verbose=True,
)

# Save and visualize
torch.save(model.state_dict(), "quickstart_model.pth")
plot_results(history, save_path="quickstart_results.png")

print("\nDone! Model saved to quickstart_model.pth")
