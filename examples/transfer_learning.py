#!/usr/bin/env python3
"""
Transfer Learning Example - Fine-tune pretrained models on your data.

Demonstrates best practices for transfer learning with limited data.

Usage:
    python examples/transfer_learning.py --data ./my_small_dataset
"""

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from cv_pipeline import load_image_folder, get_model, plot_results


def train_with_frozen_backbone(model, train_loader, val_loader, device, epochs=5):
    """Phase 1: Train only the classifier head."""
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier (works for most architectures)
    for name, param in model.named_parameters():
        if any(x in name for x in ['fc', 'classifier', 'head']):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1: Training {trainable:,} parameters (classifier only)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history


def finetune_full_model(model, train_loader, val_loader, device, epochs=10):
    """Phase 2: Fine-tune the entire model with lower learning rate."""
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 2: Fine-tuning all {total_params:,} parameters")

    # Use different learning rates for backbone and head
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if any(x in name for x in ['fc', 'classifier', 'head']):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained layers
        {'params': head_params, 'lr': 1e-4}       # Higher LR for head
    ], weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return history


def main():
    parser = argparse.ArgumentParser(description="Transfer learning example")
    parser.add_argument("--data", required=True, help="Path to image data")
    parser.add_argument("--model", default="resnet50", help="Base model")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Head-only epochs")
    parser.add_argument("--phase2-epochs", type=int, default=10, help="Full fine-tune epochs")
    args = parser.parse_args()

    # Setup device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, class_names = load_image_folder(
        args.data,
        image_size=(224, 224),
        batch_size=32,
        augment=True,
        split=0.2,
    )
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")

    # Get pretrained model
    model = get_model(args.model, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    print(f"\n{'='*60}")
    print("Transfer Learning Strategy")
    print("="*60)
    print("1. Freeze backbone, train classifier head")
    print("2. Unfreeze all, fine-tune with lower learning rate")
    print("="*60)

    # Phase 1: Train classifier only
    history1 = train_with_frozen_backbone(
        model, train_loader, val_loader, device,
        epochs=args.phase1_epochs
    )

    # Phase 2: Fine-tune everything
    history2 = finetune_full_model(
        model, train_loader, val_loader, device,
        epochs=args.phase2_epochs
    )

    # Combine histories
    full_history = {
        "train_loss": history1["train_loss"] + history2["train_loss"],
        "val_loss": history1["val_loss"] + history2["val_loss"],
        "val_acc": history1["val_acc"] + history2["val_acc"],
    }

    # Save results
    torch.save(model.state_dict(), "transfer_learning_model.pth")
    plot_results(full_history, save_path="transfer_learning_results.png")

    print(f"\nFinal accuracy: {full_history['val_acc'][-1]*100:.2f}%")
    print("Model saved to: transfer_learning_model.pth")


if __name__ == "__main__":
    main()
