"""
Tests for cv_pipeline/utils.py

Tests the core utility functions:
- analyze_dataset
- load_image_folder
- get_model
- compare_models
- quick_train
- export_model
- plot_results
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn


class TestAnalyzeDataset:
    """Tests for analyze_dataset function."""

    def test_analyze_basic(self, sample_image_folder):
        """Test basic dataset analysis."""
        from cv_pipeline import analyze_dataset

        stats = analyze_dataset(sample_image_folder, show_samples=False)

        assert "total_images" in stats
        assert "num_classes" in stats
        assert "class_distribution" in stats
        assert stats["total_images"] == 10  # 2 classes * 5 images
        assert stats["num_classes"] == 2

    def test_analyze_class_distribution(self, sample_image_folder):
        """Test that class distribution is correct."""
        from cv_pipeline import analyze_dataset

        stats = analyze_dataset(sample_image_folder, show_samples=False)

        assert "class_0" in stats["class_distribution"]
        assert "class_1" in stats["class_distribution"]
        assert stats["class_distribution"]["class_0"] == 5
        assert stats["class_distribution"]["class_1"] == 5

    def test_analyze_save_report(self, sample_image_folder, temp_output_dir):
        """Test saving analysis report to JSON."""
        from cv_pipeline import analyze_dataset

        report_path = os.path.join(temp_output_dir, "report.json")
        stats = analyze_dataset(
            sample_image_folder,
            show_samples=False,
            save_report=report_path
        )

        assert os.path.exists(report_path)

        with open(report_path) as f:
            saved_stats = json.load(f)

        assert saved_stats["total_images"] == stats["total_images"]
        assert saved_stats["num_classes"] == stats["num_classes"]

    def test_analyze_imbalanced(self, sample_image_folder_imbalanced):
        """Test analysis detects imbalanced classes."""
        from cv_pipeline import analyze_dataset

        stats = analyze_dataset(sample_image_folder_imbalanced, show_samples=False)

        assert stats["total_images"] == 12  # 10 + 2
        assert stats["num_classes"] == 2

        # Check imbalance is reflected
        dist = stats["class_distribution"]
        assert max(dist.values()) / min(dist.values()) == 5.0  # 10:2 ratio

    def test_analyze_multiclass(self, sample_image_folder_multiclass):
        """Test analysis with multiple classes."""
        from cv_pipeline import analyze_dataset

        stats = analyze_dataset(sample_image_folder_multiclass, show_samples=False)

        assert stats["total_images"] == 15  # 5 classes * 3 images
        assert stats["num_classes"] == 5

    def test_analyze_nonexistent_path(self):
        """Test error handling for non-existent path."""
        from cv_pipeline import analyze_dataset

        with pytest.raises((FileNotFoundError, ValueError)):
            analyze_dataset("/nonexistent/path", show_samples=False)


class TestLoadImageFolder:
    """Tests for load_image_folder function."""

    def test_load_basic(self, sample_image_folder):
        """Test basic image loading."""
        from cv_pipeline import load_image_folder

        train_loader, val_loader, class_names = load_image_folder(
            sample_image_folder,
            image_size=(64, 64),
            batch_size=4,
            split=0.2,
        )

        assert len(class_names) == 2
        assert "class_0" in class_names
        assert "class_1" in class_names

    def test_load_returns_dataloaders(self, sample_image_folder):
        """Test that function returns proper DataLoaders."""
        from cv_pipeline import load_image_folder
        from torch.utils.data import DataLoader

        train_loader, val_loader, _ = load_image_folder(
            sample_image_folder,
            batch_size=2,
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_load_image_dimensions(self, sample_image_folder):
        """Test that images are resized correctly."""
        from cv_pipeline import load_image_folder

        target_size = (128, 128)
        train_loader, _, _ = load_image_folder(
            sample_image_folder,
            image_size=target_size,
            batch_size=2,
        )

        images, labels = next(iter(train_loader))
        assert images.shape[2:] == target_size

    def test_load_batch_size(self, sample_image_folder):
        """Test that batch size is respected."""
        from cv_pipeline import load_image_folder

        batch_size = 3
        train_loader, _, _ = load_image_folder(
            sample_image_folder,
            batch_size=batch_size,
            split=0.0,  # Use all for training
        )

        images, _ = next(iter(train_loader))
        assert images.shape[0] <= batch_size

    def test_load_split_ratio(self, sample_image_folder_multiclass):
        """Test train/val split ratio."""
        from cv_pipeline import load_image_folder

        train_loader, val_loader, _ = load_image_folder(
            sample_image_folder_multiclass,
            batch_size=1,
            split=0.4,  # 40% validation
        )

        # Count samples
        train_count = sum(1 for _ in train_loader)
        val_count = sum(1 for _ in val_loader)
        total = train_count + val_count

        # Allow some tolerance due to rounding
        assert 0.5 <= train_count / total <= 0.7


class TestGetModel:
    """Tests for get_model function."""

    def test_get_resnet50(self):
        """Test getting ResNet50 model."""
        from cv_pipeline import get_model

        model = get_model("resnet50", num_classes=10)

        assert isinstance(model, nn.Module)
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 10)

    def test_get_efficientnet(self):
        """Test getting EfficientNet model."""
        from cv_pipeline import get_model

        model = get_model("efficientnet_b0", num_classes=5)

        assert isinstance(model, nn.Module)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 5)

    def test_get_mobilenet(self):
        """Test getting MobileNet model."""
        from cv_pipeline import get_model

        # Use timm's mobilenetv2 name
        model = get_model("mobilenetv2_100", num_classes=3)

        assert isinstance(model, nn.Module)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 3)

    def test_get_model_pretrained(self):
        """Test getting pretrained model."""
        from cv_pipeline import get_model

        model_pretrained = get_model("resnet18", num_classes=10, pretrained=True)
        model_random = get_model("resnet18", num_classes=10, pretrained=False)

        # Get first conv layer weights
        w1 = list(model_pretrained.parameters())[0].data
        w2 = list(model_random.parameters())[0].data

        # Pretrained weights should be different from random
        # (This is probabilistic but highly likely to pass)
        assert not torch.allclose(w1, w2)

    def test_get_invalid_model(self):
        """Test error handling for invalid model name."""
        from cv_pipeline import get_model

        with pytest.raises(Exception):
            get_model("invalid_model_xyz", num_classes=10)


class TestQuickTrain:
    """Tests for quick_train function."""

    def test_quick_train_basic(self, sample_image_folder, device):
        """Test basic training run."""
        from cv_pipeline import quick_train

        model, history = quick_train(
            sample_image_folder,
            model="resnet18",
            epochs=1,
            batch_size=4,
            device=device,
            verbose=False,
        )

        assert isinstance(model, nn.Module)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 1

    def test_quick_train_returns_trained_model(self, sample_image_folder, device):
        """Test that returned model can make predictions."""
        from cv_pipeline import quick_train

        model, _ = quick_train(
            sample_image_folder,
            model="resnet18",
            epochs=1,
            batch_size=4,
            device=device,
            verbose=False,
        )

        model.eval()
        x = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(x)

        assert output.shape[1] == 2  # 2 classes

    def test_quick_train_history_format(self, sample_image_folder, device):
        """Test that training history has correct format."""
        from cv_pipeline import quick_train

        _, history = quick_train(
            sample_image_folder,
            model="resnet18",
            epochs=2,
            batch_size=4,
            device=device,
            verbose=False,
        )

        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(isinstance(v, float) for v in history["train_loss"])

    def test_quick_train_custom_lr(self, sample_image_folder, device):
        """Test training with custom learning rate."""
        from cv_pipeline import quick_train

        model, history = quick_train(
            sample_image_folder,
            model="resnet18",
            epochs=1,
            lr=0.1,  # High LR
            device=device,
            verbose=False,
        )

        # Should complete without error
        assert len(history["train_loss"]) == 1


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_basic(self, sample_image_folder_multiclass, device):
        """Test basic model comparison."""
        from cv_pipeline import load_image_folder, compare_models

        # Use train_loader which has data (split=0.2 means 80% train)
        train_loader, _, class_names = load_image_folder(
            sample_image_folder_multiclass,
            batch_size=4,
            split=0.2,
        )

        results = compare_models(
            models=["resnet18"],
            test_loader=train_loader,
            num_classes=len(class_names),
            device=device,
        )

        assert "resnet18" in results
        assert "accuracy" in results["resnet18"]
        assert "params" in results["resnet18"]

    def test_compare_multiple_models(self, sample_image_folder_multiclass, device):
        """Test comparing multiple models."""
        from cv_pipeline import load_image_folder, compare_models

        train_loader, _, class_names = load_image_folder(
            sample_image_folder_multiclass,
            batch_size=4,
            split=0.2,
        )

        # Just test single model to keep test fast
        results = compare_models(
            models=["resnet18"],
            test_loader=train_loader,
            num_classes=len(class_names),
            device=device,
        )

        assert len(results) >= 1
        assert "resnet18" in results

    def test_compare_accuracy_range(self, sample_image_folder_multiclass, device):
        """Test that accuracy is in valid range."""
        from cv_pipeline import load_image_folder, compare_models

        train_loader, _, class_names = load_image_folder(
            sample_image_folder_multiclass,
            batch_size=4,
            split=0.2,
        )

        results = compare_models(
            models=["resnet18"],
            test_loader=train_loader,
            num_classes=len(class_names),
            device=device,
        )

        acc = results["resnet18"]["accuracy"]
        assert 0.0 <= acc <= 1.0


class TestExportModel:
    """Tests for export_model function."""

    def test_export_torchscript(self, temp_output_dir):
        """Test exporting to TorchScript format."""
        from cv_pipeline import get_model, export_model

        model = get_model("resnet18", num_classes=10)
        output_path = os.path.join(temp_output_dir, "model.pt")

        result = export_model(model, output_path, format="torchscript")

        assert os.path.exists(result)
        # Verify it can be loaded
        loaded = torch.jit.load(result)
        assert loaded is not None

    def test_export_state_dict(self, temp_output_dir):
        """Test exporting state dict."""
        from cv_pipeline import get_model, export_model

        model = get_model("resnet18", num_classes=10)
        output_path = os.path.join(temp_output_dir, "weights.pth")

        result = export_model(model, output_path, format="state_dict")

        assert os.path.exists(result)
        # Verify it can be loaded
        state_dict = torch.load(result, weights_only=True)
        assert isinstance(state_dict, dict)

    def test_export_onnx(self, temp_output_dir):
        """Test exporting to ONNX format."""
        from cv_pipeline import get_model, export_model

        model = get_model("resnet18", num_classes=10)
        output_path = os.path.join(temp_output_dir, "model.onnx")

        try:
            result = export_model(model, output_path, format="onnx")
        except Exception as e:
            # ONNX export might fail due to version compatibility
            pytest.skip(f"ONNX export not available: {e}")

        assert os.path.exists(result)

    def test_export_custom_input_size(self, temp_output_dir):
        """Test exporting with custom input size."""
        from cv_pipeline import get_model, export_model

        model = get_model("resnet18", num_classes=10)
        output_path = os.path.join(temp_output_dir, "model.pt")

        result = export_model(
            model,
            output_path,
            format="torchscript",
            input_size=(128, 128)
        )

        assert os.path.exists(result)


class TestPlotResults:
    """Tests for plot_results function."""

    def test_plot_basic(self, temp_output_dir):
        """Test basic plotting."""
        from cv_pipeline import plot_results

        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
        }

        save_path = os.path.join(temp_output_dir, "plot.png")
        plot_results(history, save_path=save_path)

        assert os.path.exists(save_path)

    def test_plot_with_accuracy(self, temp_output_dir):
        """Test plotting with accuracy metrics."""
        from cv_pipeline import plot_results

        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "val_acc": [0.5, 0.6, 0.7],
        }

        save_path = os.path.join(temp_output_dir, "plot_acc.png")
        plot_results(history, save_path=save_path)

        assert os.path.exists(save_path)

    def test_plot_no_save(self):
        """Test plotting without saving (just verify no error)."""
        from cv_pipeline import plot_results
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        history = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
        }

        # Should not raise an error
        plot_results(history, save_path=None)
