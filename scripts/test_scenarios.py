#!/usr/bin/env python3
"""
Quick Scenario Validation Script

Tests key cv_pipeline functionality with minimal resources:
- Uses tiny synthetic datasets (no downloads)
- Runs 1 epoch only
- Uses smallest models (resnet18)
- Skips GPU-intensive operations

Run: python scripts/test_scenarios.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_tiny_dataset(base_path: str, num_classes: int = 3, images_per_class: int = 5):
    """Create a tiny synthetic dataset for testing."""
    import numpy as np
    from PIL import Image

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_classes):
        class_dir = base_path / f"class_{i}"
        class_dir.mkdir(exist_ok=True)

        for j in range(images_per_class):
            # Create random 64x64 RGB image
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"img_{j}.jpg")

    return str(base_path)


def test_scenario_1_dataset_analysis():
    """Test: Dataset Analysis"""
    print("\n" + "="*60)
    print("SCENARIO 1: Dataset Analysis")
    print("="*60)

    from cv_pipeline import analyze_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data_path = create_tiny_dataset(f"{tmpdir}/data", num_classes=3, images_per_class=10)

        # Run analysis
        stats = analyze_dataset(data_path, show_samples=False)

        # Verify
        assert stats['total_images'] == 30, f"Expected 30 images, got {stats['total_images']}"
        assert stats['num_classes'] == 3, f"Expected 3 classes, got {stats['num_classes']}"
        assert 'class_distribution' in stats

        print("✅ Dataset analysis working correctly")
        return True


def test_scenario_2_data_loading():
    """Test: Data Loading"""
    print("\n" + "="*60)
    print("SCENARIO 2: Data Loading")
    print("="*60)

    from cv_pipeline import load_image_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = create_tiny_dataset(f"{tmpdir}/data", num_classes=2, images_per_class=8)

        train_loader, val_loader, class_names = load_image_folder(
            data_path,
            batch_size=4,
            image_size=(64, 64),
            split=0.25
        )

        assert len(class_names) == 2
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check batch shape
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (3, 64, 64), f"Unexpected shape: {images.shape}"

        print(f"✅ Loaded {len(class_names)} classes: {class_names}")
        print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return True


def test_scenario_3_model_loading():
    """Test: Model Loading"""
    print("\n" + "="*60)
    print("SCENARIO 3: Model Loading")
    print("="*60)

    from cv_pipeline import get_model
    import torch

    # Test different architectures
    models_to_test = [
        ("resnet18", 10),
        ("efficientnet_b0", 5),
    ]

    for model_name, num_classes in models_to_test:
        model = get_model(model_name, num_classes=num_classes, pretrained=False)

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, num_classes), f"Expected (1, {num_classes}), got {output.shape}"
        print(f"✅ {model_name}: output shape {output.shape}")

    return True


def test_scenario_4_quick_train():
    """Test: Quick Training (1 epoch)"""
    print("\n" + "="*60)
    print("SCENARIO 4: Quick Training")
    print("="*60)

    from cv_pipeline import quick_train

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = create_tiny_dataset(f"{tmpdir}/data", num_classes=2, images_per_class=10)

        model, history = quick_train(
            data_path,
            model="resnet18",
            epochs=1,
            batch_size=4,
            image_size=(64, 64),
            pretrained=False,  # Faster, no download
            verbose=True
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1

        print(f"✅ Training completed: loss={history['train_loss'][0]:.4f}")
        return True


def test_scenario_5_model_export():
    """Test: Model Export"""
    print("\n" + "="*60)
    print("SCENARIO 5: Model Export")
    print("="*60)

    from cv_pipeline import get_model, export_model
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        model = get_model("resnet18", num_classes=5, pretrained=False)

        # Test TorchScript export
        ts_path = f"{tmpdir}/model.pt"
        result = export_model(model, ts_path, format="torchscript", input_size=(64, 64))
        assert os.path.exists(result)

        # Verify it loads
        loaded = torch.jit.load(result)
        assert loaded is not None

        print(f"✅ TorchScript export successful: {result}")

        # Test state_dict export
        sd_path = f"{tmpdir}/weights.pth"
        result = export_model(model, sd_path, format="state_dict")
        assert os.path.exists(result)
        print(f"✅ State dict export successful: {result}")

        return True


def test_scenario_6_notebook_generation():
    """Test: Notebook Generation"""
    print("\n" + "="*60)
    print("SCENARIO 6: Notebook Generation")
    print("="*60)

    from cv_pipeline import generate_notebook
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["classification", "detection", "segmentation"]:
            output_path = f"{tmpdir}/{task}.ipynb"
            result = generate_notebook(task, output_path)

            assert os.path.exists(result)

            # Verify valid notebook format
            with open(result) as f:
                notebook = json.load(f)
            assert "cells" in notebook
            assert "nbformat" in notebook

            print(f"✅ {task.capitalize()} notebook generated")

        return True


def test_scenario_7_cli_commands():
    """Test: CLI Commands"""
    print("\n" + "="*60)
    print("SCENARIO 7: CLI Commands")
    print("="*60)

    from cv_pipeline.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = create_tiny_dataset(f"{tmpdir}/data", num_classes=2, images_per_class=5)

        # Test analyze command
        exit_code = main(["analyze", "--data", data_path, "--no-samples"])
        assert exit_code == 0, "Analyze command failed"
        print("✅ cv-pipeline analyze: OK")

        # Test notebook command
        nb_path = f"{tmpdir}/test.ipynb"
        exit_code = main(["notebook", "--task", "classification", "--output", nb_path])
        assert exit_code == 0, "Notebook command failed"
        print("✅ cv-pipeline notebook: OK")

        return True


def test_scenario_8_plot_results():
    """Test: Plot Results"""
    print("\n" + "="*60)
    print("SCENARIO 8: Plot Results")
    print("="*60)

    from cv_pipeline import plot_results
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    with tempfile.TemporaryDirectory() as tmpdir:
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.5],
            "val_loss": [1.1, 0.9, 0.7, 0.6],
            "train_acc": [0.5, 0.6, 0.7, 0.75],
            "val_acc": [0.45, 0.55, 0.65, 0.70],
        }

        plot_path = f"{tmpdir}/results.png"
        plot_results(history, save_path=plot_path)

        assert os.path.exists(plot_path)
        print(f"✅ Plot saved: {plot_path}")

        return True


def run_all_tests():
    """Run all scenario tests."""
    print("\n" + "="*60)
    print("CV PIPELINE - SCENARIO VALIDATION")
    print("="*60)
    print("Running lightweight tests to validate key functionality...")
    print("(Uses synthetic data, no downloads, minimal compute)")

    tests = [
        ("Dataset Analysis", test_scenario_1_dataset_analysis),
        ("Data Loading", test_scenario_2_data_loading),
        ("Model Loading", test_scenario_3_model_loading),
        ("Quick Training", test_scenario_4_quick_train),
        ("Model Export", test_scenario_5_model_export),
        ("Notebook Generation", test_scenario_6_notebook_generation),
        ("CLI Commands", test_scenario_7_cli_commands),
        ("Plot Results", test_scenario_8_plot_results),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, "FAIL"))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")

    print("-"*60)
    print(f"  Total: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All scenarios validated successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} scenario(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
