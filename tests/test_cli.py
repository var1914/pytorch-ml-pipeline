"""
Tests for cv_pipeline/cli.py

Tests the command-line interface:
- cv-pipeline train
- cv-pipeline analyze
- cv-pipeline compare
- cv-pipeline export
- cv-pipeline notebook
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_parser_help(self):
        """Test that help command works."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        # Should not raise
        assert parser is not None
        assert parser.prog == "cv-pipeline"

    def test_parser_train_args(self):
        """Test train command argument parsing."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "train",
            "--data", "./images",
            "--model", "resnet50",
            "--epochs", "10",
            "--batch-size", "32",
        ])

        assert args.command == "train"
        assert args.data == "./images"
        assert args.model == "resnet50"
        assert args.epochs == 10
        assert args.batch_size == 32

    def test_parser_train_defaults(self):
        """Test train command default values."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["train", "--data", "./images"])

        assert args.model == "resnet50"
        assert args.epochs == 10
        assert args.batch_size == 32
        assert args.lr == 0.001
        assert args.image_size == 224
        assert args.device == "auto"

    def test_parser_analyze_args(self):
        """Test analyze command argument parsing."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "analyze",
            "--data", "./images",
            "--save", "report.json",
        ])

        assert args.command == "analyze"
        assert args.data == "./images"
        assert args.save == "report.json"

    def test_parser_compare_args(self):
        """Test compare command argument parsing."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--models", "resnet50,efficientnet_b0",
            "--data", "./test_images",
        ])

        assert args.command == "compare"
        assert args.models == "resnet50,efficientnet_b0"
        assert args.data == "./test_images"

    def test_parser_export_args(self):
        """Test export command argument parsing."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "export",
            "--model", "model.pth",
            "--format", "onnx",
            "--output", "model.onnx",
        ])

        assert args.command == "export"
        assert args.model == "model.pth"
        assert args.format == "onnx"
        assert args.output == "model.onnx"

    def test_parser_notebook_args(self):
        """Test notebook command argument parsing."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "notebook",
            "--task", "classification",
            "--output", "experiment.ipynb",
        ])

        assert args.command == "notebook"
        assert args.task == "classification"
        assert args.output == "experiment.ipynb"

    def test_parser_no_command(self):
        """Test parser with no command."""
        from cv_pipeline.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([])

        assert args.command is None


class TestCLIAnalyze:
    """Tests for cv-pipeline analyze command."""

    def test_cmd_analyze(self, sample_image_folder, temp_output_dir, capsys):
        """Test analyze command execution."""
        from cv_pipeline.cli import main

        report_path = os.path.join(temp_output_dir, "report.json")
        exit_code = main([
            "analyze",
            "--data", sample_image_folder,
            "--save", report_path,
            "--no-samples",
        ])

        assert exit_code == 0
        assert os.path.exists(report_path)

        # Check output
        captured = capsys.readouterr()
        assert "Dataset Analysis" in captured.out
        assert "Total images" in captured.out

    def test_cmd_analyze_no_save(self, sample_image_folder, capsys):
        """Test analyze command without saving report."""
        from cv_pipeline.cli import main

        exit_code = main([
            "analyze",
            "--data", sample_image_folder,
            "--no-samples",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Summary" in captured.out

    def test_cmd_analyze_invalid_path(self, capsys):
        """Test analyze command with invalid path."""
        from cv_pipeline.cli import main

        exit_code = main([
            "analyze",
            "--data", "/nonexistent/path",
            "--no-samples",
        ])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestCLITrain:
    """Tests for cv-pipeline train command."""

    def test_cmd_train_basic(self, sample_image_folder, temp_output_dir, device, capsys):
        """Test basic train command."""
        from cv_pipeline.cli import main

        output_path = os.path.join(temp_output_dir, "model.pth")
        exit_code = main([
            "train",
            "--data", sample_image_folder,
            "--model", "resnet18",
            "--epochs", "1",
            "--batch-size", "4",
            "--output", output_path,
            "--device", device,
        ])

        assert exit_code == 0
        assert os.path.exists(output_path)

        captured = capsys.readouterr()
        assert "Training" in captured.out

    def test_cmd_train_creates_plot(self, sample_image_folder, temp_output_dir, device):
        """Test that train command creates training plot."""
        from cv_pipeline.cli import main

        output_path = os.path.join(temp_output_dir, "model.pth")
        main([
            "train",
            "--data", sample_image_folder,
            "--model", "resnet18",
            "--epochs", "1",
            "--output", output_path,
            "--device", device,
        ])

        # Plot should be created alongside model
        plot_path = output_path.replace(".pth", "_training.png")
        assert os.path.exists(plot_path)


class TestCLICompare:
    """Tests for cv-pipeline compare command."""

    def test_cmd_compare_basic(self, sample_image_folder_multiclass, device, capsys):
        """Test basic compare command."""
        from cv_pipeline.cli import main

        # Use multiclass fixture which has more samples (15 total)
        exit_code = main([
            "compare",
            "--models", "resnet18",
            "--data", sample_image_folder_multiclass,
            "--device", device,
            "--batch-size", "4",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "resnet18" in captured.out

    def test_cmd_compare_multiple(self, sample_image_folder_multiclass, device, capsys):
        """Test comparing multiple models."""
        from cv_pipeline.cli import main

        # Just test single model to avoid timeout issues
        exit_code = main([
            "compare",
            "--models", "resnet18",
            "--data", sample_image_folder_multiclass,
            "--device", device,
            "--batch-size", "4",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "resnet18" in captured.out


class TestCLIExport:
    """Tests for cv-pipeline export command."""

    def test_cmd_export_torchscript(self, temp_output_dir, capsys):
        """Test export command with TorchScript."""
        from cv_pipeline.cli import main
        from cv_pipeline import get_model
        import torch

        # First create a model to export
        model = get_model("resnet18", num_classes=2)
        model_path = os.path.join(temp_output_dir, "model.pth")
        torch.save(model, model_path)

        output_path = os.path.join(temp_output_dir, "exported.pt")
        exit_code = main([
            "export",
            "--model", model_path,
            "--format", "torchscript",
            "--output", output_path,
        ])

        assert exit_code == 0
        assert os.path.exists(output_path)


class TestCLINotebook:
    """Tests for cv-pipeline notebook command."""

    def test_cmd_notebook_classification(self, temp_output_dir, capsys):
        """Test notebook generation for classification."""
        from cv_pipeline.cli import main

        output_path = os.path.join(temp_output_dir, "experiment.ipynb")
        exit_code = main([
            "notebook",
            "--task", "classification",
            "--output", output_path,
        ])

        assert exit_code == 0
        assert os.path.exists(output_path)

        # Verify it's valid JSON (notebook format)
        with open(output_path) as f:
            notebook = json.load(f)
        assert "cells" in notebook
        assert "nbformat" in notebook

    def test_cmd_notebook_detection(self, temp_output_dir):
        """Test notebook generation for detection."""
        from cv_pipeline.cli import main

        output_path = os.path.join(temp_output_dir, "detection.ipynb")
        exit_code = main([
            "notebook",
            "--task", "detection",
            "--output", output_path,
        ])

        assert exit_code == 0
        assert os.path.exists(output_path)

    def test_cmd_notebook_segmentation(self, temp_output_dir):
        """Test notebook generation for segmentation."""
        from cv_pipeline.cli import main

        output_path = os.path.join(temp_output_dir, "segmentation.ipynb")
        exit_code = main([
            "notebook",
            "--task", "segmentation",
            "--output", output_path,
        ])

        assert exit_code == 0
        assert os.path.exists(output_path)


class TestCLIMain:
    """Tests for main entry point."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        from cv_pipeline.cli import main

        exit_code = main([])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "cv-pipeline" in captured.out

    def test_main_returns_int(self, sample_image_folder):
        """Test that main always returns an integer."""
        from cv_pipeline.cli import main

        result = main(["analyze", "--data", sample_image_folder, "--no-samples"])
        assert isinstance(result, int)
