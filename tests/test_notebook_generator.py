"""
Tests for cv_pipeline/notebook_generator.py

Tests the notebook generation functionality:
- Classification notebooks
- Detection notebooks
- Segmentation notebooks
"""

import json
import os
from pathlib import Path

import pytest


class TestGenerateNotebook:
    """Tests for generate_notebook function."""

    def test_generate_classification_notebook(self, temp_output_dir):
        """Test generating classification notebook."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "classification.ipynb")
        result = generate_notebook(
            task="classification",
            output_path=output_path,
        )

        assert os.path.exists(result)
        assert result == output_path

        # Verify notebook structure
        with open(result) as f:
            notebook = json.load(f)

        assert notebook["nbformat"] == 4
        assert "cells" in notebook
        assert len(notebook["cells"]) > 0

    def test_generate_detection_notebook(self, temp_output_dir):
        """Test generating detection notebook."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "detection.ipynb")
        result = generate_notebook(
            task="detection",
            output_path=output_path,
        )

        assert os.path.exists(result)

        with open(result) as f:
            notebook = json.load(f)

        assert "cells" in notebook
        # Should mention YOLO or detection
        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "detection" in all_source.lower() or "yolo" in all_source.lower()

    def test_generate_segmentation_notebook(self, temp_output_dir):
        """Test generating segmentation notebook."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "segmentation.ipynb")
        result = generate_notebook(
            task="segmentation",
            output_path=output_path,
        )

        assert os.path.exists(result)

        with open(result) as f:
            notebook = json.load(f)

        assert "cells" in notebook
        # Should mention segmentation or UNet
        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "segmentation" in all_source.lower() or "unet" in all_source.lower()

    def test_generate_with_data_path(self, temp_output_dir):
        """Test generating notebook with custom data path."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "custom.ipynb")
        custom_data = "/path/to/my/data"

        result = generate_notebook(
            task="classification",
            output_path=output_path,
            data_path=custom_data,
        )

        with open(result) as f:
            notebook = json.load(f)

        # Custom data path should appear in notebook
        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert custom_data in all_source

    def test_generate_with_custom_model(self, temp_output_dir):
        """Test generating notebook with custom model."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "custom_model.ipynb")
        result = generate_notebook(
            task="classification",
            output_path=output_path,
            model="efficientnet_b3",
        )

        with open(result) as f:
            notebook = json.load(f)

        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "efficientnet_b3" in all_source

    def test_generate_invalid_task(self, temp_output_dir):
        """Test error handling for invalid task."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "invalid.ipynb")

        with pytest.raises(ValueError) as exc_info:
            generate_notebook(task="invalid_task", output_path=output_path)

        assert "invalid_task" in str(exc_info.value).lower()


class TestNotebookCellStructure:
    """Tests for notebook cell structure."""

    def test_notebook_has_markdown_cells(self, temp_output_dir):
        """Test that notebooks have markdown cells for documentation."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        cell_types = [cell["cell_type"] for cell in notebook["cells"]]
        assert "markdown" in cell_types

    def test_notebook_has_code_cells(self, temp_output_dir):
        """Test that notebooks have code cells."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        cell_types = [cell["cell_type"] for cell in notebook["cells"]]
        assert "code" in cell_types

    def test_code_cells_have_outputs(self, temp_output_dir):
        """Test that code cells have outputs field."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert "outputs" in cell
                assert "execution_count" in cell

    def test_notebook_has_kernelspec(self, temp_output_dir):
        """Test that notebooks have kernel specification."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        assert "metadata" in notebook
        assert "kernelspec" in notebook["metadata"]
        assert notebook["metadata"]["kernelspec"]["language"] == "python"


class TestNotebookContent:
    """Tests for notebook content quality."""

    def test_classification_imports_cv_pipeline(self, temp_output_dir):
        """Test that classification notebook imports cv_pipeline."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "cv_pipeline" in all_source or "from cv_pipeline" in all_source

    def test_classification_has_training_section(self, temp_output_dir):
        """Test that classification notebook has training section."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        ).lower()

        assert "train" in all_source

    def test_classification_has_evaluation_section(self, temp_output_dir):
        """Test that classification notebook has evaluation section."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("classification", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        ).lower()

        assert "evaluat" in all_source or "accuracy" in all_source

    def test_detection_mentions_yolo(self, temp_output_dir):
        """Test that detection notebook mentions YOLO."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("detection", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        assert "YOLO" in all_source or "yolo" in all_source

    def test_segmentation_mentions_unet(self, temp_output_dir):
        """Test that segmentation notebook mentions UNet or segmentation models."""
        from cv_pipeline.notebook_generator import generate_notebook

        output_path = os.path.join(temp_output_dir, "test.ipynb")
        generate_notebook("segmentation", output_path)

        with open(output_path) as f:
            notebook = json.load(f)

        all_source = " ".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        ).lower()

        assert "unet" in all_source or "segmentation" in all_source


class TestHelperFunctions:
    """Tests for helper functions in notebook_generator."""

    def test_create_cell_markdown(self):
        """Test creating markdown cell."""
        from cv_pipeline.notebook_generator import _create_cell

        cell = _create_cell("markdown", "# Test Header\n\nSome text")

        assert cell["cell_type"] == "markdown"
        assert "# Test Header" in cell["source"]
        assert "outputs" not in cell

    def test_create_cell_code(self):
        """Test creating code cell."""
        from cv_pipeline.notebook_generator import _create_cell

        cell = _create_cell("code", "print('hello')")

        assert cell["cell_type"] == "code"
        assert "outputs" in cell
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
