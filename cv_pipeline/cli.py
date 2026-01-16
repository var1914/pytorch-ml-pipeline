"""
CV Pipeline CLI - Command-line interface for quick ML experimentation.

Usage:
    cv-pipeline train --data ./my_images --model resnet50 --epochs 10
    cv-pipeline analyze --data ./my_images
    cv-pipeline compare --models resnet50,efficientnet_b0 --data ./test_images
    cv-pipeline export --model model.pth --format onnx
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="cv-pipeline",
        description="A practical toolkit for computer vision research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a classifier on your image folder
  cv-pipeline train --data ./my_images --model resnet50 --epochs 10

  # Analyze your dataset before training
  cv-pipeline analyze --data ./my_images --save report.json

  # Compare multiple models on the same data
  cv-pipeline compare --models resnet50,efficientnet_b0 --data ./test_images

  # Export trained model for deployment
  cv-pipeline export --model best_model.pth --format onnx --output model.onnx

  # Generate a notebook template
  cv-pipeline notebook --task classification --output my_experiment.ipynb
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model on your data",
        description="Train a classification/detection/segmentation model"
    )
    train_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to image folder (with class subfolders)"
    )
    train_parser.add_argument(
        "--model", "-m",
        default="resnet50",
        help="Model architecture (resnet50, efficientnet_b0, vit_small, etc.)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (default: 224)"
    )
    train_parser.add_argument(
        "--output", "-o",
        default="trained_model.pth",
        help="Output path for saved model"
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)"
    )
    train_parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained weights"
    )
    train_parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a dataset",
        description="Get statistics and insights about your image dataset"
    )
    analyze_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to image folder"
    )
    analyze_parser.add_argument(
        "--save", "-s",
        help="Save report to JSON file"
    )
    analyze_parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Don't show sample images"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple models",
        description="Evaluate and compare multiple models on the same test data"
    )
    compare_parser.add_argument(
        "--models", "-m",
        required=True,
        help="Comma-separated model names (e.g., resnet50,efficientnet_b0)"
    )
    compare_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to test data folder"
    )
    compare_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    compare_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (default: 224)"
    )
    compare_parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (default: auto)"
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export model for deployment",
        description="Export a trained model to different formats"
    )
    export_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained model (.pth file)"
    )
    export_parser.add_argument(
        "--format", "-f",
        default="torchscript",
        choices=["torchscript", "onnx", "state_dict"],
        help="Export format (default: torchscript)"
    )
    export_parser.add_argument(
        "--output", "-o",
        help="Output path (default: auto-generated)"
    )
    export_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (default: 224)"
    )

    # Notebook command
    notebook_parser = subparsers.add_parser(
        "notebook",
        help="Generate experiment notebook",
        description="Create a Jupyter notebook template for your experiment"
    )
    notebook_parser.add_argument(
        "--task", "-t",
        default="classification",
        choices=["classification", "detection", "segmentation"],
        help="Task type (default: classification)"
    )
    notebook_parser.add_argument(
        "--output", "-o",
        default="experiment.ipynb",
        help="Output notebook path"
    )
    notebook_parser.add_argument(
        "--data",
        help="Data path to include in notebook"
    )
    notebook_parser.add_argument(
        "--model",
        default="resnet50",
        help="Model to use in notebook"
    )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    from .utils import quick_train, plot_results
    import torch

    print(f"\n{'='*60}")
    print("CV Pipeline - Training")
    print(f"{'='*60}")
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"{'='*60}\n")

    try:
        model, history = quick_train(
            data_path=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=(args.image_size, args.image_size),
            device=args.device,
            pretrained=not args.no_pretrained,
            augment=args.augment,
            verbose=True,
        )

        # Save model
        torch.save(model.state_dict(), args.output)
        print(f"\nModel saved to: {args.output}")

        # Save training plot
        plot_path = args.output.replace(".pth", "_training.png")
        plot_results(history, save_path=plot_path)
        print(f"Training plot saved to: {plot_path}")

        return 0

    except Exception as e:
        print(f"\nError during training: {e}")
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the analyze command."""
    from .utils import analyze_dataset
    import json

    print(f"\n{'='*60}")
    print("CV Pipeline - Dataset Analysis")
    print(f"{'='*60}")
    print(f"Analyzing: {args.data}")
    print(f"{'='*60}\n")

    try:
        stats = analyze_dataset(
            data_path=args.data,
            show_samples=not args.no_samples,
            save_report=args.save,
        )

        # Print summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Total images:  {stats['total_images']}")
        print(f"Classes:       {stats.get('num_classes', 0)}")

        if stats.get("formats"):
            print(f"Image formats: {list(stats['formats'].keys())}")

        if stats.get("class_distribution"):
            print("\nClass distribution:")
            for cls, count in stats["class_distribution"].items():
                print(f"  {cls}: {count}")

        if args.save:
            print(f"\nFull report saved to: {args.save}")

        return 0

    except Exception as e:
        print(f"\nError during analysis: {e}")
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute the compare command."""
    from .utils import compare_models, load_image_folder

    models = [m.strip() for m in args.models.split(",")]

    print(f"\n{'='*60}")
    print("CV Pipeline - Model Comparison")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models)}")
    print(f"Data:   {args.data}")
    print(f"{'='*60}\n")

    try:
        # Load test data - use train_loader since we want all data for testing
        # (split=0.2 gives us 80% train, but we'll use train for comparison
        # since it has the most samples)
        train_loader, _, class_names = load_image_folder(
            data_path=args.data,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            split=0.2,  # Small val split, use train for testing
        )

        # Compare models
        results = compare_models(
            models=models,
            test_loader=train_loader,
            num_classes=len(class_names),
            device=args.device,
        )

        # Print results
        print("\n" + "="*60)
        print("Results")
        print("="*60)
        print(f"{'Model':<25} {'Accuracy':>10} {'Params':>12}")
        print("-"*60)

        for model_name, metrics in results.items():
            acc = metrics.get("accuracy", 0) * 100
            params = metrics.get("params", 0)
            params_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.1f}K"
            print(f"{model_name:<25} {acc:>9.2f}% {params_str:>12}")

        return 0

    except Exception as e:
        print(f"\nError during comparison: {e}")
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export command."""
    from .utils import export_model
    import torch

    print(f"\n{'='*60}")
    print("CV Pipeline - Model Export")
    print(f"{'='*60}")
    print(f"Model:  {args.model}")
    print(f"Format: {args.format}")
    print(f"{'='*60}\n")

    try:
        # Load the model
        model = torch.load(args.model, weights_only=False)

        # Determine output path
        output = args.output
        if not output:
            base = Path(args.model).stem
            ext = {"torchscript": ".pt", "onnx": ".onnx", "state_dict": "_weights.pth"}
            output = f"{base}{ext.get(args.format, '.pt')}"

        # Export
        result = export_model(
            model=model,
            output_path=output,
            format=args.format,
            input_size=(args.image_size, args.image_size),
        )

        print(f"\nExported to: {result}")
        return 0

    except Exception as e:
        print(f"\nError during export: {e}")
        return 1


def cmd_notebook(args: argparse.Namespace) -> int:
    """Execute the notebook command."""
    from .notebook_generator import generate_notebook

    print(f"\n{'='*60}")
    print("CV Pipeline - Notebook Generator")
    print(f"{'='*60}")
    print(f"Task:   {args.task}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    try:
        generate_notebook(
            task=args.task,
            output_path=args.output,
            data_path=args.data,
            model=args.model,
        )

        print(f"Notebook created: {args.output}")
        print("\nOpen with: jupyter notebook " + args.output)
        return 0

    except Exception as e:
        print(f"\nError generating notebook: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Route to appropriate command
    commands = {
        "train": cmd_train,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "export": cmd_export,
        "notebook": cmd_notebook,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
