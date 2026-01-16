#!/usr/bin/env python3
"""
Model Comparison Example - Compare different architectures on your data.

Find the best model for your specific dataset with a simple comparison.

Usage:
    python examples/model_comparison.py --data ./test_images
"""

from cv_pipeline import load_image_folder, compare_models, get_model
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compare CV models")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--models", default="resnet50,efficientnet_b0,mobilenet_v2",
                        help="Comma-separated model names")
    args = parser.parse_args()

    # Load test data
    print(f"Loading data from: {args.data}")
    _, test_loader, class_names = load_image_folder(
        args.data,
        batch_size=32,
        split=0.0,  # Use all data for testing
    )
    print(f"Found {len(class_names)} classes: {class_names}")

    # Compare models
    models_to_compare = [m.strip() for m in args.models.split(",")]
    print(f"\nComparing {len(models_to_compare)} models...")

    results = compare_models(
        models=models_to_compare,
        test_loader=test_loader,
        num_classes=len(class_names),
    )

    # Print results table
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Accuracy':>12} {'Parameters':>15}")
    print("=" * 60)

    best_model = None
    best_acc = 0

    for model_name, metrics in results.items():
        acc = metrics["accuracy"] * 100
        params = metrics["params"]
        params_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"

        print(f"{model_name:<25} {acc:>11.2f}% {params_str:>15}")

        if acc > best_acc:
            best_acc = acc
            best_model = model_name

    print("=" * 60)
    print(f"\nBest model: {best_model} ({best_acc:.2f}% accuracy)")

    # Recommendations
    print("\nRecommendations:")
    if best_acc < 70:
        print("- Consider using more training data or data augmentation")
        print("- Try a larger model (resnet101, efficientnet_b3)")
    elif best_acc < 85:
        print("- Try fine-tuning with a lower learning rate")
        print("- Consider ensemble methods for better performance")
    else:
        print("- Great performance! Ready for deployment")
        print("- Consider quantization for faster inference")


if __name__ == "__main__":
    main()
