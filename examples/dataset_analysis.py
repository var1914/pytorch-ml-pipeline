#!/usr/bin/env python3
"""
Dataset Analysis Example - Understand your data before training.

Get comprehensive insights about your image dataset including:
- Class distribution and balance
- Image statistics
- Sample visualization
- Recommendations for training

Usage:
    python examples/dataset_analysis.py --data ./my_images
"""

import argparse
import json
from pathlib import Path

from cv_pipeline import analyze_dataset


def main():
    parser = argparse.ArgumentParser(description="Analyze image dataset")
    parser.add_argument("--data", required=True, help="Path to image data")
    parser.add_argument("--output", help="Save report to JSON file")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    print(f"Analyzing dataset: {args.data}\n")

    # Analyze
    stats = analyze_dataset(
        args.data,
        show_samples=not args.no_visualize,
        save_report=args.output
    )

    # Print comprehensive report
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nOverview:")
    print(f"  Total images:    {stats['total_images']:,}")
    print(f"  Number of classes: {stats['num_classes']}")

    if stats.get("class_distribution"):
        print(f"\nClass Distribution:")
        total = sum(stats["class_distribution"].values())
        for cls, count in sorted(stats["class_distribution"].items()):
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {cls:<20} {count:>6} ({pct:>5.1f}%) {bar}")

        # Check for imbalance
        counts = list(stats["class_distribution"].values())
        ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if ratio > 3:
            print(f"\n  WARNING: Class imbalance detected (ratio {ratio:.1f}:1)")
            print("  Consider: weighted loss, oversampling, or data augmentation")

    if stats.get("formats"):
        print(f"\nImage Formats:")
        for fmt, count in stats["formats"].items():
            print(f"  {fmt}: {count}")

    if stats.get("avg_size"):
        print(f"\nImage Dimensions:")
        print(f"  Average size: {stats['avg_size'][0]:.0f} x {stats['avg_size'][1]:.0f}")

    # Training recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print("=" * 60)

    n_images = stats['total_images']
    n_classes = stats['num_classes']

    if n_images < 100 * n_classes:
        print("- Limited data: Use transfer learning with pretrained models")
        print("- Enable strong data augmentation")
        print("- Consider using mobilenet_v2 or efficientnet_b0 (smaller models)")
    elif n_images < 1000 * n_classes:
        print("- Moderate data: Transfer learning recommended")
        print("- Try resnet50 or efficientnet_b0")
        print("- Use moderate augmentation")
    else:
        print("- Sufficient data for training from scratch")
        print("- Can experiment with larger models (resnet101, efficientnet_b3)")
        print("- Consider training longer epochs")

    # Suggested training command
    print(f"\nQuick Start:")
    print(f"  cv-pipeline train --data {args.data} --model resnet50 --epochs 10")

    if args.output:
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
