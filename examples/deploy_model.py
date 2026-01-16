#!/usr/bin/env python3
"""
Model Deployment Example - Export trained models for production.

Demonstrates different export formats:
- TorchScript: PyTorch native format
- ONNX: Cross-platform (TensorRT, OpenVINO, etc.)
- State Dict: For continued training

Usage:
    python examples/deploy_model.py --model trained_model.pth --format onnx
"""

import argparse
import torch
from pathlib import Path

from cv_pipeline import export_model, get_model


def benchmark_inference(model, input_size=(224, 224), device="cpu", n_runs=100):
    """Benchmark model inference speed."""
    import time

    model.eval()
    model = model.to(device)

    dummy_input = torch.randn(1, 3, *input_size).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_time = (end - start) / n_runs * 1000  # ms
    fps = 1000 / avg_time

    return avg_time, fps


def main():
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--model", required=True, help="Path to model (.pth) or model name")
    parser.add_argument("--format", default="torchscript",
                        choices=["torchscript", "onnx", "state_dict", "all"],
                        help="Export format")
    parser.add_argument("--output", help="Output path (auto-generated if not provided)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    args = parser.parse_args()

    input_size = (args.image_size, args.image_size)

    # Load or create model
    if Path(args.model).exists():
        print(f"Loading model from: {args.model}")
        # Try to load as full model first
        try:
            model = torch.load(args.model, weights_only=False)
            if isinstance(model, dict):
                # It's a state dict, need architecture
                print("Found state dict, creating ResNet50 model...")
                model = get_model("resnet50", num_classes=args.num_classes)
                model.load_state_dict(torch.load(args.model, weights_only=True))
        except Exception:
            # Assume it's a state dict
            model = get_model("resnet50", num_classes=args.num_classes)
            model.load_state_dict(torch.load(args.model, weights_only=True))
    else:
        # Assume it's a model name
        print(f"Creating model: {args.model}")
        model = get_model(args.model, num_classes=args.num_classes)

    model.eval()

    # Benchmark
    if args.benchmark:
        print("\nRunning inference benchmark...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        avg_time, fps = benchmark_inference(model, input_size, device)
        print(f"  Device: {device}")
        print(f"  Average inference: {avg_time:.2f} ms")
        print(f"  Throughput: {fps:.1f} FPS")

    # Export
    base_name = Path(args.model).stem if Path(args.model).exists() else args.model
    formats = ["torchscript", "onnx", "state_dict"] if args.format == "all" else [args.format]

    print(f"\nExporting model...")
    for fmt in formats:
        if args.output and len(formats) == 1:
            output_path = args.output
        else:
            ext = {"torchscript": ".pt", "onnx": ".onnx", "state_dict": "_weights.pth"}
            output_path = f"{base_name}{ext[fmt]}"

        try:
            result = export_model(model, output_path, format=fmt, input_size=input_size)
            file_size = Path(result).stat().st_size / 1024 / 1024  # MB
            print(f"  {fmt:<12} -> {result} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"  {fmt:<12} -> FAILED: {e}")

    # Deployment tips
    print("\nDeployment Tips:")
    print("-" * 40)
    print("TorchScript: Best for PyTorch servers, mobile (LibTorch)")
    print("ONNX: Best for TensorRT, OpenVINO, ONNX Runtime")
    print("State Dict: Best for research, continued training")

    print("\nExample: Load TorchScript model for inference")
    print("  model = torch.jit.load('model.pt')")
    print("  output = model(input_tensor)")


if __name__ == "__main__":
    main()
