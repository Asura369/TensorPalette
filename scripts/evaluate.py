"""Eval harness: content/style loss table, LPIPS diversity, latency benchmarks.

Usage:
    python scripts/evaluate.py --model-dir models/ --styles styles/catalog.yaml --output docs/benchmarks.md
"""
import argparse
import os
import time

import torch

from styleforge.engine import InferenceEngine


def measure_latency(engine, model_name, style_id, sizes, device, runs=5):
    results = {}
    dummy_img = torch.randn(1, 3, 256, 256).to(device) * 255

    for size in sizes:
        img = torch.nn.functional.interpolate(dummy_img, size=(size, size))
        model = engine.get_model(model_name)

        model.eval()
        with torch.no_grad():
            for _ in range(2):
                style_ids = torch.tensor([style_id], device=device)
                _ = model(img, style_ids)

            times = []
            for _ in range(runs):
                start = time.perf_counter()
                style_ids = torch.tensor([style_id], device=device)
                _ = model(img, style_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        results[size] = avg_ms
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--styles", type=str, default="styles/catalog.yaml")
    parser.add_argument("--output", type=str, default="docs/benchmarks.md")
    parser.add_argument("--cin-model", type=str, default=None)
    parser.add_argument("--num-styles", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine(device=device)

    lines = [
        "# StyleForge Benchmarks",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {device}",
        "",
    ]

    if args.cin_model and os.path.exists(args.cin_model):
        engine.load_cin("cin", args.cin_model, num_styles=args.num_styles)
        latencies = measure_latency(engine, "cin", 0, [512, 1280], device)

        lines.append("## CIN Model Latency (ms)")
        lines.append("")
        lines.append("| Size | Latency (ms) |")
        lines.append("|------|-------------|")
        for size, ms in latencies.items():
            lines.append(f"| {size}px | {ms:.1f} |")
        lines.append("")

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Benchmarks written to {output_path}")


if __name__ == "__main__":
    main()
