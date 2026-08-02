"""Eval harness: content/style loss table and latency benchmarks.

Usage:
    python scripts/evaluate.py --styles styles/catalog.yaml \
        --cin-model models/multistyle.pth \
        --images-dir /path/to/heldout/images --output docs/benchmarks.md
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def list_images(images_dir, max_images):
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted(
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.lower().endswith(exts)
    )
    return paths[:max_images]


def measure_quality(engine, model_name, image_paths, style_paths, device):
    """Unscaled content/style loss over held-out images, mean across images."""
    import torchvision.transforms as T

    from styleforge import utils
    from styleforge.train_cin import STYLE_LAYERS, content_loss, style_loss
    from styleforge.vgg import Vgg16

    content_transform = T.Compose([
        T.Resize(288),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255)),
    ])
    style_transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255)),
    ])

    mse = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)

    style_grams = []
    for path in style_paths:
        img = utils.load_image(path, size=512)
        t = style_transform(img).unsqueeze(0).to(device)
        feats = vgg(utils.normalize_batch(t))
        style_grams.append([utils.gram_matrix(getattr(feats, ln)) for ln in STYLE_LAYERS])

    model = engine.get_model(model_name)
    model.eval()

    content_losses = []
    style_losses = {i: [] for i in range(len(style_paths))}
    with torch.no_grad():
        for path in image_paths:
            img = utils.load_image(path)
            t = content_transform(img).unsqueeze(0).to(device)
            style_ids = torch.tensor([0], device=device)
            y = model(t, style_ids)
            y_norm = utils.normalize_batch(y)
            x_norm = utils.normalize_batch(t)
            fy = vgg(y_norm)
            fx = vgg(x_norm)
            content_losses.append(content_loss(fy, fx, mse, 1.0).item())
            for i, grams in enumerate(style_grams):
                style_losses[i].append(style_loss(fy, grams, mse, 1, 1.0).item())

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_images": len(image_paths),
        "content_loss": mean(content_losses),
        "style_loss": {i: mean(style_losses[i]) for i in style_losses},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", type=str, default="styles/catalog.yaml")
    parser.add_argument("--output", type=str, default="docs/benchmarks.md")
    parser.add_argument("--cin-model", type=str, default=None)
    parser.add_argument("--num-styles", type=int, default=5)
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory of held-out content images for the quality table")
    parser.add_argument("--max-images", type=int, default=32)
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

        if args.images_dir and os.path.isdir(args.images_dir):
            import yaml

            style_paths = []
            if os.path.exists(args.styles):
                with open(args.styles, "r") as f:
                    catalog = yaml.safe_load(f)
                styles_dir = os.path.dirname(args.styles)
                for entry in catalog.get("styles", []):
                    style_paths.append(os.path.join(styles_dir, entry["filename"]))

            image_paths = list_images(args.images_dir, args.max_images)
            if image_paths and style_paths:
                quality = measure_quality(engine, "cin", image_paths, style_paths, device)

                lines.append(f"## CIN Quality (unscaled loss, mean over {quality['n_images']} images)")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Content loss | {quality['content_loss']:.4f} |")
                lines.append("")
                lines.append("| Style | Style loss |")
                lines.append("|-------|-----------|")
                for i, loss in quality["style_loss"].items():
                    lines.append(f"| Style {i} | {loss:.4f} |")
                lines.append("")
            else:
                print(f"Warning: no images in {args.images_dir} or no styles in {args.styles}")

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Benchmarks written to {output_path}")


if __name__ == "__main__":
    main()
