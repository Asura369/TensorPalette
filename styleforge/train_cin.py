import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from styleforge import utils
from styleforge.cin import CINTransformer
from styleforge.vgg import Vgg16

STYLE_LAYERS = ("relu1_2", "relu2_2", "relu3_3", "relu4_2")
CONTENT_LAYERS = (("relu2_2", 1.0), ("relu4_2", 1.0))


def _mul255(x):
    """Scale [0,1] tensors to [0,255]; module-level so DataLoader can pickle it."""
    return x.mul(255)


def check_paths(args):
    for d in [args.save_model_dir]:
        if d and not os.path.exists(d):
            os.makedirs(d)
    if args.checkpoint_model_dir and not os.path.exists(args.checkpoint_model_dir):
        os.makedirs(args.checkpoint_model_dir)


def save_loss_plot(hist, save_dir):
    plt.figure(figsize=(10, 5))
    plt.title("CIN Training Loss")
    plt.plot(hist["content"], label="Content")
    plt.plot(hist["style"], label="Style")
    plt.plot(hist["total"], label="Total", linestyle="--")
    if hist.get("val_total") and any(v is not None for v in hist["val_total"]):
        plt.plot(hist["val_total"], label="Val Total", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "cin_loss_plot.png"))
    plt.close()


def content_loss(features_y, features_x, mse_loss, weight):
    loss = 0.0
    for layer_name, layer_weight in CONTENT_LAYERS:
        loss += layer_weight * mse_loss(
            getattr(features_y, layer_name), getattr(features_x, layer_name)
        )
    return loss * weight


def style_loss(features_y, gram_style, mse_loss, n_batch, weight):
    loss = 0.0
    for layer_name, gm_s in zip(STYLE_LAYERS, gram_style):
        gm_y = utils.gram_matrix(getattr(features_y, layer_name).float())
        loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
    return loss * weight


def split_dataset(dataset, val_fraction, seed):
    """Deterministically split a dataset into train/val subsets by index."""
    n = len(dataset)
    n_val = int(round(n * val_fraction))
    if n_val == 0:
        return dataset, None
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_indices = perm[:n_val]
    val_set = set(val_indices)
    train_indices = [i for i in range(n) if i not in val_set]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _config_snapshot(args):
    keys = ("lr", "epochs", "batch_size", "content_weight", "style_weight",
            "seed", "val_fraction", "image_size", "style_size")
    return {k: getattr(args, k) for k in keys}


def build_checkpoint(transformer, optimizer, scheduler, epoch, best_loss,
                     patience_counter, best_state, history, args, batch_id=None):
    """Full trainer state so training can resume from a checkpoint."""
    rng_numpy = np.random.get_state()
    ckpt = {
        "model": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "patience": patience_counter,
        "best_state": best_state,
        "history": history,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if args.cuda and torch.cuda.is_available() else None,
        "rng_numpy": (rng_numpy[0], rng_numpy[1].tobytes(), rng_numpy[2], rng_numpy[3], rng_numpy[4]),
        "args": _config_snapshot(args),
    }
    if batch_id is not None:
        ckpt["batch"] = batch_id
    return ckpt


def restore_rng(ckpt, args):
    torch.set_rng_state(ckpt["rng_torch"])
    rng_numpy = ckpt["rng_numpy"]
    arr = np.frombuffer(rng_numpy[1], dtype=np.uint32).copy()
    np.random.set_state((rng_numpy[0], arr, rng_numpy[2], rng_numpy[3], rng_numpy[4]))
    if ckpt.get("rng_cuda") is not None and args.cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"])


def load_style_images(style_image_paths, style_size, device):
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_mul255)
    ])
    gram_styles = []
    for path in style_image_paths:
        img = utils.load_image(path, size=style_size)
        img_t: torch.Tensor = style_transform(img).unsqueeze(0).to(device)
        gram_styles.append(img_t)
    return gram_styles


def train_cin(args):
    print("=" * 50)
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"[Setup] Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
        transforms.Resize(args.image_size + 32),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(_mul255)
    ])
    dataset = datasets.ImageFolder(args.dataset, transform)

    if args.limit:
        dataset = Subset(dataset, range(min(len(dataset), args.limit)))
        print(f"[Setup] Limit Active: {len(dataset)} images")

    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)
    if val_dataset is not None:
        print(f"[Setup] Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")
    else:
        print("[Setup] No validation set (val-fraction 0) — early stopping will use training loss")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)
    print(f"[Setup] Batches per epoch: {len(train_loader)}")

    num_styles = len(args.style_images)
    print(f"[Setup] Training CIN with {num_styles} styles")

    transformer = CINTransformer(num_styles=num_styles).to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"content": [], "style": [], "total": [], "val_total": []}

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if "model" not in ckpt or "optimizer" not in ckpt:
            raise ValueError(
                f"{args.resume} is not a full trainer checkpoint (older checkpoints "
                "saved state_dict only and cannot be resumed)."
            )
        transformer.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        patience_counter = ckpt["patience"]
        best_state = ckpt["best_state"]
        history = ckpt["history"]
        restore_rng(ckpt, args)
        for k, v in ckpt.get("args", {}).items():
            current = getattr(args, k, None)
            if current != v:
                print(f"[Resume] WARNING: saved {k}={v} differs from current {k}={current}")
        print(f"[Resume] Resuming from epoch {start_epoch + 1}/{args.epochs} "
              f"(best_loss={best_loss:.2f})")

    style_size = args.style_size or 512
    style_images = load_style_images(args.style_images, style_size, device)

    features_style_list = []
    for s_img in style_images:
        s_batch = s_img.repeat(args.batch_size, 1, 1, 1)
        feats = vgg(utils.normalize_batch(s_batch))
        grams = [utils.gram_matrix(getattr(feats, layer_name)) for layer_name in STYLE_LAYERS]
        features_style_list.append(grams)

    PATIENCE_LIMIT = 3

    print(f"[Training] Starting CIN training for {args.epochs} epochs, AMP={args.amp}...")

    for e in range(start_epoch, args.epochs):
        transformer.train()
        agg_content = 0.
        agg_style = 0.
        agg_total = 0.

        batch_iterator = tqdm(train_loader, desc=f"Epoch {e+1}/{args.epochs}", unit="batch")

        for batch_id, (x, _) in enumerate(batch_iterator):
            n_batch = len(x)
            x = x.to(device)

            style_id = random.randint(0, num_styles - 1)
            gram_style = features_style_list[style_id]

            style_ids = torch.tensor([style_id] * n_batch, device=device)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=args.amp and args.cuda, dtype=torch.float16):
                y = transformer(x, style_ids)

                y_norm = utils.normalize_batch(y)
                x_norm = utils.normalize_batch(x)

                features_y = vgg(y_norm)
                features_x = vgg(x_norm)

                # Compute style loss in fp32 outside autocast to prevent overflow
                c_loss = content_loss(features_y, features_x, mse_loss, args.content_weight)

            s_loss = style_loss(features_y, gram_style, mse_loss, n_batch, args.style_weight)

            total_loss = c_loss + s_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            agg_content += c_loss.item()
            agg_style += s_loss.item()
            agg_total += total_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "C: {:.2f} S: {:.2f}".format(
                    agg_content / (batch_id + 1),
                    agg_style / (batch_id + 1)
                )
                batch_iterator.set_description(f"Epoch {e+1} [{mesg}]")

            if args.checkpoint_interval and (batch_id + 1) % args.checkpoint_interval == 0:
                if args.checkpoint_model_dir:
                    ckpt_path = os.path.join(
                        args.checkpoint_model_dir,
                        f"cin_ckpt_e{e+1}_b{batch_id+1}.pth"
                    )
                    torch.save(build_checkpoint(
                        transformer, optimizer, scheduler, e, best_loss,
                        patience_counter, best_state, history, args,
                        batch_id=batch_id + 1), ckpt_path)

        epoch_content = agg_content / len(train_loader)
        epoch_style = agg_style / len(train_loader)
        epoch_total = agg_total / len(train_loader)

        history["content"].append(epoch_content)
        history["style"].append(epoch_style)
        history["total"].append(epoch_total)

        print(f"\n[Stats] Epoch {e+1} Avg Loss: {epoch_total:.2f}")

        epoch_val_total = None
        if val_loader is not None:
            transformer.eval()
            agg_val_content = 0.
            agg_val_style = 0.
            agg_val_total = 0.
            with torch.no_grad():
                for x, _ in val_loader:
                    n_batch = len(x)
                    x = x.to(device)
                    style_id = random.randint(0, num_styles - 1)
                    gram_style = features_style_list[style_id]
                    style_ids = torch.tensor([style_id] * n_batch, device=device)
                    y = transformer(x, style_ids)
                    y_norm = utils.normalize_batch(y)
                    x_norm = utils.normalize_batch(x)
                    features_y = vgg(y_norm)
                    features_x = vgg(x_norm)
                    v_c = content_loss(features_y, features_x, mse_loss, args.content_weight)
                    v_s = style_loss(features_y, gram_style, mse_loss, n_batch, args.style_weight)
                    agg_val_content += v_c.item()
                    agg_val_style += v_s.item()
                    agg_val_total += (v_c + v_s).item()
            transformer.train()
            n_val = max(len(val_loader), 1)
            epoch_val_total = agg_val_total / n_val
            history["val_total"].append(epoch_val_total)
            print(f"[Val] Epoch {e+1} Avg Val Loss: {epoch_val_total:.2f} "
                  f"(C: {agg_val_content / n_val:.2f} S: {agg_val_style / n_val:.2f})")
        else:
            history["val_total"].append(None)

        scheduler.step()

        if args.checkpoint_model_dir:
            ckpt_path = os.path.join(args.checkpoint_model_dir, f"cin_ckpt_epoch_{e}.pth")
            torch.save(build_checkpoint(
                transformer, optimizer, scheduler, e, best_loss,
                patience_counter, best_state, history, args), ckpt_path)

        signal = epoch_val_total if epoch_val_total is not None else epoch_total
        signal_name = "val" if epoch_val_total is not None else "train"
        if best_loss != float("inf") and signal >= best_loss * (1 - 0.01):
            patience_counter += 1
            print(f"Loss plateaued ({signal_name}). Patience: {patience_counter}/{PATIENCE_LIMIT}")
        else:
            best_loss = signal
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in transformer.state_dict().items()}
            print(f"[Best] New best {signal_name} loss: {best_loss:.2f}")

        if patience_counter >= PATIENCE_LIMIT:
            print("Early stopping triggered!")
            break

    print("\n[COMPLETE] Saving CIN model and graphs...")
    transformer.eval().cpu()

    name = args.save_model_name or f"cin_epoch_{args.epochs}.pth"
    if not name.endswith(".pth"):
        name += ".pth"

    if best_state is not None:
        best_path = os.path.join(args.save_model_dir, name)
        torch.save(best_state, best_path)
        print(f"Best model saved: {best_path}")
    else:
        best_path = os.path.join(args.save_model_dir, name)
        torch.save(transformer.state_dict(), best_path)
        print(f"Model saved: {best_path}")

    stem, ext = os.path.splitext(name)
    last_path = os.path.join(args.save_model_dir, f"{stem}_last{ext}")
    torch.save(transformer.state_dict(), last_path)
    print(f"Final-epoch model saved: {last_path}")

    save_loss_plot(history, args.save_model_dir)
    print(f"Loss plot saved: {os.path.join(args.save_model_dir, 'cin_loss_plot.png')}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    cin_parser = subparsers.add_parser("cin")

    cin_parser.add_argument("--config", type=str, default=None)
    cin_parser.add_argument("--epochs", type=int, default=None)
    cin_parser.add_argument("--batch-size", type=int, default=None)
    cin_parser.add_argument("--dataset", type=str, default=None)
    cin_parser.add_argument("--style-images", type=str, default=None,
                            help="Comma-separated list of style image paths")
    cin_parser.add_argument("--save-model-dir", type=str, default=None)
    cin_parser.add_argument("--save-model-name", type=str, default=None)
    cin_parser.add_argument("--checkpoint-model-dir", type=str, default=None)
    cin_parser.add_argument("--checkpoint-interval", type=int, default=None)
    cin_parser.add_argument("--image-size", type=int, default=None)
    cin_parser.add_argument("--style-size", type=int, default=None)
    cin_parser.add_argument("--cuda", type=int, default=None)
    cin_parser.add_argument("--amp", type=int, default=None, help="Enable AMP (1=on, 0=off)")
    cin_parser.add_argument("--seed", type=int, default=None)
    cin_parser.add_argument("--content-weight", type=float, default=None)
    cin_parser.add_argument("--style-weight", type=float, default=None)
    cin_parser.add_argument("--lr", type=float, default=None)
    cin_parser.add_argument("--log-interval", type=int, default=None)
    cin_parser.add_argument("--limit", type=int, default=None)
    cin_parser.add_argument("--val-fraction", type=float, default=None,
                            help="Fraction of the dataset held out for validation (default: 0.02)")
    cin_parser.add_argument("--resume", type=str, default=None,
                            help="Path to a full trainer checkpoint to resume from")

    args = parser.parse_args()

    if args.subcommand == "cin":
        defaults = {
            "epochs": 8,
            "batch_size": 4,
            "dataset": None,
            "style_images": None,
            "save_model_dir": None,
            "save_model_name": None,
            "checkpoint_model_dir": None,
            "checkpoint_interval": None,
            "image_size": 256,
            "style_size": 512,
            "cuda": 1,
            "amp": 1,
            "seed": 42,
            "content_weight": 1e5,
            "style_weight": 1e9,
            "lr": 1e-3,
            "log_interval": 500,
            "limit": None,
            "val_fraction": 0.02,
            "resume": None,
        }

        if args.config:
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                k_norm = k.replace("-", "_")
                if k_norm in defaults:
                    defaults[k_norm] = v

        for k, v in defaults.items():
            cli_val = getattr(args, k, None)
            if cli_val is not None:
                defaults[k] = cli_val

        for k, v in defaults.items():
            setattr(args, k, v)

        if isinstance(args.style_images, str):
            args.style_images = [s.strip() for s in args.style_images.split(",")]

        if not args.dataset:
            parser.error("--dataset is required")
        if not args.save_model_dir:
            parser.error("--save-model-dir is required")
        if not args.style_images or len(args.style_images) < 2:
            parser.error("--style-images requires at least 2 comma-separated paths")

        args.amp = bool(args.amp)
        args.cuda = bool(args.cuda)

        check_paths(args)
        train_cin(args)
    else:
        print("Unknown command.")


if __name__ == "__main__":
    main()
