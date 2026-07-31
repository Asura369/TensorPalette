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
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "cin_loss_plot.png"))
    plt.close()


def load_style_images(style_image_paths, style_size, device):
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
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

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)

    if args.limit:
        indices = range(min(len(train_dataset), args.limit))
        train_dataset = Subset(train_dataset, indices)
        print(f"[Setup] Limit Active: {len(train_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"[Setup] Batches per epoch: {len(train_loader)}")

    num_styles = len(args.style_images)
    print(f"[Setup] Training CIN with {num_styles} styles")

    transformer = CINTransformer(num_styles=num_styles).to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    style_size = args.style_size or 512
    style_images = load_style_images(args.style_images, style_size, device)

    features_style_list = []
    for s_img in style_images:
        s_batch = s_img.repeat(args.batch_size, 1, 1, 1)
        feats = vgg(utils.normalize_batch(s_batch))
        grams = [utils.gram_matrix(f) for f in feats]
        features_style_list.append(grams)

    history = {"content": [], "style": [], "total": []}
    best_loss = float("inf")
    patience_counter = 0
    PATIENCE_LIMIT = 3

    print(f"[Training] Starting CIN training for {args.epochs} epochs, AMP={args.amp}...")

    for e in range(args.epochs):
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

                content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Compute style loss in fp32 outside autocast to prevent overflow
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y.float())
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            agg_content += content_loss.item()
            agg_style += style_loss.item()
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
                    torch.save(transformer.state_dict(), ckpt_path)

        epoch_content = agg_content / len(train_loader)
        epoch_style = agg_style / len(train_loader)
        epoch_total = agg_total / len(train_loader)

        history["content"].append(epoch_content)
        history["style"].append(epoch_style)
        history["total"].append(epoch_total)

        print(f"\n[Stats] Epoch {e+1} Avg Loss: {epoch_total:.2f}")

        if args.checkpoint_model_dir:
            ckpt_path = os.path.join(args.checkpoint_model_dir, f"cin_ckpt_epoch_{e}.pth")
            torch.save(transformer.state_dict(), ckpt_path)

        if best_loss != float("inf") and epoch_total >= best_loss * (1 - 0.01):
            patience_counter += 1
            print(f"Loss plateaued. Patience: {patience_counter}/{PATIENCE_LIMIT}")
        else:
            best_loss = epoch_total
            patience_counter = 0

        if patience_counter >= PATIENCE_LIMIT:
            print("Early stopping triggered!")
            break

    print("\n[COMPLETE] Saving CIN model and graphs...")
    transformer.eval().cpu()

    name = args.save_model_name or f"cin_epoch_{args.epochs}.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    save_path = os.path.join(args.save_model_dir, name)
    torch.save(transformer.state_dict(), save_path)
    print(f"Model saved: {save_path}")

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

    args = parser.parse_args()

    if args.subcommand == "cin":
        defaults = {
            "epochs": 4,
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
            "style_weight": 1e10,
            "lr": 1e-3,
            "log_interval": 500,
            "limit": None,
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
