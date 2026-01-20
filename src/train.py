import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

import utils
from transformer import StyleTransformer
from vgg import Vgg16

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def save_loss_plot(content_losses, style_losses, total_losses, save_dir):
    """Generates and saves a training loss graph."""
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss per Epoch")
    plt.plot(content_losses, label="Content Loss")
    plt.plot(style_losses, label="Style Loss")
    plt.plot(total_losses, label="Total Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

def train(args):
    print("=" * 50)
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"[Setup] Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Data Setup ---
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    print(f"[Setup] Batches per epoch: {len(train_loader)}")

    # --- Model Setup ---
    transformer = StyleTransformer().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)

    # --- Style Image Setup ---
    style_size = args.style_size if args.style_size else 512
    style = utils.load_image(args.style_image, size=style_size)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # --- Training State Variables ---
    history = {'content': [], 'style': [], 'total': []}
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE_LIMIT = 3  # Stop if no improvement for 3 epochs

    print(f"[Training] Starting for {args.epochs} epochs...")
    
    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_total_loss = 0.
        count = 0
        
        batch_iterator = tqdm(train_loader, desc=f"Epoch {e+1}/{args.epochs}", unit="batch")
        
        for batch_id, (x, _) in enumerate(batch_iterator):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_total_loss += total_loss.item()
            
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "C: {:.2f} S: {:.2f}".format(
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1)
                )
                batch_iterator.set_description(f"Epoch {e+1} [{mesg}]")

        # --- End of Epoch Logic ---
        # 1. Calculate Average Epoch Loss
        epoch_content = agg_content_loss / len(train_loader)
        epoch_style = agg_style_loss / len(train_loader)
        epoch_total = agg_total_loss / len(train_loader)
        
        history['content'].append(epoch_content)
        history['style'].append(epoch_style)
        history['total'].append(epoch_total)

        print(f"\n[Stats] Epoch {e+1} Avg Loss: {epoch_total:.2f}")

        # 2. Checkpoint Saving
        if args.checkpoint_model_dir:
            ckpt_path = os.path.join(args.checkpoint_model_dir, f"ckpt_epoch_{e}.pth")
            torch.save(transformer.state_dict(), ckpt_path)

        # 3. Early Stopping Check
        if epoch_total < best_loss - 1.0: # Improvement threshold
            best_loss = epoch_total
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Loss plateaued. Patience: {patience_counter}/{PATIENCE_LIMIT}")
            
        if patience_counter >= PATIENCE_LIMIT:
            print("Early stopping triggered!")
            break

    # --- Final Save & Plot ---
    print("\n[COMPLETE] Saving model and graphs...")
    transformer.eval().cpu()
    
    # Save Model
    name = args.save_model_name if args.save_model_name else f"epoch_{args.epochs}.pth"
    if not name.endswith(".pth"): name += ".pth"
    save_path = os.path.join(args.save_model_dir, name)
    torch.save(transformer.state_dict(), save_path)
    print(f"Model saved: {save_path}")

    # Save Graph
    save_loss_plot(history['content'], history['style'], history['total'], args.save_model_dir)
    print(f"Loss Graph saved: {os.path.join(args.save_model_dir, 'loss_plot.png')}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    train_parser = subparsers.add_parser("train")
    
    # Arguments
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--dataset", type=str, required=True)
    train_parser.add_argument("--style-image", type=str, default="images/style/mosaic.jpg")
    train_parser.add_argument("--save-model-dir", type=str, required=True)
    train_parser.add_argument("--save-model-name", type=str, default=None)
    train_parser.add_argument("--checkpoint-model-dir", type=str, default=None)
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--style-size", type=int, default=None)
    train_parser.add_argument("--cuda", type=int, required=True)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--content-weight", type=float, default=1e5)
    train_parser.add_argument("--style-weight", type=float, default=1e10)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--log-interval", type=int, default=500)
    train_parser.add_argument("--checkpoint-interval", type=int, default=2000)
    train_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        print("Unknown command.")

if __name__ == "__main__":
    main()
