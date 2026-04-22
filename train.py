"""
Training script for the U-Net audio denoiser.

Usage:
    python train.py --data ./dataset --epochs 50 --batch 16

The dataset directory must have:
    dataset/
      clean/  *.wav
      noise/  *.wav

Checkpoints are saved to ./checkpoints/ every epoch.
TensorBoard logs go to ./runs/.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.unet import UNet, count_parameters
from model.loss import DenoisingLoss
from data.dataset import DenoisingDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net audio denoiser")
    p.add_argument("--data",       type=str,   default="./dataset",  help="Dataset root (needs clean/ and noise/)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--chunk",      type=int,   default=256,           help="Spectrogram time frames per sample")
    p.add_argument("--base-ch",    type=int,   default=64,            help="U-Net base channels")
    p.add_argument("--depth",      type=int,   default=4,             help="U-Net encoder depth")
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--resume",     type=str,   default=None,          help="Path to checkpoint to resume from")
    p.add_argument("--save-dir",   type=str,   default="./checkpoints")
    p.add_argument("--log-dir",    type=str,   default="./runs")
    return p.parse_args()


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt.get("best_val_loss", float("inf"))


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, writer, epoch
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        noisy = batch["noisy"].to(device)  # (B, 1, F, T)
        clean = batch["clean"].to(device)

        pred = model(noisy)
        losses = loss_fn(pred, clean)
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        global_step = epoch * len(loader) + step
        writer.add_scalar("train/loss", loss.item(), global_step)
        if "l1" in losses:
            writer.add_scalar("train/l1", losses["l1"].item(), global_step)

        if step % 50 == 0:
            print(f"  step {step:4d}/{len(loader)} | loss {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        pred = model(noisy)
        losses = loss_fn(pred, clean)
        total_loss += losses["total"].item()

    return total_loss / len(loader)


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    pin = device.type == "cuda"  # pin_memory only works on CUDA

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = DenoisingDataset(args.data, split="train", chunk_frames=args.chunk)
    val_ds   = DenoisingDataset(args.data, split="val",   chunk_frames=args.chunk)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
    )

    print(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet(
        base_channels=args.base_ch,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    print(f"Model — {count_parameters(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    loss_fn = DenoisingLoss()

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    writer = SummaryWriter(log_dir=args.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, writer, epoch)
        val_loss   = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | {elapsed:.1f}s")

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Save checkpoint every epoch
        ckpt = {
            "epoch":         epoch + 1,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        }
        save_checkpoint(ckpt, Path(args.save_dir) / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt, Path(args.save_dir) / "best.pt")
            print(f"  ✓ New best model saved (val loss {val_loss:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
