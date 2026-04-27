"""
Evaluate a trained U-Net denoiser on the validation split.

Metrics:
  - L1       : spectrogram L1 loss (lower = better)
  - SI-SNRi  : SI-SNR improvement over the noisy baseline (higher = better, positive = model helps)

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --data ./dataset
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.unet import UNet
from data.dataset import DenoisingDataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate U-Net denoiser")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data",       required=True, help="Dataset root (clean/ and noise/)")
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--chunk",      type=int, default=256)
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> tuple[UNet, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    model = UNet(
        base_channels=saved.get("base_ch", 64),
        depth=saved.get("depth", 4),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SNR between two tensors of any shape — flattened per batch item. Returns (B,)."""
    pred   = pred.flatten(1)
    target = target.flatten(1)
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    dot      = (pred * target).sum(dim=1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=1, keepdim=True) + eps)
    noise    = pred - s_target
    return 10 * torch.log10(s_target.pow(2).sum(dim=1) / (noise.pow(2).sum(dim=1) + eps) + eps)


@torch.no_grad()
def run_eval(model: UNet, loader: DataLoader, device: torch.device) -> dict:
    total_l1     = 0.0
    total_sisnri = 0.0
    n = 0

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        pred = model(noisy)

        total_l1 += F.l1_loss(pred, clean).item()

        sisnri = (si_snr(pred, clean) - si_snr(noisy, clean)).mean().item()
        total_sisnri += sisnri
        n += 1

    return {
        "l1":     total_l1     / n,
        "sisnri": total_sisnri / n,
    }


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss")
    print(f"Checkpoint: epoch {epoch}" + (f"  |  val loss {val_loss:.4f}" if val_loss else ""))

    val_ds = DenoisingDataset(args.data, split="val", chunk_frames=args.chunk)
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"Val samples: {len(val_ds):,}  |  Batches: {len(loader)}\n")

    metrics = run_eval(model, loader, device)

    print("── Evaluation Results ────────────────────────────────")
    print(f"  L1  (spectrogram)  : {metrics['l1']:.4f}       (lower  = better)")
    print(f"  SI-SNRi            : {metrics['sisnri']:+.2f} dB   (positive = model helps)")
    print("──────────────────────────────────────────────────────")

    if metrics["sisnri"] > 1.0:
        print("  Model is meaningfully improving signal quality.")
    elif metrics["sisnri"] > 0:
        print("  Model is slightly improving over the noisy baseline.")
    else:
        print("  Model is not yet beating the noisy baseline — more training needed.")


if __name__ == "__main__":
    main()
