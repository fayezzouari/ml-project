"""
Evaluate a trained U-Net denoiser on the validation split.

Metrics
-------
Spectrogram space  (fast, full val set):
  L1       — reconstruction accuracy            (lower = better)
  SI-SNRi  — SNR improvement over noisy input   (higher = better, positive = model helps)
  LSD      — Log Spectral Distance              (lower = better)

Waveform space  (slow, --n-audio N samples):
  PESQ     — Perceptual Evaluation of Speech Quality  (1.0 – 4.5, higher = better)
  PESQi    — PESQ improvement over noisy baseline
  STOI     — Short-Time Objective Intelligibility     (0 – 1, higher = better)
  STOIi    — STOI improvement over noisy baseline

  Vocoder used for waveform reconstruction (--vocoder):
    hifigan     (default) — neural vocoder, accurate PESQ/STOI
    griffin-lim           — fast but adds phase artifacts, underestimates PESQ

Usage:
    # Fast (spectrogram metrics only)
    python evaluate.py --checkpoint checkpoints/best.pt --data ./dataset

    # Full with HiFi-GAN (default, recommended)
    python evaluate.py --checkpoint checkpoints/best.pt --data ./dataset --n-audio 50

    # Full with Griffin-Lim (faster, less accurate perceptual metrics)
    python evaluate.py --checkpoint checkpoints/best.pt --data ./dataset --n-audio 50 --vocoder griffin-lim
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.unet import UNet
from data.dataset import DenoisingDataset
from utils.audio import mel_to_wav, mel_to_wav_hifigan, load_hifigan, SAMPLE_RATE


# ── Metric helpers ─────────────────────────────────────────────────────────

def si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred   = pred.flatten(1)
    target = target.flatten(1)
    pred   = pred   - pred.mean(1, keepdim=True)
    target = target - target.mean(1, keepdim=True)
    dot      = (pred * target).sum(1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(1, keepdim=True) + eps)
    noise    = pred - s_target
    return 10 * torch.log10(s_target.pow(2).sum(1) / (noise.pow(2).sum(1) + eps) + eps)


def lsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log Spectral Distance — lower is better."""
    return torch.sqrt(((target - pred) ** 2).mean(dim=1)).mean()


def to_wav(spec: torch.Tensor, vocoder, device: torch.device, griffin_iter: int = 64) -> np.ndarray:
    """Convert a (1, F, T) log-mel spec to a 1-D float32 numpy waveform."""
    if vocoder is not None:
        wav = mel_to_wav_hifigan(spec, vocoder, device)
    else:
        wav = mel_to_wav(spec, n_iter=griffin_iter)
    return wav.squeeze(0).numpy().astype(np.float32)


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate U-Net denoiser")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       required=True)
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--chunk",      type=int, default=256)
    p.add_argument("--n-audio",    type=int, default=0,
                   help="Number of samples for PESQ/STOI (0 = skip)")
    p.add_argument("--vocoder",    type=str, default="hifigan",
                   choices=["hifigan", "griffin-lim"],
                   help="Vocoder for waveform metrics (default: hifigan)")
    p.add_argument("--griffin-iter", type=int, default=64,
                   help="Griffin-Lim iterations (only used when --vocoder griffin-lim)")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = UNet(
        base_channels=args.get("base_ch", 64),
        depth=args.get("depth", 4),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ── Spectrogram metrics (fast) ─────────────────────────────────────────────

@torch.no_grad()
def eval_spectrogram(model, loader, device):
    total_l1, total_sisnri, total_lsd = 0.0, 0.0, 0.0
    n = 0
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        pred  = model(noisy)

        total_l1     += F.l1_loss(pred, clean).item()
        total_sisnri += (si_snr(pred, clean) - si_snr(noisy, clean)).mean().item()
        total_lsd    += lsd(pred, clean).item()
        n += 1

    return {
        "L1":     total_l1     / n,
        "SI-SNRi": total_sisnri / n,
        "LSD":    total_lsd    / n,
    }


# ── Waveform metrics: PESQ + STOI (slow) ──────────────────────────────────

@torch.no_grad()
def eval_waveform(model, dataset, device, n_samples: int, griffin_iter: int, vocoder=None):
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError:
        print("  pesq / pystoi not installed — skipping waveform metrics.")
        print("  Install with: uv add pesq pystoi")
        return {}

    pesq_clean_list, pesq_noisy_list = [], []
    stoi_clean_list, stoi_noisy_list = [], []

    indices = list(range(min(n_samples, len(dataset))))
    vocoder_name = "HiFi-GAN" if vocoder is not None else f"Griffin-Lim ({griffin_iter} iters)"
    print(f"  Computing PESQ/STOI on {len(indices)} samples ({vocoder_name})...")

    for i, idx in enumerate(indices):
        sample     = dataset[idx]
        noisy_spec = sample["noisy"].unsqueeze(0).to(device)   # (1,1,F,T)
        clean_spec = sample["clean"].unsqueeze(0)               # (1,1,F,T)

        pred_spec = model(noisy_spec).squeeze(0).cpu()          # (1,F,T)
        clean_spec = clean_spec.squeeze(0)                      # (1,F,T)
        noisy_spec = noisy_spec.squeeze(0).cpu()                # (1,F,T)

        ref       = to_wav(clean_spec,  vocoder, device, griffin_iter)
        deg       = to_wav(pred_spec,   vocoder, device, griffin_iter)
        noisy_wav = to_wav(noisy_spec,  vocoder, device, griffin_iter)

        sr = SAMPLE_RATE
        try:
            pesq_clean_list.append(pesq(sr, ref, deg,      "wb"))
            pesq_noisy_list.append(pesq(sr, ref, noisy_wav, "wb"))
        except Exception:
            pass

        try:
            stoi_clean_list.append(stoi(ref, deg,       sr, extended=False))
            stoi_noisy_list.append(stoi(ref, noisy_wav, sr, extended=False))
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(indices)} done...")

    results = {}
    if pesq_clean_list:
        results["PESQ"]  = float(np.mean(pesq_clean_list))
        results["PESQi"] = float(np.mean(pesq_clean_list) - np.mean(pesq_noisy_list))
    if stoi_clean_list:
        results["STOI"]  = float(np.mean(stoi_clean_list))
        results["STOIi"] = float(np.mean(stoi_clean_list) - np.mean(stoi_noisy_list))
    return results


# ── Main ───────────────────────────────────────────────────────────────────

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

    # ── Spectrogram metrics ────────────────────────────────────────────────
    print("Computing spectrogram metrics...")
    spec_metrics = eval_spectrogram(model, loader, device)

    # ── Waveform metrics ───────────────────────────────────────────────────
    wav_metrics = {}
    if args.n_audio > 0:
        vocoder = None
        if args.vocoder == "hifigan":
            vocoder = load_hifigan(device)
        print(f"\nComputing waveform metrics (n={args.n_audio}, vocoder={args.vocoder})...")
        wav_metrics = eval_waveform(model, val_ds, device, args.n_audio, args.griffin_iter, vocoder)

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n── Evaluation Results ──────────────────────────────────────────")
    print(f"  L1   (spectrogram)   : {spec_metrics['L1']:.4f}       ↓ lower is better")
    print(f"  LSD  (spectrogram)   : {spec_metrics['LSD']:.4f}       ↓ lower is better")
    print(f"  SI-SNRi              : {spec_metrics['SI-SNRi']:+.2f} dB   ↑ positive = model helps")

    if wav_metrics:
        print()
        if "PESQ" in wav_metrics:
            print(f"  PESQ                 : {wav_metrics['PESQ']:.3f}        ↑ range 1.0–4.5")
            print(f"  PESQ improvement     : {wav_metrics['PESQi']:+.3f}")
        if "STOI" in wav_metrics:
            print(f"  STOI                 : {wav_metrics['STOI']:.3f}        ↑ range 0–1")
            print(f"  STOI improvement     : {wav_metrics['STOIi']:+.3f}")

    print("────────────────────────────────────────────────────────────────")

    sisnri = spec_metrics["SI-SNRi"]
    if sisnri > 3:
        print("  Strong denoising result.")
    elif sisnri > 0:
        print("  Model is improving over the noisy baseline.")
    else:
        print("  Model is not yet beating the noisy baseline — more training needed.")

    if "PESQ" in wav_metrics:
        pesq_val = wav_metrics["PESQ"]
        if pesq_val >= 3.5:
            label = "Excellent"
        elif pesq_val >= 3.0:
            label = "Good"
        elif pesq_val >= 2.5:
            label = "Good (some effort)"
        elif pesq_val >= 2.0:
            label = "Fair"
        elif pesq_val >= 1.5:
            label = "Poor"
        else:
            label = "Bad"
        print(f"  PESQ quality: {label}  (ITU-T MOS-LQO scale)")


if __name__ == "__main__":
    main()
