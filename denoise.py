"""
Offline audio denoising — clean a file using a trained U-Net checkpoint.

Usage:
    python denoise.py --input noisy.wav --output clean.wav --checkpoint checkpoints/best.pt

Enhancements (no retraining required):
  --passes N      Run the U-Net N times in spectrogram space before Griffin-Lim.
                  Each pass removes more residual noise. 2-3 is a good default.
  --gate P        Zero out spectrogram bins below the Pth percentile after denoising.
                  Suppresses residual hiss. Try values between 10-30.
  --griffin-iter  More iterations = less phase artifact. 128 is noticeably better than 64.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model.unet import UNet
from utils.audio import (
    load_audio,
    wav_to_mel,
    mel_to_wav,
    mel_to_wav_hifigan,
    load_hifigan,
    normalize_spec,
    denormalize_spec,
    make_mel_transform,
    chunk_spectrogram,
    stitch_chunks,
    SAMPLE_RATE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Offline audio denoiser")
    p.add_argument("--input",        required=True,  help="Path to noisy audio file")
    p.add_argument("--output",       required=True,  help="Path to save denoised audio")
    p.add_argument("--checkpoint",   required=True,  help="Path to trained model checkpoint (.pt)")
    p.add_argument("--chunk-frames", type=int,   default=256, help="Spectrogram frames per chunk")
    p.add_argument("--hop-frames",   type=int,   default=64,  help="Hop between chunks — smaller = smoother")
    p.add_argument("--griffin-iter", type=int,   default=128, help="Griffin-Lim iterations (more = less artifact)")
    p.add_argument("--vocoder",      type=str,   default="griffin-lim", choices=["griffin-lim", "hifigan"],
                   help="Vocoder for spectrogram → waveform (hifigan = better quality, downloads ~50MB)")
    p.add_argument("--passes",       type=int,   default=2,   help="Number of denoising passes in spectrogram space")
    p.add_argument("--gate",         type=float, default=None, help="Zero out bins below this percentile (e.g. 15)")
    p.add_argument("--ns-alpha",     type=float, default=1.5, help="Noise subtraction strength (0=off, 1=subtract once, 2=aggressive)")
    p.add_argument("--device",       type=str,   default="auto")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = UNet(
        base_channels=saved_args.get("base_ch", 64),
        depth=saved_args.get("depth", 4),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def noise_subtraction(spec: torch.Tensor, alpha: float = 1.5, quiet_percentile: float = 20.0) -> torch.Tensor:
    """
    Estimate the residual noise floor from the quietest frames and subtract it.

    After the model mask, speech frames still carry noise because the mask stays
    open to preserve speech energy. We take the quietest frames (which the model
    correctly silenced) as our noise profile and subtract it everywhere.

    spec:             (1, F, T) log-mel spectrogram after model denoising
    alpha:            subtraction strength — 1.0 = subtract once, 1.5 = more aggressive
    quiet_percentile: which fraction of frames to treat as noise-only
    """
    frame_energy = spec.mean(dim=1).squeeze(0)                         # (T,)
    threshold    = torch.quantile(frame_energy, quiet_percentile / 100)
    noise_mask   = frame_energy <= threshold                            # (T,) bool

    if noise_mask.sum() == 0:
        return spec

    noise_profile = spec[:, :, noise_mask].mean(dim=-1, keepdim=True)  # (1, F, 1)
    return (spec - alpha * noise_profile).clamp(min=0.0)


def run_pass(spec: torch.Tensor, model: UNet, device: torch.device,
             chunk_frames: int, hop_frames: int) -> torch.Tensor:
    """One denoising pass: normalize → chunk → U-Net → stitch → denormalize."""
    _, _, T_frames = spec.shape
    norm, min_val, max_val = normalize_spec(spec)
    chunks, offsets = chunk_spectrogram(norm, chunk_frames, hop_frames)

    denoised_chunks = []
    for chunk in chunks:
        pred = model(chunk.unsqueeze(0).to(device))
        denoised_chunks.append(pred.squeeze(0).cpu())

    stitched = stitch_chunks(denoised_chunks, offsets, T_frames, chunk_frames, hop_frames)
    return denormalize_spec(stitched, min_val, max_val)


@torch.no_grad()
def denoise_file(
    input_path: str,
    output_path: str,
    model: UNet,
    device: torch.device,
    chunk_frames: int = 256,
    hop_frames: int = 64,
    griffin_iter: int = 128,
    passes: int = 2,
    gate: float | None = None,
    ns_alpha: float = 1.5,
    vocoder: str = "griffin-lim",
    hifigan=None,
):
    print(f"Loading: {input_path}")
    wav = load_audio(input_path)
    print(f"  Duration: {wav.shape[-1] / SAMPLE_RATE:.2f}s  |  Samples: {wav.shape[-1]:,}")

    # ── Input RMS — used to restore volume after Griffin-Lim ──────────────
    input_rms = wav.pow(2).mean().sqrt().item()

    # ── 1. Noisy wav → log-mel spectrogram ────────────────────────────────
    mel_transform = make_mel_transform()
    current_spec = wav_to_mel(wav, mel_transform)           # (1, F, T_frames)
    _, F, T_frames = current_spec.shape
    print(f"  Spectrogram: {F} mel bins × {T_frames} frames")

    # ── 2. Multi-pass denoising in spectrogram space ──────────────────────
    for p in range(passes):
        print(f"  Pass {p + 1}/{passes}...")
        current_spec = run_pass(current_spec, model, device, chunk_frames, hop_frames)

    # ── 3. Noise subtraction — remove residual noise in speech frames ─────
    if ns_alpha > 0:
        current_spec = noise_subtraction(current_spec, alpha=ns_alpha)
        print(f"  Noise subtraction: alpha={ns_alpha}")

    # ── 4. Spectral gate — suppress residual noise floor ──────────────────
    if gate is not None:
        threshold = float(np.percentile(current_spec.numpy(), gate))
        current_spec = current_spec.clamp(min=threshold)
        print(f"  Spectral gate: zeroed bins below {gate}th percentile ({threshold:.4f})")

    # ── 5. Spectrogram → waveform ─────────────────────────────────────────
    if vocoder == "hifigan" and hifigan is not None:
        print("  HiFi-GAN vocoder...")
        clean_wav = mel_to_wav_hifigan(current_spec, hifigan, device)
    else:
        print(f"  Griffin-Lim ({griffin_iter} iterations)...")
        clean_wav = mel_to_wav(current_spec, n_iter=griffin_iter)

    # ── 6. Trim / pad to original length ──────────────────────────────────
    original_len = wav.shape[-1]
    if clean_wav.shape[-1] > original_len:
        clean_wav = clean_wav[:, :original_len]
    elif clean_wav.shape[-1] < original_len:
        clean_wav = torch.nn.functional.pad(clean_wav, (0, original_len - clean_wav.shape[-1]))

    # ── 6. RMS matching — restore original loudness ───────────────────────
    output_rms = clean_wav.pow(2).mean().sqrt().item()
    if output_rms > 1e-9:
        clean_wav = clean_wav * (input_rms / output_rms)
    clean_wav = clean_wav.clamp(-1, 1)

    print(f"  Input RMS:  {input_rms:.4f}")
    print(f"  Output RMS: {clean_wav.pow(2).mean().sqrt().item():.4f}")

    # ── 7. Save ────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, clean_wav.squeeze(0).numpy().astype(np.float32), SAMPLE_RATE)
    print(f"Saved: {output_path}")

    return clean_wav


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    print(f"Checkpoint: {args.checkpoint}")

    hifigan = None
    if args.vocoder == "hifigan":
        hifigan = load_hifigan(device)

    denoise_file(
        input_path=args.input,
        output_path=args.output,
        model=model,
        device=device,
        chunk_frames=args.chunk_frames,
        hop_frames=args.hop_frames,
        griffin_iter=args.griffin_iter,
        passes=args.passes,
        gate=args.gate,
        ns_alpha=args.ns_alpha,
        vocoder=args.vocoder,
        hifigan=hifigan,
    )


if __name__ == "__main__":
    main()
