"""
Offline audio denoising — clean a file using a trained U-Net checkpoint.

Usage:
    python denoise.py --input noisy.wav --output clean.wav --checkpoint checkpoints/best.pt

How it works:
  1. Load the noisy audio file
  2. Convert to mel spectrogram
  3. Split into overlapping chunks (so any length file works)
  4. Run each chunk through the U-Net
  5. Stitch chunks back with overlap-add (Hann window to avoid seams)
  6. Convert back to waveform via Griffin-Lim
  7. Save the output

The overlap-add stitching means even very long files are handled correctly
without any seam artifacts between chunks.
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from model.unet import UNet
from utils.audio import (
    load_audio,
    wav_to_mel,
    mel_to_wav,
    normalize_spec,
    denormalize_spec,
    make_mel_transform,
    chunk_spectrogram,
    stitch_chunks,
    SAMPLE_RATE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Offline audio denoiser")
    p.add_argument("--input",       required=True,  help="Path to noisy audio file")
    p.add_argument("--output",      required=True,  help="Path to save denoised audio")
    p.add_argument("--checkpoint",  required=True,  help="Path to trained model checkpoint (.pt)")
    p.add_argument("--chunk-frames",type=int, default=256,  help="Spectrogram frames per chunk")
    p.add_argument("--hop-frames",  type=int, default=128,  help="Hop between chunks (overlap = chunk - hop)")
    p.add_argument("--griffin-iter",type=int, default=64,   help="Griffin-Lim iterations (more = better quality)")
    p.add_argument("--device",      type=str, default="auto")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Read architecture config from checkpoint if available
    saved_args = ckpt.get("args", {})
    model = UNet(
        base_channels=saved_args.get("base_ch", 64),
        depth=saved_args.get("depth", 4),
        dropout=0.0,  # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def denoise_file(
    input_path: str,
    output_path: str,
    model: UNet,
    device: torch.device,
    chunk_frames: int = 256,
    hop_frames: int = 128,
    griffin_iter: int = 64,
):
    print(f"Loading: {input_path}")
    wav = load_audio(input_path)                          # (1, T)
    print(f"  Duration: {wav.shape[-1] / SAMPLE_RATE:.2f}s  |  Samples: {wav.shape[-1]:,}")

    # ── 1. Noisy → spectrogram ─────────────────────────────────────────
    mel_transform = make_mel_transform()
    noisy_spec = wav_to_mel(wav, mel_transform)           # (1, F, T_frames)
    _, F, T_frames = noisy_spec.shape

    # Normalize and save stats to invert later
    noisy_norm, min_val, max_val = normalize_spec(noisy_spec)
    print(f"  Spectrogram: {F} mel bins × {T_frames} frames")

    # ── 2. Chunk ───────────────────────────────────────────────────────
    chunks, offsets = chunk_spectrogram(noisy_norm, chunk_frames, hop_frames)
    print(f"  Chunks: {len(chunks)} (frames per chunk: {chunk_frames}, hop: {hop_frames})")

    # ── 3. Denoise each chunk ──────────────────────────────────────────
    denoised_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_in = chunk.unsqueeze(0).to(device)          # (1, 1, F, chunk_frames)
        pred = model(chunk_in)                            # (1, 1, F, chunk_frames)
        denoised_chunks.append(pred.squeeze(0).cpu())     # (1, F, chunk_frames)

        if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
            print(f"  Denoised {i + 1}/{len(chunks)} chunks...")

    # ── 4. Stitch chunks back ──────────────────────────────────────────
    denoised_spec = stitch_chunks(denoised_chunks, offsets, T_frames, chunk_frames, hop_frames)

    # Denormalize back to original scale
    denoised_spec = denormalize_spec(denoised_spec, min_val, max_val)

    # ── 5. Spectrogram → waveform ──────────────────────────────────────
    print(f"  Converting back to waveform (Griffin-Lim, {griffin_iter} iterations)...")
    clean_wav = mel_to_wav(denoised_spec, n_iter=griffin_iter)  # (1, T_audio)

    # Match length to original
    original_len = wav.shape[-1]
    if clean_wav.shape[-1] > original_len:
        clean_wav = clean_wav[:, :original_len]
    elif clean_wav.shape[-1] < original_len:
        pad = original_len - clean_wav.shape[-1]
        clean_wav = torch.nn.functional.pad(clean_wav, (0, pad))

    # Normalize output to [-1, 1]
    peak = clean_wav.abs().max().clamp(min=1e-9)
    clean_wav = clean_wav / peak * 0.95

    # ── 6. Save ────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, clean_wav, SAMPLE_RATE)
    print(f"Saved: {output_path}")

    return clean_wav


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded from: {args.checkpoint}")

    denoise_file(
        input_path=args.input,
        output_path=args.output,
        model=model,
        device=device,
        chunk_frames=args.chunk_frames,
        hop_frames=args.hop_frames,
        griffin_iter=args.griffin_iter,
    )


if __name__ == "__main__":
    main()
