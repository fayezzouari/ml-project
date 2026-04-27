"""
Gradio app — record or upload audio, get a denoised version back.

Usage:
    uv run app.py
    uv run app.py --checkpoint checkpoints/best.pt
"""

import argparse
import sys

import gradio as gr
import numpy as np
import torch
import torchaudio.transforms as T

from denoise import load_model, run_pass, noise_subtraction
from utils.audio import (
    wav_to_mel,
    mel_to_wav,
    make_mel_transform,
    SAMPLE_RATE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--share",      action="store_true", help="Create public Gradio link")
    return p.parse_args()


def load(checkpoint_path: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    model = load_model(checkpoint_path, device)
    print(f"Model loaded from: {checkpoint_path}")
    return model, device


@torch.no_grad()
def denoise(audio_input, passes: int, ns_alpha: float, gate: float, griffin_iter: int):
    if audio_input is None:
        return None, "No audio provided."

    sr, audio_np = audio_input

    # Normalise to float32 [-1, 1]
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    else:
        audio_np = audio_np.astype(np.float32)

    # To tensor (1, T)
    wav = torch.from_numpy(audio_np)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2:
        wav = wav.mean(0, keepdim=True)

    # Resample to 16 kHz if the mic recorded at a different rate
    if sr != SAMPLE_RATE:
        wav = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)

    input_rms = wav.pow(2).mean().sqrt().item()
    if input_rms < 1e-6:
        return None, "Input audio is silent — nothing to denoise."

    duration = wav.shape[-1] / SAMPLE_RATE

    # ── Denoising pipeline ─────────────────────────────────────────────────
    mel_transform = make_mel_transform(SAMPLE_RATE)
    current_spec = wav_to_mel(wav, mel_transform)           # (1, F, T_frames)

    for i in range(passes):
        current_spec = run_pass(current_spec, MODEL, DEVICE, chunk_frames=256, hop_frames=64)

    if ns_alpha > 0:
        current_spec = noise_subtraction(current_spec, alpha=ns_alpha)

    if gate > 0:
        threshold = float(np.percentile(current_spec.numpy(), gate))
        current_spec = current_spec.clamp(min=threshold)

    clean_wav = mel_to_wav(current_spec, n_iter=griffin_iter)

    # Trim to original length
    original_len = wav.shape[-1]
    if clean_wav.shape[-1] > original_len:
        clean_wav = clean_wav[:, :original_len]
    elif clean_wav.shape[-1] < original_len:
        clean_wav = torch.nn.functional.pad(clean_wav, (0, original_len - clean_wav.shape[-1]))

    # Restore original loudness
    out_rms = clean_wav.pow(2).mean().sqrt().item()
    if out_rms > 1e-9:
        clean_wav = clean_wav * (input_rms / out_rms)
    clean_wav = clean_wav.clamp(-1, 1)

    audio_out = clean_wav.squeeze(0).numpy().astype(np.float32)
    info = (
        f"Duration: {duration:.1f}s  |  "
        f"Passes: {passes}  |  "
        f"NS alpha: {ns_alpha}  |  "
        f"Gate: {gate}th pct"
    )
    return (SAMPLE_RATE, audio_out), info


def build_ui():
    with gr.Blocks(title="Audio Denoiser") as demo:
        gr.Markdown("# Audio Denoiser\nRecord your voice or upload a file — get a denoised version back.")

        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Input (record or upload)",
                )
                with gr.Accordion("Settings", open=False):
                    passes     = gr.Slider(1, 4, value=2, step=1,   label="Denoising passes")
                    ns_alpha   = gr.Slider(0, 4, value=1.5, step=0.1, label="Noise subtraction strength")
                    gate       = gr.Slider(0, 40, value=0, step=1,  label="Spectral gate percentile (0 = off)")
                    griffin_it = gr.Slider(32, 256, value=128, step=32, label="Griffin-Lim iterations")
                run_btn = gr.Button("Denoise", variant="primary")

            with gr.Column():
                audio_out = gr.Audio(label="Denoised output", type="numpy")
                info_box  = gr.Textbox(label="Info", interactive=False)

        run_btn.click(
            fn=denoise,
            inputs=[audio_in, passes, ns_alpha, gate, griffin_it],
            outputs=[audio_out, info_box],
        )

    return demo


if __name__ == "__main__":
    args = parse_args()
    MODEL, DEVICE = load(args.checkpoint)
    demo = build_ui()
    demo.launch(share=args.share)
