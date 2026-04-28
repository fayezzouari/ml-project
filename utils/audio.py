"""
Audio utilities for the denoising pipeline.

Handles:
  - Loading and resampling audio files
  - Converting waveform ↔ mel spectrogram
  - Normalizing spectrograms to [0, 1]
  - Chunking long audio into overlapping windows
  - Stitching denoised chunks back together (overlap-add)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T


# ── Default spectrogram config ─────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # Hz — all audio resampled to this
N_FFT         = 1024
HOP_LENGTH    = 256
N_MELS        = 128      # frequency bins (height of spectrogram image)
F_MIN         = 20.0     # Hz
F_MAX         = 8_000.0  # Hz

# ── HiFi-GAN mel config (microsoft/speecht5_hifigan) ──────────────────────
_HIFIGAN_N_MELS = 80
_HIFIGAN_F_MIN  = 0.0
_HIFIGAN_F_MAX  = SAMPLE_RATE / 2

# Lazily built transforms for mel conversion (created once, reused)
_INV_MEL_128: T.InverseMelScale | None = None
_MEL_SCALE_80: T.MelScale | None = None


def _get_mel_converters():
    """
    Lazily build and cache the two transforms used for 128-mel → 80-mel conversion.

    InverseMelScale uses NNLS (non-negative least squares) to find the best
    non-negative linear STFT spectrogram consistent with the observed 128-mel
    spectrogram. MelScale then re-applies the 80-mel filterbank on top.
    This is more accurate than the pseudo-inverse (pinv) matrix approach,
    which can produce negative intermediate values and loses spectral shape.
    """
    global _INV_MEL_128, _MEL_SCALE_80
    if _INV_MEL_128 is None:
        _INV_MEL_128 = T.InverseMelScale(
            n_stft=N_FFT // 2 + 1,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            f_min=F_MIN,
            f_max=F_MAX,
        )
    if _MEL_SCALE_80 is None:
        _MEL_SCALE_80 = T.MelScale(
            n_mels=_HIFIGAN_N_MELS,
            sample_rate=SAMPLE_RATE,
            f_min=_HIFIGAN_F_MIN,
            f_max=_HIFIGAN_F_MAX,
            n_stft=N_FFT // 2 + 1,
        )
    return _INV_MEL_128, _MEL_SCALE_80


def load_hifigan(device: torch.device = torch.device("cpu")):
    """Download and return the pretrained SpeechT5 HiFi-GAN vocoder."""
    from transformers import SpeechT5HifiGan
    print("Loading pretrained HiFi-GAN (microsoft/speecht5_hifigan)...")
    model = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    model.eval().to(device)
    print("HiFi-GAN ready.")
    return model


@torch.no_grad()
def mel_to_wav_hifigan(log_mel: torch.Tensor, hifigan, device: torch.device) -> torch.Tensor:
    """
    Convert a denoised 128-mel log1p spectrogram to a waveform using HiFi-GAN.

    Pipeline:
      log_mel (128, log1p) → power mel (128) → [NNLS] → linear STFT
        → [MelScale] → power mel (80) → log → HiFi-GAN → waveform

    The NNLS step enforces non-negativity and finds the linear spectrum that
    best explains the observed mel values — much more accurate than pinv.

    log_mel: (1, 128, T)
    Returns: (1, T_audio)
    """
    inv_mel, mel_scale_80 = _get_mel_converters()

    # 1. Undo log1p → power mel (128 bins)
    mel_128 = torch.expm1(log_mel.float()).clamp(min=0)       # (1, 128, T)

    # 2. 128-mel → linear STFT via NNLS  (513, T)
    linear = inv_mel(mel_128.squeeze(0))                       # (513, T)

    # 3. Linear → 80-mel (HiFi-GAN's native filterbank config)
    mel_80 = mel_scale_80(linear.unsqueeze(0)).clamp(min=0)   # (1, 80, T)

    # 4. Log-compress — matches HiFi-GAN training: log(mel + 1e-5)
    log_mel_80 = torch.log(mel_80.clamp(min=1e-5))            # (1, 80, T)

    # 5. HiFi-GAN expects (batch, time, n_mels)
    mel_input = log_mel_80.squeeze(0).T.unsqueeze(0).to(device)  # (1, T, 80)

    wav = hifigan(mel_input)                                   # (1, T_audio)
    return wav.cpu()


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load any audio file and return a mono waveform tensor (1, T).

    Resamples to target_sr and mixes down to mono automatically.
    """
    data, sr = sf.read(str(path), always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T).float()          # (C, T)

    # Mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        wav = T.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    return wav  # (1, T)


def make_mel_transform(sr: int = SAMPLE_RATE) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,          # power spectrogram
    )


def make_inverse_mel(sr: int = SAMPLE_RATE) -> T.InverseMelScale:
    return T.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=sr,
        f_min=F_MIN,
        f_max=F_MAX,
    )


def wav_to_mel(wav: torch.Tensor, transform: T.MelSpectrogram | None = None) -> torch.Tensor:
    """
    Convert waveform to normalized log-mel spectrogram.

    Args:
        wav: (1, T) or (T,)
    Returns:
        spec: (1, N_MELS, frames) in [0, 1]
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if transform is None:
        transform = make_mel_transform()

    mel = transform(wav)                        # (1, N_MELS, T)
    log_mel = torch.log1p(mel)                  # log1p for numerical stability
    return log_mel


def normalize_spec(spec: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Normalize spectrogram to [0, 1].

    Returns (normalized, min_val, max_val) — save min/max to invert later.
    """
    min_val = spec.min().item()
    max_val = spec.max().item()
    denom = max_val - min_val
    if denom < 1e-9:
        return spec, min_val, max_val
    return (spec - min_val) / denom, min_val, max_val


def denormalize_spec(spec: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return spec * (max_val - min_val) + min_val


def mel_to_wav(
    log_mel: torch.Tensor,
    inv_mel: T.InverseMelScale | None = None,
    n_iter: int = 64,
    sr: int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Convert log-mel spectrogram back to waveform using Griffin-Lim.

    Note: Griffin-Lim is approximate — there is always some quality loss.
    For production, replace with a neural vocoder (HiFi-GAN).

    Args:
        log_mel: (1, N_MELS, T)
    Returns:
        wav: (1, T_audio)
    """
    mel = torch.expm1(log_mel).clamp(min=0)  # invert log1p

    if inv_mel is None:
        inv_mel = make_inverse_mel(sr)

    # InverseMelScale expects (freq_bins, time) without batch
    linear = inv_mel(mel.squeeze(0))           # (n_stft, T)

    griffin_lim = T.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        n_iter=n_iter,
    )
    wav = griffin_lim(linear)                  # (T_audio,)
    return wav.unsqueeze(0)                    # (1, T_audio)


# ── Chunking helpers ────────────────────────────────────────────────────────

def chunk_spectrogram(
    spec: torch.Tensor,
    chunk_frames: int = 256,
    hop_frames: int = 128,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Split a spectrogram into overlapping chunks for batch inference.

    Args:
        spec:         (1, F, T)
        chunk_frames: width of each chunk in time frames
        hop_frames:   step between chunks (overlap = chunk - hop)
    Returns:
        chunks:   list of (1, F, chunk_frames) tensors (last chunk zero-padded)
        offsets:  time-frame start index of each chunk
    """
    _, F, T = spec.shape
    chunks, offsets = [], []

    start = 0
    while start < T:
        end = start + chunk_frames
        chunk = spec[:, :, start:end]

        # Zero-pad last chunk if needed
        if chunk.shape[-1] < chunk_frames:
            pad = chunk_frames - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        chunks.append(chunk)
        offsets.append(start)
        start += hop_frames

    return chunks, offsets


def stitch_chunks(
    chunks: list[torch.Tensor],
    offsets: list[int],
    total_frames: int,
    chunk_frames: int = 256,
    hop_frames: int = 128,
) -> torch.Tensor:
    """
    Overlap-add denoised chunks back into a full spectrogram.

    Overlapping regions are averaged using a Hann window envelope
    to prevent seam artifacts.

    Args:
        chunks:       list of (1, F, chunk_frames) denoised tensors
        offsets:      start frame of each chunk
        total_frames: target total time frames
    Returns:
        (1, F, total_frames)
    """
    F = chunks[0].shape[1]
    output  = torch.zeros(1, F, total_frames)
    weights = torch.zeros(1, 1, total_frames)

    # Hann envelope to fade edges of each chunk
    env = torch.hann_window(chunk_frames).unsqueeze(0).unsqueeze(0)  # (1, 1, chunk_frames)

    for chunk, start in zip(chunks, offsets):
        end = min(start + chunk_frames, total_frames)
        length = end - start

        output[:, :, start:end]  += chunk[:, :, :length] * env[:, :, :length]
        weights[:, :, start:end] += env[:, :, :length]

    # Normalize by accumulated weights
    output = output / weights.clamp(min=1e-9)
    return output
