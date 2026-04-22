"""
Dataset for audio denoising training.

Generates (noisy, clean) spectrogram pairs on-the-fly by mixing
clean speech/music with noise clips at random SNR levels.

Supports:
  - DNS Challenge dataset structure
  - Custom folder with clean/ and noise/ subdirectories
  - On-the-fly augmentation (SNR, noise type mixing)
"""

from __future__ import annotations
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.audio import (
    load_audio,
    wav_to_mel,
    normalize_spec,
    make_mel_transform,
    chunk_spectrogram,
    SAMPLE_RATE,
)


class DenoisingDataset(Dataset):
    """
    Pairs clean audio files with noise files to synthesize noisy training data.

    Directory structure expected:
        root/
          clean/   *.wav  (speech, music — the target signal)
          noise/   *.wav  (background noise — street, HVAC, music, etc.)

    Each __getitem__ call:
      1. Picks a random clean clip and slices a random window
      2. Picks a random noise clip and slices the same length
      3. Mixes them at a random SNR between snr_min and snr_max dB
      4. Converts both to normalized mel spectrograms
      5. Returns a random chunk of chunk_frames width

    Args:
        root:         path to dataset root (must contain clean/ and noise/)
        split:        "train" or "val" — splits files 90/10 by default
        chunk_frames: spectrogram width fed to the model (default 256 ≈ 4 sec)
        snr_min:      minimum signal-to-noise ratio in dB (default -5)
        snr_max:      maximum signal-to-noise ratio in dB (default 20)
        clip_seconds: duration of audio window to sample (default 4 sec)
        sample_rate:  all audio resampled to this rate
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        chunk_frames: int = 256,
        snr_min: float = -5.0,
        snr_max: float = 20.0,
        clip_seconds: float = 4.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        root = Path(root)
        self.chunk_frames = chunk_frames
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.clip_samples = int(clip_seconds * sample_rate)
        self.sr = sample_rate
        self.mel_transform = make_mel_transform(sample_rate)

        clean_files = sorted(
            (root / "clean").glob("**/*.wav")
        ) or sorted((root / "clean").glob("**/*.flac"))
        noise_files = sorted(
            (root / "noise").glob("**/*.wav")
        ) or sorted((root / "noise").glob("**/*.flac"))

        assert len(clean_files) > 0, f"No clean .wav/.flac files found in {root / 'clean'}"
        assert len(noise_files) > 0, f"No noise .wav/.flac files found in {root / 'noise'}"

        # 90/10 split deterministically
        split_idx = int(len(clean_files) * 0.9)
        if split == "train":
            self.clean_files = clean_files[:split_idx]
        else:
            self.clean_files = clean_files[split_idx:]

        self.noise_files = noise_files

        # Each epoch item = one random chunk from one random file
        # We set a fixed length so DataLoader knows the dataset size
        self.length = len(self.clean_files) * 10  # 10 random crops per file per epoch

    def __len__(self) -> int:
        return self.length

    def _load_clip(self, path: Path) -> torch.Tensor:
        """Load audio, repeat if shorter than clip_samples, slice random window."""
        wav = load_audio(path, target_sr=self.sr)  # (1, T)
        T = wav.shape[-1]

        if T < self.clip_samples:
            # Tile the audio to fill the clip window
            repeats = (self.clip_samples // T) + 1
            wav = wav.repeat(1, repeats)

        # Random start
        max_start = wav.shape[-1] - self.clip_samples
        start = random.randint(0, max(0, max_start))
        return wav[:, start : start + self.clip_samples]

    def _mix_at_snr(
        self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """
        Mix clean + noise at a target SNR (dB).

        SNR = 10 * log10(P_signal / P_noise)
        We scale the noise so the mixture hits exactly snr_db.
        """
        eps = 1e-9
        p_clean = clean.pow(2).mean().clamp(min=eps)
        p_noise = noise.pow(2).mean().clamp(min=eps)

        target_p_noise = p_clean / (10 ** (snr_db / 10))
        scale = (target_p_noise / p_noise).sqrt()

        return clean + scale * noise

    def __getitem__(self, idx: int) -> dict:
        clean_path = self.clean_files[idx % len(self.clean_files)]
        noise_path = random.choice(self.noise_files)

        clean_wav = self._load_clip(clean_path)
        noise_wav = self._load_clip(noise_path)

        snr = random.uniform(self.snr_min, self.snr_max)
        noisy_wav = self._mix_at_snr(clean_wav, noise_wav, snr)

        # Clamp to [-1, 1] to prevent clipping artifacts
        noisy_wav = noisy_wav.clamp(-1, 1)

        # Spectrogram conversion
        clean_spec = wav_to_mel(clean_wav, self.mel_transform)   # (1, F, T)
        noisy_spec = wav_to_mel(noisy_wav, self.mel_transform)   # (1, F, T)

        # Normalize each pair using the noisy spec's statistics
        # (the model sees noisy at inference time — normalize consistently)
        noisy_norm, min_val, max_val = normalize_spec(noisy_spec)
        clean_norm, _, _ = normalize_spec(clean_spec)

        # Random chunk
        _, _, T = noisy_norm.shape
        if T > self.chunk_frames:
            start = random.randint(0, T - self.chunk_frames)
            noisy_norm = noisy_norm[:, :, start : start + self.chunk_frames]
            clean_norm = clean_norm[:, :, start : start + self.chunk_frames]
        else:
            pad = self.chunk_frames - T
            noisy_norm = torch.nn.functional.pad(noisy_norm, (0, pad))
            clean_norm = torch.nn.functional.pad(clean_norm, (0, pad))

        return {
            "noisy": noisy_norm,        # (1, F, chunk_frames)
            "clean": clean_norm,        # (1, F, chunk_frames)
            "snr":   torch.tensor(snr),
            "file":  clean_path.stem,
        }
