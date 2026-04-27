"""
Compare two audio files visually: waveform, spectrogram, and difference.

Usage:
    python plot_compare.py --input noisy.wav --output denoised.wav
"""

import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Original (noisy) wav")
    p.add_argument("--output", required=True, help="Denoised wav")
    p.add_argument("--max-seconds", type=float, default=10.0, help="Seconds to plot")
    return p.parse_args()


def load(path):
    data, sr = sf.read(path, always_2d=True)
    return data[:, 0], sr  # mono


def spectrogram(wav, sr, n_fft=1024, hop=256):
    window = np.hanning(n_fft)
    frames = (len(wav) - n_fft) // hop + 1
    spec = np.zeros((n_fft // 2 + 1, frames))
    for i in range(frames):
        frame = wav[i * hop: i * hop + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
    return np.log1p(spec)


def main():
    args = parse_args()

    noisy, sr1 = load(args.input)
    clean, sr2 = load(args.output)

    assert sr1 == sr2, f"Sample rate mismatch: {sr1} vs {sr2}"
    sr = sr1

    # Trim to max_seconds
    n = int(args.max_seconds * sr)
    noisy = noisy[:n]
    clean = clean[:n]

    # Pad shorter to same length
    L = max(len(noisy), len(clean))
    noisy = np.pad(noisy, (0, L - len(noisy)))
    clean = np.pad(clean, (0, L - len(clean)))

    t = np.linspace(0, len(noisy) / sr, len(noisy))

    spec_noisy = spectrogram(noisy, sr)
    spec_clean = spectrogram(clean, sr)
    spec_diff  = spec_noisy - spec_clean  # positive = removed energy

    print(f"Noisy  — peak: {np.abs(noisy).max():.4f}  RMS: {np.sqrt((noisy**2).mean()):.4f}")
    print(f"Denoised — peak: {np.abs(clean).max():.4f}  RMS: {np.sqrt((clean**2).mean()):.4f}")
    print(f"RMS ratio: {np.sqrt((clean**2).mean()) / (np.sqrt((noisy**2).mean()) + 1e-9):.2f}x")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Noisy vs Denoised comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Waveforms ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, noisy, linewidth=0.4, color="steelblue")
    ax.set_title("Waveform — Noisy input")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1, 1)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, clean, linewidth=0.4, color="darkorange")
    ax.set_title("Waveform — Denoised output")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1, 1)

    # ── Spectrograms ───────────────────────────────────────────────────────
    extent = [0, len(noisy) / sr, 0, sr / 2 / 1000]

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(spec_noisy, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_title("Spectrogram — Noisy input")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(spec_clean, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_title("Spectrogram — Denoised output")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")

    # ── Difference spectrogram ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    im = ax.imshow(spec_diff, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r")
    ax.set_title("Difference spectrogram (blue = removed energy, red = added energy)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
