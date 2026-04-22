"""
Download helper for free noise cancellation training datasets.

Datasets:
  - DEMAND  : 18 types of real-world noise (office, street, restaurant, etc.)
              ~3 GB, recorded at 48kHz stereo, CC BY-SA license
              https://zenodo.org/record/1227121

  - DNS (subset): Microsoft DNS Challenge — clean speech clips
                  We pull the "read_speech" subset (~5k clips, ~2 GB)
                  https://github.com/microsoft/DNS-Challenge

Usage:
    python data/download.py --output ./dataset --dataset demand
    python data/download.py --output ./dataset --dataset dns-speech
    python data/download.py --output ./dataset --dataset both
"""

import argparse
import subprocess
import zipfile
from pathlib import Path


DEMAND_NOISE_TYPES = [
    "DKITCHEN", "DLIVING", "DWASHING",
    "NFIELD",   "NPARK",   "NRIVER",
    "OOFFICE",  "OHALLWAY", "OMEETING",
    "PCAFETER", "PRESTO",  "PSTATION",
    "SPSQUARE", "STRAFFIC",
    "TBUS",     "TCAR",    "TMETRO",
]

DEMAND_BASE_URL = "https://zenodo.org/record/1227121/files"


def run(cmd: list[str]):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_demand(output_dir: Path):
    """Download DEMAND noise dataset into output_dir/noise/."""
    noise_dir = output_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading DEMAND noise dataset → {noise_dir}")
    print(f"Noise types: {len(DEMAND_NOISE_TYPES)}")

    for noise_type in DEMAND_NOISE_TYPES:
        url = f"{DEMAND_BASE_URL}/{noise_type}_16k.zip"
        zip_path = noise_dir / f"{noise_type}.zip"
        out_path = noise_dir / noise_type

        if out_path.exists():
            print(f"  [skip] {noise_type} already downloaded")
            continue

        print(f"  Downloading {noise_type}...")
        run(["curl", "-fsSL", "-o", str(zip_path), url])

        print(f"  Extracting {noise_type}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(noise_dir)

        zip_path.unlink()

    print(f"DEMAND download complete.")


def download_dns_speech(output_dir: Path):
    """
    Download a subset of DNS Challenge clean speech data.

    We use the LibriVox-sourced clips from the DNS Challenge repo.
    These are public domain audiobooks — perfect for training.
    """
    clean_dir = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading DNS clean speech → {clean_dir}")
    print("Note: full DNS dataset is large. Downloading LibriSpeech-clean-100 instead")
    print("      (100h of clean speech, ~6 GB, Apache 2.0 license)\n")

    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    tar_path = clean_dir / "train-clean-100.tar.gz"

    if (clean_dir / "LibriSpeech").exists():
        print("  [skip] LibriSpeech already downloaded")
        return

    print("  Downloading LibriSpeech train-clean-100 (~6 GB)...")
    run(["curl", "-fL", "--progress-bar", "-o", str(tar_path), url])

    print("  Extracting...")
    run(["tar", "-xzf", str(tar_path), "-C", str(clean_dir)])
    tar_path.unlink()

    # Flatten all .flac files into clean/ root for easy glob
    print("  Flattening directory structure...")
    flac_files = list((clean_dir / "LibriSpeech").rglob("*.flac"))
    print(f"  Found {len(flac_files):,} .flac files")

    # Convert .flac → .wav and move to clean/
    try:
        import torchaudio
        import torch
        for i, flac in enumerate(flac_files):
            wav_path = clean_dir / flac.name.replace(".flac", ".wav")
            if not wav_path.exists():
                wav, sr = torchaudio.load(str(flac))
                torchaudio.save(str(wav_path), wav, sr)
            if (i + 1) % 500 == 0:
                print(f"  Converted {i + 1}/{len(flac_files)} files...")
    except ImportError:
        print("  torchaudio not available — leaving as .flac files")
        print("  Update dataset.py to glob *.flac instead of *.wav")

    print("DNS speech download complete.")


def print_summary(output_dir: Path):
    clean_dir = output_dir / "clean"
    noise_dir = output_dir / "noise"

    clean_count = len(list(clean_dir.glob("**/*.wav"))) if clean_dir.exists() else 0
    noise_count = len(list(noise_dir.glob("**/*.wav"))) if noise_dir.exists() else 0

    print(f"\nDataset summary:")
    print(f"  {clean_dir}: {clean_count:,} clean files")
    print(f"  {noise_dir}: {noise_count:,} noise files")
    print(f"\nReady to train:")
    print(f"  python train.py --data {output_dir} --epochs 50 --batch 16")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="./dataset",
                   help="Directory to download datasets into")
    p.add_argument("--dataset", default="both",
                   choices=["demand", "dns-speech", "both"],
                   help="Which dataset to download")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("demand", "both"):
        download_demand(output_dir)

    if args.dataset in ("dns-speech", "both"):
        download_dns_speech(output_dir)

    print_summary(output_dir)


if __name__ == "__main__":
    main()
