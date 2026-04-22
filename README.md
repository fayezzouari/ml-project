# Audio Denoiser — U-Net offline noise cancellation

A deep learning pipeline that removes background noise from audio files
using a U-Net trained on mel spectrograms.

## How it works

1. **Noisy audio** is converted to a mel spectrogram (frequency × time image)
2. **U-Net** predicts a soft mask that suppresses noise regions
3. **Overlap-add stitching** reassembles chunks without seam artifacts
4. **Griffin-Lim** converts the clean spectrogram back to a waveform


## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download training data
```bash
# Downloads DEMAND noise (~3 GB) + LibriSpeech clean speech (~6 GB)
python data/download.py --output ./dataset --dataset both
```

### 3. Train
```bash
python train.py --data ./dataset --epochs 50 --batch 16
```

Training logs appear in `./runs/` — view with:
```bash
tensorboard --logdir ./runs
```

Checkpoints saved to `./checkpoints/best.pt` (best val loss) and `last.pt`.

### 4. Denoise a file
```bash
python denoise.py \
  --input  noisy_recording.wav \
  --output clean_recording.wav \
  --checkpoint checkpoints/best.pt
```

## Key design decisions

### Soft mask prediction
The model predicts a value in [0, 1] per spectrogram cell rather than the
clean spectrogram directly. Multiplying the mask by the noisy input is more
stable to train and avoids the model hallucinating frequencies.

### Multi-resolution STFT loss
Using three FFT scales (512, 1024, 2048) forces the model to get both
transients (fine scale) and harmonic structure (coarse scale) right.

### Overlap-add chunking
Long files are split into overlapping 256-frame chunks (~4 sec each).
A Hann window envelope is applied before summing overlapping regions,
eliminating any click or seam artifact at chunk boundaries.

### Griffin-Lim reconstruction
Griffin-Lim is an iterative phase reconstruction algorithm — it recovers
a waveform from a magnitude spectrogram by alternating between time and
frequency domains. 64 iterations gives good quality. For production,
replace with a neural vocoder (HiFi-GAN) for significantly better audio.

## Training tips

- Start with `--base-ch 32` for faster experiments, scale to 64 for quality
- SNR range `-5 to 20 dB` in the dataset covers very noisy to nearly clean
- Cosine LR schedule runs the full training — don't stop early
- Validation loss plateaus around epoch 20-30 for most setups
- A GPU cuts training from ~8h to ~45min for 50 epochs on LibriSpeech-100

## Extending to real-time

The same trained model works for real-time inference. Replace the file
loading in `denoise.py` with a `pyaudio` ring buffer and process
overlapping 20ms frames. Achievable latency: ~40ms on CPU, ~10ms on GPU.
