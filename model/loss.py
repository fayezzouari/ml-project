"""
Loss functions for spectrogram denoising.

Uses a combination of:
  1. Multi-resolution STFT loss  — penalizes spectral differences at multiple scales
  2. L1 loss on the spectrogram  — pixel-level accuracy
  3. SI-SNR loss (optional)      — signal-level quality metric

Multi-resolution STFT is key: a single STFT scale misses either
fine or coarse frequency structure. Using 3 scales simultaneously
forces the model to get both right.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Single-scale STFT loss: spectral convergence + log magnitude L1.

    spectral_convergence = ||mag_clean - mag_pred||_F / ||mag_clean||_F
    log_magnitude         = ||log(mag_clean) - log(mag_pred)||_1
    """

    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", torch.hann_window(win_size))

    def stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude. x: (B, T)"""
        B, T = x.shape
        stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
        )
        return stft.abs().clamp(min=1e-9)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   predicted waveform (B, T)
            target: clean waveform    (B, T)
        """
        mag_pred = self.stft_mag(pred)
        mag_tgt  = self.stft_mag(target)

        # Spectral convergence
        sc = torch.norm(mag_tgt - mag_pred, p="fro") / torch.norm(mag_tgt, p="fro")

        # Log magnitude L1
        lm = F.l1_loss(torch.log(mag_pred), torch.log(mag_tgt))

        return sc + lm


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss across 3 FFT scales.

    Coarse (2048): captures harmonic structure
    Mid   (1024): balances time/frequency resolution
    Fine  (512):  captures transients and attacks

    This is the standard loss used in modern audio denoising papers
    (SEGAN, MetricGAN, HiFi-GAN all use variants of this).
    """

    SCALES = [
        (2048, 512, 2048),   # (fft, hop, win)
        (1024, 256, 1024),
        (512,  128, 512),
    ]

    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(fft, hop, win) for fft, hop, win in self.SCALES
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum(loss(pred, target) for loss in self.losses) / len(self.losses)


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss.

    Higher SI-SNR = better. We return the negative so it can be minimized.
    Robust to amplitude scaling differences between pred and target.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        # Zero-mean
        target = target - target.mean(dim=-1, keepdim=True)
        pred   = pred   - pred.mean(dim=-1, keepdim=True)

        # s_target: projection of pred onto target
        dot = (pred * target).sum(dim=-1, keepdim=True)
        s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)

        # noise
        e_noise = pred - s_target

        si_snr = 10 * torch.log10(
            s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps) + eps
        )
        return -si_snr.mean()


class DenoisingLoss(nn.Module):
    """
    Combined loss for offline denoising.

    total = w_stft * MRSTFT + w_l1 * L1(spectrogram) + w_sisnr * SI-SNR

    The model outputs a denoised spectrogram. We compute waveform losses
    by converting back via Griffin-Lim (approximate) or by passing through
    a vocoder. For simplicity here we use spectrogram-space L1 + MRSTFT
    on waveforms when available.
    """

    def __init__(self, w_stft: float = 1.0, w_l1: float = 0.5, w_sisnr: float = 0.1):
        super().__init__()
        self.w_stft   = w_stft
        self.w_l1     = w_l1
        self.w_sisnr  = w_sisnr
        self.mrstft   = MultiResolutionSTFTLoss()
        self.sisnr    = SISNRLoss()

    def forward(
        self,
        pred_spec: torch.Tensor,    # (B, 1, F, T) predicted spectrogram
        target_spec: torch.Tensor,  # (B, 1, F, T) clean spectrogram
        pred_wav: torch.Tensor | None = None,    # (B, T) optional waveform
        target_wav: torch.Tensor | None = None,  # (B, T) optional waveform
    ) -> dict:

        # Spectrogram L1
        l1 = F.l1_loss(pred_spec, target_spec)

        losses = {"l1": l1, "total": self.w_l1 * l1}

        # Waveform losses (when waveforms are provided)
        if pred_wav is not None and target_wav is not None:
            stft_loss  = self.mrstft(pred_wav, target_wav)
            sisnr_loss = self.sisnr(pred_wav, target_wav)
            losses["stft"]   = stft_loss
            losses["sisnr"]  = sisnr_loss
            losses["total"] += self.w_stft * stft_loss + self.w_sisnr * sisnr_loss

        return losses
