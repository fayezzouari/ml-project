"""
U-Net architecture for audio denoising.

Operates on mel spectrograms treated as 2D images.
Encoder compresses → bottleneck learns noise pattern → decoder reconstructs.
Skip connections preserve fine-grained frequency detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two conv layers with batch norm and LeakyReLU."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock + 2x2 max pool for downsampling."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip  # return both pooled and skip


class UpBlock(nn.Module):
    """Bilinear upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd-sized dimensions from pooling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for spectrogram denoising.

    Input:  (B, 1, F, T) — single-channel mel spectrogram
    Output: (B, 1, F, T) — denoised spectrogram (same shape, sigmoid-activated mask)

    The model predicts a soft mask in [0, 1] which is multiplied
    by the noisy spectrogram to suppress noise regions.

    Args:
        base_channels: number of channels in first encoder block (doubles each level)
        depth: number of encoder/decoder levels (default 4)
        dropout: dropout rate applied in deeper blocks
    """

    def __init__(self, base_channels: int = 64, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        ch = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 1
        for i in range(depth):
            drop = dropout if i >= depth // 2 else 0.0
            self.encoders.append(DownBlock(in_ch, ch[i], drop))
            in_ch = ch[i]

        # Bottleneck
        self.bottleneck = ConvBlock(ch[depth - 1], ch[depth], dropout)

        # Decoder — receives upsampled + skip, so in_ch = ch[i] + ch[i]
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            drop = dropout if i >= depth // 2 else 0.0
            self.decoders.append(UpBlock(ch[i + 1] + ch[i], ch[i], drop))

        # Final 1x1 conv → sigmoid mask
        self.head = nn.Sequential(
            nn.Conv2d(ch[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy spectrogram (B, 1, F, T), values in [0, 1] after normalization
        Returns:
            clean: denoised spectrogram (B, 1, F, T)
        """
        skips = []
        out = x

        # Encode
        for enc in self.encoders:
            out, skip = enc(out)
            skips.append(skip)

        # Bottleneck
        out = self.bottleneck(out)

        # Decode with skip connections
        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)

        # Predict soft mask and apply to noisy input
        mask = self.head(out)
        return x * mask


def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"


if __name__ == "__main__":
    model = UNet(base_channels=64, depth=4)
    print(f"Parameters — {count_parameters(model)}")

    dummy = torch.randn(2, 1, 128, 256)  # batch=2, 128 mel bins, 256 time frames
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == dummy.shape, "Output shape must match input shape"
    print("Shape check passed.")
