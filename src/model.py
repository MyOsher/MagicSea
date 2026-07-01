"""Week 3-4 — a small U-Net for single-channel map inpainting.

Classic encoder-decoder with skip connections, sized for 64x64 maps.
Input : 2 channels  = [map with holes filled by 0, valid-pixel mask]
Output: 1 channel   = reconstructed map (normalized units)
"""
import torch
import torch.nn as nn


def _block(cin, cout):
    """Two 3x3 convs + ReLU — the standard U-Net double-conv."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()
        self.enc1 = _block(in_ch, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _block(base * 4, base * 8)

        # Transposed convs upsample; skip connections concatenate encoder maps.
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


if __name__ == "__main__":
    net = UNet()
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(2, 2, 64, 64)
    print(f"UNet params: {n_params:,}  output: {tuple(net(x).shape)}")
