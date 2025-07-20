import torch
import torch.nn as nn


class UNetSegmentationHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, features=[128, 64, 32, 16]):
        super().__init__()
        self.encoder1 = self.contract_block(in_channels, features[0])
        self.encoder2 = self.contract_block(features[0], features[1])
        self.encoder3 = self.contract_block(features[1], features[2])
        self.encoder4 = self.contract_block(features[2], features[3])

        self.middle = nn.Sequential(
            nn.Conv2d(features[3], features[3], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features[3], features[3], 3, padding=1),
            nn.ReLU()
        )

        self.upconv4 = self.expand_block(features[3], features[2])
        self.upconv3 = self.expand_block(features[2], features[1])
        self.upconv2 = self.expand_block(features[1], features[0])
        self.upconv1 = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1)
        )

    def contract_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        return block

    def expand_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        # Middle
        mid = self.middle(enc4)
        # Decoder
        dec4 = self.upconv4(mid)
        dec4 = dec4 + enc3  # skip connection
        dec3 = self.upconv3(dec4)
        dec3 = dec3 + enc2
        dec2 = self.upconv2(dec3)
        dec2 = dec2 + enc1
        out = self.upconv1(dec2)
        return out

class YOLOFeatureAdapter(nn.Module):
    """
    Adapts a fused feature map (B, 256, 64, 64) to a list of three feature maps for YOLO detection head:
    (B, 128, 80, 80), (B, 256, 40, 40), (B, 512, 20, 20)
    """
    def __init__(self):
        super().__init__()
        # For (B, 128, 80, 80)
        self.to_80 = nn.Sequential(
            nn.Upsample(scale_factor=1.25, mode='bilinear', align_corners=False),  # 64->80
            nn.Conv2d(256, 128, 1)
        )
        # For (B, 256, 40, 40)
        self.to_40 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 64->32
        )
        # For (B, 512, 20, 20)
        self.to_20 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=4, padding=1)  # 64->16
        )

    def forward(self, x):
        # x: (B, 256, 64, 64)
        f80 = self.to_80(x)  # (B, 128, 80, 80)
        f40 = self.to_40(x)  # (B, 256, 32, 32)
        f40 = nn.functional.interpolate(f40, size=(40, 40), mode='bilinear', align_corners=False)
        f20 = self.to_20(x)  # (B, 512, 16, 16)
        f20 = nn.functional.interpolate(f20, size=(20, 20), mode='bilinear', align_corners=False)
        return [f80, f40, f20]

