import torch
import torch.nn as nn
from segment_anything import sam_model_registry

class SAMEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.encoder = sam.image_encoder
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # image preprocessing notes
        # x: (B, 3, H, W), normalized as expected by SAM ViT-b
        #https://github.com/facebookresearch/segment-anything#usage
        # Example normalization:
        # mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]
        # x = (x - mean) / std
        features = self.encoder(x)
        return features


# encoder = SAMEncoder(checkpoint_path="sam_vit_b_01ec64.pth")
# features = encoder(image_tensor) 