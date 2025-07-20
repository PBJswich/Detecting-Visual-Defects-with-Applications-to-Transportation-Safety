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
        # x: (B, 3, H, W), normalized expected by SAM ViT-b
        #https://github.com/facebookresearch/segment-anything#usage
        features = self.encoder(x)
        return features

