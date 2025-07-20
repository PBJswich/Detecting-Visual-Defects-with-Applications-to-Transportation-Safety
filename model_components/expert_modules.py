import torch
import torch.nn as nn

class SegDetExpert(nn.Module):
    """
    Expert module for Segmentation + Detection.
    Concatenates features and passes through an MLP for fusion.
    """
    def __init__(self, input_dim1, input_dim2, output_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, feat1, feat2):
        # feat1, feat2: (B, N, C)
        x = torch.cat([feat1, feat2], dim=-1)
        return self.fuse(x)

class DetClsExpert(nn.Module):
    """
    Expert module for Detection + Classification.
    """
    def __init__(self, input_dim1, input_dim2, output_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, feat1, feat2):
        # feat1, feat2: (B, N, C)
        x = torch.cat([feat1, feat2], dim=-1)
        return self.fuse(x)

class ClsSegExpert(nn.Module):
    """
    Expert module for Classification + Segmentation.
    """
    def __init__(self, input_dim1, input_dim2, output_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, feat1, feat2):
        # feat1, feat2: (B, N, C)
        x = torch.cat([feat1, feat2], dim=-1)
        return self.fuse(x)

class BigExpert(nn.Module):
    """
    Big expert module that fuses outputs from SegDetExpert, DetClsExpert, and ClsSegExpert.
    Concatenates all three and passes through an MLP for fusion.
    """
    def __init__(self, input_dim1, input_dim2, input_dim3, output_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2 + input_dim3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, feat1, feat2, feat3):
        # feat1, feat2, feat3: (B, N, C)
        x = torch.cat([feat1, feat2, feat3], dim=-1)
        return self.fuse(x) 