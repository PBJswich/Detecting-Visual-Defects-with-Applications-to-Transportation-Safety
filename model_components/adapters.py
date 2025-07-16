import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationAdapter(nn.Module):
    """
    Adapter for segmentation: 2-layer MLP (Linear -> GELU -> Linear)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, N, input_dim) or (B, input_dim)
        return self.mlp(x)

class DetectionAdapter(nn.Module):
    """
    Adapter for detection: 2-layer MLP (Linear -> ReLU -> Linear)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, N, input_dim) or (B, input_dim)
        return self.mlp(x)

class ClassificationAdapter(nn.Module):
    """
    Adapter for classification: 2-layer MLP (Linear -> ReLU -> Linear)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, input_dim) or (B, N, input_dim)
        return self.mlp(x) 