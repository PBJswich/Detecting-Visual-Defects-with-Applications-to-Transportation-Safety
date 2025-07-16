import torch
import torch.nn as nn
from ultralytics import YOLO
import torch.nn.functional as F

# Load a pretrained YOLOv10 model
yolo_model = YOLO('yolov10s.pt')  

def get_device_from_pred(pred):
    if isinstance(pred, torch.Tensor):
        return pred.device
    elif isinstance(pred, dict):
        # Find the first tensor in the dict
        for v in pred.values():
            if isinstance(v, torch.Tensor):
                return v.device
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                return v[0].device
    elif isinstance(pred, (list, tuple)) and len(pred) > 0:
        if isinstance(pred[0], torch.Tensor):
            return pred[0].device
        elif isinstance(pred[0], dict):
            # Recursively check inside the dict
            return get_device_from_pred(pred[0])
    # Fallback to CPU
    return torch.device('cpu')

class YOLODetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize any parameters needed for the loss
        
    def forward(self, pred, targets):
        # Handle different prediction formats
        if isinstance(pred, dict):
            # Extract first tensor from dict
            pred_tensor = next(iter(pred.values()))
            if isinstance(pred_tensor, list):
                pred_tensor = pred_tensor[0]  # Take first element if it's a list
        elif isinstance(pred, list):
            pred_tensor = pred[0]  # Take first element if it's a list
        else:
            pred_tensor = pred
        
        # Ensure pred_tensor is a tensor
        if not isinstance(pred_tensor, torch.Tensor):
            # If it's still not a tensor, create a dummy loss
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(0.0, device=device)
        
        # Create a dummy target tensor for now
        target_tensor = torch.zeros_like(pred_tensor)
        
        # Simple MSE loss (replace with proper YOLO loss later)
        loss = F.mse_loss(pred_tensor, target_tensor)
        return loss
