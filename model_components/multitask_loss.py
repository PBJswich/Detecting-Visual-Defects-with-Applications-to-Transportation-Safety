import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_detection_loss import YOLODetectionLoss  

class MultiTaskLoss(nn.Module):
    """
    MultiTaskLoss with learnable uncertainty weighting for segmentation, detection, and classification.
    Implements Kendall et al. (2018) uncertainty weighting.
    """
    def __init__(self, yolo_model=None, seg_loss_type='bce', cls_loss_type='bce'):
        super().__init__()
        # Learnable log variances (one per task)
        self.log_vars = nn.ParameterDict({
            'segmentation': nn.Parameter(torch.zeros(1)),
            'detection': nn.Parameter(torch.zeros(1)),
            'classification': nn.Parameter(torch.zeros(1)),
        })
        # Loss functions
        if seg_loss_type == 'bce':
            self.seg_loss = nn.BCEWithLogitsLoss()
        else:
            self.seg_loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.BCEWithLogitsLoss() if cls_loss_type == 'bce' else nn.CrossEntropyLoss()
        # Detection loss with YOLO model
        if yolo_model is not None:
            self.det_loss = YOLODetectionLoss(yolo_model)
        else:
            # Fallback to dummy loss if no YOLO model provided
            self.det_loss = YOLODetectionLoss(None)

    def forward(self, outputs, targets, task_id):
        """
        outputs: dict from model (e.g., {'segmentation': ..., 'detection': ..., 'classification': ...})
        targets: dict with ground truth for each task
        task_id: str, current task
        Returns: total_loss, dict of individual losses
        """
        losses = {}
        total_loss = 0.0
        if task_id == 'segmentation':
            seg_pred = outputs['segmentation']
            seg_target = targets['mask']  # (B, H, W) or (B, 1, H, W)
            
            # Debug segmentation inputs
            print(f"[DEBUG] seg_pred shape: {seg_pred.shape}, dtype: {seg_pred.dtype}")
            print(f"[DEBUG] seg_pred min: {seg_pred.min().item()}, max: {seg_pred.max().item()}, mean: {seg_pred.mean().item()}")
            print(f"[DEBUG] seg_target shape: {seg_target.shape}, dtype: {seg_target.dtype}")
            print(f"[DEBUG] seg_target min: {seg_target.min().item()}, max: {seg_target.max().item()}, mean: {seg_target.float().mean().item()}")
            
            # Check for NaN or Inf values
            if torch.isnan(seg_pred).any() or torch.isinf(seg_pred).any():
                print("[ERROR] NaN or Inf detected in segmentation predictions!")
                seg_loss = torch.tensor(1.0, device=seg_pred.device, requires_grad=True)
            elif torch.isnan(seg_target).any() or torch.isinf(seg_target).any():
                print("[ERROR] NaN or Inf detected in segmentation targets!")
                seg_loss = torch.tensor(1.0, device=seg_target.device, requires_grad=True)
            else:
                if seg_target.dim() == 3:
                    seg_target = seg_target.unsqueeze(1)
                
                # Ensure targets are in valid range [0, 1]
                seg_target = torch.clamp(seg_target.float(), 0.0, 1.0)
                
                # Use BCEWithLogitsLoss which expects raw logits
                seg_loss = self.seg_loss(seg_pred, seg_target)
                
                # Check if loss is negative (shouldn't happen with BCE)
                if seg_loss.item() < 0:
                    print(f"[WARNING] Negative segmentation loss detected: {seg_loss.item()}")
                    print("[WARNING] This indicates a problem with the loss calculation or model outputs")
                    # Use a fallback loss
                    seg_loss = torch.tensor(1.0, device=seg_pred.device, requires_grad=True)
            
            print(f"[DEBUG] Raw segmentation loss: {seg_loss.item()}")
            print(f"[DEBUG] log_var (segmentation): {self.log_vars['segmentation'].item()}")
            
            # Clamp log_var to prevent extreme values
            log_var_clamped = torch.clamp(self.log_vars['segmentation'], min=-10, max=10)
            precision = torch.exp(-log_var_clamped)
            weighted_loss = precision * seg_loss + log_var_clamped
            
            print(f"[DEBUG] Weighted segmentation loss: {weighted_loss.item()}")
            losses['segmentation'] = seg_loss.item()
            total_loss += weighted_loss
        elif task_id == 'detection':
            det_pred = outputs['detection']
            # Pass the full targets dictionary to the YOLO loss
            det_loss = self.det_loss(det_pred, targets)
            precision = torch.exp(-self.log_vars['detection'])
            weighted_loss = precision * det_loss + self.log_vars['detection']
            losses['detection'] = det_loss.item()
            total_loss += weighted_loss
        elif task_id == 'classification':
            cls_pred = outputs['classification']
            cls_target = targets['class']  # (B,) or (B, 1)
            if cls_target.dim() == 1:
                cls_target = cls_target.unsqueeze(1)
            cls_loss = self.cls_loss(cls_pred, cls_target.float())
            precision = torch.exp(-self.log_vars['classification'])
            weighted_loss = precision * cls_loss + self.log_vars['classification']
            losses['classification'] = cls_loss.item()
            total_loss += weighted_loss
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
        return total_loss, losses

# Example usage:
# multitask_loss = MultiTaskLoss()
# total_loss, loss_dict = multitask_loss(outputs, targets, task_id='segmentation') 