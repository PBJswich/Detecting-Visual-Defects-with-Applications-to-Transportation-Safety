import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_detection_loss import YOLODetectionLoss
from .hybrid_yolo_loss import HybridYOLODetectionLoss
from .improved_segmentation_loss import CombinedSegmentationLoss  

class MultiTaskLoss(nn.Module):
    """
    MultiTaskLoss with learnable uncertainty weighting for segmentation, detection, and classification.
    Implements Kendall et al. (2018) uncertainty weighting.
    """
    def __init__(self, yolo_model=None, seg_loss_type='combined', cls_loss_type='bce', use_hybrid_loss=True):
        super().__init__()
        # Learnable log variances (one per task)
        self.log_vars = nn.ParameterDict({
            'segmentation': nn.Parameter(torch.zeros(1)),
            'detection': nn.Parameter(torch.zeros(1)),
            'classification': nn.Parameter(torch.zeros(1)),
        })
        # Loss functions
        if seg_loss_type == 'combined':
            self.seg_loss = CombinedSegmentationLoss(
                focal_weight=1.0, 
                dice_weight=1.0, 
                tversky_weight=0.5,
                focal_alpha=1.0,  
                focal_gamma=2.0,
                tversky_alpha=0.3,  
                tversky_beta=0.7    
            )
        elif seg_loss_type == 'bce':
            self.seg_loss = nn.BCEWithLogitsLoss()
        else:
            self.seg_loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.BCEWithLogitsLoss() if cls_loss_type == 'bce' else nn.CrossEntropyLoss()

        # Use hybrid loss if requested, otherwise use original
        if use_hybrid_loss:
            self.det_loss = HybridYOLODetectionLoss(yolo_model, num_classes=1)
        else:
            self.det_loss = YOLODetectionLoss(yolo_model)

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
            # Check for NaN or Inf values
            if torch.isnan(seg_pred).any() or torch.isinf(seg_pred).any():
                seg_loss = torch.tensor(1.0, device=seg_pred.device, requires_grad=True)
                seg_loss_dict = {'total': 1.0, 'focal': 0.0, 'dice': 0.0, 'tversky': 0.0}
            elif torch.isnan(seg_target).any() or torch.isinf(seg_target).any():
                seg_loss = torch.tensor(1.0, device=seg_target.device, requires_grad=True)
                seg_loss_dict = {'total': 1.0, 'focal': 0.0, 'dice': 0.0, 'tversky': 0.0}
            else:
                if seg_target.dim() == 3:
                    seg_target = seg_target.unsqueeze(1)
                # Ensure targets are in valid range [0, 1]
                seg_target = torch.clamp(seg_target.float(), 0.0, 1.0)
                
                if hasattr(self.seg_loss, 'forward') and 'CombinedSegmentationLoss' in str(type(self.seg_loss)):
                    seg_loss, seg_loss_dict = self.seg_loss(seg_pred, seg_target)
                else:
                    seg_loss = self.seg_loss(seg_pred, seg_target)
                    seg_loss_dict = {'total': seg_loss.item()}
                
                # Clamp loss to reasonable range instead of setting to 1.0
                # This prevents masking real problems while avoiding extreme values
                if seg_loss.item() < 0:
                    seg_loss = torch.clamp(seg_loss, min=0.0)
                if seg_loss.item() > 100:
                    seg_loss = torch.clamp(seg_loss, max=100.0)
            
            # Clamp log_var to prevent extreme values
            log_var_clamped = torch.clamp(self.log_vars['segmentation'], min=-10, max=10)
            precision = torch.exp(-log_var_clamped)
            weighted_loss = precision * seg_loss + log_var_clamped
            losses['segmentation'] = seg_loss_dict
            total_loss += weighted_loss
        elif task_id == 'detection':
            det_pred = outputs['detection']
            # Pass the full targets dictionary to the YOLO loss
            det_loss = self.det_loss(det_pred, targets)
            
            # Scale loss to reasonable range
            det_loss_scaled = det_loss
            
            precision = torch.exp(-self.log_vars['detection'])
            weighted_loss = precision * det_loss_scaled + self.log_vars['detection']
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