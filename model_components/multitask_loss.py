import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_detection_loss import YOLODetectionLoss  

class MultiTaskLoss(nn.Module):
    """
    MultiTaskLoss with learnable uncertainty weighting for segmentation, detection, and classification.
    Implements Kendall et al. (2018) uncertainty weighting.
    """
    def __init__(self, seg_loss_type='bce', cls_loss_type='bce'):
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
        # Detection loss placeholder (user to implement YOLO loss)
        self.det_loss = YOLODetectionLoss()

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
            if seg_target.dim() == 3:
                seg_target = seg_target.unsqueeze(1)
            seg_loss = self.seg_loss(seg_pred, seg_target.float())
            precision = torch.exp(-torch.clamp(self.log_vars['segmentation'], min=-10, max=10))
            weighted_loss = precision * seg_loss + self.log_vars['segmentation']
            losses['segmentation'] = seg_loss.item()
            total_loss += weighted_loss
        elif task_id == 'detection':
            det_pred = outputs['detection']
            det_target = targets['bboxes']
            det_loss = self.det_loss(det_pred, det_target)
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