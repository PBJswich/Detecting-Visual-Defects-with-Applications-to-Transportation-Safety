import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# utility functions and classes for YOLO loss
def xywh2xyxy(box):
    # box: (N, 4) in xywh
    x, y, w, h = box.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou(box1, box2, xywh=False, CIoU=True, eps=1e-7):
    # box1, box2: (N, 4) in xyxy
    # Returns IoU (N,)
    if xywh:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    # Intersection
    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)
    area1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    iou = inter / (union + eps)
    if not CIoU:
        return iou
    # CIoU penalty
    c2 = (
        torch.max(box1[:, None, 2:], box2[:, 2:]) - torch.min(box1[:, None, :2], box2[:, :2])
    ).clamp(0).pow(2).sum(2)
    rho2 = ((box1[:, None, :2] + box1[:, None, 2:]) / 2 - (box2[:, :2] + box2[:, 2:]) / 2).pow(2).sum(2)
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan((box1[:, None, 2] - box1[:, None, 0]) / (box1[:, None, 3] - box1[:, None, 1] + eps)) -
        torch.atan((box2[:, 2] - box2[:, 0]) / (box2[:, 3] - box2[:, 1] + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / (c2 + eps) + v * alpha)
    return ciou

def dist2bbox(distance, anchor_points, xywh=False):
    # distance: (N, 4), anchor_points: (N, 2)
    # Returns (N, 4) in xyxy
    x1 = anchor_points[:, 0] - distance[:, 0]
    y1 = anchor_points[:, 1] - distance[:, 1]
    x2 = anchor_points[:, 0] + distance[:, 2]
    y2 = anchor_points[:, 1] + distance[:, 3]
    if xywh:
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
    else:
        return torch.stack([x1, y1, x2, y2], dim=-1)

class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
    def forward(self, pred_dist, target):
        # pred_dist: (N, reg_max), target: (N,)
        target = target.clamp(0, self.reg_max - 1 - 1e-4)
        left = target.long()
        right = left + 1
        wl = right.float() - target
        wr = 1.0 - wl
        loss = (
            F.cross_entropy(pred_dist, left, reduction='none') * wl +
            F.cross_entropy(pred_dist, right, reduction='none') * wr
        )
        return loss

#  greedy matching for demo
class TaskAlignedAssigner:
    def __init__(self, topk=10, num_classes=1, alpha=0.5, beta=6.0):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        # pred_scores: (B, N, C), pred_bboxes: (B, N, 4), anchor_points: (N, 2)
        # gt_labels: (B, M, 1), gt_bboxes: (B, M, 4), mask_gt: (B, M, 1)
        B, N, C = pred_scores.shape
        M = gt_bboxes.shape[1]
        fg_mask = torch.zeros((B, N), dtype=torch.bool, device=pred_scores.device)
        target_bboxes = torch.zeros((B, N, 4), device=pred_scores.device)
        target_scores = torch.zeros((B, N, C), device=pred_scores.device)
        for b in range(B):
            for m in range(M):
                if not mask_gt[b, m]:
                    continue
                gt_box = gt_bboxes[b, m]
                gt_label = gt_labels[b, m, 0].long()
                ious = bbox_iou(pred_bboxes[b], gt_box.unsqueeze(0)).squeeze(1)
                topk_idx = torch.topk(ious, self.topk).indices
                fg_mask[b, topk_idx] = True
                target_bboxes[b, topk_idx] = gt_box
                target_scores[b, topk_idx, gt_label] = 1.0
        return None, target_bboxes, target_scores, fg_mask, None

# Utility functions 
def make_anchors(feats, strides, offset=0.5):
    # This is a simplified version. In practice, use Ultralytics' make_anchors.
    # feats: list of feature maps
    # strides: list of strides for each feature map
    # Returns anchor_points, stride_tensor
    anchor_points = []
    stride_tensor = []
    for i, f in enumerate(feats):
        _, _, h, w = f.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        points = torch.stack((x, y), dim=-1).reshape(-1, 2) + offset
        anchor_points.append(points)
        stride_tensor.append(torch.full((points.shape[0], 1), strides[i], device=f.device))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)

# Main Loss Class 
class YOLODetectionLoss(nn.Module):
    def __init__(self, yolo_model=None, grid_size=20):
        super().__init__()
        self.yolo_model = yolo_model
        self.grid_size = grid_size
        
        if yolo_model is not None:
            self.device = next(yolo_model.model.parameters()).device
            self.nc = yolo_model.model.model[-1].nc  # number of classes
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.nc = 1  # default number of classes
    
    def build_detection_target(self, det_labels, grid_size, img_size):
        """
        Convert YOLO-format labels to a dense grid tensor for detection loss.
        det_labels: list of [class, x_center, y_center, width, height] (normalized 0-1 floats)
        Returns: [5, grid_size, grid_size] tensor
        """
        target = torch.zeros(5, grid_size, grid_size, device=self.device)
        for box in det_labels:
            if len(box) >= 5:
                _, xc, yc, w, h = box[:5]
                i = int(float(xc) * grid_size)
                j = int(float(yc) * grid_size)
                i = min(max(i, 0), grid_size - 1)
                j = min(max(j, 0), grid_size - 1)
                target[0, j, i] = 1.0  # objectness
                target[1, j, i] = float(xc)  # x_center
                target[2, j, i] = float(yc)  # y_center
                target[3, j, i] = float(w)   # width
                target[4, j, i] = float(h)   # height
        return target
    
    def detection_loss_fn(self, pred, target):
        """
        Compute detection loss as objectness BCE + box MSE (only where objectness=1).
        pred, target: [batch, 5, grid_size, grid_size]
        """
        obj_loss = F.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0])
        mask = target[:, 0] > 0  # [batch, grid_size, grid_size]
        if mask.sum() > 0:
            pred_boxes = pred[:, 1:5].permute(0, 2, 3, 1)[mask]    # [num_pos, 4]
            target_boxes = target[:, 1:5].permute(0, 2, 3, 1)[mask] # [num_pos, 4]
            box_loss = F.mse_loss(pred_boxes, target_boxes)
        else:
            box_loss = torch.tensor(0.0, device=pred.device)
        return obj_loss + box_loss
    
    def forward(self, preds, targets):
        """
        Compute detection loss using simplified grid-based approach.
        preds: detection head output (will be reshaped to grid format)
        targets: dict with 'batch_idx', 'cls', 'bboxes' keys
        """
        print(f"[YOLO Loss] preds type: {type(preds)}")
        
        # Handle detection head output format
        if isinstance(preds, (list, tuple)):
            # Extract the detection results from the tuple
            detection_results = preds[0]  # This should be the actual predictions
            print(f"[YOLO Loss] detection_results shape: {detection_results.shape}")
        else:
            detection_results = preds
            print(f"[YOLO Loss] preds shape: {preds.shape}")
        
        # Debug targets
        print(f"[YOLO Loss] targets type: {type(targets)}")
        if isinstance(targets, dict):
            print(f"[YOLO Loss] targets keys: {list(targets.keys())}")
            for key, value in targets.items():
                if hasattr(value, 'shape'):
                    print(f"[YOLO Loss] targets['{key}'] shape: {value.shape}")
                else:
                    print(f"[YOLO Loss] targets['{key}'] type: {type(value)}")
        else:
            print(f"[YOLO Loss] targets is not a dict: {targets}")
        
        # Check if we have valid targets
        if not (isinstance(targets, dict) and 'batch_idx' in targets and 'cls' in targets and 'bboxes' in targets):
            print("[YOLO Loss] Missing required target keys, returning dummy loss")
            print(f"[YOLO Loss] Available keys: {list(targets.keys()) if isinstance(targets, dict) else 'not a dict'}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch_size = detection_results.shape[0] if len(detection_results.shape) > 2 else 1
        
        # convert predictions to grid format
        # assuming detection_results is [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
        # need to convert this to [batch, 5, grid_size, grid_size] format
        if len(detection_results.shape) == 3:
            # Reshape to grid format
            pred_grid = torch.zeros(batch_size, 5, self.grid_size, self.grid_size, device=self.device)
            
            # For now, use a simplified approach - take the first detection and place it in the center
            # In practice, you'd want to properly map detections to grid cells
            for b in range(batch_size):
                if detection_results[b].shape[0] > 0:
                    # Take the first detection
                    det = detection_results[b][0]
                    if len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        # Convert to center coordinates and normalize
                        xc = (x1 + x2) / 2.0
                        yc = (y1 + y2) / 2.0
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Place in center of grid
                        center_i = self.grid_size // 2
                        center_j = self.grid_size // 2
                        pred_grid[b, 0, center_j, center_i] = 1.0  # objectness
                        pred_grid[b, 1, center_j, center_i] = xc
                        pred_grid[b, 2, center_j, center_i] = yc
                        pred_grid[b, 3, center_j, center_i] = w
                        pred_grid[b, 4, center_j, center_i] = h
        else:
            # If not the expected format, return dummy loss
            print(f"[YOLO Loss] Unexpected prediction format: {detection_results.shape}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Build targets in grid format
        target_grid = torch.zeros(batch_size, 5, self.grid_size, self.grid_size, device=self.device)
        
        for i, (batch_idx, cls, bbox) in enumerate(zip(targets['batch_idx'], targets['cls'], targets['bboxes'])):
            batch_idx = batch_idx.item()
            if batch_idx < batch_size:
                # Convert bbox from [x1, y1, x2, y2] to [class, xc, yc, w, h] format
                x1, y1, x2, y2 = bbox
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                
                # Convert to grid coordinates
                grid_i = int(xc * self.grid_size)
                grid_j = int(yc * self.grid_size)
                grid_i = min(max(grid_i, 0), self.grid_size - 1)
                grid_j = min(max(grid_j, 0), self.grid_size - 1)
                
                target_grid[batch_idx, 0, grid_j, grid_i] = 1.0  # objectness
                target_grid[batch_idx, 1, grid_j, grid_i] = xc
                target_grid[batch_idx, 2, grid_j, grid_i] = yc
                target_grid[batch_idx, 3, grid_j, grid_i] = w
                target_grid[batch_idx, 4, grid_j, grid_i] = h
        
        # Compute loss
        loss = self.detection_loss_fn(pred_grid, target_grid)
        print(f"[YOLO Loss] Detection loss: {loss.item()}")
        
        return loss

# Usage:
# yolo_model = YOLO('yolov10s.pt')
# loss_fn = YOLODetectionLoss(yolo_model)
# loss = loss_fn(preds, targets)
