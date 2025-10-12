import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    Focuses learning on hard examples and down-weights easy examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for better handling of class imbalance in segmentation.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss that allows control over false positives/negatives.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Tversky coefficient
        true_pos = (probs * targets).sum()
        false_neg = (targets * (1 - probs)).sum()
        false_pos = ((1 - targets) * probs).sum()
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1 - tversky

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss function that uses multiple loss types to better handle class imbalance.
    """
    def __init__(self, focal_weight=1.0, dice_weight=1.0, tversky_weight=0.5, 
                 focal_alpha=1.0, focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.tversky_weight * tversky)
        
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(),
            'tversky': tversky.item(),
            'total': total_loss.item()
        }

class AdaptiveThresholdLoss(nn.Module):
    """
    Loss function that learns the optimal threshold during training.
    """
    def __init__(self, base_loss_fn, threshold_lr=0.01):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.threshold_lr = threshold_lr

    def forward(self, inputs, targets):
        # Calculate base loss
        base_loss, loss_dict = self.base_loss_fn(inputs, targets)
        
        # Add threshold regularization to encourage reasonable threshold values
        threshold_reg = 0.1 * (self.threshold - 0.5) ** 2
        
        total_loss = base_loss + threshold_reg
        
        loss_dict['threshold'] = self.threshold.item()
        loss_dict['threshold_reg'] = threshold_reg.item()
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
