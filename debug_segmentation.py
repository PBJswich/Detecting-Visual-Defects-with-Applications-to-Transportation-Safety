import torch
import torch.nn as nn
import torch.nn.functional as F

def test_segmentation_loss():
    """Test segmentation loss with dummy data to identify the issue."""
    
    # Create dummy predictions and targets
    batch_size = 2
    height, width = 64, 64
    
    # Dummy predictions (logits from model)
    seg_pred = torch.randn(batch_size, 1, height, width) * 0.1  # Small random values
    print(f"seg_pred shape: {seg_pred.shape}")
    print(f"seg_pred min: {seg_pred.min().item()}, max: {seg_pred.max().item()}, mean: {seg_pred.mean().item()}")
    
    # Dummy targets (binary masks)
    seg_target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    print(f"seg_target shape: {seg_target.shape}")
    print(f"seg_target min: {seg_target.min().item()}, max: {seg_target.max().item()}, mean: {seg_target.mean().item()}")
    
    # Test BCEWithLogitsLoss
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(seg_pred, seg_target)
    print(f"BCE loss: {loss.item()}")
    
    # Test with sigmoid activation (should be equivalent)
    seg_pred_sigmoid = torch.sigmoid(seg_pred)
    bce_loss_sigmoid = nn.BCELoss()
    loss_sigmoid = bce_loss_sigmoid(seg_pred_sigmoid, seg_target)
    print(f"BCE loss with sigmoid: {loss_sigmoid.item()}")
    
    # Test with extreme values
    print("\n--- Testing with extreme values ---")
    seg_pred_extreme = torch.tensor([[[[-10.0, 10.0], [0.0, 0.0]]]])  # Very extreme logits
    seg_target_extreme = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])   # Binary targets
    loss_extreme = bce_loss(seg_pred_extreme, seg_target_extreme)
    print(f"Extreme values loss: {loss_extreme.item()}")
    
    # Test with all zeros
    print("\n--- Testing with all zeros ---")
    seg_pred_zeros = torch.zeros(1, 1, 2, 2)
    seg_target_zeros = torch.zeros(1, 1, 2, 2)
    loss_zeros = bce_loss(seg_pred_zeros, seg_target_zeros)
    print(f"All zeros loss: {loss_zeros.item()}")
    
    # Test with all ones
    print("\n--- Testing with all ones ---")
    seg_pred_ones = torch.zeros(1, 1, 2, 2)  # logits for sigmoid(0) = 0.5
    seg_target_ones = torch.ones(1, 1, 2, 2)
    loss_ones = bce_loss(seg_pred_ones, seg_target_ones)
    print(f"All ones loss: {loss_ones.item()}")

if __name__ == "__main__":
    test_segmentation_loss() 