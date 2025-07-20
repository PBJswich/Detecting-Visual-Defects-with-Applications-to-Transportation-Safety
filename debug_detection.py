import torch
import torch.nn as nn
from model_components.yolo_detection_loss import YOLODetectionLoss

def test_detection_loss():
    """Test detection loss with dummy data to identify the issue."""
    
    # Create dummy detection predictions (simulating YOLO output)
    batch_size = 2
    num_detections = 300
    detection_results = torch.randn(batch_size, num_detections, 6)  # [x1, y1, x2, y2, conf, class]
    print(f"detection_results shape: {detection_results.shape}")
    
    # Create dummy targets in the expected format
    targets = {
        'batch_idx': torch.tensor([0, 0, 1, 1], dtype=torch.long),  # batch indices
        'cls': torch.tensor([0, 0, 0, 0], dtype=torch.long),        # class labels
        'bboxes': torch.tensor([                                    # bounding boxes [x1, y1, x2, y2]
            [0.1, 0.1, 0.3, 0.3],  # box 1 in batch 0
            [0.5, 0.5, 0.7, 0.7],  # box 2 in batch 0
            [0.2, 0.2, 0.4, 0.4],  # box 1 in batch 1
            [0.6, 0.6, 0.8, 0.8],  # box 2 in batch 1
        ], dtype=torch.float32)
    }
    
    print(f"targets['batch_idx']: {targets['batch_idx']}")
    print(f"targets['cls']: {targets['cls']}")
    print(f"targets['bboxes']: {targets['bboxes']}")
    
    # Create YOLO detection loss
    yolo_loss = YOLODetectionLoss(yolo_model=None, grid_size=20)
    
    # Test the loss
    try:
        loss = yolo_loss((detection_results, None), targets)  # Tuple format as expected
        print(f"Detection loss: {loss.item()}")
    except Exception as e:
        print(f"Error in detection loss: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_loss() 