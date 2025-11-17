import torch
# Fix cuDNN issues with batch normalization
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch.optim import Adam
from model_components.multitask_model import MultiTaskModel
from model_components.multitask_loss import MultiTaskLoss
from model_components.multi_task_dataloader import MultiTaskDataLoader
from model_components.sam_encoder import SAMEncoder
from model_components.adapters import SegmentationAdapter, DetectionAdapter, ClassificationAdapter
from model_components.expert_modules import SegDetExpert, DetClsExpert, ClsSegExpert, BigExpert
from model_components.dual_task_heads import UNetSegmentationHead
from model_components.yolo_detection_loss import YOLODetectionLoss  
from segment_anything import sam_model_registry
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from model_components.SUTCrackDataset import SUTCrackDataset
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")

# 1. Instantiate the encoder
encoder = SAMEncoder(checkpoint_path="sam_vit_b_01ec64.pth")

# 2. Instantiate adapters 
seg_adapter = SegmentationAdapter(input_dim=256, output_dim=256)
det_adapter = DetectionAdapter(input_dim=256, output_dim=256)
cls_adapter = ClassificationAdapter(input_dim=256, output_dim=256)

# 3. Instantiate expert modules
segdet_expert = SegDetExpert(input_dim1=256, input_dim2=256, output_dim=256)
detcls_expert = DetClsExpert(input_dim1=256, input_dim2=256, output_dim=256)
clsseg_expert = ClsSegExpert(input_dim1=256, input_dim2=256, output_dim=256)

# 3b. Instantiate the big expert module
big_expert = BigExpert(input_dim1=256, input_dim2=256, input_dim3=256, output_dim=256)

# 4. Instantiate heads 
segmentation_head = UNetSegmentationHead(in_channels=256, out_channels=1)

# Detection head: use YOLO head
from ultralytics import YOLO
yolo_model = YOLO('yolov10s.pt')
detection_head = yolo_model.model.model[-1]  

# Classification head: use ResNet-50
from torchvision.models import resnet50, ResNet50_Weights
resnet_head = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = resnet_head.fc.in_features
resnet_head.fc = torch.nn.Linear(num_ftrs, 1)  # For binary classification
# Set to eval mode to use running stats in batch norm (fixes cuDNN issues)
resnet_head.eval()
classification_head = resnet_head

# Add a feature adapter for classification to convert (B, 256, 64, 64) -> (B, 3, 224, 224)
class ClassificationFeatureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 3, 1)  # 1x1 conv to get 3 channels
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize as expected by ResNet-50
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        x = (x - mean) / std
        # Debug print
        print('[Adapter] x shape:', x.shape, 'dtype:', x.dtype, 'min:', x.min().item(), 'max:', x.max().item(), 'mean:', x.mean().item())
        if torch.isnan(x).any() or torch.isinf(x).any():
            print('[Adapter] NaN or Inf detected in classification input!')
            raise ValueError('[Adapter] NaN or Inf detected in classification input!')
        return x
classification_feature_adapter = ClassificationFeatureAdapter()


model = MultiTaskModel(
    encoder, seg_adapter, det_adapter, cls_adapter,
    segdet_expert, detcls_expert, clsseg_expert,
    big_expert,
    segmentation_head, detection_head, classification_head,
    classification_feature_adapter
).to(device)

multitask_loss = MultiTaskLoss(yolo_model=yolo_model).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

# Instantiate datasets for each task
root_dir = "SUT-Crack"  
seg_ds = SUTCrackDataset(root_dir, task='segmentation')
det_ds = SUTCrackDataset(root_dir, task='detection')
cls_ds = SUTCrackDataset(root_dir, task='classification')

# Instantiate the multitask DataLoader
loader = MultiTaskDataLoader(seg_ds, det_ds, cls_ds, batch_size=3)

num_epochs = 200

# Clear GPU memory at start of training
torch.cuda.empty_cache()

for epoch in range(num_epochs):
    # Clear GPU memory at start of each epoch
    torch.cuda.empty_cache()
    
    for batch in loader:
        # Each batch is a list of dicts, each with 'image', 'task_id', etc.
        images = torch.stack([sample['image'] for sample in batch]).to(device)
        task_ids = [sample['task_id'] for sample in batch]
        # For simplicity, assume all samples in batch are for the same task
        # (If not, group by task and do multiple forwards per batch)
        task_id = task_ids[0]
        targets = {}
        if task_id == 'segmentation':
            # Filter out samples without masks for segmentation
            valid_samples = [sample for sample in batch if sample['mask'] is not None]
            if len(valid_samples) == 0:
                print("No valid samples in segmentation batch, skipping...")
                continue  
            batch = valid_samples 
            images = torch.stack([sample['image'] for sample in batch]).to(device)
            masks = [sample['mask'] for sample in batch]
            masks_tensor = torch.stack(masks).to(device)
            # Normalize masks to [0, 1] if needed
            if masks_tensor.max() > 1:
                masks_tensor = masks_tensor.float() / 255.0
            targets['mask'] = masks_tensor
        elif task_id == 'detection':
            # Format detection targets for YOLO loss
            batch_idx = []
            cls = []
            bboxes = []
            
            for i, sample in enumerate(batch):
                if sample['bboxes'] is not None:
                    img_h = sample['image'].shape[-2]
                    img_w = sample['image'].shape[-1]
                    for bbox in sample['bboxes']:
                        # bbox: [x1, y1, x2, y2] in pixel coordinates
                        bbox = list(bbox)
                        bbox[0] /= img_w  # x1
                        bbox[2] /= img_w  # x2
                        bbox[1] /= img_h  # y1
                        bbox[3] /= img_h  # y2
                        batch_idx.append(i)
                        cls.append(sample['class'])  # Use the class label
                        bboxes.append(bbox)
            
            if len(batch_idx) == 0:
                print("No valid detection targets in batch, skipping...")
                continue
                
            targets['batch_idx'] = torch.tensor(batch_idx, device=device)
            targets['cls'] = torch.tensor(cls, device=device)
            targets['bboxes'] = torch.tensor(bboxes, device=device, dtype=torch.float32)
            
            # Debug prints
            print(f"[DEBUG] Detection targets created:")
            print(f"[DEBUG] batch_idx: {targets['batch_idx'].shape}, {targets['batch_idx']}")
            print(f"[DEBUG] cls: {targets['cls'].shape}, {targets['cls']}")
            print(f"[DEBUG] bboxes: {targets['bboxes'].shape}, {targets['bboxes']}")
        elif task_id == 'classification':
            targets['class'] = torch.tensor([sample['class'] for sample in batch]).to(device)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        outputs = model(images, task_id)
        # outputs is now a dict with only the current task's output
        loss, loss_dict = multitask_loss(outputs, targets, task_id)

        # Checking for problematic loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss}, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        print(f"Epoch {epoch} | Task {task_id} | Loss: {loss.item():.4f} | Details: {loss_dict}")

    # future: validation, checkpointing, etc.

print("Training complete.")

# Save the trained model
import os
from datetime import datetime

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model state
model_path = f'models/multitask_crack_model_{timestamp}.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'loss': loss,
    'model_config': {
        'encoder_type': 'SAM_ViT_b',
        'adapters': 'MLP',
        'expert_modules': 'pairwise_MLP',
        'heads': 'dual_task'
    }
}, model_path)

print(f"Model saved to: {model_path}")

