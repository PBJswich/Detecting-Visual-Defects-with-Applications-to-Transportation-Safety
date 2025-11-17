import torch
# Enable CUDA optimizations for better performance
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.enabled = True
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

# 1. Instantiate the encoder
encoder = SAMEncoder(checkpoint_path="sam_vit_b_01ec64.pth")
# Freeze encoder parameters
for p in encoder.parameters():
    p.requires_grad = False
encoder.eval()

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

# 4. Instantiate heads with improved architecture
from model_components.improved_segmentation_head import SimpleSegmentationHead
segmentation_head = SimpleSegmentationHead(in_channels=256, out_channels=1)

# Detection head: use YOLO head
from ultralytics import YOLO
yolo_model = YOLO('yolov10s.pt')
detection_head = yolo_model.model.model[-1]  

# Classification head: use ResNet-50
from torchvision.models import resnet50, ResNet50_Weights
resnet_head = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = resnet_head.fc.in_features
resnet_head.fc = torch.nn.Linear(num_ftrs, 1)  # For binary classification
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
        return x
classification_feature_adapter = ClassificationFeatureAdapter()


model = MultiTaskModel(
    encoder, seg_adapter, det_adapter, cls_adapter,
    segdet_expert, detcls_expert, clsseg_expert,
    big_expert,
    segmentation_head, detection_head, classification_head,
    classification_feature_adapter
).to(device)

multitask_loss = MultiTaskLoss(yolo_model=yolo_model, use_hybrid_loss=True).to(device)
optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)  # Lower learning rate for stability
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Instantiate datasets for each task
root_dir = "SUT-Crack"  
seg_ds = SUTCrackDataset(root_dir, task='segmentation')
det_ds = SUTCrackDataset(root_dir, task='detection')
cls_ds = SUTCrackDataset(root_dir, task='classification')

# Instantiate the multitask DataLoader with optimized settings
loader = MultiTaskDataLoader(seg_ds, det_ds, cls_ds, batch_size=12, num_workers=2, pin_memory=True)

num_epochs = 300  # Train longer for better convergence

# Clear GPU memory at start of training
torch.cuda.empty_cache()

print(f"Starting training for {num_epochs} epochs...")
print("=" * 60)

best_val_loss = float('inf')
patience_counter = 0
patience = 20

for epoch in range(num_epochs):
    # Clear GPU memory at start of each epoch
    torch.cuda.empty_cache()
    
    # Sync before epoch to free any cached memory
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    epoch_losses = {'segmentation': [], 'detection': [], 'classification': []}
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    batch_num = 0
    
    for batch in loader:
        # Each batch is a list of dicts, each with 'image', 'task_id', etc.
        images = torch.stack([sample['image'] for sample in batch]).to(device)
        task_ids = [sample['task_id'] for sample in batch]
        # For simplicity, assume all samples in batch are for the same task
        # (If not, group by task and do multiple forwards per batch)
        task_id = task_ids[0]
        targets = {}
        if task_id == 'segmentation':
            valid_samples = [sample for sample in batch if sample['mask'] is not None]
            if len(valid_samples) == 0:
                continue  
            batch = valid_samples 
            images = torch.stack([sample['image'] for sample in batch]).to(device)
            masks = [sample['mask'] for sample in batch]
            masks_tensor = torch.stack(masks).to(device)
            # Normalize masks to [0, 1] if needed
            if masks_tensor.max() > 1:
                masks_tensor = masks_tensor.float() / 255.0
            else:
                masks_tensor = masks_tensor.float()
            # Ensure masks are binary [0, 1]
            masks_tensor = (masks_tensor > 0.5).float()
            # Resize masks to match model output (1024x1024) if needed
            if masks_tensor.shape[-2:] != (1024, 1024):
                masks_tensor = torch.nn.functional.interpolate(
                    masks_tensor.unsqueeze(1), 
                    size=(1024, 1024), 
                    mode='nearest'
                ).squeeze(1)
            targets['mask'] = masks_tensor
        elif task_id == 'detection':
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
                        cls.append(sample['class'])  
                        bboxes.append(bbox)
            
            if len(batch_idx) == 0:
                continue
                
            targets['batch_idx'] = torch.tensor(batch_idx, device=device)
            targets['cls'] = torch.tensor(cls, device=device)
            targets['bboxes'] = torch.tensor(bboxes, device=device, dtype=torch.float32)
            

        elif task_id == 'classification':
            targets['class'] = torch.tensor([sample['class'] for sample in batch]).to(device)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        outputs = model(images, task_id)
        loss, loss_dict = multitask_loss(outputs, targets, task_id)

        # Checking for problematic loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: NaN/Inf loss detected, skipping batch")
            continue

        # Additional diagnostics for segmentation
        if task_id == 'segmentation':
            seg_pred = outputs['segmentation']
            seg_mean = seg_pred.mean().item()
            seg_std = seg_pred.std().item()
            seg_min = seg_pred.min().item()
            seg_max = seg_pred.max().item()
            # Check if output is constant (very low std)
            if seg_std < 1e-5:
                print(f"  Warning: Segmentation output is nearly constant (std={seg_std:.6f}, mean={seg_mean:.4f})")

        optimizer.zero_grad()
        loss.backward()
        
        # Check for gradient issues
        total_grad_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Clear gradient and free memory
        loss_item = loss.item()
        del loss, outputs
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Log progress
        epoch_losses[task_id].append(loss_item)
        batch_num += 1
        
        if batch_num % 50 == 0:
            avg_loss = sum(epoch_losses[task_id]) / len(epoch_losses[task_id]) if epoch_losses[task_id] else 0
            if task_id == 'segmentation' and 'segmentation' in loss_dict:
                seg_loss_info = loss_dict['segmentation']
                if isinstance(seg_loss_info, dict):
                    print(f"  Batch {batch_num}, Task: {task_id}, Avg Loss: {avg_loss:.4f}, "
                          f"Focal: {seg_loss_info.get('focal', 0):.4f}, "
                          f"Dice: {seg_loss_info.get('dice', 0):.4f}, "
                          f"Tversky: {seg_loss_info.get('tversky', 0):.4f}, "
                          f"Grad Norm: {total_grad_norm:.4f}")
                else:
                    print(f"  Batch {batch_num}, Task: {task_id}, Avg Loss: {avg_loss:.4f}, Grad Norm: {total_grad_norm:.4f}")
            else:
                print(f"  Batch {batch_num}, Task: {task_id}, Avg Loss: {avg_loss:.4f}, Grad Norm: {total_grad_norm:.4f}")

    # Print epoch summary
    print(f"\nEpoch {epoch+1} Summary:")
    total_avg_loss = 0
    for task, losses in epoch_losses.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"  {task.capitalize()}: {len(losses)} batches, avg loss: {avg_loss:.4f}")
            total_avg_loss += avg_loss
    
    # Learning rate scheduling
    if epoch_losses['segmentation'] or epoch_losses['detection'] or epoch_losses['classification']:
        scheduler.step(total_avg_loss)
    
    # Early stopping
    if total_avg_loss < best_val_loss:
        best_val_loss = total_avg_loss
        patience_counter = 0
        # Save best model
        best_model_path = 'models/multitask_crack_model_best.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': best_val_loss,
            'model_config': {
                'encoder_type': 'SAM_ViT_b',
                'adapters': 'MLP',
                'expert_modules': 'pairwise_MLP',
                'heads': 'dual_task'
            }
        }, best_model_path)
        print(f"  ✓ New best model saved (loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        

print(f"\nTraining complete. Saving model...")

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
    'best_loss': best_val_loss,
    'model_config': {
        'encoder_type': 'SAM_ViT_b',
        'adapters': 'MLP',
        'expert_modules': 'pairwise_MLP',
        'heads': 'improved_segmentation'
    }
}, model_path)

print(f"Model saved to: {model_path}")

