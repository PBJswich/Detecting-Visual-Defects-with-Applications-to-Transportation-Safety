import torch
from torch.optim import Adam
from model_components.multitask_model import MultiTaskModel
from model_components.multitask_loss import MultiTaskLoss
from model_components.multi_task_dataloader import MultiTaskDataLoader
from model_components.sam_encoder import SAMEncoder
from model_components.adapters import SegmentationAdapter, DetectionAdapter, ClassificationAdapter
from model_components.expert_modules import SegDetExpert, DetClsExpert, ClsSegExpert
from model_components.dual_task_heads import SegDetHead, DetClsHead, ClsSegHead, SimpleSegmentationHead
from model_components.yolo_detection_loss import YOLODetectionLoss  
from segment_anything import sam_model_registry
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from model_components.SUTCrackDataset import SUTCrackDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")

# 1. Instantiate the encoder
encoder = SAMEncoder(checkpoint_path="sam_vit_b_01ec64.pth")

# 2. Instantiate adapters (use correct input/output dims as discussed)
seg_adapter = SegmentationAdapter(input_dim=256, output_dim=256)
det_adapter = DetectionAdapter(input_dim=256, output_dim=256)
cls_adapter = ClassificationAdapter(input_dim=256, output_dim=256)

# 3. Instantiate expert modules
segdet_expert = SegDetExpert(input_dim1=256, input_dim2=256, output_dim=256)
detcls_expert = DetClsExpert(input_dim1=256, input_dim2=256, output_dim=256)
clsseg_expert = ClsSegExpert(input_dim1=256, input_dim2=256, output_dim=256)

# 4. Instantiate heads (pass in your actual head modules)
# sam_decoder = ... (load your SAM decoder)
# yolo_head = ... (load your YOLOv10 head)
# resnet_head = ... (load your ResNet-50 head)

# Load the full SAM model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam_decoder = sam.mask_decoder  # This is the decoder module

# Download or specify the path to the YOLOv10 weights
yolo_model = YOLO('yolov10s.pt')  # or 'yolov10n.pt', 'yolov10m.pt', etc.

# The detection head is typically part of the model, but for your custom model,
# you may want to use the full model or extract the head if possible.
yolo_head = yolo_model.model.model[-1]  # This is often the detection head (check model structure)

# Load ResNet-50 with pretrained weights
resnet_head = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = resnet_head.fc.in_features
resnet_head.fc = torch.nn.Linear(num_ftrs, 1)  # For binary classification (crack/no crack)

segmentation_head = SimpleSegmentationHead(in_channels=256, out_channels=1)
segdet_head = SegDetHead(segmentation_head, yolo_head)
detcls_head = DetClsHead(yolo_head, resnet_head)
clsseg_head = ClsSegHead(resnet_head, segmentation_head)

# Now you can instantiate your MultiTaskModel
model = MultiTaskModel(
    encoder, seg_adapter, det_adapter, cls_adapter,
    segdet_expert, detcls_expert, clsseg_expert,
    segdet_head, detcls_head, clsseg_head
).to(device)

multitask_loss = MultiTaskLoss().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

# Instantiate datasets for each task
root_dir = "SUT-Crack"  # or your actual path
seg_ds = SUTCrackDataset(root_dir, task='segmentation')
det_ds = SUTCrackDataset(root_dir, task='detection')
cls_ds = SUTCrackDataset(root_dir, task='classification')

# Instantiate the multitask DataLoader
loader = MultiTaskDataLoader(seg_ds, det_ds, cls_ds, batch_size=3)

num_epochs = 10

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
        # (If not, you can group by task and do multiple forwards per batch)
        task_id = task_ids[0]
        targets = {}
        if task_id == 'segmentation':
            # Filter out samples without masks for segmentation
            valid_samples = [sample for sample in batch if sample['mask'] is not None]
            if len(valid_samples) == 0:
                print("No valid samples in segmentation batch, skipping...")
                continue  # Skip this batch
            batch = valid_samples  # Use only valid samples
            images = torch.stack([sample['image'] for sample in batch]).to(device)
            masks = [sample['mask'] for sample in batch]
            targets['mask'] = torch.stack(masks).to(device)
        elif task_id == 'detection':
            targets['bboxes'] = [sample['bboxes'] for sample in batch]  # List of bboxes per image
        elif task_id == 'classification':
            targets['class'] = torch.tensor([sample['class'] for sample in batch]).to(device)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        outputs = model(images, task_id)
        loss, loss_dict = multitask_loss(outputs, targets, task_id)

        # Check for problematic loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss}, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        print(f"Epoch {epoch} | Task {task_id} | Loss: {loss.item():.4f} | Details: {loss_dict}")

    # Optionally: validation, checkpointing, etc.

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

# Also save just the model for easier loading
model_only_path = f'models/multitask_crack_model_{timestamp}_model_only.pth'
torch.save(model.state_dict(), model_only_path)
print(f"Model state dict saved to: {model_only_path}")
