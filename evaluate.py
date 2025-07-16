import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
import json
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Import your model components
from model_components.sam_encoder import SAMEncoder
from model_components.adapters import SegmentationAdapter, DetectionAdapter, ClassificationAdapter
from model_components.expert_modules import SegDetExpert, DetClsExpert, ClsSegExpert
from model_components.dual_task_heads import SegDetHead, DetClsHead, ClsSegHead, SimpleSegmentationHead, SimpleDetectionHead
from model_components.multitask_model import MultiTaskModel
from model_components.multitask_loss import MultiTaskLoss
from model_components.SUTCrackDataset import SUTCrackDataset
from model_components.multi_task_dataloader import MultiTaskDataLoader

class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        """Load the trained model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model components
        encoder = SAMEncoder(checkpoint_path="sam_vit_b_01ec64.pth")
        seg_adapter = SegmentationAdapter(input_dim=256, output_dim=256)
        det_adapter = DetectionAdapter(input_dim=256, output_dim=256)
        cls_adapter = ClassificationAdapter(input_dim=256, output_dim=256)
        
        segdet_expert = SegDetExpert(input_dim1=256, input_dim2=256, output_dim=256)
        detcls_expert = DetClsExpert(input_dim1=256, input_dim2=256, output_dim=256)
        clsseg_expert = ClsSegExpert(input_dim1=256, input_dim2=256, output_dim=256)
        
        # Create segmentation head
        segmentation_head = SimpleSegmentationHead(in_channels=256, out_channels=1)
        
        # Create detection head
        detection_head = SimpleDetectionHead(in_channels=256, num_classes=1, num_anchors=3)
        
        # Create dual-task heads
        segdet_head = SegDetHead(segmentation_head, detection_head)
        detcls_head = DetClsHead(detection_head, None)  # No ResNet head needed
        clsseg_head = ClsSegHead(None, segmentation_head)  # No ResNet head needed
        
        # Create the full model
        model = MultiTaskModel(
            encoder=encoder,
            seg_adapter=seg_adapter,
            det_adapter=det_adapter,
            cls_adapter=cls_adapter,
            segdet_expert=segdet_expert,
            detcls_expert=detcls_expert,
            clsseg_expert=clsseg_expert,
            segdet_head=segdet_head,
            detcls_head=detcls_head,
            clsseg_head=clsseg_head
        )
        
        # Filter out ResNet-related keys that don't exist in current model
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out keys that contain 'resnet_head' since we don't use ResNet in current model
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'resnet_head' not in key:
                filtered_state_dict[key] = value
        
        # Load filtered state dict with strict=False to handle any remaining mismatches
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        # Print information about the loading process
        print(f"Original state dict had {len(state_dict)} keys")
        print(f"Filtered state dict has {len(filtered_state_dict)} keys")
        print(f"Removed {len(state_dict) - len(filtered_state_dict)} ResNet-related keys")
        
        if missing_keys:
            print(f"Missing keys (will use default initialization): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys (will be ignored): {len(unexpected_keys)} keys")
        
        model = model.to(self.device)
        print("Model loaded successfully!")
        return model
    
    def create_test_loader(self, data_root, batch_size=6):
        """Create test data loader"""
        print("Creating test data loader...")
        
        # Create datasets for each task
        seg_dataset = SUTCrackDataset(data_root, task='segmentation')
        det_dataset = SUTCrackDataset(data_root, task='detection')
        cls_dataset = SUTCrackDataset(data_root, task='classification')
        
        # Create multi-task data loader
        test_loader = MultiTaskDataLoader(seg_dataset, det_dataset, cls_dataset, batch_size)
        
        print(f"Test loader created with {len(test_loader)} batches")
        return test_loader
    
    def evaluate_segmentation(self, test_loader, save_visualizations=True, vis_dir='visualizations'):
        """Evaluate segmentation performance"""
        print("Evaluating segmentation...")
        
        if save_visualizations:
            os.makedirs(os.path.join(vis_dir, 'segmentation'), exist_ok=True)
        
        iou_scores = []
        dice_scores = []
        pixel_accuracies = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Segmentation"):
                # Filter batch for segmentation samples
                seg_samples = [s for s in batch if s['task_id'] == 'segmentation' and s['mask'] is not None]
                
                if not seg_samples:
                    continue
                
                # Prepare batch data
                images = torch.stack([s['image'] for s in seg_samples]).to(self.device)
                targets = torch.stack([s['mask'] for s in seg_samples]).to(self.device)
                
                outputs = self.model(images, task_id='segmentation')
                predictions = outputs['segmentation']  # Already logits
                
                # Apply sigmoid and threshold
                pred_probs = torch.sigmoid(predictions)
                pred_masks = (pred_probs > 0.5).float()
                target_masks = (targets > 0).float()  # Ensure binary targets
                
                # Debug: print prediction statistics
                if len(iou_scores) == 0:  # Only for first batch
                    print(f"Segmentation prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"Segmentation probability range: [{pred_probs.min():.4f}, {pred_probs.max():.4f}]")
                    print(f"Target mask range: [{targets.min():.4f}, {targets.max():.4f}]")
                    print(f"Pred mask unique values: {torch.unique(pred_masks)}")
                    print(f"Target mask unique values: {torch.unique(target_masks)}")
                
                # Calculate metrics for each sample
                for i, (pred, target, image) in enumerate(zip(pred_masks, target_masks, images)):
                    iou = self.calculate_iou(pred, target)
                    dice = self.calculate_dice(pred, target)
                    pixel_acc = self.calculate_pixel_accuracy(pred, target)
                    
                    iou_scores.append(iou)
                    dice_scores.append(dice)
                    pixel_accuracies.append(pixel_acc)
                    
                    # Save visualization for first few samples
                    if save_visualizations and len(iou_scores) <= 10:  # Save first 10 samples
                        vis_path = os.path.join(vis_dir, 'segmentation', f'seg_sample_{len(iou_scores):03d}.png')
                        self.visualize_segmentation(image, target, pred, vis_path)
        
        if not iou_scores:
            return {'error': 'No segmentation samples found'}
        
        return {
            'iou_mean': np.mean(iou_scores),
            'iou_std': np.std(iou_scores),
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'pixel_accuracy_mean': np.mean(pixel_accuracies),
            'pixel_accuracy_std': np.std(pixel_accuracies),
            'num_samples': len(iou_scores)
        }
    
    def evaluate_classification(self, test_loader, save_visualizations=True, vis_dir='visualizations'):
        """Evaluate classification performance"""
        print("Evaluating classification...")
        
        if save_visualizations:
            os.makedirs(os.path.join(vis_dir, 'classification'), exist_ok=True)
        
        all_predictions = []
        all_targets = []
        pos_vis_count = 0
        neg_vis_count = 0
        max_vis_per_class = 5
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Classification"):
                # Filter batch for classification samples
                cls_samples = [s for s in batch if s['task_id'] == 'classification']
                
                if not cls_samples:
                    continue
                
                # Prepare batch data
                images = torch.stack([s['image'] for s in cls_samples]).to(self.device)
                targets = torch.tensor([s['class'] for s in cls_samples], dtype=torch.float32).to(self.device)
                
                outputs = self.model(images, task_id='classification')
                predictions = outputs['classification']  # Already logits
                
                # Apply sigmoid and threshold
                pred_probs = torch.sigmoid(predictions)
                pred_classes = (pred_probs > 0.5).float().cpu().numpy().flatten()
                target_classes = targets.cpu().numpy().flatten()
                
                # Debug: print classification statistics
                if len(all_predictions) == 0:  # Only for first batch
                    print(f"Classification prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"Classification probability range: [{pred_probs.min():.4f}, {pred_probs.max():.4f}]")
                    print(f"Target classes: {np.unique(target_classes)}")
                    print(f"Pred classes: {np.unique(pred_classes)}")
                    print(f"Target class distribution: {np.bincount(target_classes.astype(int))}")
                    print(f"Pred class distribution: {np.bincount(pred_classes.astype(int))}")
                
                all_predictions.extend(pred_classes)
                all_targets.extend(target_classes)
                
                # Save up to 5 positive and 5 negative visualizations
                if save_visualizations and (pos_vis_count < max_vis_per_class or neg_vis_count < max_vis_per_class):
                    for i, (pred_class, target_class, pred_prob, image) in enumerate(zip(pred_classes, target_classes, pred_probs.cpu().numpy().flatten(), images)):
                        if target_class == 1 and pos_vis_count < max_vis_per_class:
                            vis_path = os.path.join(vis_dir, 'classification', f'cls_pos_{pos_vis_count+1:02d}.png')
                            self.visualize_classification(image, target_class, pred_class, pred_prob, vis_path)
                            pos_vis_count += 1
                        elif target_class == 0 and neg_vis_count < max_vis_per_class:
                            vis_path = os.path.join(vis_dir, 'classification', f'cls_neg_{neg_vis_count+1:02d}.png')
                            self.visualize_classification(image, target_class, pred_class, pred_prob, vis_path)
                            neg_vis_count += 1
                        if pos_vis_count >= max_vis_per_class and neg_vis_count >= max_vis_per_class:
                            break
        
        if not all_predictions:
            return {'error': 'No classification samples found'}
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(all_predictions)
        }
    
    def evaluate_detection(self, test_loader, save_visualizations=True, vis_dir='visualizations'):
        """Evaluate detection performance using YOLO metrics"""
        print("Evaluating detection...")
        
        if save_visualizations:
            os.makedirs(os.path.join(vis_dir, 'detection'), exist_ok=True)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Detection"):
                # Filter batch for detection samples
                det_samples = [s for s in batch if s['task_id'] == 'detection' and s['bboxes'] is not None]
                
                if not det_samples:
                    continue
                
                # Prepare batch data
                images = torch.stack([s['image'] for s in det_samples]).to(self.device)
                bboxes_list = [s['bboxes'] for s in det_samples]
                
                outputs = self.model(images, task_id='detection')
                predictions = outputs['detection']
                
                # Debug: print prediction shape for first few batches
                if len(all_predictions) == 0:  # Only print for first batch
                    print(f"Detection output shape: {predictions.shape}")
                    print(f"Detection output type: {type(predictions)}")
                    if hasattr(predictions, 'dtype'):
                        print(f"Detection output dtype: {predictions.dtype}")
                    print(f"Detection output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"Sample detection output (first 5 values): {predictions[0, :5, :].cpu().numpy()}")
                
                # Process YOLO outputs
                for i, (pred, bboxes) in enumerate(zip(predictions, bboxes_list)):
                    if len(bboxes) > 0:
                        # Convert detection predictions to bounding boxes
                        pred_boxes = self.process_detection_predictions(pred)
                        
                        # Save visualization for first few samples
                        if save_visualizations and len(all_predictions) <= 10:  # Save first 10 samples
                            vis_path = os.path.join(vis_dir, 'detection', f'det_sample_{len(all_predictions)+1:03d}.png')
                            self.visualize_detection(images[i], bboxes, pred_boxes, vis_path)
                        
                        # Calculate IoU for each predicted box with ground truth
                        for pred_box in pred_boxes:
                            max_iou = 0
                            for gt_box in bboxes:
                                iou = self.calculate_bbox_iou(pred_box, gt_box)
                                max_iou = max(max_iou, iou)
                            
                            # Consider as positive if IoU > 0.5
                            all_predictions.append(1 if max_iou > 0.5 else 0)
                            all_targets.append(1)  # Ground truth has boxes
                        
                        # Add false negatives (missed detections)
                        for _ in range(len(bboxes) - len(pred_boxes)):
                            all_predictions.append(0)
                            all_targets.append(1)
                    
                    else:
                        # No ground truth boxes
                        pred_boxes = self.process_detection_predictions(predictions[i])
                        
                        # Save visualization for first few samples
                        if save_visualizations and len(all_predictions) <= 10:  # Save first 10 samples
                            vis_path = os.path.join(vis_dir, 'detection', f'det_sample_{len(all_predictions)+1:03d}.png')
                            self.visualize_detection(images[i], [], pred_boxes, vis_path)
                        
                        for _ in pred_boxes:
                            all_predictions.append(1)  # False positive
                            all_targets.append(0)
        
        if not all_predictions:
            return {'error': 'No detection samples found'}
        
        # Calculate detection metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(all_predictions)
        }
    
    def process_detection_predictions(self, detection_output):
        """Process detection head output to extract bounding boxes"""
        # detection_output: (B, H*W*num_anchors, 4+1+num_classes)
        # Format: [x, y, w, h, confidence, class_score]
        
        boxes = []
        if detection_output is not None and detection_output.numel() > 0:
            try:
                # Convert to numpy
                det_np = detection_output.cpu().numpy()
                
                # Handle different output shapes
                if len(det_np.shape) == 3:
                    # (B, H*W*num_anchors, 6)
                    for batch_idx in range(det_np.shape[0]):
                        batch_detections = det_np[batch_idx]  # (H*W*num_anchors, 6)
                        
                        for detection in batch_detections:
                            if len(detection) >= 6:
                                # Extract components
                                x, y, w, h, confidence, class_score = detection[:6]
                                
                                # Apply confidence threshold (lowered for debugging)
                                if confidence > 0.1:  # Lower threshold to see more detections
                                    # Convert to x1, y1, x2, y2 format
                                    x1 = x - w/2
                                    y1 = y - h/2
                                    x2 = x + w/2
                                    y2 = y + h/2
                                    
                                    # Ensure coordinates are within image bounds
                                    x1 = max(0, min(1024, x1))
                                    y1 = max(0, min(1024, y1))
                                    x2 = max(0, min(1024, x2))
                                    y2 = max(0, min(1024, y2))
                                    
                                    if x2 > x1 and y2 > y1:  # Valid box
                                        boxes.append([x1, y1, x2, y2])
                elif len(det_np.shape) == 2:
                    # (H*W*num_anchors, 6) - single batch
                    for detection in det_np:
                        if len(detection) >= 6:
                            # Extract components
                            x, y, w, h, confidence, class_score = detection[:6]
                            
                            # Apply confidence threshold
                            if confidence > 0.5:
                                # Convert to x1, y1, x2, y2 format
                                x1 = x - w/2
                                y1 = y - h/2
                                x2 = x + w/2
                                y2 = y + h/2
                                
                                # Ensure coordinates are within image bounds
                                x1 = max(0, min(1024, x1))
                                y1 = max(0, min(1024, y1))
                                x2 = max(0, min(1024, x2))
                                y2 = max(0, min(1024, y2))
                                
                                if x2 > x1 and y2 > y1:  # Valid box
                                    boxes.append([x1, y1, x2, y2])
                else:
                    print(f"Unexpected detection output shape: {det_np.shape}")
                    
            except Exception as e:
                print(f"Error processing detection output: {e}")
                print(f"Detection output shape: {detection_output.shape if hasattr(detection_output, 'shape') else 'unknown'}")
        
        return boxes
    
    def calculate_bbox_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_iou(self, pred, target):
        """Calculate Intersection over Union"""
        intersection = torch.logical_and(pred, target).sum()
        union = torch.logical_or(pred, target).sum()
        return (intersection / (union + 1e-8)).item()
    
    def calculate_dice(self, pred, target):
        """Calculate Dice coefficient"""
        intersection = torch.logical_and(pred, target).sum()
        total = pred.sum() + target.sum()
        return (2 * intersection / (total + 1e-8)).item()
    
    def calculate_pixel_accuracy(self, pred, target):
        """Calculate pixel accuracy"""
        correct = torch.eq(pred, target).sum()
        total = pred.numel()
        return (correct / total).item()
    
    def plot_confusion_matrix(self, y_true, y_pred, task_name, save_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {task_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
    
    def save_evaluation_results(self, results, save_path):
        """Save evaluation results to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def visualize_segmentation(self, image, target_mask, pred_mask, save_path):
        """Create and save segmentation visualization"""
        # Convert tensors to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        
        # Squeeze channel dimension if present (should be H, W)
        if target_mask.ndim == 3 and target_mask.shape[0] == 1:
            target_mask = target_mask[0]
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
        
        # Normalize image to 0-1 range and transpose to (H, W, C)
        if image.max() > 1:
            image = image / 255.0
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(target_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_detection(self, image, gt_boxes, pred_boxes, save_path):
        """Create and save detection visualization"""
        # Convert tensors to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Normalize image to 0-1 range
        if image.max() > 1:
            image = image / 255.0
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image with detections
        axes[1].imshow(image.transpose(1, 2, 0))
        
        # Draw ground truth boxes in green
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='green', linewidth=2, label='Ground Truth')
            axes[1].add_patch(rect)
        
        # Draw predicted boxes in red
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=2, label='Prediction')
            axes[1].add_patch(rect)
        
        axes[1].set_title('Detection Results (Green: GT, Red: Pred)')
        axes[1].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Ground Truth'),
                          Patch(facecolor='red', label='Prediction')]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_classification(self, image, true_label, pred_label, pred_prob, save_path):
        """Create and save classification visualization"""
        # Convert tensors to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Normalize image to 0-1 range
        if image.max() > 1:
            image = image / 255.0
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Show image
        ax.imshow(image.transpose(1, 2, 0))
        
        # Add text with classification results
        result_text = f'True Label: {"Crack" if true_label == 1 else "No Crack"}\n'
        result_text += f'Predicted: {"Crack" if pred_label == 1 else "No Crack"}\n'
        result_text += f'Confidence: {pred_prob:.3f}'
        
        # Color code the text
        color = 'green' if true_label == pred_label else 'red'
        
        ax.text(0.05, 0.95, result_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color=color, weight='bold')
        
        ax.set_title('Classification Result')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_evaluation(self, test_loader, save_dir='evaluation_results'):
        """Run full evaluation pipeline"""
        print("Starting model evaluation...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {}
        
        # Evaluate each task
        try:
            seg_results = self.evaluate_segmentation(test_loader, save_visualizations=True, vis_dir=save_dir)
            results['segmentation'] = seg_results
            print(f"Segmentation Results: {seg_results}")
        except Exception as e:
            print(f"Error in segmentation evaluation: {e}")
            results['segmentation'] = {'error': str(e)}
        
        try:
            cls_results = self.evaluate_classification(test_loader, save_visualizations=True, vis_dir=save_dir)
            results['classification'] = cls_results
            print(f"Classification Results: {cls_results}")
        except Exception as e:
            print(f"Error in classification evaluation: {e}")
            results['classification'] = {'error': str(e)}
        
        try:
            det_results = self.evaluate_detection(test_loader, save_visualizations=True, vis_dir=save_dir)
            results['detection'] = det_results
            print(f"Detection Results: {det_results}")
        except Exception as e:
            print(f"Error in detection evaluation: {e}")
            results['detection'] = {'error': str(e)}
        
        # Save results
        results_path = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
        self.save_evaluation_results(results, results_path)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {results_path}")
        
        return results

def main():
    """Main evaluation function"""
    # Configuration
    model_path = "models/multitask_crack_model_20250715_233538.pth"  # Update with your model path
    data_root = "SUT-Crack"  # Update with your data path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 6  # Adjust based on your GPU memory
    
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")
    print(f"Data root: {data_root}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(data_root):
        print(f"Error: Data directory not found at {data_root}")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, device)
    
    # Create test data loader
    test_loader = evaluator.create_test_loader(data_root, batch_size)
    
    # Run evaluation
    results = evaluator.run_evaluation(test_loader)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for task, metrics in results.items():
        print(f"\n{task.upper()}:")
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

if __name__ == "__main__":
    main() 