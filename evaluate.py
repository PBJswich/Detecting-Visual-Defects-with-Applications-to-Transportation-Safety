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

from model_components.sam_encoder import SAMEncoder
from model_components.adapters import SegmentationAdapter, DetectionAdapter, ClassificationAdapter
from model_components.expert_modules import SegDetExpert, DetClsExpert, ClsSegExpert, BigExpert
from model_components.dual_task_heads import UNetSegmentationHead, YOLOFeatureAdapter
from model_components.multitask_model import MultiTaskModel
from model_components.multitask_loss import MultiTaskLoss
from model_components.SUTCrackDataset import SUTCrackDataset
from model_components.multi_task_dataloader import MultiTaskDataLoader
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights

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
        
        # big expert module
        big_expert = BigExpert(input_dim1=256, input_dim2=256, input_dim3=256, output_dim=256)
        
        # segmentation head
        segmentation_head = UNetSegmentationHead(in_channels=256, out_channels=1)
        
        # detection head
        yolo_model = YOLO('yolov10s.pt')
        detection_head = yolo_model.model.model[-1]
        
        # classification head
        resnet_head = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = resnet_head.fc.in_features
        resnet_head.fc = torch.nn.Linear(num_ftrs, 1)  # For binary classification
        classification_head = resnet_head
        
        # classification feature adapter
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
                device = x.device
                mean = self.mean.to(device)
                std = self.std.to(device)
                x = (x - mean) / std
                return x
        
        classification_feature_adapter = ClassificationFeatureAdapter()
        
        # full model
        model = MultiTaskModel(
            encoder=encoder,
            seg_adapter=seg_adapter,
            det_adapter=det_adapter,
            cls_adapter=cls_adapter,
            segdet_expert=segdet_expert,
            detcls_expert=detcls_expert,
            clsseg_expert=clsseg_expert,
            big_expert=big_expert,
            segmentation_head=segmentation_head,
            detection_head=detection_head,
            classification_head=classification_head,
            classification_feature_adapter=classification_feature_adapter
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"State dict had {len(state_dict)} keys")
        
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
        
        # datasets for each task
        seg_dataset = SUTCrackDataset(data_root, task='segmentation')
        det_dataset = SUTCrackDataset(data_root, task='detection')
        cls_dataset = SUTCrackDataset(data_root, task='classification')
        
        # multi-task data loader
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
        all_pred_probs = []
        all_target_masks = []
        
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
                pred_masks = (pred_probs > 0.1).float()  # Optimal threshold from analysis
                target_masks = (targets > 0).float()  # Ensure binary targets
                
                # Debug: print prediction statistics
                if len(iou_scores) == 0:  # Only for first batch
                    print(f"Segmentation prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"Segmentation probability range: [{pred_probs.min():.4f}, {pred_probs.max():.4f}]")
                    print(f"Target mask range: [{targets.min():.4f}, {targets.max():.4f}]")
                    print(f"Pred mask unique values: {torch.unique(pred_masks)}")
                    print(f"Target mask unique values: {torch.unique(target_masks)}")
                    print(f"Number of pixels > 0.5 threshold: {(pred_probs > 0.5).sum().item()}")
                    print(f"Number of pixels > 0.3 threshold: {(pred_probs > 0.3).sum().item()}")
                    print(f"Number of pixels > 0.1 threshold: {(pred_probs > 0.1).sum().item()}")
                
                # Store all predictions and targets for threshold analysis
                all_pred_probs.extend(pred_probs.cpu())
                all_target_masks.extend(target_masks.cpu())
                
                # Calculate metrics for each sample
                for i, (pred, target, image, prob) in enumerate(zip(pred_masks, target_masks, images, pred_probs)):
                    iou = self.calculate_iou(pred, target)
                    dice = self.calculate_dice(pred, target)
                    pixel_acc = self.calculate_pixel_accuracy(pred, target)
                    
                    iou_scores.append(iou)
                    dice_scores.append(dice)
                    pixel_accuracies.append(pixel_acc)
                    
                    # Save visualization for first few samples
                    if save_visualizations and len(iou_scores) <= 10:  # Save first 10 samples
                        vis_path = os.path.join(vis_dir, 'segmentation', f'seg_sample_{len(iou_scores):03d}.png')
                        self.visualize_segmentation(image, target, pred, prob, vis_path)
        
        if not iou_scores:
            return {'error': 'No segmentation samples found'}
        
        # Test different thresholds
        if all_pred_probs:
            threshold_analysis = self.analyze_thresholds(torch.stack(all_pred_probs), torch.stack(all_target_masks))
        else:
            threshold_analysis = {}
        
        return {
            'iou_mean': np.mean(iou_scores),
            'iou_std': np.std(iou_scores),
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'pixel_accuracy_mean': np.mean(pixel_accuracies),
            'pixel_accuracy_std': np.std(pixel_accuracies),
            'num_samples': len(iou_scores),
            'threshold_analysis': threshold_analysis
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
                pred_classes = (pred_probs > 0.1).float().cpu().numpy().flatten()  # Optimal threshold
                target_classes = targets.cpu().numpy().flatten()
                
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
    
    def safe_unwrap_predictions(self, predictions):
        """Safely unwrap YOLO tuple output to get the tensor - handles nested tuples"""
        if predictions is None:
            return None
        
        # Recursively unwrap tuples/lists
        while isinstance(predictions, (list, tuple)):
            if len(predictions) == 0:
                return None
            predictions = predictions[0]
        
        # If it's now a tensor, return it
        if isinstance(predictions, torch.Tensor):
            return predictions
        
        # If we couldn't unwrap to a tensor, return None
        return None
    
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
                
                try:
                    outputs = self.model(images, task_id='detection')
                    predictions = outputs['detection']
                    
                    # Safely unwrap predictions
                    predictions_tensor = self.safe_unwrap_predictions(predictions)
                    
                    if predictions_tensor is None:
                        print(f"Warning: Could not unwrap predictions, skipping batch")
                        continue
                    
                    # Process predictions for each image in the batch
                    for i, bboxes in enumerate(bboxes_list):
                        # Get predictions for this specific image
                        if i < predictions_tensor.shape[0]:
                            pred = predictions_tensor[i]
                            pred_boxes = self.process_detection_predictions(pred)
                        else:
                            pred_boxes = []
                        
                        # Handle bbox processing logic
                        if len(bboxes) > 0:
                            if save_visualizations and len(all_predictions) <= 10: 
                                vis_path = os.path.join(vis_dir, 'detection', f'det_sample_{len(all_predictions)+1:03d}.png')
                                self.visualize_detection(images[i], bboxes, pred_boxes, vis_path)
                            
                            # Calculate IoU for each predicted box with ground truth
                            for pred_box in pred_boxes:
                                max_iou = 0
                                for gt_box in bboxes:
                                    iou = self.calculate_bbox_iou(pred_box, gt_box)
                                    max_iou = max(max_iou, iou)
                                
                                all_predictions.append(1 if max_iou > 0.1 else 0)
                                all_targets.append(1)  
                            
                            for _ in range(len(bboxes) - len(pred_boxes)):
                                all_predictions.append(0)
                                all_targets.append(1)
                        
                        else:
                            if save_visualizations and len(all_predictions) <= 10:
                                vis_path = os.path.join(vis_dir, 'detection', f'det_sample_{len(all_predictions)+1:03d}.png')
                                self.visualize_detection(images[i], [], pred_boxes, vis_path)
                            
                            for _ in pred_boxes:
                                all_predictions.append(1)
                                all_targets.append(0)
                
                except Exception as e:
                    print(f"Error processing detection batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
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
    
    
    def process_detection_predictions(self, pred):
        """
        Process YOLO detection predictions to extract bounding boxes.
        pred: tensor of shape (num_detections, 6) with [x1, y1, x2, y2, conf, class]
        """
        # Recursively unwrap if pred is still a tuple
        while isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]
        
        if pred is None or not isinstance(pred, torch.Tensor):
            return []
        
        # Check if tensor is empty
        if pred.numel() == 0 or pred.shape[0] == 0:
            return []
        
        # Ensure pred is on CPU and convert to numpy for easier processing
        pred = pred.cpu().numpy()
        
        # Filter by confidence threshold (assuming confidence is in column 4)
        # Lower threshold because YOLO head might not output high confidence scores
        confidence_threshold = 0.1
        if pred.shape[1] >= 5:  # Has confidence column
            conf_mask = pred[:, 4] > confidence_threshold
            pred = pred[conf_mask]
        
        # Extract bounding boxes [x1, y1, x2, y2]
        if pred.shape[0] > 0 and pred.shape[1] >= 4:
            boxes = pred[:, :4].tolist()  # Convert to list of [x1, y1, x2, y2]
            return boxes
        else:
            return []
    
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
    
    def analyze_thresholds(self, pred_probs, target_masks):
        """Analyze performance at different threshold values"""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        
        print("\nThreshold Analysis:")
        print("Threshold | IoU    | Dice   | Pixel Acc | Pixels > Thresh")
        print("-" * 55)
        
        best_dice = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            pred_masks = (pred_probs > thresh).float()
            
            # Calculate metrics
            ious = []
            dices = []
            pixel_accs = []
            
            for pred, target in zip(pred_masks, target_masks):
                iou = self.calculate_iou(pred, target)
                dice = self.calculate_dice(pred, target)
                pixel_acc = self.calculate_pixel_accuracy(pred, target)
                
                ious.append(iou)
                dices.append(dice)
                pixel_accs.append(pixel_acc)
            
            mean_iou = np.mean(ious)
            mean_dice = np.mean(dices)
            mean_pixel_acc = np.mean(pixel_accs)
            pixels_above_thresh = (pred_probs > thresh).sum().item()
            
            results[thresh] = {
                'iou': mean_iou,
                'dice': mean_dice,
                'pixel_accuracy': mean_pixel_acc,
                'pixels_above_threshold': pixels_above_thresh
            }
            
            # Track best threshold based on Dice score
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_threshold = thresh
            
            print(f"{thresh:8.1f} | {mean_iou:6.3f} | {mean_dice:6.3f} | {mean_pixel_acc:9.3f} | {pixels_above_thresh:13d}")
        
        print(f"\nBest threshold based on Dice score: {best_threshold:.1f} (Dice: {best_dice:.3f})")
        results['best_threshold'] = best_threshold
        results['best_dice'] = best_dice
        
        return results
    
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
    
    def visualize_segmentation(self, image, target_mask, pred_mask, pred_probs, save_path):
        """Create and save segmentation visualization with gradient image"""
        # Convert tensors to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(pred_probs, torch.Tensor):
            pred_probs = pred_probs.cpu().numpy()
        
        # Squeeze channel dimension if present (should be H, W)
        if target_mask.ndim == 3 and target_mask.shape[0] == 1:
            target_mask = target_mask[0]
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
        if pred_probs.ndim == 3 and pred_probs.shape[0] == 1:
            pred_probs = pred_probs[0]
        
        # Normalize image to 0-1 range and transpose to (H, W, C)
        if image.max() > 1:
            image = image / 255.0
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        # Compute an adaptive threshold (Otsu) from probabilities for visualization
        def _otsu_threshold(probs_np):
            hist, bin_edges = np.histogram(probs_np.flatten(), bins=256, range=(0.0, 1.0))
            hist = hist.astype(np.float64)
            total = probs_np.size
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            weight1 = np.cumsum(hist)
            weight2 = total - weight1
            valid = (weight1 > 0) & (weight2 > 0)
            mean1 = np.cumsum(hist * bin_mids) / np.maximum(weight1, 1e-12)
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
            between_var = weight1 * weight2 * (mean1 - mean2) ** 2
            idx = np.argmax(np.where(valid, between_var, -1))
            return float(bin_mids[idx])

        otsu_thr = _otsu_threshold(pred_probs if isinstance(pred_probs, np.ndarray) else pred_probs)
        otsu_mask = (pred_probs > otsu_thr).astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(target_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted probability map (gradient image) - greyscale
        im = axes[2].imshow(pred_probs, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Predicted Probabilities\nMin: {pred_probs.min():.3f}, Max: {pred_probs.max():.3f}')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Predicted mask (adaptive Otsu threshold)
        axes[3].imshow(otsu_mask, cmap='gray')
        axes[3].set_title(f'Predicted Mask (Otsu thr={otsu_thr:.3f})')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_detection(self, image, gt_boxes, pred_boxes, save_path):
        """Create and save detection visualization"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.max() > 1:
            image = image / 255.0
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(image.transpose(1, 2, 0))
        
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=2, label='Ground Truth')
            axes[1].add_patch(rect)
        
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='green', linewidth=2, label='Prediction')
            axes[1].add_patch(rect)
        
        axes[1].set_title('Detection Results (Green: GT, Red: Pred)')
        axes[1].axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Ground Truth'),
                          Patch(facecolor='green', label='Prediction')]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_classification(self, image, true_label, pred_label, pred_prob, save_path):
        """Create and save classification visualization"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.max() > 1:
            image = image / 255.0
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.imshow(image.transpose(1, 2, 0))
        
        result_text = f'True Label: {"Crack" if true_label == 1 else "No Crack"}\n'
        result_text += f'Predicted: {"Crack" if pred_label == 1 else "No Crack"}\n'
        result_text += f'Confidence: {pred_prob:.3f}'
        
        color = 'green'
        
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
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {}
        
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
        
        results_path = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
        self.save_evaluation_results(results, results_path)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {results_path}")
        
        return results

def main():
    """Main evaluation function"""
    # Configuration
    model_path = "models/multitask_crack_model_best.pth"  # Update with your model
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
    
    # Evaluation
    results = evaluator.run_evaluation(test_loader)
    
    # Summary
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