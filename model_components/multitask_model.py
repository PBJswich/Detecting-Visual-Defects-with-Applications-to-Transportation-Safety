import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components.dual_task_heads import YOLOFeatureAdapter

class MultiTaskModel(nn.Module):
    """
    MultiTaskModel wires together the encoder, adapters, expert modules, and the big expert module.
    Routing is determined by task_id ('segmentation', 'detection', 'classification').
    """
    def __init__(self, encoder,
                 seg_adapter, det_adapter, cls_adapter,
                 segdet_expert, detcls_expert, clsseg_expert,
                 big_expert,
                 segmentation_head, detection_head, classification_head,
                 classification_feature_adapter):
        super().__init__()
        self.encoder = encoder
        self.seg_adapter = seg_adapter
        self.det_adapter = det_adapter
        self.cls_adapter = cls_adapter
        self.segdet_expert = segdet_expert
        self.detcls_expert = detcls_expert
        self.clsseg_expert = clsseg_expert
        self.big_expert = big_expert
        self.segmentation_head = segmentation_head
        self.detection_head = detection_head
        self.classification_head = classification_head
        self.classification_feature_adapter = classification_feature_adapter
        self.yolo_feature_adapter = YOLOFeatureAdapter()

    def forward(self, image, task_id=None, return_all=True):
        """
        image: (B, 3, 512, 512)
        task_id: (optional) if provided, only return that task's output
        return_all: if True, return all three outputs
        """
        features = self.encoder(image)  # (B, 256, 64, 64)
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, 4096, 256)

        seg_feat = self.seg_adapter(features_flat)
        det_feat = self.det_adapter(features_flat)
        cls_feat = self.cls_adapter(features_flat)

        segdet_out = self.segdet_expert(seg_feat, det_feat)
        detcls_out = self.detcls_expert(det_feat, cls_feat)
        clsseg_out = self.clsseg_expert(cls_feat, seg_feat)

        fused = self.big_expert(segdet_out, detcls_out, clsseg_out)

        # Heads expect (B, 4096, C) or similar
        B, N, C = fused.shape
        H = W = int(N ** 0.5)
        fused_spatial = fused.permute(0, 2, 1).reshape(B, C, H, W)

        outputs = {}
        if return_all or task_id is None:
            seg_out = self.segmentation_head(fused_spatial)
            seg_out = F.interpolate(seg_out, size=(1024, 1024), mode='bilinear', align_corners=False)
            yolo_feats = self.yolo_feature_adapter(fused_spatial)
            det_out = self.detection_head(yolo_feats)
            cls_img = self.classification_feature_adapter(fused_spatial)
            cls_out = self.classification_head(cls_img)
            outputs = {
                'segmentation': seg_out,
                'detection': det_out,
                'classification': cls_out
            }
            return outputs
        else:
            if task_id == 'segmentation':
                seg_out = self.segmentation_head(fused_spatial)
                seg_out = F.interpolate(seg_out, size=(1024, 1024), mode='bilinear', align_corners=False)
                return {'segmentation': seg_out}
            elif task_id == 'detection':
                yolo_feats = self.yolo_feature_adapter(fused_spatial)
                det_out = self.detection_head(yolo_feats)
                return {'detection': det_out}
            elif task_id == 'classification':
                cls_img = self.classification_feature_adapter(fused_spatial)
                # Debugging: print shape, dtype, min, max, NaN/Inf check
                print('cls_img shape:', cls_img.shape, 'dtype:', cls_img.dtype)
                print('cls_img min:', cls_img.min().item(), 'max:', cls_img.max().item(), 'mean:', cls_img.mean().item())
                if torch.isnan(cls_img).any() or torch.isinf(cls_img).any():
                    print('NaN or Inf detected in classification input!')
                    raise ValueError('NaN or Inf detected in classification input!')
                cls_img = cls_img.to(self.classification_head.fc.weight.device)
                cls_out = self.classification_head(cls_img)
                return {'classification': cls_out}
            else:
                raise ValueError(f"Unknown task_id: {task_id}")

# Example usage:
# model = MultiTaskModel(encoder, seg_adapter, det_adapter, cls_adapter,
#                       segdet_expert, detcls_expert, clsseg_expert,
#                       big_expert,
#                       segmentation_head, detection_head, classification_head)
# out = model(image, task_id='segmentation') 