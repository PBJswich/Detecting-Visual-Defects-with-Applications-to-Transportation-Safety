import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    MultiTaskModel wires together the encoder, adapters, expert modules, and dual-task heads.
    Routing is determined by task_id ('segmentation', 'detection', 'classification').
    """
    def __init__(self, encoder,
                 seg_adapter, det_adapter, cls_adapter,
                 segdet_expert, detcls_expert, clsseg_expert,
                 segdet_head, detcls_head, clsseg_head):
        super().__init__()
        self.encoder = encoder
        self.seg_adapter = seg_adapter
        self.det_adapter = det_adapter
        self.cls_adapter = cls_adapter
        self.segdet_expert = segdet_expert
        self.detcls_expert = detcls_expert
        self.clsseg_expert = clsseg_expert
        self.segdet_head = segdet_head
        self.detcls_head = detcls_head
        self.clsseg_head = clsseg_head

    def forward(self, image, task_id, return_all=False):
        """
        image: (B, 3, 512, 512)
        task_id: 'segmentation', 'detection', or 'classification'
        return_all: if True, return both outputs from the dual-task head
        """
        features = self.encoder(image)  # (B, 256, 64, 64)
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, 4096, 256)
        outputs = {}
        if task_id == 'segmentation':
            seg_feat = self.seg_adapter(features_flat)
            det_feat = self.det_adapter(features_flat)
            fused = self.segdet_expert(seg_feat, det_feat)
            head_out = self.segdet_head(fused)
            # Optionally also get ClsSeg head for auxiliary output
            cls_feat = self.cls_adapter(features_flat)
            fused_aux = self.clsseg_expert(cls_feat, seg_feat)
            aux_out = self.clsseg_head(fused_aux)
            outputs['segmentation'] = head_out['segmentation']
            if return_all:
                outputs['detection'] = head_out['detection']
                outputs['classification'] = aux_out['classification']
        elif task_id == 'detection':
            seg_feat = self.seg_adapter(features_flat)
            det_feat = self.det_adapter(features_flat)
            fused = self.segdet_expert(seg_feat, det_feat)
            head_out = self.segdet_head(fused)
            # Optionally also get DetCls head for auxiliary output
            cls_feat = self.cls_adapter(features_flat)
            fused_aux = self.detcls_expert(det_feat, cls_feat)
            aux_out = self.detcls_head(fused_aux)
            outputs['detection'] = head_out['detection']
            if return_all:
                outputs['segmentation'] = head_out['segmentation']
                outputs['classification'] = aux_out['classification']
        elif task_id == 'classification':
            det_feat = self.det_adapter(features_flat)
            cls_feat = self.cls_adapter(features_flat)
            fused = self.detcls_expert(det_feat, cls_feat)
            head_out = self.detcls_head(fused)
            # Optionally also get ClsSeg head for auxiliary output
            seg_feat = self.seg_adapter(features_flat)
            fused_aux = self.clsseg_expert(cls_feat, seg_feat)
            aux_out = self.clsseg_head(fused_aux)
            outputs['classification'] = head_out['classification']
            if return_all:
                outputs['detection'] = head_out['detection']
                outputs['segmentation'] = aux_out['segmentation']
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
        return outputs

# Example usage:
# model = MultiTaskModel(encoder, seg_adapter, det_adapter, cls_adapter,
#                       segdet_expert, detcls_expert, clsseg_expert,
#                       segdet_head, detcls_head, clsseg_head)
# out = model(image, task_id='segmentation') 