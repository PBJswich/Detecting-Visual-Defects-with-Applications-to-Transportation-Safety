import torch
import torch.nn as nn

class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 -> 128x128
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x128 -> 256x256
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256x256 -> 512x512
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 512x512 -> 1024x1024
            nn.Conv2d(16, out_channels, 1)
        )
    
    def forward(self, x):
        return self.up(x)

class SimpleDetectionHead(nn.Module):
    """
    Simple detection head that outputs bounding box predictions
    """
    def __init__(self, in_channels=256, num_classes=1, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Detection layers
        self.detect_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
        )
        
        # Output layers: 4 coordinates + 1 confidence + num_classes
        self.bbox_conv = nn.Conv2d(32, num_anchors * 4, 1)  # x, y, w, h
        self.conf_conv = nn.Conv2d(32, num_anchors * 1, 1)  # confidence
        self.cls_conv = nn.Conv2d(32, num_anchors * num_classes, 1)  # class scores
    
    def forward(self, x):
        # x: (B, C, H, W)
        features = self.detect_conv(x)
        
        # Generate outputs
        bbox_pred = self.bbox_conv(features)  # (B, num_anchors*4, H, W)
        conf_pred = self.conf_conv(features)  # (B, num_anchors*1, H, W)
        cls_pred = self.cls_conv(features)    # (B, num_anchors*num_classes, H, W)
        
        # Reshape to (B, H*W*num_anchors, 4+1+num_classes)
        B, _, H, W = bbox_pred.shape
        bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        conf_pred = conf_pred.view(B, self.num_anchors, 1, H, W).permute(0, 3, 4, 1, 2).contiguous()
        cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W).permute(0, 3, 4, 1, 2).contiguous()
        
        # Flatten spatial dimensions
        bbox_pred = bbox_pred.view(B, H*W*self.num_anchors, 4)
        conf_pred = conf_pred.view(B, H*W*self.num_anchors, 1)
        cls_pred = cls_pred.view(B, H*W*self.num_anchors, self.num_classes)
        
        # Combine all predictions
        detections = torch.cat([bbox_pred, conf_pred, cls_pred], dim=2)
        
        return detections

class SegDetHead(nn.Module):
    """
    Dual-task head for Segmentation + Detection.
    Uses simple segmentation head and custom detection head.
    """
    def __init__(self, segmentation_head, detection_head):
        super().__init__()
        self.segmentation_head = segmentation_head
        self.detection_head = detection_head

    def forward(self, x):
        # x: (B, 4096, 256) - flattened features
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Should be 64 for 4096
        x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 256, 64, 64)
        
        # Use segmentation head with original features
        seg_out = self.segmentation_head(x_spatial)
        
        # Use detection head with original features
        det_out = self.detection_head(x_spatial)
        
        return {'segmentation': seg_out, 'detection': det_out}

class DetClsHead(nn.Module):
    """
    Dual-task head for Detection + Classification.
    Uses custom detection head and simple classifier.
    """
    def __init__(self, detection_head, resnet_head):
        super().__init__()
        self.detection_head = detection_head
        
        # Add a simple classifier that can handle 1D input
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)  # Assuming binary classification
        )

    def forward(self, x):
        # x: (B, 4096, 256) - flattened features
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Should be 64 for 4096
        x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 256, 64, 64)
        
        # Use detection head
        det_out = self.detection_head(x_spatial)
        
        # For classification, global average pool then pass to our custom classifier
        cls_in = torch.mean(x_spatial, dim=[2, 3])  # (B, 256)
        cls_out = self.classifier(cls_in)  # (B, 1)
        
        return {'detection': det_out, 'classification': cls_out}

class ClsSegHead(nn.Module):
    """
    Dual-task head for Classification + Segmentation.
    Uses simple classifier and segmentation head.
    """
    def __init__(self, resnet_head, segmentation_head):
        super().__init__()
        self.segmentation_head = segmentation_head
        # Add a simple classifier that can handle 1D input
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)  # Assuming binary classification
        )

    def forward(self, x):
        # x: (B, 4096, 256) - flattened features
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Should be 64 for 4096
        x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 256, 64, 64)
        
        # For classification, global average pool then pass to our custom classifier
        cls_in = torch.mean(x_spatial, dim=[2, 3])  # (B, 256)
        cls_out = self.classifier(cls_in)  # (B, 1)
        
        # Use segmentation head with spatial features
        seg_out = self.segmentation_head(x_spatial)  # (B, 1, 512, 512)
        
        return {'classification': cls_out, 'segmentation': seg_out}

# Example usage:
# segmentation_head = SimpleSegmentationHead(256)
# detection_head = SimpleDetectionHead(256)
# segdet_head = SegDetHead(segmentation_head, detection_head)
# detcls_head = DetClsHead(detection_head, None)
# clsseg_head = ClsSegHead(None, segmentation_head)
# outputs = segdet_head(fused_features)