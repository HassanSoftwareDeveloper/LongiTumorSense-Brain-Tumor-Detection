# import os
# import logging
# from typing import List, Dict, Tuple, Optional, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import efficientnet_b4, resnet50, densenet121
# import joblib

# logger = logging.getLogger(__name__)


# class EnhancedAttentionBlock(nn.Module):
#     """
#     Attention block with residual connections for 3D U-Net
#     """
#     def __init__(self, F_g: int, F_l: int, F_int: int):
#         super().__init__()
#         # Make absolutely sure these are ints (guards against accidental floats)
#         F_g = int(F_g)
#         F_l = int(F_l)
#         F_int = int(F_int)

#         self.W_g = nn.Sequential(
#             nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
#             nn.InstanceNorm3d(F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
#             nn.InstanceNorm3d(F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
#             nn.InstanceNorm3d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass with residual attention gating"""
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi + x  # Residual connection


# class Enhanced3DUNet(nn.Module):
#     """
#     Enhanced 3D U-Net with deep supervision, attention mechanisms, and patch-based inference support
#     """
#     def __init__(
#         self,
#         in_channels: int = 1,
#         out_channels: int = 4,
#         use_attention: bool = True,
#         dropout_rate: float = 0.3,
#         use_deep_supervision: bool = True,
#         base_channels: int = 64
#     ):
#         super().__init__()
#         self.use_attention = use_attention
#         self.use_deep_supervision = use_deep_supervision
#         self.out_channels = out_channels

#         # Encoder with residual connections
#         self.enc1 = self._make_layer(in_channels, base_channels)
#         self.enc2 = self._make_layer(base_channels, base_channels * 2)
#         self.enc3 = self._make_layer(base_channels * 2, base_channels * 4)
#         self.enc4 = self._make_layer(base_channels * 4, base_channels * 8)

#         # Bottleneck with dilated convolutions
#         self.bottleneck = nn.Sequential(
#             nn.Conv3d(base_channels * 8, base_channels * 16, 3, padding=2, dilation=2),
#             nn.InstanceNorm3d(base_channels * 16),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(base_channels * 16, base_channels * 16, 3, padding=4, dilation=4),
#             nn.InstanceNorm3d(base_channels * 16),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(base_channels * 16, base_channels * 16, 3, padding=1),
#             nn.InstanceNorm3d(base_channels * 16),
#             nn.ReLU(inplace=True)
#         )

#         # Decoder with attention gates
#         self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, stride=2)
#         self.att4 = EnhancedAttentionBlock(base_channels * 8, base_channels * 8, base_channels * 4) if use_attention else None
#         self.dec4 = self._make_layer(base_channels * 16, base_channels * 8)

#         self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
#         self.att3 = EnhancedAttentionBlock(base_channels * 4, base_channels * 4, base_channels * 2) if use_attention else None
#         self.dec3 = self._make_layer(base_channels * 8, base_channels * 4)

#         self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
#         self.att2 = EnhancedAttentionBlock(base_channels * 2, base_channels * 2, base_channels) if use_attention else None
#         self.dec2 = self._make_layer(base_channels * 4, base_channels * 2)

#         self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
#         # FIX: use integer division for channels (no floats!)
#         self.att1 = EnhancedAttentionBlock(base_channels, base_channels, base_channels // 2) if use_attention else None
#         self.dec1 = self._make_layer(base_channels * 2, base_channels)

#         # Final convolution (returns logits)
#         self.final_conv = nn.Conv3d(base_channels, out_channels, 1)

#         # Deep supervision outputs
#         if use_deep_supervision:
#             self.ds_conv1 = nn.Conv3d(base_channels * 8, out_channels, 1)
#             self.ds_conv2 = nn.Conv3d(base_channels * 4, out_channels, 1)
#             self.ds_conv3 = nn.Conv3d(base_channels * 2, out_channels, 1)

#         self.pool = nn.MaxPool3d(2)
#         self.dropout = nn.Dropout3d(dropout_rate)

#     def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
#         """Create a convolutional block with two layers"""
#         return nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, 3, padding=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, 3, padding=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         """
#         Forward pass with optional deep supervision
#         Returns logits for flexibility during training/inference
#         """
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))

#         # Bottleneck
#         b = self.bottleneck(self.pool(e4))
#         b = self.dropout(b)

#         # Decoder with attention and deep supervision
#         ds_outputs: List[torch.Tensor] = []

#         d4 = self.upconv4(b)
#         if self.use_attention:
#             e4_att = self.att4(d4, e4)
#             d4 = torch.cat([e4_att, d4], dim=1)
#         else:
#             d4 = torch.cat([e4, d4], dim=1)
#         d4 = self.dec4(d4)
#         if self.use_deep_supervision:
#             ds_outputs.append(self.ds_conv1(d4))

#         d3 = self.upconv3(d4)
#         if self.use_attention:
#             e3_att = self.att3(d3, e3)
#             d3 = torch.cat([e3_att, d3], dim=1)
#         else:
#             d3 = torch.cat([e3, d3], dim=1)
#         d3 = self.dec3(d3)
#         if self.use_deep_supervision:
#             ds_outputs.append(self.ds_conv2(d3))

#         d2 = self.upconv2(d3)
#         if self.use_attention:
#             e2_att = self.att2(d2, e2)
#             d2 = torch.cat([e2_att, d2], dim=1)
#         else:
#             d2 = torch.cat([e2, d2], dim=1)
#         d2 = self.dec2(d2)
#         if self.use_deep_supervision:
#             ds_outputs.append(self.ds_conv3(d2))

#         d1 = self.upconv1(d2)
#         if self.use_attention:
#             e1_att = self.att1(d1, e1)
#             d1 = torch.cat([e1_att, d1], dim=1)
#         else:
#             d1 = torch.cat([e1, d1], dim=1)
#         d1 = self.dec1(d1)

#         # Final output (logits)
#         final_output = self.final_conv(d1)

#         if self.use_deep_supervision:
#             return [final_output] + ds_outputs[::-1]  # Return in order of resolution
#         else:
#             return final_output


# class EnhancedGliomaClassifier(nn.Module):
#     """
#     Multi-task glioma classifier with clinical data integration and transfer learning support
#     """
#     def __init__(
#         self,
#         num_classes: int = 4,
#         backbone: str = 'efficientnet',
#         clinical_feature_size: Optional[int] = None,
#         dropout_rate: float = 0.4,
#         freeze_backbone: bool = False,
#         pretrained: bool = True
#     ):
#         super().__init__()

#         self.backbone_name = backbone
#         self.clinical_feature_size = clinical_feature_size

#         # Initialize backbone
#         if backbone == 'efficientnet':
#             self.backbone = efficientnet_b4(pretrained=pretrained)
#             in_features = 1792
#             self.backbone.classifier = nn.Identity()
#         elif backbone == 'resnet':
#             self.backbone = resnet50(pretrained=pretrained)
#             in_features = 2048
#             self.backbone.fc = nn.Identity()
#         elif backbone == 'densenet':
#             self.backbone = densenet121(pretrained=pretrained)
#             in_features = 1024
#             self.backbone.classifier = nn.Identity()
#         else:
#             raise ValueError(f"Unsupported backbone: {backbone}")

#         # Freeze backbone if requested
#         if freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         # Adapt first layer for grayscale input
#         self._adapt_first_conv(backbone)

#         # Feature extraction
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(in_features, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate * 0.75),
#         )

#         # Dynamic clinical feature handling
#         self.clinical_adapter = nn.Linear(clinical_feature_size, 64) if clinical_feature_size else None
#         total_features = 512 + (64 if clinical_feature_size else 0)

#         # Multi-task heads
#         self.tumor_classifier = nn.Sequential(
#             nn.Linear(total_features, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes)
#         )

#         self.survival_predictor = nn.Sequential(
#             nn.Linear(total_features, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#         self.progression_predictor = nn.Sequential(
#             nn.Linear(total_features, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 2)
#         )

#         # Attention mechanism with normalized weights (scalar gate per sample)
#         self.attention = nn.Sequential(
#             nn.Linear(in_features, 512),
#             nn.Tanh(),
#             nn.Linear(512, 1)
#         )

#         # Temperature scaling for calibration
#         self.temperature = nn.Parameter(torch.ones(1))

#     def _adapt_first_conv(self, backbone: str):
#         """Adapt first convolution layer for grayscale input"""
#         try:
#             if backbone == 'efficientnet':
#                 original_conv = self.backbone.features[0][0]
#                 bias_flag = original_conv.bias is not None
#                 new_conv = nn.Conv2d(
#                     1,
#                     original_conv.out_channels,
#                     kernel_size=original_conv.kernel_size,
#                     stride=original_conv.stride,
#                     padding=original_conv.padding,
#                     bias=bias_flag,
#                 )
#                 with torch.no_grad():
#                     new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
#                 self.backbone.features[0][0] = new_conv

#             elif backbone == 'resnet':
#                 original_conv = self.backbone.conv1
#                 bias_flag = original_conv.bias is not None
#                 new_conv = nn.Conv2d(
#                     1,
#                     original_conv.out_channels,
#                     kernel_size=original_conv.kernel_size,
#                     stride=original_conv.stride,
#                     padding=original_conv.padding,
#                     bias=bias_flag,
#                 )
#                 with torch.no_grad():
#                     new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
#                 self.backbone.conv1 = new_conv

#             elif backbone == 'densenet':
#                 original_conv = self.backbone.features.conv0
#                 bias_flag = original_conv.bias is not None
#                 new_conv = nn.Conv2d(
#                     1,
#                     original_conv.out_channels,
#                     kernel_size=original_conv.kernel_size,
#                     stride=original_conv.stride,
#                     padding=original_conv.padding,
#                     bias=bias_flag,
#                 )
#                 with torch.no_grad():
#                     new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
#                 self.backbone.features.conv0 = new_conv

#         except Exception as e:
#             logger.warning(f"Could not adapt first conv layer: {e}")

#     def forward(self, x: torch.Tensor, clinical_data: Optional[torch.Tensor] = None) -> Tuple:
#         """
#         Forward pass with optional clinical data integration
#         Returns: (tumor_pred, survival_pred, progression_pred, attention_weights)
#         """
#         try:
#             # Extract imaging features
#             if hasattr(self.backbone, 'features'):
#                 features = self.backbone.features(x)
#                 features = F.adaptive_avg_pool2d(features, (1, 1))
#             else:
#                 features = self.backbone(x)

#             features = features.view(features.size(0), -1)

#             # Scalar attention per sample (sigmoid gate)
#             att_logits = self.attention(features)            # (B, 1)
#             att_weights = torch.sigmoid(att_logits)          # (B, 1) in [0,1]
#             attended_features = features * att_weights       # broadcast

#             # Extract deep features
#             imaging_features = self.feature_extractor(attended_features)

#             # Handle clinical data
#             if clinical_data is not None and self.clinical_adapter:
#                 if clinical_data.size(1) != self.clinical_feature_size:
#                     logger.warning(
#                         f"Clinical data size mismatch: expected {self.clinical_feature_size}, got {clinical_data.size(1)}"
#                     )
#                     # Adaptive resizing
#                     if clinical_data.size(1) > self.clinical_feature_size:
#                         clinical_data = clinical_data[:, : self.clinical_feature_size]
#                     else:
#                         padding = torch.zeros(
#                             clinical_data.size(0),
#                             self.clinical_feature_size - clinical_data.size(1),
#                             device=clinical_data.device,
#                             dtype=clinical_data.dtype,
#                         )
#                         clinical_data = torch.cat([clinical_data, padding], dim=1)

#                 clinical_features = self.clinical_adapter(clinical_data)
#                 combined_features = torch.cat([imaging_features, clinical_features], dim=1)
#             else:
#                 combined_features = imaging_features

#             # Multi-task predictions with temperature scaling
#             tumor_logits = self.tumor_classifier(combined_features)
#             tumor_pred = F.softmax(tumor_logits / self.temperature, dim=1)

#             survival_pred = self.survival_predictor(combined_features)
#             progression_pred = self.progression_predictor(combined_features)

#             return tumor_pred, survival_pred, progression_pred, att_weights

#         except Exception as e:
#             logger.error(f"Error in classifier forward pass: {e}")
#             # Return fallback predictions
#             batch_size = x.size(0)
#             device = x.device

#             fallback_tumor = torch.ones(batch_size, 4, device=device) / 4
#             fallback_survival = torch.ones(batch_size, 1, device=device) * 12.0
#             fallback_progression = torch.ones(batch_size, 2, device=device) / 2
#             fallback_attention = torch.ones(batch_size, 1, device=device)

#             return fallback_tumor, fallback_survival, fallback_progression, fallback_attention


# def load_model_weights(model, checkpoint_path, device='cpu', strict=True):
#     """
#     Load model weights from a checkpoint file with error handling
#     """
#     try:
#         if not os.path.exists(checkpoint_path):
#             logger.warning(f"Checkpoint not found: {checkpoint_path}")
#             return model

#         checkpoint = torch.load(checkpoint_path, map_location=device)

#         if 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint

#         # Remove 'module.' prefix if present (from DataParallel)
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 k = k[7:]  # remove 'module.' prefix
#             new_state_dict[k] = v

#         model.load_state_dict(new_state_dict, strict=strict)
#         logger.info(f"Successfully loaded weights from {checkpoint_path}")

#     except Exception as e:
#         logger.error(f"Error loading weights from {checkpoint_path}: {e}")
#         # Initialize with default weights if available
#         if hasattr(model, 'reset_parameters'):
#             model.reset_parameters()
#             logger.info("Initialized model with default parameters")

#     return model


# # Model loading functions
# def load_segmentation_model(checkpoint_path=None, device='cpu'):
#     """Load the segmentation model"""
#     model = Enhanced3DUNet(
#         in_channels=1,
#         out_channels=4,
#         use_attention=True,
#         dropout_rate=0.3,
#         use_deep_supervision=True,
#         base_channels=64
#     )
    
#     if checkpoint_path:
#         model = load_model_weights(model, checkpoint_path, device)
    
#     model.to(device)
#     model.eval()
#     return model


# def load_classification_model(checkpoint_path=None, device='cpu'):
#     """Load the classification model"""
#     model = EnhancedGliomaClassifier(
#         num_classes=4,
#         backbone='efficientnet',
#         clinical_feature_size=10,  # Adjust based on your clinical data
#         dropout_rate=0.4,
#         freeze_backbone=False,
#         pretrained=True
#     )
    
#     if checkpoint_path:
#         model = load_model_weights(model, checkpoint_path, device)
    
#     model.to(device)
#     model.eval()
#     return model


# # Example usage
# if __name__ == "__main__":
#     # Test the models
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Test segmentation model
#     seg_model = load_segmentation_model(device=device)
#     dummy_input = torch.randn(1, 1, 128, 128, 128)  # Batch, Channels, Depth, Height, Width
#     with torch.no_grad():
#         output = seg_model(dummy_input)
#     print(f"Segmentation output shape: {[o.shape for o in output] if isinstance(output, list) else output.shape}")
    
#     # Test classification model
#     cls_model = load_classification_model(device=device)
#     dummy_input = torch.randn(1, 1, 224, 224)  # Batch, Channels, Height, Width
#     clinical_data = torch.randn(1, 10)  # Adjust based on your clinical data
#     with torch.no_grad():
#         tumor_pred, survival_pred, progression_pred, att_weights = cls_model(dummy_input, clinical_data)
#     print(f"Tumor prediction shape: {tumor_pred.shape}")
#     print(f"Survival prediction shape: {survival_pred.shape}")
#     print(f"Progression prediction shape: {progression_pred.shape}")


























import os
import logging
from typing import List, Dict, Tuple, Optional, Union
import requests
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, resnet50, densenet121, vgg16, mobilenet_v3_large
from torchvision import transforms
import timm  # For additional pretrained models
import cv2
from PIL import Image
import joblib

logger = logging.getLogger(__name__)

class BrainTumorDetector(nn.Module):
    """
    Ensemble Brain Tumor Detector using multiple pretrained models
    Combines EfficientNet, ResNet, DenseNet, and specialized medical models
    """
    def __init__(
        self,
        num_classes: int = 4,  # 0: Normal, 1: Glioma, 2: Meningioma, 3: Pituitary
        clinical_feature_size: Optional[int] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.clinical_feature_size = clinical_feature_size
        self.device = device
        
        # Initialize multiple pretrained backbones
        self.models = nn.ModuleDict()
        self.feature_extractors = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        
        # EfficientNet-B4 (Best for medical images)
        self._init_efficientnet()
        
        # ResNet-50 (Robust feature extraction)
        self._init_resnet()
        
        # DenseNet-121 (Good for small datasets)
        self._init_densenet()
        
        # Medical-specific model (Using Vision Transformer)
        self._init_medical_vit()
        
        # Ensemble fusion layers
        total_features = 1792 + 2048 + 1024 + 768  # From all models
        if clinical_feature_size:
            total_features += 64
            self.clinical_adapter = nn.Linear(clinical_feature_size, 64)
        
        # Ensemble classifier
        self.ensemble_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # Attention mechanism for model weighting
        self.attention_weights = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # 4 models
            nn.Softmax(dim=1)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def _init_efficientnet(self):
        """Initialize EfficientNet-B4 for medical imaging"""
        model = efficientnet_b4(pretrained=True)
        
        # Adapt for grayscale input
        original_conv = model.features[0][0]
        new_conv = nn.Conv2d(1, original_conv.out_channels,
                           kernel_size=original_conv.kernel_size,
                           stride=original_conv.stride,
                           padding=original_conv.padding,
                           bias=original_conv.bias is not None)
        
        # Initialize with mean of RGB weights
        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        
        model.features[0][0] = new_conv
        model.classifier = nn.Identity()
        
        self.models['efficientnet'] = model
        self.feature_extractors['efficientnet'] = nn.Sequential(
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def _init_resnet(self):
        """Initialize ResNet-50"""
        model = resnet50(pretrained=True)
        
        # Adapt for grayscale
        original_conv = model.conv1
        new_conv = nn.Conv2d(1, original_conv.out_channels,
                           kernel_size=original_conv.kernel_size,
                           stride=original_conv.stride,
                           padding=original_conv.padding,
                           bias=original_conv.bias is not None)
        
        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        
        model.conv1 = new_conv
        model.fc = nn.Identity()
        
        self.models['resnet'] = model
        self.feature_extractors['resnet'] = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def _init_densenet(self):
        """Initialize DenseNet-121"""
        model = densenet121(pretrained=True)
        
        # Adapt for grayscale
        original_conv = model.features.conv0
        new_conv = nn.Conv2d(1, original_conv.out_channels,
                           kernel_size=original_conv.kernel_size,
                           stride=original_conv.stride,
                           padding=original_conv.padding,
                           bias=original_conv.bias is not None)
        
        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        
        model.features.conv0 = new_conv
        model.classifier = nn.Identity()
        
        self.models['densenet'] = model
        self.feature_extractors['densenet'] = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def _init_medical_vit(self):
        """Initialize Vision Transformer for medical imaging"""
        try:
            # Using TIMM for Vision Transformer
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            
            # Adapt for grayscale by modifying patch embedding
            original_proj = model.patch_embed.proj
            new_proj = nn.Conv2d(1, original_proj.out_channels,
                               kernel_size=original_proj.kernel_size,
                               stride=original_proj.stride,
                               padding=original_proj.padding)
            
            with torch.no_grad():
                new_proj.weight = nn.Parameter(original_proj.weight.mean(dim=1, keepdim=True))
            
            model.patch_embed.proj = new_proj
            
            self.models['vit'] = model
            self.feature_extractors['vit'] = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
        except Exception as e:
            logger.warning(f"Could not load ViT model: {e}. Using MobileNet instead.")
            # Fallback to MobileNet
            model = mobilenet_v3_large(pretrained=True)
            
            # Adapt for grayscale
            original_conv = model.features[0][0]
            new_conv = nn.Conv2d(1, original_conv.out_channels,
                               kernel_size=original_conv.kernel_size,
                               stride=original_conv.stride,
                               padding=original_conv.padding,
                               bias=original_conv.bias is not None)
            
            with torch.no_grad():
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
            
            model.features[0][0] = new_conv
            model.classifier = nn.Identity()
            
            self.models['mobilenet'] = model
            self.feature_extractors['mobilenet'] = nn.Sequential(
                nn.Linear(960, 512),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
    
    def forward(self, x: torch.Tensor, clinical_data: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through ensemble of models
        """
        batch_size = x.size(0)
        device = x.device
        
        # Extract features from all models
        all_features = []
        individual_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get raw features
                if 'vit' in model_name:
                    features = model(x)
                else:
                    features = model(x)
                    if features.dim() > 2:
                        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                
                # Process through feature extractor
                processed_features = self.feature_extractors[model_name](features)
                all_features.append(processed_features)
                
                # Individual model prediction for interpretability
                individual_pred = F.softmax(
                    nn.Linear(processed_features.size(1), self.num_classes).to(device)(processed_features),
                    dim=1
                )
                individual_predictions[model_name] = individual_pred
                
            except Exception as e:
                logger.warning(f"Error in {model_name}: {e}")
                # Use zero features as fallback
                fallback_features = torch.zeros(batch_size, 512, device=device)
                all_features.append(fallback_features)
        
        # Concatenate all features
        combined_features = torch.cat(all_features, dim=1)
        
        # Add clinical data if available
        if clinical_data is not None and hasattr(self, 'clinical_adapter'):
            clinical_features = self.clinical_adapter(clinical_data)
            combined_features = torch.cat([combined_features, clinical_features], dim=1)
        
        # Ensemble prediction
        ensemble_logits = self.ensemble_classifier(combined_features)
        ensemble_probs = F.softmax(ensemble_logits / self.temperature, dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention_weights(combined_features)
        
        # Confidence estimation
        confidence = torch.max(ensemble_probs, dim=1)[0]
        
        # Uncertainty estimation (entropy-based)
        entropy = -torch.sum(ensemble_probs * torch.log(ensemble_probs + 1e-8), dim=1)
        uncertainty = entropy / np.log(self.num_classes)
        
        return {
            'predictions': ensemble_probs,
            'logits': ensemble_logits,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'individual_predictions': individual_predictions,
            'features': combined_features
        }

class BrainTumorSegmenter(nn.Module):
    """
    Segmentation model using pretrained encoder with U-Net decoder
    """
    def __init__(self, encoder_name='efficientnet-b4', num_classes=4):
        super().__init__()
        
        try:
            import segmentation_models_pytorch as smp
            
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=1,
                classes=num_classes,
                activation=None  # We'll apply softmax separately
            )
            
        except ImportError:
            logger.warning("segmentation_models_pytorch not available. Using basic U-Net.")
            self.model = self._create_basic_unet(num_classes)
    
    def _create_basic_unet(self, num_classes):
        """Create a basic U-Net if SMP is not available"""
        class BasicUNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                
                # Encoder (using pretrained EfficientNet as backbone)
                backbone = efficientnet_b4(pretrained=True)
                
                # Adapt for grayscale
                original_conv = backbone.features[0][0]
                new_conv = nn.Conv2d(1, original_conv.out_channels,
                                   kernel_size=original_conv.kernel_size,
                                   stride=original_conv.stride,
                                   padding=original_conv.padding,
                                   bias=original_conv.bias is not None)
                
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
                
                backbone.features[0][0] = new_conv
                self.encoder = backbone.features
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(1792, 512, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1)
                )
            
            def forward(self, x):
                features = self.encoder(x)
                output = self.decoder(features)
                return F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return BasicUNet(num_classes)
    
    def forward(self, x):
        return self.model(x)

def download_pretrained_weights(model_name: str, save_path: str) -> bool:
    """
    Download pretrained weights for brain tumor detection
    """
    # URLs for pretrained medical imaging models (you would need to find actual URLs)
    model_urls = {
        'brain_tumor_detector': 'https://github.com/your-repo/brain_tumor_weights.pth',
        'medical_segmenter': 'https://github.com/your-repo/segmentation_weights.pth'
    }
    
    if model_name not in model_urls:
        logger.warning(f"No pretrained weights available for {model_name}")
        return False
    
    try:
        url = model_urls[model_name]
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded pretrained weights for {model_name} to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download weights for {model_name}: {e}")
        return False

def load_pretrained_detector(device='cpu', checkpoint_path=None):
    """Load pretrained brain tumor detector"""
    model = BrainTumorDetector(
        num_classes=4,
        clinical_feature_size=10,
        device=device
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Using base pretrained models.")
    
    model.to(device)
    model.eval()
    return model

def load_pretrained_segmenter(device='cpu', checkpoint_path=None, encoder_name='efficientnet-b4'):
    """Load pretrained brain tumor segmenter"""
    model = BrainTumorSegmenter(encoder_name=encoder_name, num_classes=4)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained segmentation weights from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load segmentation checkpoint: {e}. Using base model.")
    
    model.to(device)
    model.eval()
    return model

def preprocess_for_model(image: np.ndarray, target_size=(224, 224)) -> torch.Tensor:
    """Preprocess image for model input"""
    if len(image.shape) == 3:
        # Take middle slice if 3D
        image = image[:, :, image.shape[2] // 2]
    
    # Normalize
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Resize
    image = cv2.resize(image.astype(np.float32), target_size)
    
    # Convert to tensor
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    return tensor

class MedicalImagePreprocessor:
    """Advanced preprocessing for medical images"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet stats adapted for grayscale
        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Handle different input formats
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                image = image.squeeze(2)
            elif image.shape[2] > 1:
                # Take middle slice for 3D volumes
                image = image[:, :, image.shape[2] // 2]
        
        # Normalize to 0-255 range
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply transforms
        tensor = self.transform(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension

# Initialize global preprocessor
preprocessor = MedicalImagePreprocessor()

def get_ensemble_models(device='cpu'):
    """Get ensemble of pretrained models for brain tumor detection"""
    models = {}
    
    try:
        # Main detector
        models['detector'] = load_pretrained_detector(device)
        
        # Segmenter
        models['segmenter'] = load_pretrained_segmenter(device)
        
        logger.info(f"Loaded {len(models)} models for ensemble prediction")
        
    except Exception as e:
        logger.error(f"Error loading ensemble models: {e}")
        # Return dummy models as fallback
        models['detector'] = BrainTumorDetector(device=device)
        models['segmenter'] = BrainTumorSegmenter()
    
    return models

# Model download and setup
def setup_models(models_dir: str = './models'):
    """Setup and download pretrained models if needed"""
    os.makedirs(models_dir, exist_ok=True)
    
    model_files = {
        'brain_tumor_detector.pth': 'brain_tumor_detector',
        'brain_segmenter.pth': 'medical_segmenter'
    }
    
    for filename, model_name in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            logger.info(f"Downloading {model_name}...")
            download_pretrained_weights(model_name, filepath)
    
    logger.info("Model setup complete")

# Example usage
if __name__ == "__main__":
    # Setup models
    setup_models()
    
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test detector
    detector = load_pretrained_detector(device)
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    
    with torch.no_grad():
        results = detector(dummy_input)
    
    print(f"Detector predictions shape: {results['predictions'].shape}")
    print(f"Confidence: {results['confidence'].mean().item():.3f}")
    print(f"Uncertainty: {results['uncertainty'].mean().item():.3f}")
    
    # Test segmenter
    segmenter = load_pretrained_segmenter(device)
    
    with torch.no_grad():
        seg_output = segmenter(dummy_input)
    
    print(f"Segmentation output shape: {seg_output.shape}")
    
    logger.info("Model testing complete")