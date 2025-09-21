
# model.py
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
try:
    import timm  # For additional pretrained models
except ImportError:
    timm = None
import cv2
from PIL import Image
import joblib
from datetime import datetime, timedelta

# Additional imports for advanced analysis
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    pydicom = None

from skimage import exposure, measure, morphology, filters
from sklearn.cluster import KMeans
from scipy import ndimage
from typing import Tuple, List

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
        
        # Medical-specific model (Using Vision Transformer or MobileNet fallback)
        self._init_medical_vit()
        
        # Calculate total features
        total_features = 1792 + 2048 + 1024  # From EfficientNet + ResNet + DenseNet
        
        # Add VIT/MobileNet features
        if 'vit' in self.models:
            total_features += 768
        elif 'mobilenet' in self.models:
            total_features += 960
        
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
        try:
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
        except Exception as e:
            logger.warning(f"Could not load EfficientNet: {e}. Using ResNet instead.")
            self._init_resnet_fallback()
    
    def _init_resnet(self):
        """Initialize ResNet-50"""
        try:
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
        except Exception as e:
            logger.warning(f"Could not load ResNet: {e}")
    
    def _init_densenet(self):
        """Initialize DenseNet-121"""
        try:
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
        except Exception as e:
            logger.warning(f"Could not load DenseNet: {e}")
    
    def _init_medical_vit(self):
        """Initialize Vision Transformer for medical imaging"""
        try:
            if timm is not None:
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
                logger.info("Successfully loaded ViT model")
            else:
                raise ImportError("TIMM not available")
                
        except Exception as e:
            logger.warning(f"Could not load ViT model: {e}. Using MobileNet instead.")
            # Fallback to MobileNet
            try:
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
                logger.info("Using MobileNet as fallback")
            except Exception as e2:
                logger.error(f"Could not load MobileNet fallback: {e2}")
    
    def _init_resnet_fallback(self):
        """Fallback ResNet initialization for EfficientNet failure"""
        try:
            model = resnet50(pretrained=True)
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
            
            self.models['resnet_fallback'] = model
            self.feature_extractors['resnet_fallback'] = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        except Exception as e:
            logger.error(f"Fallback ResNet also failed: {e}")
    
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
                with torch.no_grad():
                    temp_classifier = nn.Linear(processed_features.size(1), self.num_classes).to(device)
                    individual_pred = F.softmax(temp_classifier(processed_features), dim=1)
                    individual_predictions[model_name] = individual_pred
                
            except Exception as e:
                logger.warning(f"Error in {model_name}: {e}")
                # Use zero features as fallback
                fallback_features = torch.zeros(batch_size, 512, device=device)
                all_features.append(fallback_features)
        
        # Ensure we have at least some features
        if not all_features:
            # Create dummy features if all models failed
            all_features = [torch.zeros(batch_size, 512, device=device) for _ in range(4)]
        
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
            logger.info(f"Loaded segmentation model with {encoder_name}")
            
        except ImportError:
            logger.warning("segmentation_models_pytorch not available. Using basic U-Net.")
            self.model = self._create_basic_unet(num_classes)
    
    def _create_basic_unet(self, num_classes):
        """Create a basic U-Net if SMP is not available"""
        class BasicUNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                
                # Encoder (using pretrained EfficientNet as backbone)
                try:
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
                except Exception as e:
                    logger.warning(f"Could not create EfficientNet-based UNet: {e}")
                    # Fallback to simple CNN
                    self.encoder = nn.Sequential(
                        nn.Conv2d(1, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    
                    self.decoder = nn.Sequential(
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

# Enhanced 3D U-Net Model
class Enhanced3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(Enhanced3DUNet, self).__init__()
        self.encoder1 = self._make_encoder(in_channels, 64)
        self.encoder2 = self._make_encoder(64, 128)
        self.encoder3 = self._make_encoder(128, 256)
        self.encoder4 = self._make_encoder(256, 512)
     
        self.bottleneck = self._make_encoder(512, 1024)
     
        self.decoder4 = self._make_decoder(1024, 512)
        self.decoder3 = self._make_decoder(512, 256)
        self.decoder2 = self._make_decoder(256, 128)
        self.decoder1 = self._make_decoder(128, 64)
     
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        try:
            self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        except:
            # Fallback if MultiheadAttention is not available
            self.attention = None
     
    def _make_encoder(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def _make_decoder(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
     
        b = self.bottleneck(F.max_pool2d(e4, 2))
     
        d4 = self.decoder4(b)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
     
        return self.final_conv(d1)

# Enhanced Glioma Classifier
class EnhancedGliomaClassifier(nn.Module):
    def __init__(self, backbone='efficientnet', num_classes=4):
        super(EnhancedGliomaClassifier, self).__init__()
     
        try:
            if backbone == 'efficientnet':
                self.backbone = efficientnet_b4(pretrained=True)
                num_features = 1792
                self.backbone.classifier = nn.Identity()
            elif backbone == 'resnet':
                self.backbone = resnet50(pretrained=True)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif backbone == 'densenet':
                self.backbone = densenet121(pretrained=True)
                num_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            else:
                num_features = 512
                self.backbone = self._create_custom_backbone()
        except Exception as e:
            logger.warning(f"Could not load {backbone} backbone: {e}. Using custom backbone.")
            num_features = 512
            self.backbone = self._create_custom_backbone()
     
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
     
        self.attention = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
     
        self.feature_enhancer = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features)
        )

    def _create_custom_backbone(self):
        """Create custom backbone when pretrained models fail"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU()
        )
     
    def forward(self, x):
        features = self.backbone(x)
        enhanced_features = self.feature_enhancer(features)
        features = features + enhanced_features
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        classification = self.classifier(attended_features)
        return classification, features, attention_weights, attended_features

# ===== MODEL LOADING AND UTILITY FUNCTIONS =====

def download_pretrained_weights(model_name: str, save_path: str) -> bool:
    """
    Download pretrained weights for brain tumor detection
    """
    # URLs for pretrained medical imaging models
    model_urls = {
        'brain_tumor_detector': 'D:\\Brain_Tumor_Detection\\model.pt',
        'medical_segmenter': 'D:\\Brain_Tumor_Detection\\model.pt'
    }
    
    if model_name not in model_urls:
        logger.warning(f"No pretrained weights available for {model_name}")
        return False
    
    try:
        file_path = model_urls[model_name]
        if os.path.exists(file_path):
            logger.info(f"Found pretrained weights for {model_name} at {file_path}")
            return True
        else:
            logger.warning(f"Pretrained weights file not found: {file_path}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to access weights for {model_name}: {e}")
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
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
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
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained segmentation weights from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load segmentation checkpoint: {e}. Using base model.")
    
    model.to(device)
    model.eval()
    return model

def load_brats_model(device='cpu'):
    """Load your specific BraTS model"""
    brats_model_path = "D:\\Brain_Tumor_Detection\\model.pt"
    
    if not os.path.exists(brats_model_path):
        logger.error(f"BraTS model file not found at: {brats_model_path}")
        return None
    
    try:
        # Try to load the model
        model = torch.load(brats_model_path, map_location=device)
        if hasattr(model, 'to'):
            model.to(device)
        if hasattr(model, 'eval'):
            model.eval()
        logger.info("Successfully loaded BraTS model")
        return model
    except Exception as e:
        logger.error(f"Failed to load BraTS model: {e}")
        return None

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
        # Load your BraTS model first
        brats_model = load_brats_model(device)
        if brats_model:
            models['brats_model'] = brats_model
            logger.info("Loaded BraTS model for ensemble prediction")
        
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

# Model setup and download
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
            logger.info(f"Checking for {model_name}...")
            download_pretrained_weights(model_name, filepath)
    
    logger.info("Model setup complete")

# ===== MEDICAL IMAGE PROCESSING FUNCTIONS =====

def normalize_volume(volume: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize volume using 'zscore' (percentile clipping + z-score), 'percentile' (0-1), or 'minmax'.
    """
    volume = volume.ast(np.float32)
    if method == 'zscore':
        # percentile clipping to remove outliers, then z-score
        p1, p99 = np.percentile(volume, [1, 99])
        clipped = np.clip(volume, p1, p99)
        mean = np.mean(clipped)
        std = np.std(clipped) + 1e-8
        return (clipped - mean) / std
    elif method == 'percentile':
        # Percentile-based normalization (0-1)
        p1, p99 = np.percentile(volume, [1, 99])
        clipped = np.clip(volume, p1, p99)
        return (clipped - p1) / (p99 - p1 + 1e-8)
    else:
        mn, mx = np.min(volume), np.max(volume)
        return (volume - mn) / (mx - mn + 1e-8)

def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resize 3D volume to target shape with linear interpolation (order=1)."""
    try:
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        resized = ndimage.zoom(volume, zoom=factors, order=1, mode='nearest')
        return resized
    except Exception as e:
        logger.exception(f"Error resizing volume: {e}")
        raise

def remove_small_components(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Remove connected components smaller than min_size. Return cleaned mask."""
    try:
        from scipy.ndimage import label
        labeled, num = label(mask)
        if num <= 0:
            return mask
        counts = np.bincount(labeled.ravel())
        small_ids = np.where(counts < min_size)[0]
        cleaned = mask.copy()
        for comp_id in small_ids:
            if comp_id == 0:
                continue
            cleaned[labeled == comp_id] = 0
        return cleaned
    except Exception as e:
        logger.exception(f"Error in remove_small_components (scipy): {e}")
        # fallback to skimage
        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)
        cleaned = mask.copy()
        for reg in regions:
            if reg.area < min_size:
                cleaned[labeled == reg.label] = 0
        return cleaned

# ===== ENHANCEMENT AND PROCESSING FUNCTIONS =====

def enhance_brain_image(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Apply CLAHE enhancement to brain image."""
    try:
        if len(image.shape) == 3:
            # For 3D volumes, enhance each slice
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                slice_2d = image[:, :, i]
                enhanced_slice = exposure.equalize_adapthist(slice_2d, clip_limit=clip_limit)
                enhanced[:, :, i] = enhanced_slice
            return enhanced
        else:
            # For 2D images
            return exposure.equalize_adapthist(image, clip_limit=clip_limit)
    except Exception as e:
        logger.warning(f"CLAHE enhancement failed: {e}. Using original image.")
        return image

def extract_brain_region(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Extract brain region using intensity thresholding."""
    try:
        # Normalize image
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Threshold to get brain mask
        brain_mask = normalized > threshold
        
        # Apply morphological operations to clean up the mask
        from skimage import morphology
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
        brain_mask = morphology.remove_small_holes(brain_mask, area_threshold=100)
        
        # Apply mask to original image
        brain_image = image.copy()
        brain_image[~brain_mask] = 0
        
        return brain_image, brain_mask
    except Exception as e:
        logger.warning(f"Brain extraction failed: {e}")
        return image, np.ones_like(image, dtype=bool)

def create_image_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
    """Create image pyramid for multi-scale processing."""
    pyramid = [image]
    for i in range(1, levels):
        # Reduce size by half for each level
        smaller = cv2.resize(pyramid[i-1], 
                            (pyramid[i-1].shape[1] // 2, pyramid[i-1].shape[0] // 2),
                            interpolation=cv2.INTER_AREA)
        pyramid.append(smaller)
    return pyramid

def register_images(fixed_image: np.ndarray, moving_image: np.ndarray) -> np.ndarray:
    """Simple image registration using phase correlation."""
    try:
        # Convert to float32
        fixed = fixed_image.astype(np.float32)
        moving = moving_image.astype(np.float32)
        
        # Calculate phase correlation
        shift, error, phase_diff = cv2.phaseCorrelate(fixed, moving)
        
        # Apply translation
        rows, cols = fixed.shape
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        registered = cv2.warpAffine(moving, M, (cols, rows))
        
        logger.info(f"Image registration applied: shift={shift}, error={error}")
        return registered
        
    except Exception as e:
        logger.warning(f"Image registration failed: {e}")
        return moving_image

# Example usage and testing
if __name__ == "__main__":
    # Setup models
    setup_models()
    
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test BraTS model loading
    brats_model = load_brats_model(device)
    if brats_model:
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        try:
            with torch.no_grad():
                brats_output = brats_model(dummy_input)
            print(f"BraTS model output: {type(brats_output)}")
        except Exception as e:
            print(f"BraTS model test failed: {e}")
    
    # Test detector
    try:
        detector = load_pretrained_detector(device)
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        
        with torch.no_grad():
            results = detector(dummy_input)
        
        print(f"Detector predictions shape: {results['predictions'].shape}")
        print(f"Confidence: {results['confidence'].mean().item():.3f}")
        print(f"Uncertainty: {results['uncertainty'].mean().item():.3f}")
    except Exception as e:
        print(f"Detector test failed: {e}")
    
    # Test segmenter
    try:
        segmenter = load_pretrained_segmenter(device)
        
        with torch.no_grad():
            seg_output = segmenter(dummy_input)
        
        print(f"Segmentation output shape: {seg_output.shape}")
    except Exception as e:
        print(f"Segmenter test failed: {e}")
    
    # Test new image processing functions
    try:
        test_image = np.random.rand(64, 64, 32)
        normalized = normalize_volume(test_image)
        resized = resize_volume(test_image, (32, 32, 16))
        enhanced = enhance_brain_image(test_image[:, :, 0])
        
        print(f"Normalization test: {normalized.shape}")
        print(f"Resize test: {resized.shape}")
        print(f"Enhancement test: {enhanced.shape}")
        
    except Exception as e:
        print(f"Image processing test failed: {e}")
    
    logger.info("Model testing complete")
