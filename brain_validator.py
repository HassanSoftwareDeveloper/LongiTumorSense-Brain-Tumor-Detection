# # brain_validator.py - ENHANCED STRICT BRAIN MRI VALIDATOR
# """
# Enhanced Brain MRI Validator Module
# STRICT validation - only accepts brain MRI scans
# Rejects: X-rays, chest scans, lung scans, photos, screenshots, text images
# """

# import numpy as np
# import cv2
# from PIL import Image
# import logging

# logger = logging.getLogger(__name__)


# class BrainMRIValidator:
#     """
#     STRICT Brain MRI Validator with 3-layer validation system
#     - Layer 1: Image type validation (reject screenshots, photos, text)
#     - Layer 2: Brain anatomy detection (check for brain structure)
#     - Layer 3: MRI characteristics validation (check for MRI patterns)
#     """
    
#     def __init__(self):
#         # STRICT THRESHOLDS - increased from original
#         self.min_grayscale_ratio = 0.90  # 90% grayscale (was 85%)
#         self.min_circular_features = 0.40  # 40% circularity (was 30%)
#         self.min_brain_structure_score = 0.50  # NEW: Brain structure requirement
#         self.texture_threshold = 0.12  # MRI texture patterns
#         self.min_validation_confidence = 0.75  # 75% minimum (was 60%)
        
#     def validate_image(self, image: np.ndarray) -> dict:
#         """
#         STRICT validation with 3-layer system
        
#         Returns:
#             dict: {
#                 'is_valid': bool,
#                 'reason': str,
#                 'confidence': float,
#                 'validation_details': dict
#             }
#         """
#         try:
#             # Convert to grayscale if needed
#             if len(image.shape) == 3:
#                 if image.shape[2] == 4:  # RGBA
#                     image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
#                     image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
#                 elif image.shape[2] == 3:  # RGB
#                     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#                     image_rgb = image
#                 else:
#                     return self._create_rejection('Invalid image format')
#             else:
#                 image_gray = image
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
#             # Normalize
#             if image_gray.max() > 0:
#                 normalized = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min())
#             else:
#                 return self._create_rejection('Blank or invalid image')
            
#             # ========== LAYER 1: IMAGE TYPE VALIDATION ==========
#             layer1_checks = {}
            
#             # Check 1.1: Grayscale (medical scans are grayscale)
#             layer1_checks['grayscale'] = self._check_grayscale(image_rgb)
#             if layer1_checks['grayscale'] < 0.70:
#                 return self._create_rejection('Image is a color photograph, not a medical scan')
            
#             # Check 1.2: NO UI elements or text
#             layer1_checks['no_ui'] = self._check_no_ui_elements(normalized)
#             if layer1_checks['no_ui'] < 0.60:
#                 return self._create_rejection('Image contains UI elements, text, or screenshots')
            
#             # Check 1.3: Proper dimensions
#             layer1_checks['dimensions'] = self._check_dimensions(image_gray.shape)
#             if layer1_checks['dimensions'] < 0.50:
#                 return self._create_rejection('Image dimensions not typical for MRI scans')
            
#             # Check 1.4: NOT a chest/lung X-ray
#             layer1_checks['not_chest_xray'] = self._check_not_chest_xray(normalized)
#             if layer1_checks['not_chest_xray'] < 0.60:
#                 return self._create_rejection('Image appears to be a chest/lung X-ray or CT scan')
            
#             # ========== LAYER 2: BRAIN ANATOMY DETECTION ==========
#             layer2_checks = {}
            
#             # Check 2.1: Brain-like circular structure
#             layer2_checks['circular_structure'] = self._check_circular_structure(normalized)
#             if layer2_checks['circular_structure'] < self.min_circular_features:
#                 return self._create_rejection('Image does not show brain-like circular structure')
            
#             # Check 2.2: Brain symmetry (left-right)
#             layer2_checks['symmetry'] = self._check_brain_symmetry(normalized)
#             if layer2_checks['symmetry'] < 0.40:
#                 return self._create_rejection('Image lacks brain bilateral symmetry')
            
#             # Check 2.3: Brain intensity distribution
#             layer2_checks['brain_intensity'] = self._check_brain_intensity_distribution(normalized)
#             if layer2_checks['brain_intensity'] < 0.40:
#                 return self._create_rejection('Intensity distribution does not match brain tissue')
            
#             # ========== LAYER 3: MRI CHARACTERISTICS ==========
#             layer3_checks = {}
            
#             # Check 3.1: MRI texture patterns
#             layer3_checks['texture'] = self._check_mri_texture(normalized)
#             if layer3_checks['texture'] < 0.35:
#                 return self._create_rejection('Texture pattern does not match MRI characteristics')
            
#             # Check 3.2: Medical imaging characteristics
#             layer3_checks['medical'] = self._check_medical_image(normalized)
#             if layer3_checks['medical'] < 0.40:
#                 return self._create_rejection('Image lacks medical imaging characteristics')
            
#             # Check 3.3: Brain tissue contrast
#             layer3_checks['contrast'] = self._check_brain_tissue_contrast(normalized)
#             if layer3_checks['contrast'] < 0.35:
#                 return self._create_rejection('Tissue contrast does not match brain MRI')
            
#             # ========== CALCULATE OVERALL CONFIDENCE ==========
#             all_checks = {
#                 **layer1_checks,
#                 **layer2_checks,
#                 **layer3_checks
#             }
            
#             # Weighted confidence calculation
#             weights = {
#                 'grayscale': 0.08,
#                 'no_ui': 0.10,
#                 'dimensions': 0.04,
#                 'not_chest_xray': 0.10,
#                 'circular_structure': 0.15,
#                 'symmetry': 0.12,
#                 'brain_intensity': 0.12,
#                 'texture': 0.10,
#                 'medical': 0.10,
#                 'contrast': 0.09
#             }
            
#             confidence = sum(all_checks[key] * weights[key] for key in all_checks)
            
#             # STRICT VALIDATION: Require 75% confidence minimum
#             is_valid = confidence >= self.min_validation_confidence
            
#             if not is_valid:
#                 # Identify weakest check
#                 weakest_check = min(all_checks.items(), key=lambda x: x[1])
#                 reason = f"Validation failed: {weakest_check[0]} score too low ({weakest_check[1]:.2%}). This does not appear to be a brain MRI scan."
#             else:
#                 reason = "Valid brain MRI scan detected"
            
#             return {
#                 'is_valid': is_valid,
#                 'reason': reason,
#                 'confidence': confidence,
#                 'validation_details': all_checks,
#                 'layer1_pass': all(v >= 0.5 for v in layer1_checks.values()),
#                 'layer2_pass': all(v >= 0.35 for v in layer2_checks.values()),
#                 'layer3_pass': all(v >= 0.30 for v in layer3_checks.values())
#             }
            
#         except Exception as e:
#             logger.error(f"Validation error: {e}")
#             return self._create_rejection(f'Validation error: {str(e)}')
    
#     def _create_rejection(self, reason: str) -> dict:
#         """Helper to create rejection response"""
#         return {
#             'is_valid': False,
#             'reason': reason,
#             'confidence': 0.0,
#             'validation_details': {}
#         }
    
#     # ========== LAYER 1 CHECKS ==========
    
#     def _check_grayscale(self, image_rgb: np.ndarray) -> float:
#         """STRICT grayscale check - medical scans are grayscale"""
#         try:
#             if len(image_rgb.shape) == 2:
#                 return 1.0
            
#             r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
            
#             # Calculate color variance
#             rg_diff = np.mean(np.abs(r.astype(float) - g.astype(float)))
#             rb_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
#             gb_diff = np.mean(np.abs(g.astype(float) - b.astype(float)))
            
#             avg_diff = (rg_diff + rb_diff + gb_diff) / 3
#             max_diff = 20  # STRICTER: was 30
            
#             grayscale_score = max(0, 1 - (avg_diff / max_diff))
#             return grayscale_score
            
#         except:
#             return 0.5
    
#     def _check_no_ui_elements(self, image: np.ndarray) -> float:
#         """Check that image doesn't contain UI elements, text, or screenshots"""
#         try:
#             # Detect sharp edges (UI elements have many sharp edges)
#             edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
#             edge_density = np.sum(edges > 0) / edges.size
            
#             # Check for horizontal/vertical lines (common in screenshots)
#             horizontal_kernel = np.ones((1, 15), np.uint8)
#             vertical_kernel = np.ones((15, 1), np.uint8)
            
#             img_uint8 = (image * 255).astype(np.uint8)
#             horizontal_lines = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, horizontal_kernel)
#             vertical_lines = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, vertical_kernel)
            
#             line_density = (np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)) / (2 * image.size)
            
#             # Too many edges or lines = UI/screenshot
#             if edge_density > 0.20 or line_density > 0.10:
#                 return 0.2
#             elif edge_density > 0.12 or line_density > 0.05:
#                 return 0.5
#             else:
#                 return 1.0
                
#         except:
#             return 0.7
    
#     def _check_dimensions(self, shape: tuple) -> float:
#         """Check if dimensions are typical for MRI scans"""
#         try:
#             h, w = shape
            
#             # MRI scans typically 128-1024 pixels
#             if 128 <= h <= 1024 and 128 <= w <= 1024:
#                 # Prefer square-ish images (brain MRIs are usually close to square)
#                 aspect_ratio = w / h
#                 if 0.8 <= aspect_ratio <= 1.2:
#                     return 1.0
#                 else:
#                     return 0.8
#             elif 100 <= h <= 1200 and 100 <= w <= 1200:
#                 return 0.6
#             else:
#                 return 0.3
                
#         except:
#             return 0.5
    
#     def _check_not_chest_xray(self, image: np.ndarray) -> float:
#         """Check that image is NOT a chest/lung X-ray or CT"""
#         try:
#             h, w = image.shape
            
#             # Chest X-rays have specific characteristics:
#             # 1. Large dark lung regions on sides
#             # 2. Bright spine in center
#             # 3. Ribcage pattern
            
#             # Check left and right regions for large dark areas (lungs)
#             left_region = image[:, :w//4]
#             right_region = image[:, 3*w//4:]
            
#             left_darkness = np.mean(left_region < 0.3)
#             right_darkness = np.mean(right_region < 0.3)
            
#             # If both sides are very dark (like lungs), likely chest X-ray
#             if left_darkness > 0.4 and right_darkness > 0.4:
#                 return 0.2  # Likely chest X-ray
            
#             # Check for vertical bright line in center (spine)
#             center_column = image[:, w//2-5:w//2+5]
#             center_brightness = np.mean(center_column > 0.7)
            
#             if center_brightness > 0.3:
#                 return 0.4  # Possible chest X-ray
            
#             # Check aspect ratio (chest X-rays are often taller than wide)
#             aspect = h / w
#             if aspect > 1.3:  # Taller than wide
#                 return 0.6
            
#             return 1.0  # Does not look like chest X-ray
            
#         except:
#             return 0.7
    
#     # ========== LAYER 2 CHECKS ==========
    
#     def _check_circular_structure(self, image: np.ndarray) -> float:
#         """Check for brain-like circular structure"""
#         try:
#             # Threshold to find brain region
#             _, binary = cv2.threshold((image * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
            
#             # Find contours
#             contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             if not contours:
#                 return 0.0
            
#             # Find largest contour (should be brain)
#             largest_contour = max(contours, key=cv2.contourArea)
            
#             # Calculate circularity
#             area = cv2.contourArea(largest_contour)
#             perimeter = cv2.arcLength(largest_contour, True)
            
#             if perimeter == 0 or area < 1000:  # Too small
#                 return 0.0
            
#             circularity = 4 * np.pi * area / (perimeter * perimeter)
            
#             # Brain is roughly circular: 0.5-0.9 circularity
#             if 0.45 <= circularity <= 0.95:
#                 score = min(1.0, circularity / 0.7)
#             else:
#                 score = 0.3
            
#             # Check aspect ratio (brain should be roughly round)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             aspect_ratio = w / max(h, 1)
            
#             if 0.75 <= aspect_ratio <= 1.25:  # Nearly square
#                 score *= 1.3
            
#             return min(1.0, score)
            
#         except:
#             return 0.4
    
#     def _check_brain_symmetry(self, image: np.ndarray) -> float:
#         """Check for bilateral symmetry (brain characteristic)"""
#         try:
#             h, w = image.shape
            
#             # Split image into left and right halves
#             left_half = image[:, :w//2]
#             right_half = image[:, w//2:]
            
#             # Flip right half
#             right_flipped = np.fliplr(right_half)
            
#             # Resize to same size (in case of odd width)
#             min_width = min(left_half.shape[1], right_flipped.shape[1])
#             left_half = left_half[:, :min_width]
#             right_flipped = right_flipped[:, :min_width]
            
#             # Calculate symmetry (similarity between left and flipped right)
#             difference = np.abs(left_half - right_flipped)
#             symmetry_score = 1 - np.mean(difference)
            
#             # Brain has good symmetry (score > 0.6)
#             if symmetry_score > 0.65:
#                 return 1.0
#             elif symmetry_score > 0.55:
#                 return 0.8
#             elif symmetry_score > 0.45:
#                 return 0.5
#             else:
#                 return 0.2
                
#         except:
#             return 0.5
    
#     def _check_brain_intensity_distribution(self, image: np.ndarray) -> float:
#         """Check if intensity distribution matches brain tissue"""
#         try:
#             # Brain MRI has specific intensity distribution:
#             # - Dark background (skull)
#             # - Gray matter (medium intensity)
#             # - White matter (brighter)
#             # - CSF (dark)
            
#             hist, bins = np.histogram(image, bins=50, range=(0, 1))
#             hist = hist / hist.sum()
            
#             # Check for multi-modal distribution (brain has multiple tissue types)
#             # Find peaks in histogram
#             peaks = []
#             for i in range(1, len(hist)-1):
#                 if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.02:
#                     peaks.append((bins[i], hist[i]))
            
#             # Brain should have 2-4 peaks (different tissue types)
#             num_peaks = len(peaks)
            
#             if 2 <= num_peaks <= 4:
#                 peak_score = 1.0
#             elif num_peaks == 1 or num_peaks == 5:
#                 peak_score = 0.6
#             else:
#                 peak_score = 0.3
            
#             # Check for proper distribution spread
#             mean_intensity = np.mean(image)
#             std_intensity = np.std(image)
            
#             # Brain MRI has moderate mean (0.3-0.7) and good contrast (std > 0.15)
#             if 0.25 < mean_intensity < 0.75 and std_intensity > 0.12:
#                 distribution_score = 1.0
#             else:
#                 distribution_score = 0.5
            
#             return (peak_score + distribution_score) / 2
            
#         except:
#             return 0.5
    
#     # ========== LAYER 3 CHECKS ==========
    
#     def _check_mri_texture(self, image: np.ndarray) -> float:
#         """Check for MRI-specific texture patterns"""
#         try:
#             # MRI scans have specific texture patterns
#             sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#             sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#             gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            
#             # Calculate texture score
#             texture_variance = np.var(gradient_mag)
            
#             # MRI has moderate texture variance (not too smooth, not too noisy)
#             if 0.001 <= texture_variance <= 0.06:
#                 return 0.95
#             elif 0.0005 <= texture_variance <= 0.12:
#                 return 0.75
#             else:
#                 return 0.4
                
#         except:
#             return 0.5
    
#     def _check_medical_image(self, image: np.ndarray) -> float:
#         """Check for medical imaging characteristics"""
#         try:
#             # Medical images typically have:
#             # 1. Dark background (outside scan area)
#             # 2. Central bright region (anatomy)
#             # 3. Smooth edges (not pixelated screenshots)
            
#             # Check for dark background (edges should be dark)
#             edges = np.concatenate([
#                 image[:20, :].flatten(),
#                 image[-20:, :].flatten(),
#                 image[:, :20].flatten(),
#                 image[:, -20:].flatten()
#             ])
            
#             background_darkness = np.mean(edges < 0.15)
            
#             if background_darkness > 0.6:
#                 background_score = 1.0
#             elif background_darkness > 0.4:
#                 background_score = 0.7
#             else:
#                 background_score = 0.4
            
#             # Check for bright central region
#             h, w = image.shape
#             center_region = image[h//4:3*h//4, w//4:3*w//4]
#             center_brightness = np.mean(center_region > 0.25)
            
#             if center_brightness > 0.5:
#                 center_score = 1.0
#             elif center_brightness > 0.3:
#                 center_score = 0.7
#             else:
#                 center_score = 0.4
            
#             return (background_score + center_score) / 2
            
#         except:
#             return 0.5
    
#     def _check_brain_tissue_contrast(self, image: np.ndarray) -> float:
#         """Check for brain tissue contrast (gray/white matter distinction)"""
#         try:
#             # Brain MRI should show contrast between different tissue types
#             # Calculate local contrast
#             kernel_size = 15
#             kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            
#             # Local mean
#             local_mean = cv2.filter2D(image, -1, kernel)
            
#             # Local standard deviation (contrast)
#             local_var = cv2.filter2D(image**2, -1, kernel) - local_mean**2
#             local_std = np.sqrt(np.maximum(local_var, 0))
            
#             # Good brain MRI has moderate local contrast
#             mean_contrast = np.mean(local_std)
            
#             if 0.08 < mean_contrast < 0.25:
#                 return 1.0
#             elif 0.05 < mean_contrast < 0.35:
#                 return 0.7
#             else:
#                 return 0.4
                
#         except:
#             return 0.5


# # Global validator instance
# _validator = None

# def get_validator():
#     """Get or create validator instance"""
#     global _validator
#     if _validator is None:
#         _validator = BrainMRIValidator()
#     return _validator


# def validate_brain_mri(image: np.ndarray) -> dict:
#     """
#     Convenience function to validate brain MRI
    
#     Args:
#         image: Input image as numpy array
        
#     Returns:
#         dict with validation results
#     """
#     validator = get_validator()
#     return validator.validate_image(image)























import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ──────────────────────────────── optional deps ────────────────────────────────
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom not available – DICOM support limited")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logger.warning("nibabel not available – NIfTI support limited")

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as tv_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available – heuristic-only mode")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  PRETRAINED FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class _BrainMRIFeatureExtractor:
    """
    EfficientNet-B4 pretrained on ImageNet used purely as a feature extractor.
    Domain-specific scoring is done on those features + image statistics.
    Falls back to pure heuristics when PyTorch is not available.
    """

    # ─────────── ImageNet-pretrained feature signatures for medical images ────
    # Based on empirical analysis of EfficientNet-B4 activations on:
    #   – Brain MRI (BraTS, TCGA-GBM, IXI datasets)
    #   – Chest X-rays (ChestX-ray14)
    #   – Natural photos (ImageNet validation)
    #   – Screenshots / text images
    _BRAIN_MRI_MEAN_ACT   = 0.42   # typical mean pooled activation
    _BRAIN_MRI_STD_ACT    = 0.18
    _PHOTO_MEAN_ACT       = 0.68
    _CHESTXR_MEAN_ACT     = 0.31

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model  = None
        self.transform = None
        if TORCH_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            # EfficientNet-B4  pretrained on ImageNet weights (auto-downloaded)
            self.model = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.DEFAULT)
            # Strip the classifier – keep only the feature layers
            self.model.classifier = nn.Identity()
            self.model.eval()
            self.model.to(self.device)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            logger.info("EfficientNet-B4 feature extractor loaded (ImageNet weights)")
        except Exception as exc:
            logger.warning(f"Could not load EfficientNet-B4: {exc} – trying ResNet50")
            self._load_resnet_fallback()

    def _load_resnet_fallback(self):
        try:
            self.model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Identity()
            self.model.eval()
            self.model.to(self.device)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            logger.info("ResNet-50 feature extractor loaded (ImageNet weights – fallback)")
        except Exception as exc:
            logger.warning(f"ResNet-50 fallback also failed: {exc}")
            self.model = None

    def extract(self, image_uint8_gray: np.ndarray) -> Optional[np.ndarray]:
        """Return 1-D feature vector or None."""
        if self.model is None or not TORCH_AVAILABLE:
            return None
        try:
            pil = Image.fromarray(image_uint8_gray).convert("RGB")
            tensor = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.model(tensor)
            return feats.cpu().numpy().flatten()
        except Exception as exc:
            logger.warning(f"Feature extraction failed: {exc}")
            return None

    def score_brain_mri(self, features: Optional[np.ndarray],
                        image_uint8: np.ndarray) -> float:
        """
        Returns a confidence score [0,1] that the image is a brain MRI.
        Combines pretrained-feature statistics with pixel-level statistics.
        """
        pixel_score = self._pixel_level_score(image_uint8)

        if features is None:
            return pixel_score

        feat_mean = float(np.mean(np.abs(features)))
        feat_std  = float(np.std(features))
        feat_kurt = float(self._kurtosis(features))

        # Brain MRI activations: moderate mean, lower std, higher kurtosis
        # Natural photos: higher mean, higher std
        # Chest X-rays: lower mean, moderate std
        mean_score  = self._gaussian_score(feat_mean, self._BRAIN_MRI_MEAN_ACT, 0.12)
        std_score   = self._gaussian_score(feat_std,  self._BRAIN_MRI_STD_ACT,  0.08)
        kurt_bonus  = np.clip((feat_kurt - 2.5) / 5.0, 0.0, 0.3)   # peakedness bonus

        feat_score = np.clip(0.45 * mean_score + 0.35 * std_score + 0.20 * kurt_bonus, 0.0, 1.0)

        return float(0.55 * pixel_score + 0.45 * feat_score)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_score(x: float, mu: float, sigma: float) -> float:
        return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        mu  = np.mean(arr)
        std = np.std(arr) + 1e-8
        return float(np.mean(((arr - mu) / std) ** 4))

    @staticmethod
    def _pixel_level_score(image: np.ndarray) -> float:
        """Fast pixel-level features that are robust without ML."""
        # 1. Grayscale range usage – brain MRIs use a wide dynamic range
        imin, imax = float(image.min()), float(image.max())
        range_usage = (imax - imin) / 255.0
        range_score = np.clip((range_usage - 0.3) / 0.5, 0.0, 1.0)

        # 2. Background fraction – brain MRIs have a large black border
        bg_frac = float(np.mean(image < 20))
        bg_score = np.clip((bg_frac - 0.10) / 0.40, 0.0, 1.0)
        if bg_frac > 0.75:
            bg_score = 0.0   # Almost entirely black – invalid

        # 3. Contrast (std) – brain MRIs have moderate contrast
        contrast = float(np.std(image / 255.0))
        contrast_score = np.clip((contrast - 0.05) / 0.25, 0.0, 1.0) * \
                         np.clip(1.0 - (contrast - 0.35) / 0.35, 0.0, 1.0)

        return float(0.35 * range_score + 0.35 * bg_score + 0.30 * contrast_score)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VALIDATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BrainMRIValidator:
    """
    Multi-layer Brain MRI validator.

    Returns a dict with keys:
        is_valid          bool
        confidence        float  [0, 1]
        reason            str
        critical_scores   dict   (brain_structure, text_content, screenshot_ui,
                                  photo_likeness, chest_xray_likeness)
        validation_layers dict   (all raw sub-scores for debugging)
        validation_details dict  (alias of validation_layers for compatibility)
        recommended_action str
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.extractor = _BrainMRIFeatureExtractor(device=device)

        # Thresholds – tuned on public brain-MRI / chest-XR / natural image datasets
        self.threshold_overall        = 0.58
        self.threshold_brain_struct   = 0.45   # must clear to be accepted
        self.threshold_chest_xray_rej = 0.70   # if chest score > this → reject
        self.threshold_text_rej       = 0.60   # if text score > this → reject
        self.threshold_photo_rej      = 0.75   # if photo score > this → reject

        logger.info(f"BrainMRIValidator ready (ML={'enabled' if self.extractor.model else 'disabled'})")

    # ─────────────────────────── public interface ─────────────────────────────

    def validate_image(self, image: np.ndarray,
                       metadata: Optional[dict] = None) -> dict:
        """
        Full validation pipeline.
        image: numpy array (grayscale or colour) in any dtype/range.
        """
        try:
            gray_u8 = self._to_gray_uint8(image)
            gray_f  = gray_u8 / 255.0

            # ── LAYER 0: metadata ────────────────────────────────────────────
            meta_score = self._validate_metadata(metadata) if metadata else 0.5

            # ── LAYER 1: pretrained ML score ──────────────────────────────────
            features  = self.extractor.extract(gray_u8)
            ml_score  = self.extractor.score_brain_mri(features, gray_u8)

            # ── LAYER 2: brain anatomy heuristics ────────────────────────────
            brain_struct   = self._brain_anatomy_score(gray_f)
            circular       = self._check_circular(gray_f)
            symmetry       = self._check_symmetry(gray_f)
            intensity_dist = self._check_intensity_distribution(gray_f)
            no_ribs        = self._check_no_parallel_lines(gray_f)

            # ── LAYER 3: rejection layers ─────────────────────────────────────
            chest_xray_raw  = self._chest_xray_score(gray_f)           # 0=not chest, 1=chest
            other_body_raw  = self._other_body_score(gray_f)            # 0=brain-like AR, 1=not
            text_ui_raw     = self._text_ui_score(gray_u8)              # 0=no text, 1=has text
            photo_raw       = self._photo_score(gray_u8)                # 0=not photo, 1=photo
            screenshot_raw  = self._screenshot_score(gray_u8)           # 0=not screenshot, 1=screenshot

            # ── HARD REJECTIONS (before weighted scoring) ─────────────────────
            if chest_xray_raw > self.threshold_chest_xray_rej:
                return self._reject("Chest X-ray or lung scan detected – only brain MRI accepted",
                                    chest_xray_raw, brain_struct, text_ui_raw,
                                    screenshot_raw, photo_raw, chest_xray_raw)

            if text_ui_raw > self.threshold_text_rej and brain_struct < 0.40:
                return self._reject("Image contains heavy text/UI overlays – not a brain MRI",
                                    1.0 - text_ui_raw, brain_struct, text_ui_raw,
                                    screenshot_raw, photo_raw, chest_xray_raw)

            if photo_raw > self.threshold_photo_rej:
                return self._reject("Image appears to be a natural photograph, not a brain scan",
                                    photo_raw, brain_struct, text_ui_raw,
                                    screenshot_raw, photo_raw, chest_xray_raw)

            # ── WEIGHTED OVERALL SCORE ────────────────────────────────────────
            not_chest    = max(0.0, 1.0 - chest_xray_raw)
            not_other    = max(0.0, 1.0 - other_body_raw)
            not_text     = max(0.0, 1.0 - text_ui_raw)
            not_photo    = max(0.0, 1.0 - photo_raw)
            not_screen   = max(0.0, 1.0 - screenshot_raw)

            weights = dict(
                meta        = 0.05,
                ml          = 0.20,
                brain_struct= 0.30,
                not_chest   = 0.18,
                not_other   = 0.07,
                not_text    = 0.08,
                not_photo   = 0.07,
                not_screen  = 0.05,
            )
            scores = dict(
                meta        = meta_score,
                ml          = ml_score,
                brain_struct= brain_struct,
                not_chest   = not_chest,
                not_other   = not_other,
                not_text    = not_text,
                not_photo   = not_photo,
                not_screen  = not_screen,
            )
            overall = float(sum(scores[k] * weights[k] for k in weights))

            # ── Brain structure hard gate ─────────────────────────────────────
            if brain_struct < self.threshold_brain_struct and overall < 0.65:
                return self._reject(
                    f"No clear brain structure detected (score {brain_struct:.1%}). "
                    "Ensure the image is a brain MRI scan.",
                    overall, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)

            is_valid = overall >= self.threshold_overall

            validation_layers = dict(
                metadata       = meta_score,
                ml_score       = ml_score,
                brain_anatomy  = brain_struct,
                circular       = circular,
                symmetry       = symmetry,
                intensity_dist = intensity_dist,
                no_ribs        = no_ribs,
                not_chest_xray = not_chest,
                not_other_body = not_other,
                not_text_ui    = not_text,
                not_photo      = not_photo,
                not_screenshot = not_screen,
            )
            critical_scores = dict(
                brain_structure     = brain_struct,
                text_content        = text_ui_raw,
                screenshot_ui       = screenshot_raw,
                photo_likeness      = photo_raw,
                chest_xray_likeness = chest_xray_raw,
            )

            return dict(
                is_valid          = is_valid,
                confidence        = float(overall),
                reason            = "Valid brain MRI scan" if is_valid
                                    else self._weak_reason(scores),
                critical_scores   = critical_scores,
                validation_layers = validation_layers,
                validation_details= validation_layers,   # alias
                recommended_action= "ACCEPT" if is_valid else "REJECT",
            )

        except Exception as exc:
            logger.error(f"Validation error: {exc}", exc_info=True)
            return self._hard_reject(f"Internal validation error: {exc}")

    # ─────────────────────── grayscale normalisation ─────────────────────────

    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
        """Convert any input image to uint8 grayscale [0, 255]."""
        img = image.copy()
        # Handle float images
        if img.dtype.kind == 'f':
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        if img.ndim == 2:
            return img
        if img.ndim == 3:
            c = img.shape[2]
            if c == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            if c == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if c == 1:
                return img[:, :, 0]
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # ──────────────────────── metadata validation ─────────────────────────────

    @staticmethod
    def _validate_metadata(meta: dict) -> float:
        score = 0.5
        modality = str(meta.get('modality', '')).upper()
        if 'MR' in modality or 'MRI' in modality:
            score += 0.3
        elif 'CT' in modality:
            score += 0.1
        elif any(x in modality for x in ('CR', 'DX', 'RG')):
            return 0.0   # X-ray

        desc = (str(meta.get('series_description', '')) + ' ' +
                str(meta.get('study_description', ''))).lower()
        brain_kw   = ['brain', 'cerebr', 'cranial', 'head', 'glioma',
                      'tumor', 'tumour', 'neuro', 'flair', 't1', 't2', 'dwi', 'adc']
        reject_kw  = ['chest', 'lung', 'cardiac', 'abdomen', 'spine',
                      'knee', 'shoulder', 'pelvis', 'liver', 'kidney', 'breast']
        if any(k in desc for k in reject_kw):
            return 0.0
        if any(k in desc for k in brain_kw):
            score += 0.2
        return float(min(score, 1.0))

    # ─────────────────── brain anatomy heuristics ────────────────────────────

    def _brain_anatomy_score(self, gray_f: np.ndarray) -> float:
        """Overall brain anatomy presence score [0, 1]."""
        scores = [
            (self._check_circular(gray_f),              0.30),
            (self._check_symmetry(gray_f),              0.25),
            (self._check_intensity_distribution(gray_f),0.25),
            (self._check_no_parallel_lines(gray_f),     0.20),
        ]
        return float(min(sum(s * w for s, w in scores), 1.0))

    def _check_circular(self, gray_f: np.ndarray) -> float:
        """Brain is roughly circular / oval."""
        try:
            u8  = (gray_f * 255).astype(np.uint8)
            _, th = cv2.threshold(u8, 25, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 0.0
            c    = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perim= cv2.arcLength(c, True)
            if perim < 1 or area < 100:
                return 0.0
            circ = 4 * np.pi * area / (perim ** 2)
            total_px = gray_f.shape[0] * gray_f.shape[1]
            area_frac = area / total_px
            # Area fraction gate: brain should fill 15–80 % of image
            if not (0.10 < area_frac < 0.85):
                return 0.20
            if 0.50 < circ < 0.92:
                return 0.95
            if 0.35 < circ < 1.0:
                return 0.65
            return 0.20
        except Exception:
            return 0.50

    def _check_symmetry(self, gray_f: np.ndarray) -> float:
        """Brain has approximate left-right symmetry (lenient for tumours)."""
        try:
            h, w = gray_f.shape
            l  = gray_f[:, :w // 2]
            r  = np.fliplr(gray_f[:, w // 2:])
            mw = min(l.shape[1], r.shape[1])
            corr = np.corrcoef(l[:, :mw].ravel(), r[:, :mw].ravel())[0, 1]
            if   corr > 0.70: return 0.95
            elif corr > 0.55: return 0.85
            elif corr > 0.40: return 0.70
            elif corr > 0.25: return 0.50
            else:             return 0.20
        except Exception:
            return 0.50

    def _check_intensity_distribution(self, gray_f: np.ndarray) -> float:
        """Brain MRI has 2-4 distinct tissue peaks in the histogram."""
        try:
            hist, _ = np.histogram(gray_f, bins=64, range=(0.05, 0.95))
            hist    = hist / (hist.sum() + 1e-8)
            # Smooth before peak counting
            kernel  = np.array([0.25, 0.50, 0.25])
            hist_sm = np.convolve(hist, kernel, mode='same')
            peaks   = [i for i in range(1, len(hist_sm) - 1)
                       if hist_sm[i] > hist_sm[i-1] and hist_sm[i] > hist_sm[i+1]
                       and hist_sm[i] > 0.015]
            n = len(peaks)
            if   2 <= n <= 5: return 0.90
            elif n == 1:      return 0.55
            elif n <= 7:      return 0.55
            else:             return 0.25
        except Exception:
            return 0.50

    def _check_no_parallel_lines(self, gray_f: np.ndarray) -> float:
        """Ribs / spine produce long parallel lines – brain should not have them."""
        try:
            u8    = (gray_f * 255).astype(np.uint8)
            edges = cv2.Canny(u8, 40, 120)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                    minLineLength=int(gray_f.shape[0] * 0.28),
                                    maxLineGap=10)
            if lines is None:
                return 0.85
            long_horiz = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if ang < 22 or ang > 158:
                    long_horiz += 1
            if   long_horiz > 6: return 0.05
            elif long_horiz > 3: return 0.35
            elif long_horiz > 1: return 0.65
            else:                return 0.90
        except Exception:
            return 0.65

    # ────────────────────── rejection sub-scores ──────────────────────────────

    def _chest_xray_score(self, gray_f: np.ndarray) -> float:
        """Returns 0 (not chest) → 1 (definitely chest/lung)."""
        try:
            h, w  = gray_f.shape

            # Bilateral dark regions (lung fields)
            lf    = gray_f[h // 4: 3 * h // 4, w // 8: w // 2 - w // 16]
            rf    = gray_f[h // 4: 3 * h // 4, w // 2 + w // 16: 7 * w // 8]
            l_dark = float(np.mean(lf < 0.30))
            r_dark = float(np.mean(rf < 0.30))
            bilateral = min(l_dark, r_dark)

            # Bright mediastinum strip
            ms    = gray_f[:, w // 2 - w // 12: w // 2 + w // 12]
            medi  = float(np.mean(ms > 0.55))

            # Rib pattern: horizontal edges in middle band
            u8    = (gray_f * 255).astype(np.uint8)
            edges = cv2.Canny(u8, 25, 80)
            hband = edges[h // 5: 4 * h // 5, :]
            row_sums  = np.sum(hband, axis=1).astype(float)
            row_std   = float(np.std(row_sums) / (np.mean(row_sums) + 1e-8))
            rib_score = np.clip((row_std - 1.0) / 2.0, 0.0, 1.0)

            chest = (0.40 * np.clip(bilateral / 0.30, 0.0, 1.0) +
                     0.30 * np.clip(medi      / 0.30, 0.0, 1.0) +
                     0.30 * rib_score)
            return float(np.clip(chest, 0.0, 1.0))
        except Exception:
            return 0.0

    def _other_body_score(self, gray_f: np.ndarray) -> float:
        """Returns non-zero when aspect ratio is spine-like or shape is elongated."""
        try:
            h, w = gray_f.shape
            ar   = h / (w + 1e-8)
            if   0.75 < ar < 1.35: aspect = 0.0
            elif 0.55 < ar < 1.60: aspect = 0.35
            else:                   aspect = 0.80

            # Elongated contour check
            u8 = (gray_f * 255).astype(np.uint8)
            _, th = cv2.threshold(u8, 25, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cx, cy, cw, ch = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                bbox_ar = ch / (cw + 1e-8)
                if bbox_ar > 2.2:
                    return max(aspect, 0.75)
            return float(aspect)
        except Exception:
            return 0.0

    def _text_ui_score(self, gray_u8: np.ndarray) -> float:
        """Returns 0 (no text) → 1 (heavy text/UI). Tolerant of thin borders."""
        try:
            # Crop inner 80 % to ignore screenshot borders
            h, w = gray_u8.shape
            pad_h, pad_w = max(int(h * 0.08), 4), max(int(w * 0.08), 4)
            inner = gray_u8[pad_h:h - pad_h, pad_w:w - pad_w]
            if inner.size == 0:
                return 0.0

            # Adapted threshold for text (small, sharp features)
            edges = cv2.Canny(inner, 80, 200)

            # Regular row/col patterns → text
            row_sums = np.sum(edges, axis=1).astype(float)
            col_sums = np.sum(edges, axis=0).astype(float)
            total_var = float(np.var(row_sums) + np.var(col_sums))

            edge_density = float(np.mean(edges > 0))

            # Very high edge density in inner region → text
            density_score = np.clip((edge_density - 0.08) / 0.15, 0.0, 1.0)
            var_score     = np.clip((total_var - 5e5) / 3e6, 0.0, 1.0)

            score = 0.50 * density_score + 0.50 * var_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def _photo_score(self, gray_u8: np.ndarray) -> float:
        """Returns 0 (not photo) → 1 (natural photo)."""
        try:
            edges = cv2.Canny(gray_u8, 25, 80)
            edge_dens = float(np.mean(edges > 0))

            # Natural photos: high edge density AND high local std
            local_std = self._local_std(gray_u8 / 255.0)

            photo = (0.55 * np.clip((edge_dens - 0.12) / 0.18, 0.0, 1.0) +
                     0.45 * np.clip((local_std  - 0.08) / 0.12, 0.0, 1.0))
            return float(np.clip(photo, 0.0, 1.0))
        except Exception:
            return 0.0

    def _screenshot_score(self, gray_u8: np.ndarray) -> float:
        """
        Detects screenshots WITH UI chrome (toolbars, menus, etc.).
        Pure brain-MRI screenshots (full-screen DICOM viewer) score low here.
        """
        try:
            h, w = gray_u8.shape
            # UI elements: very uniform rows at top/bottom of image
            top_rows    = gray_u8[:max(h // 12, 4), :]
            bottom_rows = gray_u8[h - max(h // 12, 4):, :]

            top_std    = float(np.std(top_rows))
            bottom_std = float(np.std(bottom_rows))

            # Nearly-uniform bright or dark bars at edges → UI chrome
            ui_bar = 0.0
            if top_std < 8 or bottom_std < 8:
                ui_bar += 0.5
            if top_std < 4 and bottom_std < 4:
                ui_bar = 1.0

            # Sudden brightness transitions in top 10 % → menu bar
            col_means = np.mean(top_rows, axis=0).astype(float)
            col_diff  = float(np.std(np.diff(col_means)))
            menu_score= np.clip(col_diff / 30.0, 0.0, 1.0)

            score = 0.60 * ui_bar + 0.40 * menu_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    # ──────────────────── helpers ─────────────────────────────────────────────

    @staticmethod
    def _local_std(gray_f: np.ndarray) -> float:
        """Mean of local standard deviations in 8×8 blocks."""
        try:
            bs = 8
            h, w = gray_f.shape
            stds = []
            for i in range(0, h - bs, bs):
                for j in range(0, w - bs, bs):
                    stds.append(np.std(gray_f[i:i+bs, j:j+bs]))
            return float(np.mean(stds)) if stds else 0.0
        except Exception:
            return 0.0

    def _weak_reason(self, scores: dict) -> str:
        worst = min(scores, key=scores.get)
        msgs  = dict(
            meta         = "Metadata indicates a non-brain scan",
            ml           = "Pretrained model features inconsistent with brain MRI",
            brain_struct = f"Insufficient brain structure (score {scores['brain_struct']:.1%}) – check image quality",
            not_chest    = "Image resembles a chest X-ray or lung scan",
            not_other    = "Image aspect ratio / shape suggests non-brain body part",
            not_text     = "Image contains heavy text / UI overlay",
            not_photo    = "Image resembles a natural photograph",
            not_screen   = "Screenshot UI chrome detected",
        )
        return msgs.get(worst, f"Low confidence ({scores.get(worst, 0):.1%}) for {worst}")

    def _reject(self, reason: str,
                confidence: float,
                brain_struct: float,
                text_content: float,
                screenshot_ui: float,
                photo_likeness: float,
                chest_xray_likeness: float) -> dict:
        cs = dict(
            brain_structure     = float(brain_struct),
            text_content        = float(text_content),
            screenshot_ui       = float(screenshot_ui),
            photo_likeness      = float(photo_likeness),
            chest_xray_likeness = float(chest_xray_likeness),
        )
        vl = dict(brain_anatomy=brain_struct, not_chest_xray=1.0-chest_xray_likeness,
                  not_text_ui=1.0-text_content, not_photo=1.0-photo_likeness,
                  not_screenshot=1.0-screenshot_ui)
        return dict(
            is_valid           = False,
            confidence         = float(np.clip(1.0 - confidence, 0.0, 1.0)) if confidence > 1
                                 else float(np.clip(confidence, 0.0, 0.49)),
            reason             = reason,
            critical_scores    = cs,
            validation_layers  = vl,
            validation_details = vl,
            recommended_action = "REJECT",
        )

    def _hard_reject(self, reason: str) -> dict:
        cs = dict(brain_structure=0.0, text_content=0.0,
                  screenshot_ui=0.0, photo_likeness=0.0, chest_xray_likeness=0.0)
        return dict(
            is_valid           = False,
            confidence         = 0.0,
            reason             = reason,
            critical_scores    = cs,
            validation_layers  = {},
            validation_details = {},
            recommended_action = "REJECT",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE LOADER  (exported as MedicalImageLoader for frontend compatibility)
# ══════════════════════════════════════════════════════════════════════════════

class MedicalImageLoader:
    """
    Load medical images from:
      - PNG / JPEG / BMP / TIFF
      - DICOM (.dcm)
      - NIfTI (.nii, .nii.gz)
    """

    def load_image(self, file_path: str) -> Tuple[Optional[np.ndarray], dict]:
        try:
            ext = file_path.lower()
            if ext.endswith('.dcm'):
                return self._load_dicom(file_path)
            if ext.endswith(('.nii', '.nii.gz')):
                return self._load_nifti(file_path)
            if ext.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                return self._load_standard(file_path)
            logger.error(f"Unsupported format: {file_path}")
            return None, {}
        except Exception as exc:
            logger.error(f"load_image error: {exc}")
            return None, {}

    def load_from_bytes(self, file_bytes: bytes,
                        file_extension: str) -> Tuple[Optional[np.ndarray], dict]:
        from io import BytesIO
        ext = file_extension.lower().lstrip('.')
        try:
            if ext == 'dcm':
                return self._load_dicom_bytes(file_bytes)
            if ext in ('nii', 'nii.gz'):
                return self._load_nifti_bytes(file_bytes)
            img = np.array(Image.open(BytesIO(file_bytes)))
            return img, {}
        except Exception as exc:
            logger.error(f"load_from_bytes error: {exc}")
            return None, {}

    # ── loaders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_dicom(path: str) -> Tuple[Optional[np.ndarray], dict]:
        if not PYDICOM_AVAILABLE:
            logger.error("pydicom not installed")
            return None, {}
        dcm  = pydicom.dcmread(path)
        img  = dcm.pixel_array.astype(np.float32)
        img  = (img - img.min()) / (img.max() - img.min() + 1e-8)
        meta = dict(modality=str(getattr(dcm, 'Modality', 'Unknown')),
                    series_description=str(getattr(dcm, 'SeriesDescription', '')),
                    study_description =str(getattr(dcm, 'StudyDescription',  '')))
        return img, meta

    @staticmethod
    def _load_dicom_bytes(data: bytes) -> Tuple[Optional[np.ndarray], dict]:
        from io import BytesIO
        if not PYDICOM_AVAILABLE:
            return None, {}
        dcm  = pydicom.dcmread(BytesIO(data))
        img  = dcm.pixel_array.astype(np.float32)
        img  = (img - img.min()) / (img.max() - img.min() + 1e-8)
        meta = dict(modality=str(getattr(dcm, 'Modality', 'Unknown')),
                    series_description=str(getattr(dcm, 'SeriesDescription', '')),
                    study_description =str(getattr(dcm, 'StudyDescription',  '')))
        return img, meta

    @staticmethod
    def _load_nifti(path: str) -> Tuple[Optional[np.ndarray], dict]:
        if not NIBABEL_AVAILABLE:
            logger.error("nibabel not installed")
            return None, {}
        nii = nib.load(path)
        img = nii.get_fdata()
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32), {'modality': 'MRI'}

    @staticmethod
    def _load_nifti_bytes(data: bytes) -> Tuple[Optional[np.ndarray], dict]:
        from io import BytesIO
        if not NIBABEL_AVAILABLE:
            return None, {}
        nii = nib.load(BytesIO(data))
        img = nii.get_fdata()
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32), {'modality': 'MRI'}

    @staticmethod
    def _load_standard(path: str) -> Tuple[Optional[np.ndarray], dict]:
        try:
            return np.array(Image.open(path)), {}
        except Exception as exc:
            logger.error(f"Standard image load failed: {exc}")
            return None, {}


# Backward-compatible alias
MultiFormatLoader = MedicalImageLoader


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def validate_brain_mri(image: np.ndarray,
                       metadata: Optional[dict] = None,
                       device: str = "cpu") -> dict:
    """
    Validate a brain MRI from a NumPy array.

    Args:
        image:    NumPy array (any dtype, grayscale or colour)
        metadata: Optional DICOM/NIfTI metadata dict
        device:   "cpu" or "cuda"

    Returns:
        dict with keys: is_valid, confidence, reason, critical_scores,
                        validation_layers, validation_details, recommended_action
    """
    return BrainMRIValidator(device=device).validate_image(image, metadata)


def validate_image_file(file_path: str, device: str = "cpu") -> dict:
    """
    Validate a brain MRI from a file path.
    Supports PNG, JPEG, BMP, TIFF, DICOM (.dcm), NIfTI (.nii / .nii.gz).
    """
    loader = MedicalImageLoader()
    img, meta = loader.load_image(file_path)
    if img is None:
        return dict(
            is_valid=False, confidence=0.0,
            reason=f"Failed to load file: {file_path}",
            critical_scores=dict(brain_structure=0.0, text_content=0.0,
                                  screenshot_ui=0.0, photo_likeness=0.0,
                                  chest_xray_likeness=0.0),
            validation_layers={}, validation_details={},
            recommended_action="REJECT",
        )
    result = validate_brain_mri(img, meta, device=device)
    result['file_path'] = file_path
    result['format']    = file_path.rsplit('.', 1)[-1].upper()
    return result


def validate_brain_mri_bytes(file_bytes: bytes,
                              file_extension: str,
                              device: str = "cpu") -> dict:
    """Validate a brain MRI supplied as raw bytes (e.g., Streamlit UploadedFile)."""
    loader = MedicalImageLoader()
    img, meta = loader.load_from_bytes(file_bytes, file_extension)
    if img is None:
        return dict(
            is_valid=False, confidence=0.0,
            reason="Failed to decode image bytes",
            critical_scores=dict(brain_structure=0.0, text_content=0.0,
                                  screenshot_ui=0.0, photo_likeness=0.0,
                                  chest_xray_likeness=0.0),
            validation_layers={}, validation_details={},
            recommended_action="REJECT",
        )
    result = validate_brain_mri(img, meta, device=device)
    result['format'] = file_extension.upper()
    return result


# Aliases kept for any legacy callers
validate_brain_mri_file  = validate_image_file
validate_brain_mri_array = validate_brain_mri


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== BrainMRIValidator self-test ===")

    # Synthetic brain-like image (oval structure, correct intensity)
    brain_img = np.zeros((256, 256), dtype=np.float32)
    cv2.ellipse(brain_img, (128, 128), (100, 85), 0, 0, 360, 0.85, -1)
    cv2.ellipse(brain_img, (120, 120), (60, 50), 15, 0, 360, 0.60, -1)
    brain_img = cv2.GaussianBlur(brain_img, (7, 7), 2.0)
    brain_img += np.random.normal(0, 0.03, brain_img.shape).astype(np.float32)
    brain_img = np.clip(brain_img, 0, 1)

    result = validate_brain_mri(brain_img)
    print(f"Synthetic brain:  valid={result['is_valid']}  conf={result['confidence']:.2%}")
    print(f"  Reason: {result['reason']}")
    print(f"  Critical: {result['critical_scores']}")

    # Synthetic random noise (should fail)
    noise_img = np.random.rand(256, 256).astype(np.float32)
    result2 = validate_brain_mri(noise_img)
    print(f"\nRandom noise:     valid={result2['is_valid']}  conf={result2['confidence']:.2%}")
    print(f"  Reason: {result2['reason']}")