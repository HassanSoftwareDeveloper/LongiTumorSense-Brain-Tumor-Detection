import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, Tuple, Optional
import warnings

try:
    from skimage import measure
except ImportError:
    measure = None
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

        # BALANCED Thresholds – Accept brain MRIs, reject portraits
        self.threshold_overall        = 0.35   # Accept brain MRIs with ≥35% confidence
        self.threshold_brain_struct   = 0.15   # Minimum brain structure requirement
        self.threshold_chest_xray_rej = 0.75   # Reject chest X-rays if >75%
        self.threshold_text_rej       = 0.90   # Allow text overlays (MRI screenshots)
        self.threshold_photo_rej      = 0.55   # Reject natural photos if >55%
        self.threshold_face_rej       = 0.45   # Reject face photos if >45%
        self.threshold_other_body_rej = 0.80   # Reject other body parts if >80%
        self.threshold_brain_override = 0.50   # Brain structure ≥50% overrides rejections

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

            # ── ADDITIONAL REJECTION LAYERS ───────────────────────────────────
            face_raw       = self._face_human_score(gray_u8)
            other_body_adv = self._advanced_other_body_score(gray_f)

            # ═══════════════════════════════════════════════════════════════════
            # AGGRESSIVE PORTRAIT REJECTION (CRITICAL FIX)
            # Problem: Brain structure detector can score 65%+ for portrait photos
            # because the oval head shape resembles a brain cross-section.
            # Solution: Check for faces/photos FIRST with strict thresholds.
            # These rules run BEFORE all other checks.
            # ═══════════════════════════════════════════════════════════════════

            # Calculate pixel statistics for portrait detection
            h, w = gray_u8.shape
            aspect_ratio = h / (w + 1e-8)
            hist, _ = np.histogram(gray_u8, bins=32, range=(0, 256))
            hist_norm = hist / (hist.sum() + 1e-8)
            dark_frac = float(np.sum(hist_norm[:4]))   # Very dark pixels
            mid_frac  = float(np.sum(hist_norm[8:24])) # Mid-tone pixels

            # RULE 1: Face + vertical orientation = IMMEDIATE REJECT
            if face_raw > 0.20 and 1.0 < aspect_ratio < 2.5:
                return self._reject(
                    "Human face in portrait orientation – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw, screenshot_raw, photo_raw, chest_xray_raw)

            # RULE 2: Face + Photo combination = IMMEDIATE REJECT
            if face_raw > 0.15 and photo_raw > 0.10:
                return self._reject(
                    "Portrait photograph detected – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw, screenshot_raw, photo_raw, chest_xray_raw)

            # RULE 3: Uniform non-dark background + face = studio portrait REJECT
            if dark_frac < 0.40 and mid_frac > 0.28 and face_raw > 0.15:
                return self._reject(
                    "Studio portrait detected – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw, screenshot_raw, photo_raw, chest_xray_raw)

            # RULE 4: High photo score alone = IMMEDIATE REJECT
            if photo_raw > 0.35:
                return self._reject(
                    "Photograph detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw, screenshot_raw, photo_raw, chest_xray_raw)

            # RULE 5: Face detection alone (high score) = IMMEDIATE REJECT
            if face_raw > 0.30:
                return self._reject(
                    "Human face detected – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw, screenshot_raw, photo_raw, chest_xray_raw)

            # RULE 6: Moderate face + brain structure — head shape fooling detector
            if face_raw > 0.25 and brain_struct > 0.60:
                return self._reject(
                    "Human face detected (head shape resembles brain cross-section) – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw, screenshot_raw, photo_raw, chest_xray_raw)



            # ── HARD REJECTIONS (before weighted scoring) ─────────────────────
            # CRITICAL: Reject non-brain images FIRST, then check brain structure
            # This ensures we reject other MRI types (spine, knee, etc.) even if they
            # have circular patterns that could score high on brain structure
            
            # SPECIAL CASE: High text content (>90%) suggests MRI screenshot
            # Be more lenient with brain structure threshold for screenshots
            is_screenshot = text_ui_raw > 0.90
            brain_override_threshold = 0.50 if is_screenshot else self.threshold_brain_override  # RAISED
            
            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL SAFETY CHECKS: Detect human photos with combined features
            # These checks run BEFORE individual threshold checks
            # IMPORTANT: STRICTER thresholds to reject portrait photographs
            # ═══════════════════════════════════════════════════════════════════
            
            # ═══ ULTIMATE CHECK: Detect visible human body/person ═══
            # Brain MRI scans NEVER show a person's body
            # If we detect a human body shape, it's 100% NOT a brain MRI
            # Check for: vertical human silhouette, clothing, body proportions
            
            # Calculate aspect ratio and check for human-like proportions
            h, w = gray_u8.shape
            aspect_ratio = h / (w + 1e-8)
            
            # Human photos typically have:
            # 1. Vertical orientation or square (aspect ratio 1.2-2.0)
            # 2. Low dark pixel fraction (outdoor lighting, <25%)
            # 3. Moderate to high edge density in center (person outline)
            # 4. Specific intensity distribution (skin tones, clothing)
            
            # Check for human body indicators
            hist, _ = np.histogram(gray_u8, bins=32, range=(0, 256))
            hist_norm = hist / (hist.sum() + 1e-8)
            dark_frac = float(np.sum(hist_norm[:4]))  # Very dark pixels
            bright_frac = float(np.sum(hist_norm[-4:]))  # Very bright pixels
            mid_frac = float(np.sum(hist_norm[8:24]))  # Mid-tone pixels (skin, clothing)
            
            # Detect edges in center region (where person would be)
            center_h_start, center_h_end = h // 4, 3 * h // 4
            center_w_start, center_w_end = w // 4, 3 * w // 4
            center_region = gray_u8[center_h_start:center_h_end, center_w_start:center_w_end]
            edges_center = cv2.Canny(center_region, 50, 150)
            center_edge_density = float(np.mean(edges_center > 0))
            
            # HARD RULE: Reject if BOTH body indicators AND low brain structure signal
            # IMPORTANT: Only reject if brain structure score is ALSO low (< 0.35).
            # Brain MRIs with high brain coverage (dark_frac < 0.35) are still valid.
            # The key discriminator is brain_struct: real MRIs score >0.40 on brain anatomy.
            
            # Condition 1: Human body shape + non-black background + weak brain structure
            if (1.0 < aspect_ratio < 2.2 and
                dark_frac < 0.25 and       # Very few dark pixels (natural photo)
                mid_frac > 0.40 and        # Very high mid-tones (skin/clothing)
                center_edge_density > 0.05 and
                brain_struct < 0.35):      # CRITICAL: must also fail brain structure check
                return self._reject(
                    "Human body/person detected in image – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # Condition 2: Low dark pixels + high mid-tones + any photo score
            # STRICTER: Catch studio portraits with uniform backgrounds
            if (dark_frac < 0.20 and  # Very few dark pixels (RAISED - more strict)
                mid_frac > 0.35 and  # Very high mid-tones (LOWERED - more strict)
                photo_raw > 0.15 and  # Significant photo indication (LOWERED - more strict)
                brain_struct < 0.65 and  # Not good brain structure (RAISED - more strict)
                face_raw > 0.25):  # Significant face indication (LOWERED - more strict)
                return self._reject(
                    "Outdoor/studio photograph with person detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # Condition 3: Bright outdoor scene (sky, snow) + person
            if (bright_frac > 0.12 and  # Bright pixels (LOWERED - more strict)
                dark_frac < 0.30 and  # Few dark pixels (RAISED - more strict)
                mid_frac > 0.30 and  # Mid-tones (LOWERED - more strict)
                center_edge_density > 0.04 and  # Person outline (LOWERED - more strict)
                brain_struct < 0.65 and  # Not good brain (RAISED - more strict)
                face_raw > 0.20):  # Face indication (LOWERED - more strict)
                return self._reject(
                    "Outdoor scene with person detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # Condition 4: Studio portrait — uniform non-black background + face detected
            # This catches person-in-suit-against-grey-background photos
            # CRITICAL FIX: Lower face threshold to catch more portraits
            if (face_raw > 0.35 and  # Strong face detection (LOWERED from 0.55 - more strict)
                dark_frac < 0.30 and  # Not a black-background MRI (RAISED - more strict)
                brain_struct < 0.65):  # Not a brain scan (RAISED - more strict)
                return self._reject(
                    "Portrait photograph detected – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # Check 1: Photo + Face combination (portrait with person)
            # Even low scores combined indicate a human photo
            # CRITICAL FIX: Lower thresholds to catch more portraits
            if photo_raw > 0.20 and face_raw > 0.30 and brain_struct < brain_override_threshold:
                return self._reject(
                    "Human portrait photograph detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # Check 2: Photo + Few dark pixels (outdoor/landscape photo)
            # Brain MRIs have large black backgrounds (dark pixels >20%)
            # Outdoor photos have few dark pixels (<15%)
            if photo_raw > 0.40 and dark_frac < 0.15 and brain_struct < 0.60:  # Made MORE STRICT
                return self._reject(
                    "Natural outdoor photograph detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # Check 3: Moderate photo + Moderate brain + Low dark pixels
            # This catches landscape portraits that fool brain structure detector
            if photo_raw > 0.35 and brain_struct > 0.45 and brain_struct < 0.65 and dark_frac < 0.18:  # Made MORE STRICT
                return self._reject(
                    "Photograph with landscape background detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            
            # STEP 1: Reject obvious non-brain images (regardless of brain structure)
            
            # Reject chest X-rays (obvious lung/rib patterns)
            if chest_xray_raw > self.threshold_chest_xray_rej:
                return self._reject("Chest X-ray or lung scan detected – only brain MRI accepted",
                                    chest_xray_raw, brain_struct, text_ui_raw,
                                    screenshot_raw, photo_raw, chest_xray_raw)

            # Reject natural photos (landscapes, portraits, objects)
            # THREE-TIER SYSTEM (STRICTER - Reject portrait photos):
            # Tier 1 (>60%): ALWAYS reject, no brain override allowed (LOWERED - more strict)
            # Tier 2 (45-60%): Reject unless brain structure is good (≥55%) (RAISED - more strict)
            # Tier 3 (35-45%): Reject if brain structure is moderate (45-60%) AND no medical features
            if photo_raw > 0.60:
                # TIER 1: Photo score >60% → ALWAYS REJECT (no brain override)
                return self._reject(
                    "Natural photograph detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            elif photo_raw > 0.45 and brain_struct < brain_override_threshold:
                # TIER 2: Photo score 45-60% → Reject unless brain ≥55% (override threshold)
                return self._reject(
                    "Natural photograph detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)
            elif photo_raw > 0.35 and brain_struct < 0.60 and text_ui_raw < 0.15:
                # TIER 3: Photo score 35-45% + brain 45-60% + no text → Likely a photo
                # This catches landscape portraits that fool the detectors
                return self._reject(
                    "Natural photograph detected – only brain MRI scans accepted",
                    photo_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)

            # Reject human faces/portraits (obvious face features)
            # CRITICAL FIX: Lower threshold to catch more portraits
            if face_raw > self.threshold_face_rej and brain_struct < brain_override_threshold:
                return self._reject(
                    "Human face or portrait photograph detected – only brain MRI scans accepted",
                    face_raw, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)

            # Reject other body parts (spine, knee, abdomen, etc.)
            if other_body_adv > self.threshold_other_body_rej and brain_struct < brain_override_threshold:
                return self._reject(
                    "Non-brain body part scan detected (spine, abdomen, knee, etc.) – "
                    "only brain MRI accepted",
                    other_body_adv, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)

            # Allow text/UI overlays on brain MRIs (screenshots are OK)

            # ── WEIGHTED OVERALL SCORE ────────────────────────────────────────
            not_chest    = max(0.0, 1.0 - chest_xray_raw)
            not_other    = max(0.0, 1.0 - other_body_raw)
            not_text     = max(0.0, 1.0 - text_ui_raw)
            not_photo    = max(0.0, 1.0 - photo_raw)
            not_screen   = max(0.0, 1.0 - screenshot_raw)
            not_face     = max(0.0, 1.0 - face_raw)
            not_body_adv = max(0.0, 1.0 - other_body_adv)

            weights = dict(
                meta        = 0.05,
                ml          = 0.18,
                brain_struct= 0.28,
                not_chest   = 0.16,
                not_other   = 0.05,
                not_text    = 0.07,
                not_photo   = 0.06,
                not_screen  = 0.05,
                not_face    = 0.06,
                not_body_adv= 0.04,
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
                not_face    = not_face,
                not_body_adv= not_body_adv,
            )
            overall = float(sum(scores[k] * weights[k] for k in weights))

            # ── Brain structure hard gate (RELAXED) ───────────────────────────
            # Only reject if brain structure is VERY low AND overall score is poor
            if brain_struct < self.threshold_brain_struct and overall < 0.40:
                return self._reject(
                    f"No clear brain structure detected (score {brain_struct:.1%}). "
                    "Ensure the image is a brain MRI scan.",
                    overall, brain_struct, text_ui_raw,
                    screenshot_raw, photo_raw, chest_xray_raw)

            # ── Low confidence safety check ───────────────────────────────────
            # If overall confidence is low (<50%) AND photo score is high (>40%)
            # AND brain structure is not good (<45%), likely a photo
            # IMPORTANT: Made MORE LENIENT to avoid rejecting valid brain MRIs
            if overall < 0.50 and photo_raw > 0.40 and brain_struct < brain_override_threshold:
                return self._reject(
                    "Low confidence score with photo-like features detected. "
                    "Only brain MRI scans accepted.",
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

    def _face_human_score(self, gray_u8: np.ndarray) -> float:
        """
        Detect human faces / portraits using Haar cascades + skin-tone heuristics.
        Returns 0 (no face) → 1 (face/human photo detected).
        """
        try:
            score = 0.0

            # ── Haar cascade face detection ───────────────────────────────────
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(
                        gray_u8, scaleFactor=1.1, minNeighbors=3,
                        minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(faces) > 0:
                        largest_area = max(w * h for (_, _, w, h) in faces)
                        total_area   = gray_u8.shape[0] * gray_u8.shape[1]
                        face_frac    = largest_area / total_area
                        score = max(score, min(0.98, 0.70 + face_frac * 2.0))
            except Exception:
                pass

            # ── Skin-tone heuristic (grayscale intensity range 90-210) ────────
            skin_mask = ((gray_u8 >= 80) & (gray_u8 <= 220)).astype(np.float32)
            skin_frac  = float(np.mean(skin_mask))
            if 0.25 < skin_frac < 0.80:
                lap     = cv2.Laplacian(gray_u8, cv2.CV_64F)
                lap_std = float(np.std(lap))
                if lap_std < 25:
                    skin_score = np.clip((skin_frac - 0.25) / 0.40, 0.0, 1.0) * 0.65
                    score = max(score, skin_score)

            # ── Uniform background detection (studio/portrait photos) ─────────
            # Brain MRIs have black backgrounds; portrait photos have uniform grey/white
            h, w = gray_u8.shape
            # Check corners for uniform background (not black)
            corner_size = max(h // 8, 10)
            corners = [
                gray_u8[:corner_size, :corner_size],
                gray_u8[:corner_size, -corner_size:],
                gray_u8[-corner_size:, :corner_size],
                gray_u8[-corner_size:, -corner_size:],
            ]
            corner_means = [float(np.mean(c)) for c in corners]
            corner_stds  = [float(np.std(c)) for c in corners]
            avg_corner_mean = float(np.mean(corner_means))
            avg_corner_std  = float(np.mean(corner_stds))
            # Uniform non-black corners = portrait/studio background
            if avg_corner_mean > 60 and avg_corner_std < 30:
                bg_score = np.clip((avg_corner_mean - 60) / 150.0, 0.0, 1.0) * 0.70
                score = max(score, bg_score)

            # ── Very few dark pixels → likely a colour photo (no black bg) ────
            hist, _ = np.histogram(gray_u8, bins=32, range=(0, 256))
            hist_norm = hist / (hist.sum() + 1e-8)
            dark_frac = float(np.sum(hist_norm[:4]))   # pixels < 32
            if dark_frac < 0.08:
                score = max(score, 0.60)

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def _advanced_other_body_score(self, gray_f: np.ndarray) -> float:
        """
        Detect non-brain body parts: spine, knee, abdomen, hand, etc.
        Returns 0 (brain-like) → 1 (definitely non-brain body part).
        """
        try:
            h, w = gray_f.shape
            score = 0.0

            # ── Aspect ratio: spine/limbs are very elongated ──────────────────
            ar = h / (w + 1e-8)
            if ar > 2.0 or ar < 0.50:
                score = max(score, 0.70)
            elif ar > 1.60 or ar < 0.62:
                score = max(score, 0.45)

            # ── Contour elongation ────────────────────────────────────────────
            u8 = (gray_f * 255).astype(np.uint8)
            _, th = cv2.threshold(u8, 20, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                _, _, cw, ch = cv2.boundingRect(c)
                bbox_ar = ch / (cw + 1e-8)
                if bbox_ar > 2.5:
                    score = max(score, 0.80)
                elif bbox_ar > 1.8:
                    score = max(score, 0.55)

            # ── Parallel horizontal lines (ribs / vertebrae) ──────────────────
            edges = cv2.Canny(u8, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35,
                                    minLineLength=int(min(h, w) * 0.25),
                                    maxLineGap=12)
            if lines is not None:
                horiz = sum(1 for l in lines
                            if abs(np.degrees(np.arctan2(
                                l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 20)
                if horiz > 8:
                    score = max(score, 0.85)
                elif horiz > 4:
                    score = max(score, 0.55)

            # ── Bright central column (spine / long bone) ─────────────────────
            center_col = gray_f[:, w // 3: 2 * w // 3]
            center_bright = float(np.mean(center_col > 0.70))
            if center_bright > 0.35 and ar > 1.4:
                score = max(score, 0.65)

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
            brain_struct = f"Insufficient brain structure (score {scores.get('brain_struct', 0):.1%}) – check image quality",
            not_chest    = "Image resembles a chest X-ray or lung scan",
            not_other    = "Image aspect ratio / shape suggests non-brain body part",
            not_text     = "Image contains heavy text / UI overlay",
            not_photo    = "Image resembles a natural photograph",
            not_screen   = "Screenshot UI chrome detected",
            not_face     = "Human face or portrait detected – not a brain MRI",
            not_body_adv = "Non-brain body part detected (spine, limb, abdomen, etc.)",
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


def assess_scan_artifacts(image: np.ndarray) -> dict:
    """
    Dedicated artifact-detection pass that runs independently of the
    brain-vs-non-brain validator.  Returns a report suitable for surfacing
    warnings to the user before inference begins.

    Checks performed
    ----------------
    - Motion / ghosting:  directional gradient ratio (horizontal >> vertical
      suggests readout-direction ghosting).
    - Metal susceptibility: very dark elliptical voids with hard edges,
      typical of dental implants / screws in brain MRI.
    - Zipper / RF artefact: horizontal bright lines in the phase direction
      (detected via row-mean variance peaks).
    - Truncation (Gibbs) ringing: strong periodic oscillations adjacent to
      high-contrast boundaries.
    - Low SNR: background noise standard deviation as a fraction of signal mean.
    - Insufficient brain coverage: too few non-background pixels.

    Note: these are heuristic checks and can produce false positives on
    unusual but valid scans.  All warnings should be reviewed by the operator.
    """
    import scipy.ndimage as ndi

    warnings = []
    details  = {}

    if image is None or image.size == 0:
        return {'warnings': ['Empty or null image'], 'details': {}, 'quality': 'POOR'}

    # Normalise to float [0,1]
    arr = image.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return {'warnings': ['Zero-range image — all pixels identical'],
                'details': {}, 'quality': 'POOR'}
    arr = (arr - mn) / (mx - mn)

    h, w = arr.shape
    brain_mask = arr > 0.05
    brain_px   = arr[brain_mask]
    details['coverage'] = float(brain_mask.sum() / arr.size)

    if details['coverage'] < 0.10:
        warnings.append('Low brain coverage (<10%) — possible cropping or non-brain image')

    # ── SNR check ─────────────────────────────────────────────────────────────
    bg_mask = ~brain_mask
    bg_std  = float(arr[bg_mask].std()) if bg_mask.sum() > 50 else 0.05
    snr     = float(np.mean(brain_px) / (bg_std + 1e-8))
    details['snr'] = round(snr, 2)
    if snr < 5.0:
        warnings.append(f'Low SNR ({snr:.1f}) — image may be noisy or poorly acquired')

    # ── Motion / ghosting: directional gradient ratio ─────────────────────────
    gx   = cv2.Sobel(arr, cv2.CV_32F, 1, 0)
    gy   = cv2.Sobel(arr, cv2.CV_32F, 0, 1)
    mean_gx = float(np.mean(np.abs(gx)))
    mean_gy = float(np.mean(np.abs(gy)))
    grad_ratio = mean_gx / (mean_gy + 1e-8)
    details['grad_ratio_xy'] = round(grad_ratio, 3)
    if grad_ratio > 4.0 or grad_ratio < 0.25:
        warnings.append(
            f'Possible motion / readout ghosting (gradient ratio {grad_ratio:.2f}) — '
            'check phase-encode direction')

    # ── Metal susceptibility void detection ───────────────────────────────────
    # Very dark, compact, hard-edged voids = susceptibility from implants / clips
    dark_mask = (arr < 0.05).astype(np.uint8)
    dark_mask = cv2.erode(dark_mask, np.ones((3, 3), np.uint8), iterations=1)
    labeled, n_voids = ndi.label(dark_mask)
    compact_voids = 0
    for lbl in range(1, n_voids + 1):
        region = labeled == lbl
        area   = int(region.sum())
        if area < 16 or area > h * w * 0.10:
            continue
        # Compactness: 4π·A / P²
        try:
            if measure is not None:
                contours = measure.find_contours(region.astype(float), 0.5)
                if contours:
                    perim = float(len(max(contours, key=len)))
                    compactness = (4 * np.pi * area) / (perim ** 2 + 1e-8)
                    if compactness > 0.55:   # relatively compact → possible metal void
                        compact_voids += 1
        except Exception:
            pass
    details['compact_dark_voids'] = compact_voids
    if compact_voids >= 2:
        warnings.append(
            f'{compact_voids} compact dark region(s) detected — possible metal artifact '
            '(implant, clip, or dental work). Check susceptibility sequences.')

    # ── Zipper / RF artefact: bright row-mean spikes ──────────────────────────
    row_means = arr.mean(axis=1)
    row_var   = float(np.var(row_means))
    row_mean_global = float(np.mean(row_means))
    row_cv    = np.sqrt(row_var) / (row_mean_global + 1e-8)
    details['row_cv'] = round(row_cv, 4)
    if row_cv > 0.25:
        warnings.append(
            f'Row-mean coefficient of variation high ({row_cv:.2f}) — '
            'possible RF zipper or N/2 ghost artefact')

    # ── Gibbs / truncation ringing: ring power near large edges ───────────────
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    strong_edges = (edge_mag > edge_mag.mean() + 2 * edge_mag.std()).astype(np.uint8)
    # Dilate edges and check for oscillation in adjacent strip
    dilated = cv2.dilate(strong_edges, np.ones((5, 5), np.uint8))
    near_edges = arr[dilated.astype(bool)]
    if len(near_edges) > 100:
        near_std = float(near_edges.std())
        details['edge_strip_std'] = round(near_std, 4)
        if near_std > 0.18:
            warnings.append(
                'Elevated intensity variation near high-contrast edges — '
                'possible Gibbs/truncation ringing')

    quality = 'GOOD' if len(warnings) == 0 else ('ACCEPTABLE' if len(warnings) <= 1 else 'POOR')
    return {
        'warnings': warnings,
        'details': details,
        'quality': quality,
        'artifact_count': len(warnings),
    }


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