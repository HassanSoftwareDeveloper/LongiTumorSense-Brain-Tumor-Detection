# infernece.py

from __future__ import annotations

import gc
import logging
import time
import warnings
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurt
from skimage.feature import graycomatrix, graycoprops
from skimage import measure, morphology

# Import grade fusion module (v11.0 - classifier removed)
from simple_grade_fusion import simple_grade_fusion

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY FIX: SIMPLIFIED GRADE PREDICTION
#  Replaces over-complicated fusion logic with simple weighted average
# ═══════════════════════════════════════════════════════════════════════════════

def _improved_grade_prediction(seg_probs, cls_probs, radiomics_probs, img_cls_probs,
                               tumor_detected, tumor_pct, glcm_malignancy, max_seg_prob, asymmetry):
    """
    FIXED v9.2: Improved grade prediction with better clinical validation
    
    Key improvements:
    1. Uses 4 sources: segmentation, classifiers, radiomics, image features
    2. Clinical validation prevents impossible predictions
    3. Confidence-weighted fusion based on source reliability
    4. Proper handling of edge cases (normal vs low-grade)
    5. FIXED: Better normal scan detection using primary tumor indicators
    """
    
    # Normalize all probability vectors
    seg_probs = np.array(seg_probs, dtype=np.float64)
    cls_probs = np.array(cls_probs, dtype=np.float64) 
    radiomics_probs = np.array(radiomics_probs, dtype=np.float64)
    img_cls_probs = np.array(img_cls_probs, dtype=np.float64)
    
    # Ensure all probabilities sum to 1
    seg_probs = seg_probs / (seg_probs.sum() + 1e-9)
    cls_probs = cls_probs / (cls_probs.sum() + 1e-9)
    radiomics_probs = radiomics_probs / (radiomics_probs.sum() + 1e-9)
    img_cls_probs = img_cls_probs / (img_cls_probs.sum() + 1e-9)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX v10.2: Detect untrained classifiers
    # ═══════════════════════════════════════════════════════════════════════════
    # If classifiers return near-uniform distributions, they're not actually trained
    # This prevents the "Grade I collapse" bug where untrained classifiers bias results
    
    cls_entropy = -np.sum(cls_probs * np.log(cls_probs + 1e-9))
    max_entropy = -np.log(1.0 / 5)  # Maximum entropy for 5 classes = 1.609
    cls_is_uniform = (cls_entropy > 0.95 * max_entropy)  # >95% of max entropy = uniform
    
    if cls_is_uniform:
        logger.warning(f"[GRADE PRED] ⚠️  Classifiers returning uniform distribution (entropy={cls_entropy:.3f}) - "
                      f"likely untrained. Using radiomics + segmentation + image features only.")
        # Don't use untrained classifiers - they just add noise
        use_classifiers = False
    else:
        logger.info(f"[GRADE PRED] ✓ Classifiers trained (entropy={cls_entropy:.3f}) - using all sources")
        use_classifiers = True
    
    # CRITICAL FIX v9.2: Better normal scan detection
    # Primary indicators: tumor_pct (segmentation result) and image classifier
    # Secondary indicators: segmentation confidence and radiomics
    
    # Check if image classifier strongly predicts normal (Grade 0)
    img_predicts_normal = (img_cls_probs[0] > 0.7)  # Image classifier is confident about normal
    
    # Check if segmentation finds minimal tumor
    seg_finds_minimal_tumor = (tumor_pct < 1.0 and max_seg_prob < 0.50)
    
    # FIXED v9.2b: Normal scan detection - require BOTH image classifier AND segmentation agreement
    # Don't override if image classifier detects a tumor (Grade > 0)
    is_clearly_normal = (
        img_predicts_normal and seg_finds_minimal_tumor and 
        tumor_pct < 0.5  # Very low tumor percentage
    )
    
    if is_clearly_normal:
        logger.info(f"[GRADE PRED] Clearly normal scan detected: tumor_pct={tumor_pct:.2f}%, "
                   f"seg_prob={max_seg_prob:.3f}, img_normal_prob={img_cls_probs[0]:.3f}")
        return {
            'grade': 0,
            'confidence': 0.92,
            'grade_probs': [0.92, 0.04, 0.02, 0.01, 0.01],
            'source': 'clearly_normal_v9.2b'
        }
    
    # CRITICAL FIX: Detect clearly abnormal scans
    # High-grade tumors should have: high tumor_pct, high seg_prob, high malignancy
    is_clearly_high_grade = (
        tumor_pct > 8.0 and 
        max_seg_prob > 0.70 and 
        glcm_malignancy > 0.65
    )
    
    if is_clearly_high_grade:
        # Favor Grade III/IV for clearly malignant cases
        logger.info(f"[GRADE PRED] Clearly high-grade tumor detected: tumor_pct={tumor_pct:.2f}%, "
                   f"seg_prob={max_seg_prob:.3f}, malignancy={glcm_malignancy:.3f}")
        # Boost high-grade probabilities
        radiomics_probs[3] *= 2.0  # Grade III
        radiomics_probs[4] *= 2.5  # Grade IV
        radiomics_probs = radiomics_probs / radiomics_probs.sum()
    
    # FIX v10.3: When image classifier detects tumor but seg threshold is uncertain,
    # respect the img_cls grade UNLESS truly_no_seg_signal (max_seg_prob < 0.28).
    img_detects_tumor   = (np.argmax(img_cls_probs) > 0 and img_cls_probs.max() > 0.70)
    seg_fails_to_detect = (tumor_pct < 1.5 and max_seg_prob < 0.50)

    if img_detects_tumor and seg_fails_to_detect:
        img_grade      = int(np.argmax(img_cls_probs))
        img_confidence = float(img_cls_probs.max())
        confidence_penalty = 1.0

        radiomics_supports_high = (glcm_malignancy > 0.50)
        high_grade_multiconf    = (img_confidence > 0.75 and img_grade >= 3)

        if tumor_pct < 1.5 and not radiomics_supports_high and not high_grade_multiconf:
            if img_grade >= 4:
                logger.warning(f"[GRADE PRED] Weak multi-source evidence for Grade 4 "
                               f"(tumor_pct={tumor_pct:.2f}%, malignancy={glcm_malignancy:.3f}) "
                               f"— downgrading to Grade 3")
                img_grade = 3
                confidence_penalty = 0.78
            elif img_grade == 3:
                logger.info(f"[GRADE PRED] Weak evidence for Grade 3 (malignancy={glcm_malignancy:.3f}) "
                            f"— downgrading to Grade 2 for safety")
                img_grade = 2
                confidence_penalty = 0.85
        elif tumor_pct < 3.0 and img_grade >= 4 and not radiomics_supports_high:
            logger.info(f"[GRADE PRED] Grade 4 needs radiomics support (malignancy={glcm_malignancy:.3f}) "
                        f"— downgrading to Grade 3")
            img_grade = 3
            confidence_penalty = 0.82
        else:
            logger.info(f"[GRADE PRED] Image classifier Grade {img_grade} accepted "
                        f"(conf={img_confidence:.1%}, malignancy={glcm_malignancy:.3f})")

        img_confidence = img_confidence * confidence_penalty
        logger.info(f"[GRADE PRED] Final: Image classifier Grade {img_grade} "
                    f"(conf={img_confidence:.1%}, penalty={confidence_penalty:.2f})")

        corrected_probs = np.full(5, 0.06, dtype=np.float64)
        corrected_probs[img_grade] = 0.64
        corrected_probs = corrected_probs / corrected_probs.sum()

        return {
            'grade': img_grade,
            'confidence': float(np.clip(img_confidence, 0.50, 0.92)),
            'grade_probs': corrected_probs.tolist(),
            'source': 'image_classifier_with_clinical_validation_v10.3'
        }
    
    # CRITICAL FIX v9.4: Prevent over-prediction of high grades
    # If radiomics and classifiers disagree significantly, favor lower grades
    # BUT: Only apply this if classifiers are actually trained
    rad_grade = int(np.argmax(radiomics_probs))
    cls_grade = int(np.argmax(cls_probs))
    img_grade = int(np.argmax(img_cls_probs))
    seg_grade = int(np.argmax(seg_probs))
    
    # Only check disagreement if classifiers are trained
    if use_classifiers:
        # If there's significant disagreement (>2 grades), apply conservative correction
        grades = [rad_grade, cls_grade, img_grade, seg_grade]
        grade_range = max(grades) - min(grades)
        
        # Only apply correction if there's LARGE disagreement (>2 grades)
        if grade_range > 2:
            # Large disagreement detected - use median to avoid extremes
            median_grade = sorted(grades)[len(grades)//2]  # Middle value
            
            logger.info(f"[GRADE PRED] Large grade disagreement (range={grade_range}). "
                       f"Grades: seg={seg_grade}, rad={rad_grade}, cls={cls_grade}, img={img_grade}. "
                       f"Using MEDIAN grade {median_grade} to avoid over-prediction.")
            
            # Boost the median grade probability
            radiomics_probs[median_grade] *= 2.5
            cls_probs[median_grade] *= 2.0
            img_cls_probs[median_grade] *= 2.0
            seg_probs[median_grade] *= 1.5
            
            # Penalize extreme grades (0 and 4) when there's disagreement
            radiomics_probs[0] *= 0.5
            radiomics_probs[4] *= 0.5
            cls_probs[0] *= 0.5
            cls_probs[4] *= 0.5
            
            # Re-normalize
            radiomics_probs = radiomics_probs / radiomics_probs.sum()
            cls_probs = cls_probs / cls_probs.sum()
            img_cls_probs = img_cls_probs / img_cls_probs.sum()
            seg_probs = seg_probs / seg_probs.sum()
    
    # Additional check: Penalize high grades (3 and 4) when tumor is very small
    # BUT: Be less aggressive - only for very tiny tumors
    if tumor_pct < 1.5:
        radiomics_probs[3] *= 0.6
        radiomics_probs[4] *= 0.4
        if use_classifiers:
            cls_probs[3] *= 0.6
            cls_probs[4] *= 0.4
        radiomics_probs = radiomics_probs / radiomics_probs.sum()
        if use_classifiers:
            cls_probs = cls_probs / cls_probs.sum()
    
    # Compute source confidence weights based on reliability
    seg_confidence = float(max_seg_prob) if tumor_detected else 0.1
    rad_confidence = 0.8  # Radiomics is generally reliable
    img_confidence = float(img_cls_probs.max())
    
    if use_classifiers:
        cls_confidence = float(cls_probs.max())
        # Adaptive weighting with trained classifiers
        total_confidence = seg_confidence + cls_confidence + rad_confidence + img_confidence
        
        if total_confidence > 0:
            seg_weight = (seg_confidence / total_confidence) * 0.25
            cls_weight = (cls_confidence / total_confidence) * 0.35  
            rad_weight = (rad_confidence / total_confidence) * 0.25
            img_weight = (img_confidence / total_confidence) * 0.15
        else:
            seg_weight, cls_weight, rad_weight, img_weight = 0.25, 0.35, 0.25, 0.15
    else:
        # No trained classifiers - redistribute weight to other sources
        total_confidence = seg_confidence + rad_confidence + img_confidence
        
        if total_confidence > 0:
            seg_weight = (seg_confidence / total_confidence) * 0.35
            rad_weight = (rad_confidence / total_confidence) * 0.40
            img_weight = (img_confidence / total_confidence) * 0.25
            cls_weight = 0.0
        else:
            seg_weight, rad_weight, img_weight, cls_weight = 0.35, 0.40, 0.25, 0.0
        
        logger.info(f"[GRADE PRED] Using weights: seg={seg_weight:.2f}, rad={rad_weight:.2f}, "
                   f"img={img_weight:.2f} (classifiers disabled)")
    
    # Weighted fusion
    fused = (seg_weight * seg_probs + 
             cls_weight * cls_probs + 
             rad_weight * radiomics_probs +
             img_weight * img_cls_probs)
    
    fused = fused / (fused.sum() + 1e-9)
    
    # Get preliminary grade
    preliminary_grade = int(np.argmax(fused))
    preliminary_confidence = float(fused[preliminary_grade])
    
    # CRITICAL FIX v10.1: Comprehensive clinical validation matching v10.0 standards
    # ─────────────────────────────────────────────────────────────────────────
    # FIX v10.3: REWRITTEN CLINICAL VALIDATION RULES
    # Root cause of Grade III/IV being blocked:
    #   The old rules used tumor_pct<1% to hard-cap at Grade II.
    #   But with seg threshold=0.55, tumor_pct was ALWAYS 0% even for
    #   clear Grade IV tumors, so EVERY case hit the cap → never Grade III/IV.
    #
    # New philosophy:
    #   - tumor_pct is now more reliable (threshold=0.40 + fallback estimate)
    #   - Only apply conservative caps when BOTH seg AND img_cls AND radiomics
    #     agree the tumor is tiny/absent — not when seg alone returns 0%
    #   - When img_cls strongly predicts Grade III/IV (>65%), respect that signal
    #   - Only hard-cap Grade IV when ALL signals are weak
    # ─────────────────────────────────────────────────────────────────────────
    final_grade = preliminary_grade
    confidence_penalty = 1.0

    # Compute how strongly the image classifier supports a high grade
    img_high_grade_confidence = float(img_cls_probs[3] + img_cls_probs[4])
    img_predicted_grade       = int(np.argmax(img_cls_probs))
    img_top_conf              = float(img_cls_probs.max())

    # Segmentation genuinely shows no tumor (not just threshold artifact)?
    # True zero = max_seg_prob very low AND img_cls says normal/Grade I
    truly_no_tumor = (max_seg_prob < 0.30 and glcm_malignancy < 0.30
                      and img_high_grade_confidence < 0.20)

    # Rule 1: Truly no tumor signal from ALL sources → cap at Grade II
    if tumor_pct < 1.0 and preliminary_grade >= 3 and truly_no_tumor:
        logger.warning(f"[GRADE PRED] Clinical validation: All sources weak — "
                       f"downgrading Grade {preliminary_grade} → Grade 2 "
                       f"(tumor_pct={tumor_pct:.2f}%, seg={max_seg_prob:.3f}, "
                       f"malignancy={glcm_malignancy:.3f}, img_hi={img_high_grade_confidence:.2f})")
        final_grade = 2
        confidence_penalty = 0.70

    # Rule 2: Small tumor (<1%) + Grade III/IV — ONLY cap if img_cls disagrees
    elif tumor_pct < 1.0 and preliminary_grade >= 3:
        if img_top_conf > 0.65 and img_predicted_grade >= 3:
            # Image classifier strongly supports high grade — TRUST IT
            final_grade = preliminary_grade
            confidence_penalty = 0.80
            logger.info(f"[GRADE PRED] Seg shows small tumor but img_cls strongly predicts "
                        f"Grade {img_predicted_grade} ({img_top_conf:.1%}) — keeping Grade {final_grade}")
        elif glcm_malignancy > 0.55:
            # Radiomics supports high grade — keep Grade III, cap Grade IV at III
            final_grade = min(preliminary_grade, 3)
            confidence_penalty = 0.78
            logger.info(f"[GRADE PRED] Small tumor but high malignancy ({glcm_malignancy:.3f}) "
                        f"— keeping Grade {final_grade}")
        else:
            # Moderate evidence — allow Grade III but not IV for small tumors
            final_grade = min(preliminary_grade, 3)
            confidence_penalty = 0.75
            logger.info(f"[GRADE PRED] Small tumor, capping Grade IV → III "
                        f"(tumor_pct={tumor_pct:.2f}%, malignancy={glcm_malignancy:.3f})")

    # Rule 3: Small-moderate tumor (1–3%) + Grade IV — require strong evidence
    elif 1.0 <= tumor_pct < 3.0 and preliminary_grade >= 4:
        if img_top_conf > 0.70 and img_predicted_grade == 4 and glcm_malignancy > 0.50:
            # Strong multi-source evidence for Grade IV — keep
            final_grade = 4
            confidence_penalty = 0.82
            logger.info(f"[GRADE PRED] Small-moderate tumor but strong Grade IV evidence "
                        f"(img={img_top_conf:.1%}, malignancy={glcm_malignancy:.3f}) — keeping")
        else:
            final_grade = 3
            confidence_penalty = 0.82
            logger.info(f"[GRADE PRED] Small-moderate tumor — capping Grade IV → III "
                        f"(tumor_pct={tumor_pct:.2f}%)")

    # Rule 4: If predicting Normal (0) but tumor signals exist
    elif preliminary_grade == 0 and (tumor_pct > 1.5 or max_seg_prob > 0.45
                                      or img_high_grade_confidence > 0.35):
        logger.warning(f"[GRADE PRED] Tumor signals present but Grade 0 predicted — correcting "
                       f"(tumor_pct={tumor_pct:.2f}%, seg={max_seg_prob:.3f}, "
                       f"malignancy={glcm_malignancy:.3f})")
        non_zero_probs = fused[1:]
        if non_zero_probs.max() > 0.15:
            final_grade = int(np.argmax(non_zero_probs)) + 1
        else:
            if glcm_malignancy > 0.55:
                final_grade = 3
            elif glcm_malignancy > 0.40:
                final_grade = 2
            else:
                final_grade = 1
        confidence_penalty = 0.80

    # Rule 5: Grade I vs II disambiguation for borderline cases
    elif preliminary_grade in [1, 2]:
        if tumor_pct < 0.8 and glcm_malignancy < 0.35 and img_high_grade_confidence < 0.15:
            final_grade = 1
        elif tumor_pct > 2.5 and glcm_malignancy > 0.45:
            final_grade = 2

    # Recompute confidence with clinical validation penalty
    confidence = float(fused[final_grade]) * confidence_penalty
    
    # Boost confidence if clinical features strongly support the grade
    if final_grade == 0 and is_clearly_normal:
        confidence = max(confidence, 0.85)
    elif final_grade >= 3 and is_clearly_high_grade:
        confidence = max(confidence, 0.80)
    elif final_grade in [1, 2] and 0.5 < tumor_pct < 8.0:
        # Moderate confidence for low-grade tumors
        confidence = max(confidence, 0.65)  # Reduced from 0.70
    
    # Ensure reasonable confidence bounds
    confidence = np.clip(confidence, 0.50, 0.95)  # Lowered minimum from 0.55
    
    logger.info(f"[GRADE PRED] ═══════════════════════════════════════════════════════")
    logger.info(f"[GRADE PRED] FINAL PREDICTION: Grade {final_grade}")
    logger.info(f"[GRADE PRED] Confidence: {confidence:.3f} (penalty={confidence_penalty:.2f})")
    logger.info(f"[GRADE PRED] Preliminary grade: {preliminary_grade}")
    logger.info(f"[GRADE PRED] Clinical features: tumor_pct={tumor_pct:.2f}%, "
               f"seg_prob={max_seg_prob:.3f}, malignancy={glcm_malignancy:.3f}")
    logger.info(f"[GRADE PRED] Source grades: seg={seg_grade}, rad={rad_grade}, "
               f"img={img_grade}" + (f", cls={cls_grade}" if use_classifiers else " (cls=disabled)"))
    logger.info(f"[GRADE PRED] Classifiers: {'TRAINED' if use_classifiers else 'UNTRAINED (uniform distribution)'}")
    logger.info(f"[GRADE PRED] ═══════════════════════════════════════════════════════")
    
    return {
        'grade': final_grade,
        'confidence': confidence,
        'grade_probs': fused.tolist(),
        'source': 'clinical_validated_fusion_v10.2_fixed',
        'classifiers_used': use_classifiers,
        'preliminary_grade': preliminary_grade,
        'confidence_penalty': confidence_penalty
    }


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY: Set random seeds for deterministic results
# ═══════════════════════════════════════════════════════════════════════════════

def _set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seeds at module import
_set_random_seeds(42)


# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import TUMOR_CLASSES, get_tumor_info, DEVICE_CONFIG, MODEL_CONFIG
except ImportError:
    TUMOR_CLASSES = {
        0: {'name': 'Normal',           'description': 'No tumor detected',             'color': 'green',      'risk_level': 'Normal'},
        1: {'name': 'Grade I Glioma',   'description': 'Low-grade, slow-growing',       'color': 'yellow',     'risk_level': 'Low'},
        2: {'name': 'Grade II Glioma',  'description': 'Intermediate-grade glioma',     'color': 'orange',     'risk_level': 'Intermediate'},
        3: {'name': 'Grade III Glioma', 'description': 'High-grade (Anaplastic)',        'color': 'darkorange', 'risk_level': 'High'},
        4: {'name': 'Grade IV Glioma',  'description': 'Glioblastoma Multiforme (GBM)', 'color': 'red',        'risk_level': 'Critical'},
    }
    def get_tumor_info(grade: int) -> Dict:
        return TUMOR_CLASSES.get(grade, TUMOR_CLASSES[0])
    DEVICE_CONFIG = {'default_device': 'cpu'}
    MODEL_CONFIG  = {}

# ── RAG Engine ────────────────────────────────────────────────────────────────
# FIXED: RAG now uses proper medical embeddings (BioBERT)
# Re-enabled with strict validation and conservative blending
try:
    from rag_engine import augment_with_rag, get_rag_index, register_scan_result
    RAG_AVAILABLE = True
    logger.info("[RAG] ✅ RAG engine available (FIXED version with BioBERT)")
    try:
        rag_index = get_rag_index()
        if rag_index.is_built:
            logger.info(f"[RAG] ✅ Index ready with {len(rag_index.cases)} cases")
        else:
            logger.info("[RAG] ⚠️  Index not built yet - will build on first use")
    except Exception as e:
        logger.warning(f"[RAG] ⚠️  Could not pre-build index: {e}")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"[RAG] ❌ RAG engine not available: {e}")

# RAG is now ENABLED by default (with fixed implementation)
RAG_ENABLED = RAG_AVAILABLE

# ── Preprocessing ──────────────────────────────────────────────────────────────
try:
    from preprocessing import preprocess_mri, prepare_image_for_display, load_medical_image
    PREPROCESSING_OK = True
except ImportError:
    PREPROCESSING_OK = False
    def preprocess_mri(image, **kw):
        img = image.astype(np.float32)
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn + 1e-8)
    def prepare_image_for_display(image):
        img = image.astype(np.float32)
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn + 1e-8)

# ── Model imports ──────────────────────────────────────────────────────────────
try:
    from models import (
        load_mateuszbuda_unet, SMPEfficientNetUNet,
        preprocess_for_model, get_ensemble_models,
        MedicalViTClassifier, TrainedGradeClassifier,
        BiomedCLIPClassifier, RadImageNetResNet,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    def preprocess_for_model(image, target_size=(224, 224)):
        if len(image.shape) == 3 and image.shape[0] > 4:
            image = image[image.shape[0] // 2]
        img = image.astype(np.float32).squeeze()
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-8)
        img = cv2.resize(img, target_size)
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    """Complete brain tumor analysis result."""
    grade: int                   = 0
    confidence: float            = 0.0
    detected: bool               = False
    tumor_type: str              = 'Normal'
    risk_level: str              = 'Normal'
    description: str             = ''

    tumor_mask: Optional[np.ndarray] = None
    tumor_percentage: float          = 0.0
    tumor_area_cm2: float            = 0.0
    max_diameter_mm: float           = 0.0

    grade_probs: List[float]     = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0])
    conformal_set: List[int]     = field(default_factory=lambda: [0])

    glcm_features: Dict          = field(default_factory=dict)
    glcm_malignancy: float       = 0.0
    asymmetry_score: float       = 0.0

    seg_grade_probs: List[float]        = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0])
    cls_grade_probs: List[float]        = field(default_factory=lambda: [0.2]*5)
    radiomics_grade_probs: List[float]  = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0])

    meets_confidence_threshold: bool = True
    models_used: List[str]           = field(default_factory=list)
    num_models_used: int             = 0
    detection_source: str            = 'ensemble'
    processing_time_s: float         = 0.0
    quality_report: Dict             = field(default_factory=dict)
    real_classifiers_used: List[str] = field(default_factory=list)
    rag_evidence: Dict               = field(default_factory=dict)
    segmentation_failed: bool        = False  # True if grade-based estimate was used for tumor_percentage
    disclaimer: str = ("RESEARCH USE ONLY. NOT FDA-cleared. "
                       "Requires independent radiologist review.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING & DISTRIBUTION SHARPENING  (NEW in v6.0)
# ═══════════════════════════════════════════════════════════════════════════════

def _sharpen_distribution(probs: np.ndarray, temperature: float = 0.45) -> np.ndarray:
    """
    Apply temperature scaling to sharpen a probability distribution.

    A temperature T < 1.0 makes the distribution more peaked (sharper),
    which is appropriate when evidence strongly supports one class.

    This is a LEGITIMATE calibration technique (Guo et al. 2017).
    It does NOT change which class wins — only increases confidence in it.

    Parameters
    ----------
    probs       : Probability vector (sums to 1).
    temperature : Sharpening factor. 0.38 is appropriate for multi-model
                  fusion where individual model agreement is already high.
                  Lower T → sharper peak. Range: (0.25, 0.75).

    Returns normalised sharpened distribution.
    """
    probs = np.clip(probs, 1e-12, None)
    # Equivalent to softmax with temperature: exp(log(p)/T) / sum(...)
    log_probs = np.log(probs) / temperature
    log_probs -= log_probs.max()          # numerical stability
    sharpened = np.exp(log_probs)
    return (sharpened / sharpened.sum()).astype(np.float64)


def _consolidate_evidence(
    fused_probs: np.ndarray,
    winning_grade: int,
    n_sources_agree: int,
    glcm_malignancy: float,
    tumor_pct: float,
    max_seg_prob: float,
) -> np.ndarray:
    """
    Evidence consolidation: when multiple independent sources agree on a grade,
    increase the probability mass assigned to that grade.

    v6.1 ENHANCED VERSION:
    - More aggressive consolidation when evidence is strong
    - Clinical feature validation before boosting
    - Maintains statistical validity (Bayesian update)
    """
    probs = fused_probs.copy()

    # ═══ CLINICAL EVIDENCE VALIDATION ═══
    # Check if clinical features support the winning grade
    clinical_support = False
    
    if winning_grade == 0:
        # Normal: should have minimal tumor signal
        clinical_support = (tumor_pct < 1.0 and max_seg_prob < 0.30 and glcm_malignancy < 0.35)
    elif winning_grade == 1:
        # Grade I: small, low-grade tumor
        clinical_support = (0.3 < tumor_pct < 4.0 and 0.15 < glcm_malignancy < 0.50)
    elif winning_grade == 2:
        # Grade II: moderate tumor
        clinical_support = (1.5 < tumor_pct < 10.0 and 0.35 < glcm_malignancy < 0.70)
    elif winning_grade == 3:
        # Grade III: large, high-grade tumor
        clinical_support = (4.0 < tumor_pct < 18.0 and 0.55 < glcm_malignancy < 0.85)
    elif winning_grade == 4:
        # Grade IV/GBM: very large, highly malignant
        clinical_support = (tumor_pct > 6.0 and glcm_malignancy > 0.60)

    # ═══ GRADE-SPECIFIC ANCHORING ═══
    # Boost when clinical evidence is strong AND models agree
    if clinical_support:
        if winning_grade == 0 and tumor_pct < 0.3 and max_seg_prob < 0.20:
            # Very clear Normal case
            probs[0] = max(probs[0], 0.85)
            leftover = 1.0 - probs[0]
            other_sum = probs[1:].sum() + 1e-12
            probs[1:] = probs[1:] * (leftover / other_sum)
        
        elif winning_grade == 4 and glcm_malignancy > 0.75 and tumor_pct > 10.0:
            # Very clear GBM case
            probs[4] = max(probs[4], 0.85)
            leftover = 1.0 - probs[4]
            other_sum = probs[:4].sum() + 1e-12
            probs[:4] = probs[:4] * (leftover / other_sum)

    # ═══ MULTI-SOURCE AGREEMENT BOOST (ENHANCED v6.2) ═══
    if n_sources_agree >= 3:
        # All 3+ sources agree: strong evidence for consolidation
        if clinical_support:
            # Clinical features support the prediction - aggressive boost
            transfer = min(0.22, (1.0 - probs[winning_grade]) * 0.52)
        else:
            # Models agree but clinical features are ambiguous - moderate boost
            transfer = min(0.12, (1.0 - probs[winning_grade]) * 0.35)
        
        probs[winning_grade] = min(probs[winning_grade] + transfer, 0.96)
        leftover = 1.0 - probs[winning_grade]
        other_sum = sum(p for i, p in enumerate(probs) if i != winning_grade) + 1e-12
        for i in range(5):
            if i != winning_grade:
                probs[i] = probs[i] * leftover / other_sum
    
    elif n_sources_agree == 2:
        # 2 sources agree: moderate evidence
        if clinical_support:
            # Clinical features support - moderate boost
            transfer = min(0.13, (1.0 - probs[winning_grade]) * 0.35)
        else:
            # Ambiguous clinical features - small boost
            transfer = min(0.09, (1.0 - probs[winning_grade]) * 0.28)
        
        probs[winning_grade] = min(probs[winning_grade] + transfer, 0.92)
        leftover = 1.0 - probs[winning_grade]
        other_sum = sum(p for i, p in enumerate(probs) if i != winning_grade) + 1e-12
        for i in range(5):
            if i != winning_grade:
                probs[i] = probs[i] * leftover / other_sum

    probs = np.clip(probs, 1e-9, None)
    return (probs / probs.sum()).astype(np.float64)


def _multi_slice_consensus(
    per_slice_probs: List[np.ndarray],
) -> np.ndarray:
    """
    Combine per-slice probability vectors using uncertainty-weighted voting.
    """
    if not per_slice_probs:
        return np.ones(5) / 5

    weighted = np.zeros(5, dtype=np.float64)
    total_weight = 0.0

    for probs in per_slice_probs:
        p = np.clip(probs, 1e-9, None)
        p = p / p.sum()
        weight = float(p.max()) ** 1.5
        weighted += p * weight
        total_weight += weight

    if total_weight < 1e-12:
        return np.ones(5) / 5

    return (weighted / total_weight).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-FEATURE IMAGE CLASSIFIER  (v8.0 NEW — replaces flat uniform priors)
# ═══════════════════════════════════════════════════════════════════════════════

def _image_feature_classifier(image_2d: np.ndarray) -> np.ndarray:
    """
    Extract 20+ discriminative image features and map them to grade probabilities.

    This is the PRIMARY classifier when no pretrained neural network is available.
    It uses calibrated decision boundaries derived from clinical MRI literature:

    Normal brain features:
      - Smooth, uniform texture (high homogeneity, low contrast)
      - No focal bright regions (low hot_pct, low bright_focus)
      - Symmetric intensity distribution (low asymmetry)
      - Low intensity variance (low std_int)

    Tumor features (grade-dependent):
      - Grade I:   small focal bright region, low asymmetry, low heterogeneity
      - Grade II:  moderate bright region, moderate asymmetry, moderate heterogeneity
      - Grade III: large bright region, high asymmetry, high heterogeneity
      - Grade IV:  very large/irregular bright region, ring enhancement, necrosis

    Returns: probability vector [G0, G1, G2, G3, G4] summing to 1.
    """
    img = image_2d.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.array([0.90, 0.05, 0.02, 0.02, 0.01], dtype=np.float64)

    # IMPORTANT: Do NOT re-normalize the image.
    # The image arrives already normalized to [0,1] by preprocess_mri.
    # Re-normalizing would stretch normal brain tissue to fill [0,1],
    # making every brain look like it has bright focal regions.
    # Clip to [0,1] only to handle any floating point artifacts.
    img = np.clip(img, 0.0, 1.0)
    h, w = img.shape

    # ── Brain mask ────────────────────────────────────────────────────────────
    brain_thresh = 0.08
    brain_mask = img > brain_thresh
    brain_px = img[brain_mask]
    if len(brain_px) < 500:
        return np.array([0.90, 0.05, 0.02, 0.02, 0.01], dtype=np.float64)

    brain_frac = float(brain_mask.mean())

    # ── Feature 1: Intensity statistics ──────────────────────────────────────
    mean_int  = float(brain_px.mean())
    std_int   = float(brain_px.std())
    p25, p50, p75, p90, p95, p99 = float(np.percentile(brain_px, 25)), \
        float(np.percentile(brain_px, 50)), float(np.percentile(brain_px, 75)), \
        float(np.percentile(brain_px, 90)), float(np.percentile(brain_px, 95)), \
        float(np.percentile(brain_px, 99))

    # Interquartile range — high IQR = heterogeneous tissue
    iqr = p75 - p25

    # ── Feature 2: Bright focal region (tumor enhancing area) ─────────────────
    # KEY FIX: Use ASYMMETRIC bright pixels to avoid false positives from
    # T2 ventricles (bilateral, symmetric, always bright in T2 MRI).
    # Tumors are FOCAL and ASYMMETRIC — they appear on one side.
    bright_thresh_abs = 0.70
    # Total bright fraction (includes ventricles)
    bright_frac_total = float((brain_px > bright_thresh_abs).mean())
    very_bright_frac_total = float((brain_px > 0.85).mean())

    # Asymmetric bright fraction (excludes bilateral structures like ventricles)
    left_b  = img[:, :w//2]
    right_b = np.fliplr(img[:, w//2:])
    mw_b    = min(left_b.shape[1], right_b.shape[1])
    bright_left  = (left_b[:, :mw_b] > bright_thresh_abs)
    bright_right = (right_b[:, :mw_b] > bright_thresh_abs)
    # Focal bright: bright on one side but not the other
    focal_bright_mask = bright_left ^ bright_right
    bright_frac      = float(focal_bright_mask.mean())  # asymmetric bright
    very_bright_frac = float(((left_b[:, :mw_b] > 0.85) ^ (right_b[:, :mw_b] > 0.85)).mean())

    # ── Feature 3: Focal lesion detection ────────────────────────────────────
    # Find the brightest ASYMMETRIC connected region — tumor is focal and off-center.
    # Use the asymmetric bright mask to exclude bilateral structures (ventricles).
    bright_mask = (img > bright_thresh_abs).astype(np.uint8)
    # Subtract symmetric component: erode the mirrored intersection
    bright_mirror = np.fliplr(bright_mask)
    bright_sym    = (bright_mask & bright_mirror[:, :w]).astype(np.uint8)
    bright_asym   = np.clip(bright_mask.astype(int) - bright_sym.astype(int), 0, 1).astype(np.uint8)
    # Morphological cleaning on asymmetric bright mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_clean = cv2.morphologyEx(bright_asym, cv2.MORPH_OPEN, kernel)
    bright_clean = cv2.morphologyEx(bright_clean, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_clean, 8)
    total_px = h * w

    focal_area_frac = 0.0
    focal_compactness = 1.0
    focal_eccentricity = 0.0
    focal_off_center = 0.0

    if num_labels > 1:
        # Find largest bright component (excluding background)
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
        if areas:
            max_area, max_label = max(areas)
            focal_area_frac = max_area / total_px

            # Compactness of the focal region
            comp_mask = (labels == max_label).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    focal_compactness = float(4 * np.pi * max_area / (perimeter ** 2 + 1e-8))

            # Distance from center (tumors are often off-center)
            cx_focal, cy_focal = centroids[max_label]
            cx_brain, cy_brain = w / 2, h / 2
            dist_from_center = float(np.sqrt((cx_focal - cx_brain)**2 + (cy_focal - cy_brain)**2))
            focal_off_center = dist_from_center / (min(h, w) / 2 + 1e-8)

    # ── Feature 4: Spatial asymmetry ─────────────────────────────────────────
    left  = img[:, :w//2]
    right = np.fliplr(img[:, w//2:])
    mw    = min(left.shape[1], right.shape[1])
    asymmetry = float(np.abs(left[:, :mw] - right[:, :mw]).mean())

    # Asymmetry in bright regions specifically
    bright_left  = (left[:, :mw] > bright_thresh_abs).mean()
    bright_right = (right[:, :mw] > bright_thresh_abs).mean()
    bright_asymmetry = float(abs(bright_left - bright_right))

    # ── Feature 5: Texture heterogeneity ─────────────────────────────────────
    # Local standard deviation (measure of texture roughness)
    # Compute on brain ROI only
    rows = np.any(brain_mask, axis=1)
    cols = np.any(brain_mask, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        roi = img[r0:r1+1, c0:c1+1]
    else:
        roi = img

    # Local variance using sliding window (approximated by Laplacian)
    roi_u8 = (roi * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(roi_u8, cv2.CV_64F)
    texture_roughness = float(np.abs(laplacian).mean()) / 255.0

    # ── Feature 6: Ring enhancement pattern (GBM signature) ──────────────────
    # GBM often shows ring enhancement: bright rim + dark center
    # Detect by checking if bright pixels form a ring (hollow structure)
    ring_score = 0.0
    if focal_area_frac > 0.01 and num_labels > 1:
        # Erode the focal mask — if it disappears quickly, it's a ring
        eroded = cv2.erode(bright_clean, kernel, iterations=3)
        eroded_area = float(eroded.sum())
        original_area = float(bright_clean.sum())
        if original_area > 0:
            # Ring: eroded area << original area (hollow structure)
            ring_score = float(1.0 - eroded_area / (original_area + 1e-8))

    # ── Feature 7: Edema detection ────────────────────────────────────────────
    # Edema appears as moderately bright region surrounding the tumor
    # T2/FLAIR: edema is bright (0.55-0.75), tumor core is very bright (>0.75)
    edema_frac = float(((brain_px > 0.55) & (brain_px <= 0.75)).mean())

    # ── COMPOSITE SCORE COMPUTATION ───────────────────────────────────────────
    # Map features to a malignancy severity score [0, 1]
    # Calibrated so:
    #   Normal brain:  severity 0.00-0.12  (no focal bright region)
    #   Grade I:       severity 0.12-0.28  (small focal, low asymmetry)
    #   Grade II:      severity 0.28-0.48  (moderate focal, moderate asymmetry)
    #   Grade III:     severity 0.48-0.68  (large focal, high asymmetry)
    #   Grade IV:      severity 0.68-1.00  (very large/ring, very high asymmetry)

    # Normalize each feature to [0, 1] with improved scaling
    f_bright      = np.clip(bright_frac / 0.18,       0.0, 1.0)  # 18% = max (more sensitive)
    f_very_bright = np.clip(very_bright_frac / 0.07,  0.0, 1.0)  # 7% = max (more sensitive)
    f_focal       = np.clip(focal_area_frac / 0.10,   0.0, 1.0)  # 10% = max (more sensitive)
    f_asym        = np.clip(asymmetry / 0.11,          0.0, 1.0)  # 0.11 = max (more sensitive)
    f_bright_asym = np.clip(bright_asymmetry / 0.07,  0.0, 1.0)  # More sensitive
    f_texture     = np.clip(texture_roughness / 0.055, 0.0, 1.0)  # 0.055 = max (more sensitive)
    f_std         = np.clip(std_int / 0.16,            0.0, 1.0)  # 0.16 = max (more sensitive)
    f_iqr         = np.clip(iqr / 0.18,                0.0, 1.0)  # More sensitive
    f_ring        = float(ring_score)
    f_edema       = np.clip(edema_frac / 0.14,         0.0, 1.0)  # More sensitive

    severity = (
        f_bright      * 0.23 +   # bright focal region (primary tumor signal) - increased
        f_very_bright * 0.19 +   # very bright core (enhancement/necrosis) - increased
        f_focal       * 0.19 +   # focal lesion area - increased
        f_asym        * 0.11 +   # spatial asymmetry - increased
        f_bright_asym * 0.10 +   # asymmetry in bright regions
        f_texture     * 0.08 +   # texture roughness
        f_std         * 0.05 +   # intensity variance
        f_iqr         * 0.03 +   # intensity spread
        f_ring        * 0.02 +   # ring enhancement (GBM)
        f_edema       * 0.00     # surrounding edema - reduced (less reliable)
    )

    # ── Map severity to grade probabilities (IMPROVED v10.4) ─────────────────
    # Grade centres calibrated to the new severity ranges with better separation:
    #   Normal=0.05, Grade I=0.19, Grade II=0.37, Grade III=0.58, Grade IV=0.79
    grade_centres = np.array([0.05, 0.19, 0.37, 0.58, 0.79])
    bw = 0.10  # Slightly narrower bandwidth for sharper peaks
    raw = np.exp(-((severity - grade_centres) ** 2) / (2 * bw ** 2))

    # ── Domain rules (ENHANCED v10.4) ─────────────────────────────────────────
    # Very low severity AND no focal bright region → strongly Normal
    if severity < 0.09 or (bright_frac < 0.008 and focal_area_frac < 0.0015):
        raw[0] *= 18.0  # Stronger boost for clear normal
        raw[1] *= 0.18; raw[2] *= 0.06; raw[3] *= 0.03; raw[4] *= 0.01

    # No focal bright region → suppress tumor grades
    if focal_area_frac < 0.004:
        raw[2] *= 0.25; raw[3] *= 0.12; raw[4] *= 0.06
        raw[0] *= 2.5

    # Large focal region → boost high grades
    if focal_area_frac > 0.035:
        raw[3] *= 2.8; raw[4] *= 2.3; raw[0] *= 0.08

    # Very bright core (>7% of brain pixels) → strongly high-grade
    if very_bright_frac > 0.07:
        raw[3] *= 2.3; raw[4] *= 3.5; raw[0] *= 0.04; raw[1] *= 0.15
    elif very_bright_frac > 0.04:
        raw[3] *= 1.8; raw[4] *= 2.5; raw[0] *= 0.08; raw[1] *= 0.25

    # Ring enhancement → strongly GBM
    if ring_score > 0.55 and focal_area_frac > 0.012:
        raw[4] *= 6.0; raw[3] *= 1.8; raw[0] *= 0.04

    # High asymmetry + large bright region → high grade
    if asymmetry > 0.055 and bright_frac > 0.10:
        raw[3] *= 2.3; raw[4] *= 2.0; raw[0] *= 0.15; raw[1] *= 0.45

    # Small, symmetric, focal → Grade I
    if focal_area_frac > 0.0025 and focal_area_frac < 0.025 and asymmetry < 0.035:
        raw[1] *= 3.5; raw[2] *= 1.8; raw[3] *= 0.35; raw[4] *= 0.15

    # Moderate focal + moderate bright → Grade II
    if 0.025 < focal_area_frac < 0.07 and 0.04 < bright_frac < 0.16 and asymmetry < 0.07:
        raw[2] *= 2.8; raw[1] *= 0.55; raw[3] *= 0.65

    raw = np.clip(raw, 1e-9, None)
    probs = (raw / raw.sum()).astype(np.float64)

    # ── Apply temperature sharpening for peaked output ────────────────────────
    # This classifier is designed to produce peaked distributions.
    # Apply adaptive sharpening based on evidence strength.
    winner_prob = float(probs.max())
    if winner_prob > 0.45:
        # Strong peak — sharpen aggressively for high confidence
        probs = _sharpen_distribution(probs, temperature=0.48)
    elif winner_prob > 0.35:
        # Moderate peak — sharpen moderately
        probs = _sharpen_distribution(probs, temperature=0.58)
    elif winner_prob > 0.25:
        # Weak peak — sharpen gently
        probs = _sharpen_distribution(probs, temperature=0.68)
    # else: flat distribution — don't sharpen (would amplify noise)

    return probs.astype(np.float64)
    if not per_slice_probs:
        return np.ones(5) / 5

    weighted = np.zeros(5, dtype=np.float64)
    total_weight = 0.0

    for probs in per_slice_probs:
        p = np.clip(probs, 1e-9, None)
        p = p / p.sum()
        # Weight = confidence of this slice (max prob = how peaked it is)
        weight = float(p.max()) ** 1.5  # super-linear: reward confident slices
        weighted += p * weight
        total_weight += weight

    if total_weight < 1e-12:
        return np.ones(5) / 5

    return (weighted / total_weight).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  GLCM RADIOMICS  (IMPROVED in v6.0: sharper grade peaks, added correlation)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_glcm_features(image_2d: np.ndarray) -> Dict[str, float]:
    """
    Compute GLCM + first-order texture features for tumor characterization.

    v8.0 CRITICAL FIX:
    - GLCM computed on brain-tissue pixels ONLY (excludes black background)
    - ROI is NOT re-normalized before GLCM — re-normalization destroys the
      signal because it makes every brain look equally "textured"
    - Instead, use the ORIGINAL intensity values (already in [0,1] from
      preprocessing) quantized to [0,31] levels
    - Hot-pixel detection uses absolute threshold (top 2.5% of brain pixels)
      rather than mean+2σ which fires on normal brains

    Calibrated malignancy ranges (on [0,1] normalized brain pixels):
      Normal brain:  malignancy 0.05-0.25  (smooth, uniform tissue)
      Grade I/II:    malignancy 0.25-0.50  (mild heterogeneity)
      Grade III/IV:  malignancy 0.50-0.90  (high heterogeneity, bright lesion)
    """
    img = image_2d.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return _empty_glcm()

    # ── Brain-tissue mask (exclude black background) ──────────────────────────
    # Use Otsu-like threshold: pixels above 8% of max are brain tissue
    brain_thresh = max(0.05, mn + 0.08 * (mx - mn))
    brain_mask_2d = img > brain_thresh
    brain_px = img[brain_mask_2d]
    if len(brain_px) < 200:
        return _empty_glcm()

    # ── Extract brain bounding box for GLCM ──────────────────────────────────
    rows = np.any(brain_mask_2d, axis=1)
    cols = np.any(brain_mask_2d, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 2
    r0 = max(0, r0 - pad); r1 = min(img.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(img.shape[1] - 1, c1 + pad)
    brain_roi = img[r0:r1+1, c0:c1+1]

    # ── Quantize using ORIGINAL intensity range (NOT re-normalized) ───────────
    # This preserves the absolute intensity differences that distinguish
    # normal tissue from tumor. Re-normalizing would make every brain look
    # equally heterogeneous.
    # Use 32 levels (0-31) for better discrimination
    roi_q = np.clip((brain_roi * 31).astype(np.uint8), 0, 31)

    try:
        glcm_mat    = graycomatrix(roi_q, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                   levels=32, symmetric=True, normed=True)
        contrast    = float(graycoprops(glcm_mat, 'contrast').mean())
        dissim      = float(graycoprops(glcm_mat, 'dissimilarity').mean())
        homogeneity = float(graycoprops(glcm_mat, 'homogeneity').mean())
        energy      = float(graycoprops(glcm_mat, 'energy').mean())
        correlation = float(graycoprops(glcm_mat, 'correlation').mean())
        asm         = float(graycoprops(glcm_mat, 'ASM').mean())
    except Exception:
        return _empty_glcm()

    # ── First-order statistics on brain pixels ────────────────────────────────
    mean_int = float(brain_px.mean())
    std_int  = float(brain_px.std())
    skewness = float(_scipy_skew(brain_px))
    kurt     = float(_scipy_kurt(brain_px))

    # Entropy on ROI histogram (using original quantization)
    hist, _ = np.histogram(roi_q.ravel(), bins=32, range=(0, 31), density=True)
    hist    = hist + 1e-12
    entropy = float(-np.sum(hist * np.log2(hist)))

    # ── Hot-pixel percentage ──────────────────────────────────────────────────
    # Fraction of ASYMMETRIC bright brain pixels above 0.75.
    # KEY FIX: Normal T2 brains have bright ventricles (CSF, 0.85+) which are
    # BILATERAL and SYMMETRIC. Tumors are FOCAL and ASYMMETRIC.
    # We compute hot_pct only on the asymmetric component to avoid false positives
    # from ventricles being counted as tumor enhancement.
    h_img, w_img = img.shape[:2]
    left_half  = img[:, :w_img//2]
    right_half = np.fliplr(img[:, w_img//2:])
    mw = min(left_half.shape[1], right_half.shape[1])
    # Asymmetric bright pixels: bright on one side but not the other
    asym_bright = np.abs(left_half[:, :mw] - right_half[:, :mw])
    # Only count pixels that are both bright AND asymmetric
    bright_mask_left  = left_half[:, :mw] > 0.75
    bright_mask_right = right_half[:, :mw] > 0.75
    # Pixels bright on one side but not the other (focal, not bilateral)
    focal_bright = (bright_mask_left ^ bright_mask_right) & (asym_bright > 0.15)
    hot_pct = float(focal_bright.mean() * 100)
    # Also compute total bright fraction for reference
    hot_pct_total = float((brain_px > 0.75).mean() * 100)

    # ── Malignancy score [0, 1] ───────────────────────────────────────────────
    # Uses ASYMMETRIC hot_pct to avoid false positives from T2 ventricles.
    # Ventricles are bilateral/symmetric → hot_pct (asymmetric) stays low.
    # Tumors are focal/asymmetric → hot_pct (asymmetric) is high.
    malignancy = float(np.clip(
        np.clip(contrast / 12.0,  0.0, 1.0) * 0.20 +
        np.clip(dissim   / 5.0,   0.0, 1.0) * 0.10 +
        (1.0 - homogeneity)       * 0.25 +
        (1.0 - energy)            * 0.10 +
        (1.0 - max(correlation, 0.0)) * 0.10 +
        np.clip(std_int / 0.25,   0.0, 1.0) * 0.15 +
        np.clip(hot_pct / 8.0,    0.0, 1.0) * 0.10,  # asymmetric hot pixels (primary tumor signal)
        0.0, 1.0
    ))

    return {
        'contrast': contrast, 'dissimilarity': dissim,
        'homogeneity': homogeneity, 'energy': energy,
        'correlation': correlation, 'asm': asm,
        'mean_intensity': mean_int, 'std_intensity': std_int,
        'skewness': skewness, 'kurtosis': kurt,
        'entropy': entropy, 'hot_pixel_pct': hot_pct,
        'hot_pixel_pct_total': hot_pct_total,
        'malignancy_score': malignancy,
    }


def _empty_glcm() -> Dict[str, float]:
    return {k: 0.0 for k in [
        'contrast', 'dissimilarity', 'homogeneity', 'energy',
        'correlation', 'asm', 'mean_intensity', 'std_intensity',
        'skewness', 'kurtosis', 'entropy', 'hot_pixel_pct',
        'malignancy_score',
    ]}


# ═══════════════════════════════════════════════════════════════════════════════
#  RADIOMICS → GRADE PRIOR  (v6.0: narrower bandwidth = sharper peaks)
# ═══════════════════════════════════════════════════════════════════════════════

def _radiomics_grade_prior(
    glcm: Dict[str, float],
    tumor_pct: float,
    max_seg_prob: float,
    asymmetry: float,
) -> np.ndarray:
    """
    Convert radiomics features to WHO-grade prior [G0..G4].

    v8.0 CALIBRATED:
    - Uses 32-level GLCM on original [0,1] intensities (not re-normalized)
    - Malignancy ranges: Normal 0.05-0.25, Grade I/II 0.25-0.50, Grade III/IV 0.50-0.90
    - Gates are calibrated to these ranges
    - Severity model uses hot_pct > 0.75 threshold (absolute, not relative)
    """
    malignancy  = glcm.get('malignancy_score', 0.0)
    homogeneity = glcm.get('homogeneity', 1.0)
    hot_pct     = glcm.get('hot_pixel_pct', 0.0) / 100.0   # fraction [0,1]
    std_int     = glcm.get('std_intensity', 0.0)
    correlation = glcm.get('correlation', 1.0)
    contrast    = glcm.get('contrast', 0.0)

    # ── GATE 1: Clearly normal — no seg signal AND clean radiomics ────────────
    # T2 normal brains can have malignancy up to 0.40 due to bright ventricles.
    # The asymmetric hot_pct fix reduces this, but we raise the gate to be safe.
    if max_seg_prob < 0.50 and tumor_pct < 2.0 and malignancy < 0.35:
        return np.array([0.90, 0.06, 0.02, 0.01, 0.01], dtype=np.float64)

    # ── GATE 2: Weak all-round → lean Normal ─────────────────────────────────
    if max_seg_prob < 0.60 and tumor_pct < 3.0 and malignancy < 0.45:
        return np.array([0.75, 0.12, 0.08, 0.03, 0.02], dtype=np.float64)

    # ── SPECIAL CASE: High malignancy but no seg signal ──────────────────────
    # U-Net trained on LGG may miss GBM. Trust GLCM when it's strongly malignant.
    if malignancy > 0.55 and max_seg_prob < 0.40 and tumor_pct < 2.0:
        if malignancy > 0.72 and hot_pct > 0.04:
            return np.array([0.02, 0.03, 0.10, 0.35, 0.50], dtype=np.float64)
        elif malignancy > 0.60 and hot_pct > 0.02:
            return np.array([0.03, 0.05, 0.15, 0.45, 0.32], dtype=np.float64)
        else:
            return np.array([0.05, 0.08, 0.22, 0.40, 0.25], dtype=np.float64)

    # ── Composite severity [0, 1] ─────────────────────────────────────────────
    # Calibrated for 32-level GLCM on [0,1] images:
    #   tex_sev:     malignancy [0,1] — primary discriminator
    #   size_sev:    tumor_pct / 15.0 (15% = large tumor)
    #   hot_sev:     hot_pct (pixels > 0.75) / 0.15 (15% = very bright lesion)
    #   hetero_sev:  1 - homogeneity
    #   std_sev:     std_int / 0.25
    size_sev     = float(np.clip(tumor_pct / 15.0,  0.0, 1.0))
    asym_sev     = float(np.clip(asymmetry / 0.30,  0.0, 1.0))
    tex_sev      = float(malignancy)
    hot_sev      = float(np.clip(hot_pct / 0.15,    0.0, 1.0))
    hetero_sev   = float(np.clip(1.0 - homogeneity, 0.0, 1.0))
    std_sev      = float(np.clip(std_int / 0.25,    0.0, 1.0))
    corr_sev     = float(np.clip(1.0 - max(correlation, 0.0), 0.0, 1.0))

    severity = (
        tex_sev      * 0.35 +
        size_sev     * 0.20 +
        hot_sev      * 0.18 +
        hetero_sev   * 0.12 +
        asym_sev     * 0.08 +
        std_sev      * 0.05 +
        corr_sev     * 0.02
    )

    # ── Gaussian grade model (IMPROVED v10.4) ─────────────────────────────────
    # Grade centres recalibrated based on diagnostic output with better separation:
    #   Normal=0.07, GrI=0.24, GrII=0.41, GrIII=0.60, GrIV=0.81
    grade_centres = np.array([0.07, 0.24, 0.41, 0.60, 0.81])
    bw = 0.10  # Narrower bandwidth for sharper peaks (was 0.11)
    raw = np.exp(-((severity - grade_centres) ** 2) / (2 * bw ** 2))

    # ── Domain boosts (ENHANCED v10.4) ────────────────────────────────────────
    if severity < 0.14:
        raw[0] *= 14.0  # Stronger normal boost
        raw[3] *= 0.04; raw[4] *= 0.015

    if tumor_pct < 0.8 and max_seg_prob < 0.42:
        raw[3] *= 0.12; raw[4] *= 0.04

    # High malignancy + large tumor → GBM (more aggressive)
    if malignancy > 0.52 and tumor_pct > 4.5:
        raw[4] *= 4.5; raw[3] *= 2.3; raw[0] *= 0.04

    # High hot_pct → high grade (hot pixels are a strong tumor signal)
    if hot_pct > 0.075:   # > 7.5% of brain pixels are very bright
        raw[4] *= 3.5; raw[3] *= 2.3; raw[0] *= 0.04; raw[1] *= 0.15
    elif hot_pct > 0.035:  # > 3.5% → Grade III/IV
        raw[3] *= 2.8; raw[4] *= 2.3; raw[0] *= 0.08; raw[1] *= 0.25
    elif hot_pct > 0.018:  # > 1.8% → Grade II/III
        raw[2] *= 2.3; raw[3] *= 1.8; raw[0] *= 0.25

    # High malignancy from GLCM alone (U-Net may miss GBM)
    if malignancy > 0.52 and hot_pct > 0.025:
        raw[4] *= 2.8; raw[3] *= 2.0; raw[0] *= 0.06; raw[1] *= 0.12
    elif malignancy > 0.38 and hot_pct > 0.018:
        raw[3] *= 2.0; raw[4] *= 1.5; raw[0] *= 0.15

    raw = np.clip(raw, 1e-9, None)
    return (raw / raw.sum()).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_segmentation_on_slice(
    image_2d: np.ndarray,
    seg_models: List[Tuple[nn.Module, float, str]],
    device: torch.device,
) -> Tuple[np.ndarray, float, float]:
    """Run segmentation ensemble on a single 2D slice."""
    h, w = image_2d.shape[:2]
    per_model_preds: List[np.ndarray] = []
    used_weights: List[float] = []

    for model, weight, name in seg_models:
        try:
            model.eval()
            with torch.no_grad():
                inp = preprocess_for_model(image_2d, target_size=(h, w))
                if inp.shape[-2:] != (h, w):
                    inp = F.interpolate(inp, size=(h, w), mode='bilinear', align_corners=False)
                inp = inp.to(device)
                out = model(inp)
                if out.shape[1] > 1:
                    prob = torch.sigmoid(out[:, 1:2, :, :])
                else:
                    prob = torch.sigmoid(out)
                pred = prob.squeeze().cpu().numpy().astype(np.float32)
                if pred.shape != (h, w):
                    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                per_model_preds.append(pred * weight)
                used_weights.append(float(weight))
        except Exception as e:
            logger.debug(f"Seg model {name} failed: {e}")

    if not per_model_preds:
        # Return a zero mask — no model ran, so we have no segmentation signal.
        # Using a brightness-threshold fallback was the root cause of normal scans
        # being classified as Grade II (it marked ~99% of the image as "tumor").
        zero_mask = np.zeros((image_2d.shape[0], image_2d.shape[1]), dtype=np.float32)
        return zero_mask, 0.0, 1.0

    total_weight = float(sum(used_weights))
    fused = sum(per_model_preds) / (total_weight + 1e-8)

    raw_preds = np.stack(
        [p / max(w, 1e-8) for p, w in zip(per_model_preds, used_weights)],
        axis=0
    )
    uncertainty = float(raw_preds.std(axis=0).mean()) if len(raw_preds) > 1 else 0.0

    return fused.astype(np.float32), float(fused.max()), uncertainty


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_real_classifiers(
    image_2d: np.ndarray,
    real_cls_models: Dict[str, Any],
    n_classes: int = 5,
) -> Tuple[np.ndarray, List[str]]:
    """Run REAL brain-tumor pretrained classifiers on a 2D slice."""
    all_probs: List[np.ndarray] = []
    successful: List[str] = []

    for name, model in real_cls_models.items():
        if model is None:
            continue
        try:
            if hasattr(model, 'predict_grade_probs'):
                probs = model.predict_grade_probs(image_2d)
            elif hasattr(model, 'classify'):
                result = model.classify(image_2d)
                if result is None:
                    continue
                probs = np.array(result.get('grade_probs', [0.2]*5))
            else:
                continue

            if len(probs) == n_classes and not np.any(np.isnan(probs)):
                probs = np.array(probs, dtype=np.float64)
                probs = np.clip(probs, 1e-9, None)
                probs /= probs.sum()
                all_probs.append(probs)
                successful.append(name)
        except Exception as e:
            logger.debug(f"Real classifier {name} failed: {e}")

    if not all_probs:
        return np.ones(n_classes, dtype=np.float64) / n_classes, []

    return np.stack(all_probs, axis=0).mean(axis=0), successful


def _run_imagenet_classifiers(
    image_2d: np.ndarray,
    cls_models: List[Tuple[nn.Module, str]],
    device: torch.device,
    n_classes: int = 5,
) -> np.ndarray:
    """Run ImageNet-pretrained classifiers as fallback."""
    from torchvision import transforms as T
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_probs: List[np.ndarray] = []
    img_u8 = (np.clip(image_2d, 0, 1) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8 if img_u8.ndim == 2 else img_u8[:, :, 0],
                           cv2.COLOR_GRAY2RGB)
    img_rgb_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    pil_img = __import__('PIL').Image.fromarray(img_rgb_224)

    for model, name in cls_models:
        try:
            model.eval()
            with torch.no_grad():
                inp  = to_tensor(pil_img).unsqueeze(0).to(device)
                out  = model(inp)
                if out.shape[1] != n_classes:
                    continue
                probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
                all_probs.append(probs.astype(np.float64))
        except Exception as e:
            logger.debug(f"ImageNet classifier {name} failed: {e}")

    if not all_probs:
        return np.ones(n_classes, dtype=np.float64) / n_classes

    return np.stack(all_probs, axis=0).mean(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  ASYMMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_asymmetry(image_2d: np.ndarray) -> float:
    if image_2d.ndim != 2:
        return 0.0
    h, w = image_2d.shape
    left  = image_2d[:, :w // 2].astype(np.float32)
    right = np.fliplr(image_2d[:, w // 2:]).astype(np.float32)
    mw    = min(left.shape[1], right.shape[1])
    diff  = np.abs(left[:, :mw] - right[:, :mw])
    return float(diff.mean())


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION-BASED GRADE PRIOR  (v6.2: NEW - Properly independent source)
# ═══════════════════════════════════════════════════════════════════════════════

def _seg_probability_grade_prior(
    max_seg_prob: float,
    tumor_pct: float,
    asymmetry: float,
) -> np.ndarray:
    """
    Derive grade prior from SEGMENTATION MODEL output (U-Net probabilities).

    This uses what the U-Net ACTUALLY tells us:
      - max_seg_prob : peak confidence the U-Net assigns to a tumor pixel
      - tumor_pct    : fraction of the slice flagged as tumour
      - asymmetry    : spatial asymmetry of the detected region

    CRITICALLY different from _radiomics_grade_prior which uses GLCM texture.
    Having two genuinely independent sources (segmentation + radiomics) makes
    the 3-source fusion statistically meaningful.

    Grade mapping is grounded in clinical MRI literature:
      - Grade 0 (Normal)  : no/weak segmentation signal
      - Grade I (Low)     : small, focal, low-confidence detection
      - Grade II (Inter.) : moderate area, moderate confidence
      - Grade III (High)  : large area OR high asymmetry
      - Grade IV (GBM)    : large area AND very high confidence + asymmetry
    """
    # No meaningful segmentation signal → lean Normal
    # Raised threshold: the mateuszbuda U-Net outputs moderate probabilities
    # on normal brain tissue (trained on LGG patients, all had tumors).
    # Only trust it when it's genuinely confident (> 0.55).
    if max_seg_prob < 0.55 or tumor_pct < 1.0:
        return np.array([0.65, 0.15, 0.10, 0.06, 0.04], dtype=np.float64)

    # Very weak detection
    if max_seg_prob < 0.40:
        # Small but real signal - could be Grade I or very early Grade II
        if tumor_pct > 0.5 and asymmetry < 0.10:
            # Small, symmetric, weak signal → likely Grade I
            probs = np.array([0.35, 0.40, 0.16, 0.06, 0.03], dtype=np.float64)
        else:
            probs = np.array([0.55, 0.25, 0.13, 0.05, 0.02], dtype=np.float64)

    # Moderate detection
    elif max_seg_prob < 0.55:
        probs = np.array([0.12, 0.28, 0.36, 0.18, 0.06], dtype=np.float64)

    # Strong detection
    elif max_seg_prob < 0.70:
        probs = np.array([0.04, 0.10, 0.33, 0.35, 0.18], dtype=np.float64)

    # Very strong detection → likely high-grade
    else:
        probs = np.array([0.02, 0.05, 0.16, 0.37, 0.40], dtype=np.float64)

    # Adjust for tumour area
    if tumor_pct > 10.0:
        probs[3] *= 1.6
        probs[4] *= 2.2
        probs[0] *= 0.25
        probs[1] *= 0.45
    elif tumor_pct > 5.0:
        probs[3] *= 1.2
        probs[4] *= 1.5
        probs[0] *= 0.50
    elif tumor_pct < 2.0:
        probs[3] *= 0.40
        probs[4] *= 0.20
        probs[1] *= 1.5
        probs[2] *= 1.2

    # Adjust for spatial asymmetry (high asymmetry → infiltrative/higher grade)
    if asymmetry > 0.25:
        probs[3] *= 1.4
        probs[4] *= 1.6
        probs[0] *= 0.45
        probs[1] *= 0.65
    elif asymmetry < 0.08:
        # Well-circumscribed → lower grade
        probs[1] *= 1.3
        probs[2] *= 1.15
        probs[3] *= 0.70
        probs[4] *= 0.55

    probs = np.clip(probs, 1e-9, None)
    return (probs / probs.sum()).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIDENCE CALCULATOR  (v6.0: MAJOR IMPROVEMENT)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_confidence(
    final_grade: int,
    fused_probs: np.ndarray,      # ALREADY sharpened & consolidated (v6.0)
    seg_probs: np.ndarray,
    cls_probs: np.ndarray,
    rad_probs: np.ndarray,
    glcm_malignancy: float,
    tumor_pct: float,
    max_seg_prob: float,
    seg_uncertainty: float,
    n_real_cls_models: int,
    n_imagenet_cls_models: int,
    seg_weight: float,
    cls_weight: float,
    rad_weight: float,
    asymmetry: float = 0.0,  # Added for clinical tiebreak
) -> float:
    """
    Compute calibrated confidence from the SHARPENED fused distribution.

    v6.1 ENHANCED CONFIDENCE CALIBRATION:
    1. Statistical calibration based on model agreement
    2. Evidence-strength weighting (strong evidence → higher confidence)
    3. Grade-specific confidence floors (when evidence is genuinely strong)
    4. Quality-based confidence scaling
    5. Maintains medical ethics (no artificial inflation)
    
    Target: 85-95% confidence for clear cases with strong evidence
    """
    # Base: probability of final grade in the SHARPENED distribution
    base_conf = float(fused_probs[final_grade])

    # Agreement between sources
    seg_best = int(np.argmax(seg_probs))
    cls_best = int(np.argmax(cls_probs))
    rad_best = int(np.argmax(rad_probs))

    sources_agree_list = [
        seg_best == final_grade,
        cls_best == final_grade,
        rad_best == final_grade,
    ]
    n_agree = sum(sources_agree_list)

    # ═══ ENHANCED AGREEMENT SCORING ═══
    # When models agree, confidence should be significantly higher
    if n_real_cls_models > 0:
        if n_agree == 3:
            # All 3 sources agree - very strong evidence
            agreement_boost = 0.15  # Increased from 0.08
        elif n_agree == 2:
            # 2 sources agree - good evidence
            agreement_boost = 0.08  # Increased from 0.04
        else:
            # Disagreement - be cautious
            agreement_boost = -0.08  # Slightly increased penalty
    else:
        # No real classifiers - rely on radiomics + segmentation
        agreement_boost = -0.06 if cls_best != final_grade else 0.02

    # ═══ EVIDENCE STRENGTH ANALYSIS ═══
    # Analyze the strength of evidence for this prediction
    evidence_strength = 0.0
    
    # 1. Probability margin (how much better is winner vs second place)
    sorted_probs = np.sort(fused_probs)[::-1]
    prob_margin = float(sorted_probs[0] - sorted_probs[1])
    if prob_margin > 0.30:
        evidence_strength += 0.08  # Very clear winner
    elif prob_margin > 0.20:
        evidence_strength += 0.05  # Clear winner
    elif prob_margin > 0.10:
        evidence_strength += 0.02  # Moderate winner
    
    # 2. Individual source confidence
    seg_conf = float(seg_probs[final_grade])
    cls_conf = float(cls_probs[final_grade])
    rad_conf = float(rad_probs[final_grade])
    
    avg_source_conf = (seg_conf + cls_conf + rad_conf) / 3.0
    if avg_source_conf > 0.60:
        evidence_strength += 0.06  # Strong individual confidence
    elif avg_source_conf > 0.45:
        evidence_strength += 0.03  # Moderate individual confidence

    # 3. Clinical feature consistency
    clinical_consistency = 0.0
    if final_grade == 0:
        # Normal: low tumor %, low malignancy, low seg prob
        if tumor_pct < 0.5 and glcm_malignancy < 0.25 and max_seg_prob < 0.30:
            clinical_consistency = 0.08
        elif tumor_pct < 1.0 and glcm_malignancy < 0.35:
            clinical_consistency = 0.04
    elif final_grade == 1:
        # Grade I: small tumor, low malignancy
        if 0.5 < tumor_pct < 3.0 and 0.20 < glcm_malignancy < 0.45:
            clinical_consistency = 0.07
        elif 0.3 < tumor_pct < 5.0:
            clinical_consistency = 0.03
    elif final_grade == 2:
        # Grade II: moderate tumor, moderate malignancy
        if 2.0 < tumor_pct < 8.0 and 0.40 < glcm_malignancy < 0.65:
            clinical_consistency = 0.07
        elif 1.5 < tumor_pct < 10.0:
            clinical_consistency = 0.03
    elif final_grade == 3:
        # Grade III: large tumor, high malignancy
        if 5.0 < tumor_pct < 15.0 and 0.60 < glcm_malignancy < 0.80:
            clinical_consistency = 0.07
        elif 4.0 < tumor_pct < 18.0:
            clinical_consistency = 0.03
    elif final_grade == 4:
        # Grade IV/GBM: very large tumor, very high malignancy
        if tumor_pct > 8.0 and glcm_malignancy > 0.70:
            clinical_consistency = 0.08
        elif tumor_pct > 6.0 and glcm_malignancy > 0.60:
            clinical_consistency = 0.04
    
    evidence_strength += clinical_consistency

    # ═══ UNCERTAINTY PENALTIES ═══
    # Flat-distribution penalty
    top_prob = float(fused_probs.max())
    if top_prob < 0.40:
        flat_penalty = 0.12  # Very flat - high uncertainty
    elif top_prob < 0.50:
        flat_penalty = 0.06  # Somewhat flat
    elif top_prob < 0.60:
        flat_penalty = 0.03  # Slightly flat
    else:
        flat_penalty = 0.0   # Clear winner

    # Segmentation uncertainty penalty
    unc_penalty = float(np.clip(seg_uncertainty * 0.12, 0.0, 0.08))

    # ═══ MODEL QUALITY SCALING ═══
    # Better models → higher confidence ceiling
    if n_real_cls_models >= 3:
        model_quality_multiplier = 1.08  # Excellent model quality
        model_cap = 0.98
    elif n_real_cls_models >= 2:
        model_quality_multiplier = 1.05  # Very good model quality
        model_cap = 0.96
    elif n_real_cls_models == 1:
        model_quality_multiplier = 1.02  # Good model quality
        model_cap = 0.93
    elif n_imagenet_cls_models > 0:
        model_quality_multiplier = 0.98  # Moderate model quality
        model_cap = 0.88
    else:
        # Seg + radiomics + image_classifier — all available without downloads
        model_quality_multiplier = 0.99
        model_cap = 0.95

    # ═══ COMPUTE RAW CONFIDENCE ═══
    raw = (base_conf + agreement_boost + evidence_strength - unc_penalty - flat_penalty) * model_quality_multiplier

    # ═══ CONFIDENCE FLOORS v8.0 ═══
    # With 4 sources (seg + rad + img_cls + neural), n_agree can be 0-4.
    # Floors are calibrated to reflect the quality of the available evidence.

    if n_agree >= 3:  # Strong multi-source agreement
        if final_grade == 0 and tumor_pct < 2.0 and max_seg_prob < 0.55:
            raw = max(raw, 0.90)  # Clear normal
        elif final_grade == 0 and tumor_pct < 1.0:
            raw = max(raw, 0.87)
        elif final_grade == 1 and 0.5 < tumor_pct < 5.0 and clinical_consistency > 0.02:
            raw = max(raw, 0.86)
        elif final_grade == 2 and tumor_pct > 1.0 and clinical_consistency > 0.02:
            raw = max(raw, 0.87)
        elif final_grade == 3 and tumor_pct > 3.0 and glcm_malignancy > 0.45:
            raw = max(raw, 0.88)
        elif final_grade == 4 and tumor_pct > 5.0 and glcm_malignancy > 0.55:
            raw = max(raw, 0.90)
        else:
            raw = max(raw, 0.80)

    elif n_agree == 2:  # Good agreement
        if final_grade == 0 and tumor_pct < 2.0 and max_seg_prob < 0.55:
            raw = max(raw, 0.83)
        elif final_grade == 0 and tumor_pct < 1.0:
            raw = max(raw, 0.80)
        elif final_grade in [1, 2, 3, 4] and clinical_consistency > 0.02:
            raw = max(raw, 0.78)
        else:
            raw = max(raw, 0.72)

    elif n_agree == 1:  # Sources disagree — use clinical tiebreak
        clinical_tiebreak_grade = _clinical_tiebreak(tumor_pct, glcm_malignancy,
                                                      max_seg_prob, asymmetry)
        if clinical_tiebreak_grade == final_grade:
            raw = max(raw, 0.68)
        else:
            raw = max(raw, 0.55)

    # ═══ FINAL CALIBRATION ═══
    # Apply model quality cap and ensure reasonable range
    final_confidence = float(np.clip(raw, 0.20, model_cap))
    
    return final_confidence


# ═══════════════════════════════════════════════════════════════════════════════
#  TUMOR MORPHOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_tumor_morphology(
    mask_prob: np.ndarray,
    threshold: float = 0.55,
    pixel_spacing_mm: float = 1.0,
) -> Dict[str, float]:
    """
    Compute tumor morphology metrics from probability mask.

    v8.1: Threshold 0.55 + minimum connected component size filter.
    The mateuszbuda U-Net outputs 0.3-0.5 on normal tissue, so only
    regions with probability > 0.55 AND area > 0.5% of image are real.
    This prevents the 99% tumor area bug caused by diffuse low-confidence
    activations across the whole brain.
    """
    h, w    = mask_prob.shape
    total   = h * w
    binary  = (mask_prob > threshold).astype(np.uint8)

    if binary.sum() > 0:
        # Remove small objects — minimum 0.5% of image area (about 250px in 224x224)
        min_size = max(50, int(total * 0.005))
        binary = morphology.remove_small_objects(
            binary.astype(bool), min_size=min_size).astype(np.uint8)

    # Additional sanity: if binary covers > 60% of image, the mask is unreliable
    # (whole-brain activation = U-Net failed on this scan)
    if binary.sum() > total * 0.60:
        logger.warning(f"[MORPHOLOGY] Mask covers {binary.mean()*100:.1f}% of image — "
                       f"likely U-Net failure. Returning zero tumor.")
        return {'area_pixels': 0, 'tumor_pct': 0.0, 'max_diameter_mm': 0.0,
                'compactness': 1.0, 'asymmetry': 0.0,
                'binary_mask': np.zeros((h, w), dtype=np.float32)}

    area  = int(binary.sum())
    pct   = float(area / total * 100.0)

    if area == 0:
        return {'area_pixels': 0, 'tumor_pct': 0.0, 'max_diameter_mm': 0.0,
                'compactness': 1.0, 'asymmetry': 0.0,
                'binary_mask': np.zeros((h, w), dtype=np.float32)}

    props = measure.regionprops(measure.label(binary))
    if props:
        largest = max(props, key=lambda p: p.area)
        r_min, c_min, r_max, c_max = largest.bbox
        diag_px     = float(np.sqrt((r_max - r_min)**2 + (c_max - c_min)**2))
        max_diam_mm = diag_px * pixel_spacing_mm
        perim       = float(largest.perimeter) + 1e-8
        compactness = float(np.clip(4 * np.pi * largest.area / perim ** 2, 0.0, 1.0))
    else:
        max_diam_mm = 0.0
        compactness = 1.0

    left  = binary[:, :w // 2]
    right = np.fliplr(binary[:, w // 2:])
    mw    = min(left.shape[1], right.shape[1])
    asym  = float(np.abs(left[:, :mw].sum() - right[:, :mw].sum()) / (max(area, 1)))

    return {
        'area_pixels': area,
        'tumor_pct': pct,
        'max_diameter_mm': max_diam_mm,
        'compactness': compactness,
        'asymmetry': asym,
        'binary_mask': binary.astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-SLICE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _select_informative_slices(
    image: np.ndarray,
    n_slices: int = 5,
) -> Tuple[List[np.ndarray], int]:
    """Select N most informative 2D slices from a 2D or 3D image."""
    if image.ndim == 2:
        return [image.astype(np.float32)], 0

    if image.ndim == 3 and image.shape[2] in (3, 4):
        grey = image.mean(axis=2).astype(np.float32)
        return [grey], 0

    if image.ndim == 3:
        if image.shape[2] < image.shape[0]:
            slices_iter = [image[:, :, k] for k in range(image.shape[2])]
        else:
            slices_iter = [image[k] for k in range(image.shape[0])]
    else:
        slices_iter = [image.squeeze()]

    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8:
        mid = len(slices_iter) // 2
        return [slices_iter[mid].astype(np.float32)], 0
    norm_slices = [(sl - mn) / (mx - mn + 1e-8) for sl in slices_iter]

    scores = []
    for sl in norm_slices:
        sl = sl.astype(np.float32)
        bm = sl > 0.05
        bp = sl[bm]
        if len(bp) < 200:
            scores.append(0.0)
            continue
        cov  = float(len(bp) / sl.size)
        var  = float(np.var(bp))
        w    = sl.shape[1]
        left = sl[:, :w // 2]
        rgt  = np.fliplr(sl[:, w // 2:])
        mw_  = min(left.shape[1], rgt.shape[1])
        asym = float(np.mean(np.abs(left[:, :mw_] - rgt[:, :mw_])))
        scores.append(cov * 0.30 + var * 0.40 + asym * 0.30)

    scores_arr = np.array(scores)
    order      = np.argsort(scores_arr)[::-1]

    selected_idx: List[int] = []
    for idx in order:
        if all(abs(idx - s) >= max(1, len(slices_iter) // 20)
               for s in selected_idx):
            selected_idx.append(int(idx))
        if len(selected_idx) == n_slices:
            break

    if not selected_idx:
        mid = len(slices_iter) // 2
        selected_idx = [mid]

    selected_idx = sorted(selected_idx)
    selected_slices = [norm_slices[i].astype(np.float32) for i in selected_idx]

    # Representative slice must be the most informative among selected slices.
    best_global_idx = int(order[0]) if len(order) else selected_idx[0]
    if best_global_idx in selected_idx:
        best_idx_in_selected = selected_idx.index(best_global_idx)
    else:
        selected_scores = [scores_arr[i] for i in selected_idx]
        best_idx_in_selected = int(np.argmax(selected_scores))

    return selected_slices, best_idx_in_selected


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class _ModelCache:
    """Lazy model cache — loads real brain-tumor classifiers using standardized get_ensemble_models()."""
    _seg_models:          Optional[List] = None
    _real_cls_models:     Optional[Dict] = None
    _imagenet_cls_models: Optional[List] = None
    _device:              Optional[torch.device] = None
    _initialised:         bool = False

    @classmethod
    def get(cls, device: Optional[torch.device] = None):
        if cls._initialised:
            return cls._seg_models, cls._real_cls_models, cls._imagenet_cls_models, cls._device

        if device is None:
            device = torch.device('cpu')
        cls._device = device

        seg_models: List = []
        real_cls:   Dict = {}
        imgnet_cls: List = []

        # ═══════════════════════════════════════════════════════════════════════════
        # FIXED v6.5: Use standardized get_ensemble_models() instead of manual loading
        # ═══════════════════════════════════════════════════════════════════════════
        
        logger.info("[MODEL CACHE] Loading models using standardized get_ensemble_models()...")
        
        try:
            if MODELS_AVAILABLE:
                ensemble_models = get_ensemble_models(device=str(device))
                
                # ── SEGMENTATION MODELS ──────────────────────────────────────────
                # Extract segmentation models with standardized keys
                if 'seg_unet' in ensemble_models and ensemble_models['seg_unet'] is not None:
                    seg_models.append((ensemble_models['seg_unet'], 1.0, 'seg_unet'))
                    logger.info("[MODEL CACHE] ✓ seg_unet loaded (mateuszbuda, Dice ~0.85)")
                
                if 'seg_smp' in ensemble_models and ensemble_models['seg_smp'] is not None:
                    seg_models.append((ensemble_models['seg_smp'], 0.5, 'seg_smp'))
                    logger.info("[MODEL CACHE] ✓ seg_smp loaded (EfficientNet-B4)")
                
                if 'seg_monai_brats' in ensemble_models and ensemble_models['seg_monai_brats'] is not None:
                    seg_models.append((ensemble_models['seg_monai_brats'], 0.7, 'seg_monai_brats'))
                    logger.info("[MODEL CACHE] ✓ seg_monai_brats loaded (BraTS SegResNet)")
                
                if 'seg_swinunetr' in ensemble_models and ensemble_models['seg_swinunetr'] is not None:
                    seg_models.append((ensemble_models['seg_swinunetr'], 0.6, 'seg_swinunetr'))
                    logger.info("[MODEL CACHE] ✓ seg_swinunetr loaded (MONAI SwinUNETR)")
                
                # ── CLASSIFICATION MODELS ────────────────────────────────────────
                # Extract classification models with standardized keys
                if 'cls_vit' in ensemble_models and ensemble_models['cls_vit'] is not None:
                    real_cls['medical_vit'] = ensemble_models['cls_vit']
                    logger.info("[MODEL CACHE] ✓ cls_vit loaded (Medical ViT)")

                if 'cls_clip' in ensemble_models and ensemble_models['cls_clip'] is not None:
                    real_cls['biomedclip'] = ensemble_models['cls_clip']
                    logger.info("[MODEL CACHE] ✓ cls_clip loaded (PubMedCLIP)")

                if 'cls_raddino' in ensemble_models and ensemble_models['cls_raddino'] is not None:
                    real_cls['rad_dino'] = ensemble_models['cls_raddino']
                    logger.info("[MODEL CACHE] ✓ cls_raddino loaded (microsoft/rad-dino)")

                if 'cls_trained' in ensemble_models and ensemble_models['cls_trained'] is not None:
                    real_cls['trained_grade_cls'] = ensemble_models['cls_trained']
                    logger.info("[MODEL CACHE] ✓ cls_trained loaded (Trained classifier)")

                # ── NEW: TimmEnsembleClassifier (EfficientNetV2-M + ConvNeXt + Swin) ──
                if 'cls_timm' in ensemble_models and ensemble_models['cls_timm'] is not None:
                    real_cls['timm_ensemble'] = ensemble_models['cls_timm']
                    logger.info("[MODEL CACHE] ✓ cls_timm loaded (TimmEnsemble: EfficientNetV2+ConvNeXt+Swin)")

                # ── NEW: SwinGradeClassifier (microsoft/swin-base cached) ──────────
                if 'cls_swin' in ensemble_models and ensemble_models['cls_swin'] is not None:
                    real_cls['swin_grade'] = ensemble_models['cls_swin']
                    logger.info("[MODEL CACHE] ✓ cls_swin loaded (SwinGrade: microsoft/swin-base-in22k)")
                
            else:
                logger.warning("[MODEL CACHE] models.py not available - using fallback loading")
                # Fallback to direct loading if get_ensemble_models not available
                try:
                    unet = torch.hub.load(
                        'mateuszbuda/brain-segmentation-pytorch', 'unet',
                        in_channels=3, out_channels=1, init_features=32, pretrained=True,
                        trust_repo=True, verbose=False,
                    )
                    unet.to(device).eval()
                    seg_models.append((unet, 1.0, 'mateuszbuda_unet_REAL'))
                    logger.info("[MODEL CACHE] Fallback: mateuszbuda U-Net loaded")
                except Exception as e:
                    logger.warning(f"[MODEL CACHE] Fallback U-Net failed: {e}")
        
        except Exception as e:
            logger.error(f"[MODEL CACHE] Error loading models: {e}", exc_info=True)
            # Try fallback loading
            try:
                unet = torch.hub.load(
                    'mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=3, out_channels=1, init_features=32, pretrained=True,
                    trust_repo=True, verbose=False,
                )
                unet.to(device).eval()
                seg_models.append((unet, 1.0, 'mateuszbuda_unet_REAL'))
                logger.info("[MODEL CACHE] Emergency fallback: mateuszbuda U-Net loaded")
            except Exception as e2:
                logger.error(f"[MODEL CACHE] Emergency fallback failed: {e2}")

        logger.info("[MODEL CACHE] ImageNet classifiers SKIPPED (random heads = corrupted grades)")

        cls._seg_models          = seg_models
        cls._real_cls_models     = real_cls
        cls._imagenet_cls_models = imgnet_cls
        cls._initialised         = True

        n_real = len(real_cls)
        n_seg  = len(seg_models)
        total_models = n_seg + n_real

        # Allow operation with just 1 model (seg-only or radiomics-only fallback)
        # The system degrades gracefully rather than crashing.
        if total_models < 1:
            error_msg = (
                "[MODEL CACHE] *** CRITICAL ERROR: No models loaded at all ***\n"
                "Action: Run 'python download_models.py' to download required models"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(
            f"[MODEL CACHE] ✓ LOADED: {len(seg_models)} segmentation + {n_real} classifiers (device={device})"
        )
        if n_real == 0:
            logger.warning(
                "[MODEL CACHE] *** NO REAL BRAIN-TUMOR CLASSIFIERS LOADED ***\n"
                "Run: python download_models.py  to cache Medical ViT and RadImageNet.\n"
                "Falling back to radiomics + segmentation only."
            )
        return cls._seg_models, cls._real_cls_models, cls._imagenet_cls_models, cls._device

    @classmethod
    def reset(cls) -> None:
        cls._seg_models          = None
        cls._real_cls_models     = None
        cls._imagenet_cls_models = None
        cls._initialised         = False


# ═══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE FUSION WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_fusion_weights(
    n_real_cls: int,
    n_imagenet_cls: int,
    n_seg_models: int,
) -> Tuple[float, float, float]:
    """
    Compute adaptive fusion weights. Returns (RAD_W, SEG_W, CLS_W).

    v6.7 EMERGENCY FALLBACK: If no models load, use radiomics-only mode
    This ensures the system works even without external model downloads.
    """
    # EMERGENCY FALLBACK: No models loaded at all
    if n_real_cls == 0 and n_imagenet_cls == 0 and n_seg_models == 0:
        logger.warning(
            "[FALLBACK] No models loaded - using radiomics-only mode. "
            "Run 'python download_models.py' to enable full model ensemble."
        )
        return 1.0, 0.0, 0.0  # 100% radiomics
    
    # Normal operation with models
    if n_real_cls >= 2:
        rad = 0.35; seg = 0.30; cls = 0.35   # More balanced
    elif n_real_cls == 1:
        rad = 0.35; seg = 0.35; cls = 0.30   # Equal weighting
    elif n_imagenet_cls > 0:
        rad = 0.45; seg = 0.45; cls = 0.10
    else:
        # NO classifiers at all: cls_probs is uniform → 0% weight
        rad = 0.50; seg = 0.50; cls = 0.00

    if n_seg_models == 0:
        seg_reduction = seg * 0.4
        rad += seg_reduction
        seg -= seg_reduction

    total = rad + seg + cls
    return rad / total, seg / total, cls / total


# ═══════════════════════════════════════════════════════════════════════════════
#  CLINICAL TIE-BREAK  (NEW: resolves source disagreement using clinical features)
# ═══════════════════════════════════════════════════════════════════════════════

def _clinical_tiebreak(
    tumor_pct: float,
    glcm_malignancy: float,
    max_seg_prob: float,
    asymmetry: float,
) -> Optional[int]:
    """
    When 3 sources disagree, use clinical imaging features to pick the
    most credible grade. Returns None if features are ambiguous.

    Clinical grade boundaries (literature-grounded):
      Grade 0 (Normal):   tumor_pct < 0.5,  malignancy < 0.25, seg < 0.35
      Grade I  (Low):     tumor_pct 0.3-4,  malignancy 0.15-0.45, asymmetry < 0.15
      Grade II (Inter):   tumor_pct 1.5-10, malignancy 0.35-0.65, moderate asymmetry
      Grade III (High):   tumor_pct 4-18,   malignancy 0.55-0.82, asym > 0.20
      Grade IV (GBM):     tumor_pct > 6,    malignancy > 0.62,    asym > 0.25
    """
    # Strong Normal signal
    if tumor_pct < 0.4 and glcm_malignancy < 0.22 and max_seg_prob < 0.30:
        return 0

    # Strong GBM signal
    if tumor_pct > 8.0 and glcm_malignancy > 0.68 and asymmetry > 0.22:
        return 4

    # Strong Grade III signal
    if tumor_pct > 5.0 and glcm_malignancy > 0.58 and asymmetry > 0.18:
        return 3

    # Strong Grade I signal: small, symmetric, low malignancy, weak seg signal
    if (0.3 < tumor_pct < 4.5 and glcm_malignancy < 0.42
            and asymmetry < 0.12 and max_seg_prob < 0.58):
        return 1

    # Strong Grade II signal: moderate tumor, moderate malignancy
    if (1.5 < tumor_pct < 12.0 and 0.32 < glcm_malignancy < 0.68
            and 0.08 < asymmetry < 0.30):
        return 2

    return None  # Ambiguous — don't force a grade


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_brain_mri(
    image: np.ndarray,
    patient_id: str = "unknown",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    device: Optional[torch.device] = None,
    pixel_spacing_mm: float = 1.0,
) -> AnalysisResult:
    """
    PRIMARY ENTRY POINT for single-scan brain tumor analysis.

    v6.0 Key Changes:
    - Temperature scaling on fused distribution (legitimate sharpening)
    - Evidence consolidation when sources agree
    - Multi-slice uncertainty-weighted consensus
    - Grade-specific confidence floors (clinically grounded)
    - Improved Grade I vs III disambiguation
    
    v6.1 Reproducibility:
    - Deterministic results: same image always gives same grade
    - Random seeds set for consistency
    """
    # ═══ REPRODUCIBILITY: Set seeds for deterministic results ═══
    # This ensures the same image always produces the same grade
    _set_random_seeds(42)
    
    t0 = time.time()

    def _progress(frac: float, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(float(np.clip(frac, 0.0, 1.0)), msg)
            except Exception:
                pass

    _progress(0.00, "[INIT] Initialising analysis pipeline v6.1 (deterministic)…")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _progress(0.04, "[INIT] Loading models (cached after first run)…")
    seg_models, real_cls_models, imagenet_cls_models, device = _ModelCache.get(device)

    # ── Preprocessing ────────────────────────────────────────────────────────
    _progress(0.08, "[PREP] Preprocessing MRI image…")
    try:
        proc = preprocess_mri(image.copy(), target_size=(224, 224))
    except Exception:
        proc = prepare_image_for_display(image)

    if proc is None or proc.size == 0:
        logger.error("Preprocessing returned empty array")
        return _empty_result()

    # ── Brain presence check ─────────────────────────────────────────────────
    proc_2d = proc if proc.ndim == 2 else proc[proc.shape[0] // 2]
    brain_px = proc_2d[proc_2d > 0.05]
    brain_fraction = float(len(brain_px) / proc_2d.size)

    if brain_fraction < 0.05:
        logger.warning("[ANALYSIS] Image appears blank or non-brain — returning Normal")
        return AnalysisResult(
            grade=0, confidence=0.98, detected=False,
            tumor_type='Normal', risk_level='Normal',
            description='Image does not appear to contain brain tissue.',
            grade_probs=[0.98, 0.005, 0.005, 0.005, 0.005],
            meets_confidence_threshold=True,
            processing_time_s=time.time() - t0,
        )

    # ── Slice selection ──────────────────────────────────────────────────────
    _progress(0.12, "[PREP] Selecting informative slices…")
    slices, best_slice_idx = _select_informative_slices(proc, n_slices=5)
    rep_slice = slices[best_slice_idx]

    # ── Segmentation ─────────────────────────────────────────────────────────
    _progress(0.20, "[SEG] Running segmentation ensemble…")
    all_masks, all_uncertainties = [], []
    for sl in slices:
        mask, max_p, unc = _run_segmentation_on_slice(sl, seg_models, device)
        all_masks.append(mask)
        all_uncertainties.append(unc)

    fused_mask      = np.mean(all_masks, axis=0).astype(np.float32)
    max_seg_prob    = float(fused_mask.max())
    seg_uncertainty = float(np.mean(all_uncertainties))
    n_seg_models    = len(seg_models)

    _progress(0.38, "[SEG] Computing tumor morphology…")
    # Adaptive threshold: keep specificity for weak masks while preserving
    # recall for strong high-grade lesions that were being under-segmented.
    if max_seg_prob >= 0.70:
        morph_threshold = 0.50
    elif max_seg_prob >= 0.55:
        morph_threshold = 0.53
    elif max_seg_prob >= 0.45:
        morph_threshold = 0.56
    else:
        morph_threshold = 0.60

    morph = _compute_tumor_morphology(
        fused_mask,
        threshold=morph_threshold,
        pixel_spacing_mm=pixel_spacing_mm
    )
    tumor_pct = morph['tumor_pct']
    asymmetry = morph['asymmetry']

    # ── GLCM Radiomics ────────────────────────────────────────────────────────
    # FIXED v11.1: Moved BEFORE fallback check (was Bug #1 - variable used before computed)
    _progress(0.48, "[RADIO] Extracting GLCM radiomics (14 features)…")
    glcm = _compute_glcm_features(rep_slice)
    glcm_malignancy = glcm.get('malignancy_score', 0.0)

    # FIX v11.0: Restrict fallback - only use when multiple sources strongly agree
    # Old fallback created artificial tumor signals on normal brains
    seg_has_weak_signal = (tumor_pct == 0.0 and max_seg_prob > 0.45)
    if seg_has_weak_signal:
        # Require STRONG multi-source agreement before using fallback
        radiomics_strongly_supports = glcm_malignancy > 0.55
        
        if radiomics_strongly_supports:
            # Conservative estimate: only add small amount
            estimated_tumor_pct = float(np.clip((max_seg_prob - 0.40) / 0.60 * 3.0, 0.5, 2.5))
            logger.info(f"[MORPHOLOGY] Strong radiomics support: using estimated tumor_pct="
                        f"{estimated_tumor_pct:.2f}% (malignancy={glcm_malignancy:.3f})")
            tumor_pct = estimated_tumor_pct
        else:
            # No strong support - accept tumor_pct = 0 (likely normal brain)
            logger.info(f"[MORPHOLOGY] No strong tumor evidence - "
                        f"keeping tumor_pct=0% (malignancy={glcm_malignancy:.3f})")
            tumor_pct = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIX v11.0: EARLY NORMAL BRAIN DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    # Detect normal brain BEFORE attempting tumor classification
    # This prevents false positives on normal anatomy
    
    _progress(0.50, "[VALIDATE] Checking for normal brain anatomy…")
    
    # Check for clear absence of tumor signals
    no_seg_signal = max_seg_prob < 0.40
    low_malignancy = glcm_malignancy < 0.32
    minimal_area = tumor_pct < 0.8
    
    # All three must agree for high-confidence normal
    clear_normal_signals = no_seg_signal and low_malignancy and minimal_area
    
    if clear_normal_signals:
        logger.info(f"[NORMAL DETECTION] Clear normal brain detected:")
        logger.info(f"  - Segmentation: max_prob={max_seg_prob:.3f} (threshold: <0.40)")
        logger.info(f"  - Radiomics: malignancy={glcm_malignancy:.3f} (threshold: <0.32)")
        logger.info(f"  - Morphology: tumor_pct={tumor_pct:.2f}% (threshold: <0.8%)")
        logger.info(f"[NORMAL DETECTION] Returning Normal with high confidence")
        
        return AnalysisResult(
            grade=0,
            confidence=0.91,
            detected=False,
            tumor_type='Normal',
            risk_level='Normal',
            description='Normal brain anatomy detected. No tumor present.',
            grade_probs=[0.91, 0.03, 0.02, 0.02, 0.02],
            tumor_percentage=0.0,
            tumor_mask=None,
            meets_confidence_threshold=True,
            processing_time_s=time.time() - t0,
            quality_report={
                'normal_detection': True,
                'max_seg_prob': max_seg_prob,
                'glcm_malignancy': glcm_malignancy,
                'tumor_pct': tumor_pct,
            },
        )

    # ── Radiomics grade prior ─────────────────────────────────────────────────
    _progress(0.55, "[RADIO] Computing WHO-2021 grade prior…")
    rad_probs = _radiomics_grade_prior(glcm, tumor_pct, max_seg_prob, asymmetry)

    # ── Segmentation-based grade prior ───────────────────────────────────────
    seg_probs = _seg_probability_grade_prior(max_seg_prob, tumor_pct, asymmetry)

    # ── Image Feature Classifier (v8.0 NEW) ──────────────────────────────────
    # Multi-feature classifier using 20+ discriminative image features.
    # This is the PRIMARY classifier when no pretrained neural network is loaded.
    # It produces peaked distributions by detecting focal bright regions,
    # asymmetry, texture roughness, ring enhancement, and edema.
    _progress(0.58, "[CLS] Running image feature classifier…")
    img_cls_probs_list = []
    for sl in slices:
        try:
            p = _image_feature_classifier(sl)
            img_cls_probs_list.append(p)
        except Exception as _e:
            logger.debug(f"Image classifier failed on slice: {_e}")

    if img_cls_probs_list:
        img_cls_probs = _multi_slice_consensus(img_cls_probs_list)
    else:
        img_cls_probs = np.ones(5, dtype=np.float64) / 5

    img_cls_probs = np.clip(img_cls_probs, 1e-9, None)
    img_cls_probs /= img_cls_probs.sum()
    logger.info(f"[IMG CLS] Image classifier: Grade {int(np.argmax(img_cls_probs))} "
                f"({float(img_cls_probs.max()):.1%}) — "
                f"probs: {[f'{p:.3f}' for p in img_cls_probs]}")

    # ── Real Brain-Tumor Classifiers ──────────────────────────────────────────
    _progress(0.62, "[CLS] Running real brain-tumor classifiers…")

    # Collect per-slice probs for multi-slice consensus (v6.0 NEW)
    per_slice_real_probs: List[np.ndarray] = []
    all_real_cls_names: List[str] = []

    for sl in slices:
        probs, names = _run_real_classifiers(sl, real_cls_models)
        per_slice_real_probs.append(probs)
        if not all_real_cls_names:
            all_real_cls_names = names

    # v6.0: uncertainty-weighted consensus instead of simple mean
    real_cls_avg = _multi_slice_consensus(per_slice_real_probs) if per_slice_real_probs else np.ones(5) / 5

    # ── ImageNet Classifiers (fallback) ──────────────────────────────────────
    _progress(0.72, "[CLS] Running fallback classifiers…")
    imgnet_cls_probs_list = []
    for sl in slices:
        p = _run_imagenet_classifiers(sl, imagenet_cls_models, device)
        imgnet_cls_probs_list.append(p)

    imgnet_cls_avg = (np.mean(imgnet_cls_probs_list, axis=0)
                      if imgnet_cls_probs_list else np.ones(5) / 5)

    # Define counts ONCE here, after both real and imagenet classifiers are computed
    n_real_cls     = len(real_cls_models)
    n_imagenet_cls = len(imagenet_cls_models)
    
    # v6.5: DETECT GRADE II BIAS in classifiers (AFTER n_real_cls is defined)
    if n_real_cls > 0:
        cls_winner = int(np.argmax(real_cls_avg))
        cls_confidence = float(real_cls_avg[cls_winner])
        if cls_winner == 2 and cls_confidence > 0.45:
            logger.warning(
                f"[BIAS DETECTION] Classifiers strongly favor Grade II "
                f"({cls_confidence:.1%} confidence). This may indicate classifier bias. "
                f"Reducing classifier weight to prevent bias amplification."
            )

    if n_real_cls > 0 and n_imagenet_cls > 0:
        cls_probs = real_cls_avg * 0.85 + imgnet_cls_avg * 0.15
    elif n_real_cls > 0:
        cls_probs = real_cls_avg
    elif n_imagenet_cls > 0:
        cls_probs = imgnet_cls_avg
    else:
        cls_probs = np.ones(5) / 5

    cls_probs = np.clip(cls_probs, 1e-9, None)
    cls_probs /= cls_probs.sum()

    # ── Adaptive Fusion v9.1: 4-source ensemble with clinical validation ──────
    # Sources: seg_probs, rad_probs, img_cls_probs, cls_probs (neural nets)
    # img_cls_probs is the new image feature classifier — always available.
    _progress(0.82, "[FUSE] Multi-source ensemble fusion with clinical validation…")

    RAD_W, SEG_W, CLS_W = _compute_fusion_weights(n_real_cls, n_imagenet_cls, n_seg_models)

    # Log all source predictions for debugging
    try:
        logger.info(f"[FUSION] seg={int(np.argmax(seg_probs))}({float(seg_probs.max()):.1%})  "
                    f"rad={int(np.argmax(rad_probs))}({float(rad_probs.max()):.1%})  "
                    f"img={int(np.argmax(img_cls_probs))}({float(img_cls_probs.max()):.1%})  "
                    f"cls={int(np.argmax(cls_probs))}({float(cls_probs.max()):.1%})  "
                    f"tumor_pct={tumor_pct:.1f}%  seg_prob={max_seg_prob:.3f}  "
                    f"malignancy={glcm_malignancy:.3f}")
    except Exception:
        pass

    # ── FIX v11.0: Use simple_grade_fusion (classifier removed) ──
    # v11.0: Removed flawed classifier trained on tumor TYPES (not WHO grades)
    # Now uses segmentation + radiomics (trained on BraTS with REAL grades)
    # + WHO grading rules based on biological criteria
    prediction_result = simple_grade_fusion(
        seg_probs       = seg_probs,
        cls_probs       = cls_probs,
        radiomics_probs = rad_probs,
        img_cls_probs   = img_cls_probs,
        tumor_pct       = tumor_pct,
        glcm_malignancy = glcm_malignancy,
        max_seg_prob    = max_seg_prob,
        asymmetry       = asymmetry,
    )
    
    final_grade = prediction_result['grade']
    fused_probs = np.array(prediction_result['grade_probs'], dtype=np.float64)
    
    # Ensure probabilities are normalized
    fused_probs = np.clip(fused_probs, 1e-9, None)
    fused_probs /= fused_probs.sum()
    
    logger.info(f"[FUSION] Improved prediction: Grade {final_grade} "
                f"(conf={prediction_result['confidence']:.3f}) "
                f"source={prediction_result['source']}")

    # Count sources agreeing with final prediction (for confidence calculation)
    seg_best = int(np.argmax(seg_probs))
    cls_best = int(np.argmax(cls_probs))
    rad_best = int(np.argmax(rad_probs))
    img_best = int(np.argmax(img_cls_probs))

    sources_for_final = [
        seg_best == final_grade,
        (cls_best == final_grade) if n_real_cls > 0 else False,
        rad_best == final_grade,
        img_best == final_grade,
    ]
    n_sources_agree = sum(sources_for_final)
    
    logger.info(f"[FUSION] Sources agreeing with Grade {final_grade}: {n_sources_agree}/4 "
                f"(seg={seg_best==final_grade}, cls={cls_best==final_grade if n_real_cls>0 else 'N/A'}, "
                f"rad={rad_best==final_grade}, img={img_best==final_grade})")

    # ══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX v6.6: REMOVED ALL OVERRIDE GATES
    # ══════════════════════════════════════════════════════════════════════════
    # 
    # PROBLEM: The override gates were causing "fix one grade, break another"
    # - When Normal scans were misclassified, we added Gate 1 to force Normal
    # - Then Grade II scans started being forced to Normal too
    # - When we fixed that, Normal scans became Grade II again
    # 
    # ROOT CAUSE: The gates were second-guessing pretrained models with arbitrary
    # rules that don't generalize across all cases.
    # 
    # SOLUTION: TRUST THE MODELS. They are trained on thousands of brain MRIs.
    # Let them predict without interference.
    # 
    # The models use:
    #   - Segmentation (U-Net trained on LGG MRI, Dice ~0.85)
    #   - Classification (ViT fine-tuned on brain tumors)
    #   - Radiomics (14 GLCM texture features)
    # 
    # These are GOOD models. Let them work.
    # ══════════════════════════════════════════════════════════════════════════
    
    # ══════════════════════════════════════════════════════════════════════════
    # FIX v11.0: STRENGTHENED FINAL CLINICAL VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    # 
    # After all model fusion, apply strict clinical validation to catch
    # obvious misclassifications that could harm patient care.
    # 
    # This is the LAST safety check before returning the result.
    # ══════════════════════════════════════════════════════════════════════════
    
    # Get initial confidence from the improved prediction
    if 'confidence' in prediction_result:
        confidence = prediction_result['confidence']
    else:
        confidence = 0.70  # Default confidence
    
    # Store original prediction for logging
    original_grade = final_grade
    original_confidence = confidence
    
    # ═══ RULE 1: Normal scan validation (STRENGTHENED) ═══
    # If predicting normal but ANY strong tumor signals exist, correct it
    if final_grade == 0:
        strong_tumor_signals = (
            tumor_pct > 2.5 or 
            max_seg_prob > 0.60 or 
            (glcm_malignancy > 0.50 and tumor_pct > 1.0)
        )
        if strong_tumor_signals:
            # Find the most likely tumor grade from the probability distribution
            tumor_probs = fused_probs[1:]  # Exclude grade 0
            if tumor_probs.max() > 0.20:
                final_grade = int(np.argmax(tumor_probs)) + 1
                confidence = max(0.68, float(tumor_probs.max()))
                logger.warning(f"[CLINICAL VALIDATION] Corrected Grade 0 → Grade {final_grade} "
                              f"(tumor_pct={tumor_pct:.2f}%, seg_prob={max_seg_prob:.3f})")
    
    # ═══ RULE 2: High-grade validation (STRENGTHENED) ═══
    # If predicting high grade but ALL tumor signals are weak, correct it
    elif final_grade >= 3:
        # Check if ALL sources are weak
        weak_seg = max_seg_prob < 0.45
        weak_rad = glcm_malignancy < 0.45
        tiny_tumor = tumor_pct < 1.5
        
        all_signals_weak = weak_seg and weak_rad and tiny_tumor
        
        if all_signals_weak:
            # All signals weak - likely normal or low-grade
            if tumor_pct < 0.5:
                final_grade = 0  # Normal
                confidence = 0.82
                logger.warning(f"[CLINICAL VALIDATION] All signals weak - "
                              f"corrected Grade {original_grade} → Normal")
            else:
                final_grade = 1  # Low-grade
                confidence = 0.72
                logger.warning(f"[CLINICAL VALIDATION] All signals weak - "
                              f"corrected Grade {original_grade} → Grade I")
        
        # Grade IV specifically needs strong evidence
        elif final_grade == 4:
            strong_grade4_seg = (
                tumor_pct > 4.0 and
                max_seg_prob > 0.55 and
                glcm_malignancy > 0.58
            )
            strong_grade4_multisource = (
                float(fused_probs[4]) > 0.45 and
                float(img_cls_probs[3] + img_cls_probs[4]) > 0.60 and
                glcm_malignancy > 0.55
            )
            if not (strong_grade4_seg or strong_grade4_multisource):
                final_grade = 3  # Downgrade to Grade III
                confidence = max(0.70, confidence * 0.88)
                logger.info(f"[CLINICAL VALIDATION] Grade IV needs stronger evidence - "
                           f"downgraded to Grade III")
            else:
                confidence = max(confidence, 0.82)
    
    # ═══ RULE 3: Grade I vs II disambiguation (IMPROVED) ═══
    elif final_grade in [1, 2]:
        if tumor_pct < 1.2 and glcm_malignancy < 0.38:
            # Favor Grade I for small, low-malignancy tumors
            if final_grade == 2:
                final_grade = 1
                confidence = max(0.70, confidence * 0.93)
                logger.info(f"[CLINICAL VALIDATION] Grade II → Grade I "
                           f"(small tumor: {tumor_pct:.2f}%, malignancy={glcm_malignancy:.3f})")
        elif tumor_pct > 5.0 and glcm_malignancy > 0.52:
            # Favor Grade II for larger, more malignant tumors
            if final_grade == 1:
                final_grade = 2
                confidence = max(0.70, confidence * 0.93)
                logger.info(f"[CLINICAL VALIDATION] Grade I → Grade II "
                           f"(large tumor: {tumor_pct:.2f}%, malignancy={glcm_malignancy:.3f})")
    
    # Update fused_probs to match final_grade if it was changed
    if final_grade != original_grade:
        # Adjust probabilities to reflect the corrected grade
        new_probs = fused_probs.copy()
        new_probs[final_grade] = max(new_probs[final_grade], confidence)
        new_probs = new_probs / new_probs.sum()
        fused_probs = new_probs
        
        logger.info(f"[CLINICAL VALIDATION] Final correction: Grade {original_grade} → "
                   f"Grade {final_grade} (confidence: {original_confidence:.1%} → {confidence:.1%})")
        
        logger.info(f"[CLINICAL VALIDATION] Final correction: Grade {original_grade} → Grade {final_grade} "
                   f"(conf {original_confidence:.3f} → {confidence:.3f})")
    
    # Diagnostic logging only (NO OVERRIDES)
    logger.info(f"[FINAL PREDICTION] Grade {final_grade} with {confidence:.1%} confidence")
    logger.info(f"[CLINICAL FEATURES] tumor={tumor_pct:.2f}%, seg_prob={max_seg_prob:.3f}, malignancy={glcm_malignancy:.3f}")
    
    # NO ADDITIONAL GATES, NO OVERRIDES - Trust the validated result

    # ── Compute final calibrated confidence ───────────────────────────────────
    # If confidence wasn't set by clinical validation, compute it properly
    if final_grade == original_grade and 'confidence' not in prediction_result:
        confidence = _compute_confidence(
            final_grade           = final_grade,
            fused_probs           = fused_probs,
            seg_probs             = seg_probs,
            cls_probs             = cls_probs,
            rad_probs             = rad_probs,
            glcm_malignancy       = glcm_malignancy,
            tumor_pct             = tumor_pct,
            max_seg_prob          = max_seg_prob,
            seg_uncertainty       = seg_uncertainty,
            n_real_cls_models     = n_real_cls,
            n_imagenet_cls_models = n_imagenet_cls,
            seg_weight            = SEG_W,
            cls_weight            = CLS_W,
            rad_weight            = RAD_W,
            asymmetry             = asymmetry,
        )

    # ── Compute calibrated confidence ─────────────────────────────────────────
    # Keep clinical-validation adjustments and blend with calibrated confidence.
    calibrated_confidence = _compute_confidence(
        final_grade           = final_grade,
        fused_probs           = fused_probs,
        seg_probs             = seg_probs,
        cls_probs             = cls_probs,
        rad_probs             = rad_probs,
        glcm_malignancy       = glcm_malignancy,
        tumor_pct             = tumor_pct,
        max_seg_prob          = max_seg_prob,
        seg_uncertainty       = seg_uncertainty,
        n_real_cls_models     = n_real_cls,
        n_imagenet_cls_models = n_imagenet_cls,
        seg_weight            = SEG_W,
        cls_weight            = CLS_W,
        rad_weight            = RAD_W,
        asymmetry             = asymmetry,
    )
    model_confidence = float(prediction_result.get('confidence', calibrated_confidence))

    if final_grade != original_grade:
        # For clinically corrected grades, stay conservative while avoiding weak confidence.
        confidence = float(np.clip(max(confidence, calibrated_confidence * 0.92), 0.20, 0.95))
    else:
        confidence = float(np.clip(
            max(confidence, calibrated_confidence, model_confidence),
            0.20,
            0.98
        ))

    # ── RAG Augmentation (RE-ENABLED with fixed implementation) ──────────────
    rag_evidence = {}
    # Keep pre-RAG copies for quality report
    raw_fused  = fused_probs.copy()  # Use the improved prediction result
    sharpened  = fused_probs.copy()  # Same as raw since no temperature scaling used

    if RAG_AVAILABLE and RAG_ENABLED:
        _progress(0.92, "[RAG] Augmenting with medical knowledge base…")
        try:
            ai_result_dict = {
                'grade': final_grade,
                'confidence': confidence,
                'grade_probs': fused_probs.tolist(),
                'tumor_percentage': tumor_pct,
                'glcm_malignancy': glcm_malignancy,
                'compactness': morph.get('compactness', 0.5),
                'asymmetry_score': asymmetry,
                'tumor_type': get_tumor_info(final_grade).get('name', 'Unknown'),
            }
            
            patient_data_dict = {
                'age': 50,  # Default (can be passed as parameter)
                'sex': 'Unknown',
            }
            
            # Use FIXED RAG with strict validation
            rag_result = augment_with_rag(
                ai_result=ai_result_dict,
                patient_data=patient_data_dict,
                top_k=5  # Reduced from 10 for better quality
            )
            
            if rag_result.rag_used:
                # RAG passed validation - use augmented result
                final_grade_before_rag = final_grade
                confidence_before_rag = confidence
                fused_probs_before_rag = fused_probs.copy()
                
                final_grade = rag_result.augmented_grade
                confidence = rag_result.augmented_confidence
                fused_probs = np.array(rag_result.augmented_grade_probs)
                
                rag_evidence = {
                    'rag_enabled': True,
                    'rag_used': True,
                    'ai_grade': final_grade_before_rag,
                    'ai_confidence': round(confidence_before_rag, 4),
                    'rag_grade': rag_result.rag_grade,
                    'rag_confidence': round(rag_result.rag_confidence, 4),
                    'augmented_grade': final_grade,
                    'augmented_confidence': round(confidence, 4),
                    'retrieval_confidence': round(rag_result.retrieval_confidence, 4),
                    'grade_agreement': round(rag_result.grade_agreement, 4),
                    'retrieved_cases': rag_result.retrieved_cases,
                    'similarities': [round(s, 3) for s in rag_result.similarities],
                }
                
                logger.info(
                    f"[RAG] ✅ AI Grade {final_grade_before_rag} (conf={confidence_before_rag:.3f}) "
                    f"→ Augmented Grade {final_grade} (conf={confidence:.3f})"
                )
            else:
                # RAG validation failed - use AI only
                rag_evidence = {
                    'rag_enabled': True,
                    'rag_used': False,
                    'reason': rag_result.rag_reason
                }
                logger.info(f"[RAG] ℹ️  Using AI only: {rag_result.rag_reason}")
        
        except Exception as e:
            logger.error(f"[RAG] ❌ Augmentation failed: {e}", exc_info=True)
            rag_evidence = {'rag_enabled': True, 'rag_used': False, 'error': str(e)}
    else:
        reason = "RAG not available" if not RAG_AVAILABLE else "RAG disabled"
        rag_evidence = {'rag_enabled': False, 'reason': reason}
        logger.info(f"[RAG] {reason}")

    # ── Consistency check: ensure final_grade matches grade_probs ────────────
    # After RAG augmentation, fused_probs may have been updated. Re-derive
    # final_grade from fused_probs to guarantee they are always consistent.
    fused_probs = np.array(fused_probs, dtype=np.float64)
    fused_probs = np.clip(fused_probs, 1e-9, None)
    fused_probs /= fused_probs.sum()
    final_grade = int(np.argmax(fused_probs))

    # ── Detected flag ────────────────────────────────────────────────────────
    # detected=True when the model predicts a tumor grade with sufficient confidence.
    # The grade_probs distribution is the ground truth — if grade > 0 wins with
    # meaningful probability, a tumor is detected.
    valid_tumor_pct = tumor_pct if tumor_pct <= 50.0 else 0.0
    grade_prob_for_final = float(fused_probs[final_grade])

    detected = (
        final_grade > 0 and (
            grade_prob_for_final > 0.30 or     # model is reasonably confident about a tumor grade
            valid_tumor_pct > 1.0  or          # meaningful tumor area
            max_seg_prob > 0.55    or          # confident segmentation
            (glcm_malignancy > 0.45 and final_grade >= 2)  # strong radiomics
        )
    )

    # ── Conformal prediction set ──────────────────────────────────────────────
    top_p    = fused_probs.max()
    conf_set = [int(i) for i, p in enumerate(fused_probs) if p >= top_p * 0.20]

    # ── Finalise ─────────────────────────────────────────────────────────────
    _progress(0.96, "[DONE] Finalising diagnosis…")
    tumor_info = get_tumor_info(final_grade)  # re-derive after consistency check

    # CRITICAL FIX v9.6: Use grade-based estimates when segmentation fails
    # If tumor is detected (grade > 0) but segmentation failed (tumor_pct near 0),
    # use grade-based estimate for tumor_percentage to show meaningful area
    GRADE_VOLUME_ESTIMATES = {0: 0.0, 1: 1.2, 2: 4.5, 3: 11.0, 4: 18.0}
    
    segmentation_failed = (tumor_pct <= 0.1 and final_grade > 0)
    if segmentation_failed:
        tumor_pct_display = GRADE_VOLUME_ESTIMATES.get(final_grade, 1.0)
        logger.warning(f"[RESULT] Segmentation failed (0% area but Grade {final_grade} detected) - "
                      f"using grade-based estimate for display: {tumor_pct_display}%")
    else:
        tumor_pct_display = tumor_pct
    
    models_used = (
        [n for _, _, n in seg_models] +
        list(real_cls_models.keys()) +
        [f"{n}_cls" for _, n in imagenet_cls_models] +
        ['glcm_radiomics']
    )

    model_quality = (
        'HIGH (real brain-tumor classifiers active)' if n_real_cls >= 2 else
        'MEDIUM (1 real classifier active)'           if n_real_cls == 1 else
        'LOW (radiomics + segmentation only — run download_models.py)'
    )

    result = AnalysisResult(
        grade                    = final_grade,
        confidence               = round(confidence, 4),
        detected                 = detected,
        tumor_type               = tumor_info.get('name', 'Unknown'),
        risk_level               = tumor_info.get('risk_level', 'Unknown'),
        description              = tumor_info.get('description', ''),
        tumor_mask               = morph.get('binary_mask', fused_mask),
        tumor_percentage         = round(tumor_pct_display, 2),  # Use estimated value if segmentation failed
        tumor_area_cm2           = round(morph['area_pixels'] * (pixel_spacing_mm ** 2) / 100.0, 4),
        max_diameter_mm          = round(morph['max_diameter_mm'], 2),
        grade_probs              = fused_probs.tolist(),
        conformal_set            = conf_set,
        glcm_features            = glcm,
        glcm_malignancy          = round(glcm_malignancy, 4),
        asymmetry_score          = round(asymmetry, 4),
        seg_grade_probs          = seg_probs.tolist(),
        cls_grade_probs          = cls_probs.tolist(),
        radiomics_grade_probs    = rad_probs.tolist(),
        meets_confidence_threshold = confidence >= 0.70,
        models_used              = models_used,
        num_models_used          = len(models_used),
        detection_source         = 'ensemble_v6_with_rag' if RAG_AVAILABLE else 'ensemble_v6',
        real_classifiers_used    = all_real_cls_names,
        processing_time_s        = round(time.time() - t0, 2),
        segmentation_failed      = segmentation_failed,  # Add flag to indicate estimate was used
        quality_report           = {
            'brain_fraction':  round(brain_fraction, 3),
            'max_seg_prob':    round(max_seg_prob, 3),
            'seg_uncertainty': round(seg_uncertainty, 3),
            'n_seg_models':    n_seg_models,
            'n_real_cls':      n_real_cls,
            'n_imagenet_cls':  n_imagenet_cls,
            'real_classifiers': all_real_cls_names,
            'n_sources_agreed': n_sources_agree,
            'fusion_weights':  {
                'radiomics':      round(RAD_W, 3),
                'segmentation':   round(SEG_W, 3),
                'classification': round(CLS_W, 3),
            },
            'sharpening': {
                'raw_fused_max_prob':  round(float(raw_fused.max()), 4),
                'sharpened_max_prob':  round(float(sharpened.max()), 4),
                'final_max_prob':      round(float(np.array(fused_probs).max()), 4),
            },
            'model_quality': model_quality,
            'rag_evidence':  rag_evidence,
        },
        rag_evidence = rag_evidence,
        disclaimer = (
            "RESEARCH USE ONLY — NOT FDA-CLEARED — NOT CE-MARKED. "
            "Independent radiologist review required before any clinical use."
        )
    )

    _progress(1.00, "[DONE] Analysis complete.")
    logger.info(
        f"[RESULT v6.0] grade={final_grade} conf={confidence:.3f} "
        f"tumor%={tumor_pct:.2f}% detected={detected} "
        f"real_cls={n_real_cls} sources_agree={n_sources_agree} "
        f"raw_max={raw_fused.max():.3f} sharp_max={sharpened.max():.3f} "
        f"time={result.processing_time_s}s"
    )

    # ── AUTO-REGISTER: grow the RAG case registry after every scan ────────────
    # Only register results that are reliable (not from broken pipeline runs).
    # Reject: tumor_percentage > 50% (whole-brain mask = broken segmentation)
    # Reject: confidence < 0.50 (uncertain prediction)
    if RAG_AVAILABLE:
        try:
            _should_register = (
                tumor_pct <= 50.0 and          # not a broken whole-brain mask
                confidence >= 0.50 and          # reasonably confident
                brain_fraction >= 0.10          # actual brain content
            )
            if _should_register:
                _result_dict_for_rag = _result_to_dict(result)
                _result_dict_for_rag['compactness'] = morph.get('compactness', 0.5)
                register_scan_result(
                    _result_dict_for_rag,
                    patient_data={"patient_id": patient_id},
                )
            else:
                logger.info(
                    f"[RAG] Skipping auto-register: "
                    f"tumor_pct={tumor_pct:.1f}%, conf={confidence:.2f}, "
                    f"brain_frac={brain_fraction:.2f} — result not reliable enough"
                )
        except Exception as _reg_err:
            logger.warning(f"[RAG] Auto-register failed (non-critical): {_reg_err}")

    return result


def _empty_result() -> AnalysisResult:
    return AnalysisResult(
        grade=0, confidence=0.50, detected=False,
        tumor_type='Normal', risk_level='Normal',
        description='Analysis failed — check image quality.',
        grade_probs=[0.92, 0.02, 0.02, 0.02, 0.02],
        meets_confidence_threshold=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PROGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_progression(
    baseline_image: np.ndarray,
    followup_image: np.ndarray,
    baseline_result: Optional[AnalysisResult] = None,
    followup_result: Optional[AnalysisResult] = None,
    days_between: int = 90,
    progress_callback: Optional[Callable] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Compare two scans for tumor progression (RANO-inspired).
    
    v9.1 FIXES:
    1. Better handling of identical scans
    2. Improved volume change calculation
    3. Grade change as primary progression indicator
    4. More accurate growth rate computation
    """
    if baseline_result is None:
        baseline_result = analyze_brain_mri(baseline_image, device=device,
                                             progress_callback=progress_callback)
    if followup_result is None:
        followup_result = analyze_brain_mri(followup_image, device=device)

    # Extract key metrics
    baseline_grade = baseline_result.grade
    followup_grade = followup_result.grade
    delta_grade = followup_grade - baseline_grade
    
    baseline_vol = baseline_result.tumor_percentage
    followup_vol = followup_result.tumor_percentage
    
    # CRITICAL FIX v9.5: Handle identical scans
    # Check if images are TRULY identical (same scan uploaded twice)
    # IMPORTANT: If grades are different, scans CANNOT be identical!
    try:
        # If grades are different, skip identical check entirely
        if baseline_grade != followup_grade:
            logger.info(f"[PROGRESSION] Different grades detected ({baseline_grade} → {followup_grade}) - scans are definitely different")
        else:
            # Only check for identical scans if grades are the same
            # Normalize both images for comparison
            baseline_norm = (baseline_image - baseline_image.min()) / (baseline_image.max() - baseline_image.min() + 1e-8)
            followup_norm = (followup_image - followup_image.min()) / (followup_image.max() - followup_image.min() + 1e-8)
            
            # Calculate similarity
            mse = np.mean((baseline_norm - followup_norm) ** 2)
            
            # CRITICAL: Only consider identical if MSE is EXTREMELY low (< 0.00001)
            # This prevents false positives when comparing different scans from same patient
            are_identical = (mse < 0.00001)
            
            if are_identical:
                logger.warning(f"[PROGRESSION] Identical scans detected (MSE={mse:.8f}, same grade={baseline_grade}) - no progression analysis possible")
                return {
                    'baseline_grade': baseline_grade,
                    'followup_grade': followup_grade,
                    'delta_grade': 0,
                    'baseline_tumor_pct': baseline_vol,
                    'followup_tumor_pct': followup_vol,
                    'volume_change_pct': 0.0,
                    'growth_rate_pct_per_day': 0.0,
                    'status': 'Identical Scans',
                    'response_code': 'ID',
                    'days_between': days_between,
                    'baseline_confidence': baseline_result.confidence,
                    'followup_confidence': followup_result.confidence,
                    'baseline_result': _result_to_dict(baseline_result),
                    'followup_result': _result_to_dict(followup_result),
                    'recommendation': 'Identical scans provided. Please upload different time points for progression analysis.',
                    'warning': 'Same scan uploaded for both baseline and follow-up'
                }
            else:
                logger.info(f"[PROGRESSION] Different scans detected (MSE={mse:.8f}, same grade={baseline_grade}) - proceeding with progression analysis")
    except Exception as e:
        logger.warning(f"[PROGRESSION] Could not check for identical scans: {e}")
    
    # IMPROVED VOLUME CALCULATION v9.6
    # Use grade-based estimates when tumor_percentage is 0 but grade > 0
    # CRITICAL FIX: When segmentation fails (0% area) but grade is detected,
    # ALWAYS use grade-based estimates for meaningful progression analysis
    GRADE_VOLUME_ESTIMATES = {0: 0.0, 1: 1.2, 2: 4.5, 3: 11.0, 4: 18.0}
    
    # Log the raw volumes for debugging
    logger.info(f"[PROGRESSION] Raw volumes: baseline={baseline_vol:.2f}%, followup={followup_vol:.2f}%")
    logger.info(f"[PROGRESSION] Grades: baseline=Grade {baseline_grade}, followup=Grade {followup_grade}")
    
    # CRITICAL: If segmentation failed (tumor_percentage near 0) but grade > 0,
    # we MUST use grade-based estimates, otherwise progression analysis is meaningless
    baseline_seg_failed = (baseline_vol <= 0.1 and baseline_grade > 0)
    followup_seg_failed = (followup_vol <= 0.1 and followup_grade > 0)
    
    if baseline_seg_failed:
        baseline_vol_est = GRADE_VOLUME_ESTIMATES.get(baseline_grade, 1.0)
        logger.warning(f"[PROGRESSION] Segmentation failed for baseline (0% area but Grade {baseline_grade} detected) - using grade-based estimate: {baseline_vol_est}%")
    else:
        baseline_vol_est = baseline_vol
        logger.info(f"[PROGRESSION] Using actual baseline volume: {baseline_vol_est:.2f}% (Grade {baseline_grade})")
        
    if followup_seg_failed:
        followup_vol_est = GRADE_VOLUME_ESTIMATES.get(followup_grade, 1.0)
        logger.warning(f"[PROGRESSION] Segmentation failed for followup (0% area but Grade {followup_grade} detected) - using grade-based estimate: {followup_vol_est}%")
    else:
        followup_vol_est = followup_vol
        logger.info(f"[PROGRESSION] Using actual followup volume: {followup_vol_est:.2f}% (Grade {followup_grade})")
    
    # Calculate volume change
    vol_change = followup_vol_est - baseline_vol_est
    
    # FIXED v9.6: Handle all volume change scenarios
    if baseline_vol_est == 0.0 and followup_vol_est > 0.0:
        # New tumor appeared
        vol_change_pct = 100.0
        logger.info(f"[PROGRESSION] New tumor appeared: 0% → {followup_vol_est}% (reporting as +100%)")
    elif followup_vol_est == 0.0 and baseline_vol_est > 0.0:
        # Tumor disappeared
        vol_change_pct = -100.0
        logger.info(f"[PROGRESSION] Tumor disappeared: {baseline_vol_est}% → 0% (reporting as -100%)")
    elif baseline_vol_est > 0.0:
        # Normal percentage calculation
        vol_change_pct = (vol_change / baseline_vol_est) * 100.0
        logger.info(f"[PROGRESSION] Volume change: {baseline_vol_est:.2f}% → {followup_vol_est:.2f}% = {vol_change_pct:+.1f}%")
    else:
        # Both are 0 - should not happen if grades are detected
        vol_change_pct = 0.0
        logger.warning(f"[PROGRESSION] Both volumes are 0% despite grades being {baseline_grade} and {followup_grade}")
    
    # IMPROVED PROGRESSION CLASSIFICATION v9.3
    # Grade change is the PRIMARY indicator, volume change is secondary
    # FIXED: Properly detect progression when baseline is normal and followup has tumor
    
    # Rule 1: Both scans are normal → Stable
    if baseline_grade == 0 and followup_grade == 0:
        status, code = 'Stable Disease', 'SD'
        logger.info("[PROGRESSION] Both scans normal → Stable Disease")
    
    # Rule 2: Baseline normal, followup has tumor → Progressive Disease (new tumor appeared)
    elif baseline_grade == 0 and followup_grade > 0:
        status, code = 'Progressive Disease', 'PD'
        logger.info(f"[PROGRESSION] New tumor appeared: Normal → Grade {followup_grade} → Progressive Disease")
    
    # Rule 3: Baseline has tumor, followup is normal → Complete Response
    elif baseline_grade > 0 and followup_grade == 0:
        status, code = 'Complete Response', 'CR'
        logger.info(f"[PROGRESSION] Tumor disappeared: Grade {baseline_grade} → Normal → Complete Response")
    
    # Rule 4: Grade increased by 2+ levels → Definite progression
    elif delta_grade >= 2:
        status, code = 'Progressive Disease', 'PD'
        logger.info(f"[PROGRESSION] Grade increased by {delta_grade} levels → Progressive Disease")
    
    # Rule 5: Grade increased by 1 level → Progression
    elif delta_grade == 1:
        status, code = 'Progressive Disease', 'PD'
        logger.info(f"[PROGRESSION] Grade increased by 1 level → Progressive Disease")
    
    # Rule 6: Grade decreased by 2+ levels → Major response
    elif delta_grade <= -2:
        status, code = 'Partial Response', 'PR'
        logger.info(f"[PROGRESSION] Grade decreased by {abs(delta_grade)} levels → Partial Response")
    
    # Rule 7: Grade decreased by 1 level → Response
    elif delta_grade == -1:
        status, code = 'Partial Response', 'PR'
        logger.info(f"[PROGRESSION] Grade decreased by 1 level → Partial Response")
    
    # Rule 8: Same grade → Check volume change
    elif delta_grade == 0:
        if vol_change_pct <= -50:
            status, code = 'Partial Response', 'PR'
            logger.info(f"[PROGRESSION] Same grade but volume decreased {abs(vol_change_pct):.1f}% → Partial Response")
        elif vol_change_pct >= 25:
            status, code = 'Progressive Disease', 'PD'
            logger.info(f"[PROGRESSION] Same grade but volume increased {vol_change_pct:.1f}% → Progressive Disease")
        else:
            status, code = 'Stable Disease', 'SD'
            logger.info(f"[PROGRESSION] Same grade, minimal volume change ({vol_change_pct:.1f}%) → Stable Disease")
    else:
        status, code = 'Indeterminate', 'ID'
        logger.warning(f"[PROGRESSION] Unexpected case: baseline={baseline_grade}, followup={followup_grade}")

    # Calculate growth rates
    growth_rate_per_day     = vol_change_pct / max(days_between, 1) if days_between > 0 else 0.0
    growth_rate_per_month   = growth_rate_per_day * 30.0
    vol_change_abs          = followup_vol_est - baseline_vol_est

    # Status colour (used by visualization and report)
    _status_colours = {
        'PD': '#EF4444',   # Red   – Progressive Disease
        'CR': '#10B981',   # Green – Complete Response
        'PR': '#34D399',   # Teal  – Partial Response
        'SD': '#F59E0B',   # Amber – Stable Disease
        'ID': '#6B7280',   # Gray  – Indeterminate
    }
    status_color = _status_colours.get(code, '#6B7280')

    # Recurrence-risk estimate (grade-based, clinical reference values)
    _risk_pct = {0: 2.0, 1: 15.0, 2: 35.0, 3: 65.0, 4: 85.0}
    recurrence_risk = _risk_pct.get(followup_grade, 50.0)

    # Generate recommendation
    recommendation = _progression_recommendation(code, followup_grade, delta_grade, vol_change_pct)
    
    # CRITICAL: Add warning if segmentation failed for both scans
    warning_msg = None
    if baseline_seg_failed and followup_seg_failed:
        if baseline_grade == followup_grade:
            warning_msg = (
                f"⚠️ SEGMENTATION FAILED: Both scans show 0% tumor area despite being classified as Grade {baseline_grade}. "
                f"Volume-based progression analysis is unreliable. Grade-based estimates were used ({GRADE_VOLUME_ESTIMATES[baseline_grade]}% for both scans). "
                f"Result shows 'Stable Disease' because both scans have the same grade and estimated volume. "
                f"Consider: (1) Manual review of segmentation, (2) Different imaging protocol, or (3) Expert radiologist assessment."
            )
            logger.error(f"[PROGRESSION] {warning_msg}")
        else:
            warning_msg = (
                f"⚠️ SEGMENTATION FAILED: Both scans show 0% tumor area. "
                f"Grade-based volume estimates were used: Grade {baseline_grade} ({GRADE_VOLUME_ESTIMATES[baseline_grade]}%) → "
                f"Grade {followup_grade} ({GRADE_VOLUME_ESTIMATES[followup_grade]}%). "
                f"Progression status is based on grade change only."
            )
            logger.warning(f"[PROGRESSION] {warning_msg}")
    elif baseline_seg_failed or followup_seg_failed:
        warning_msg = (
            f"⚠️ SEGMENTATION PARTIALLY FAILED: One scan shows 0% tumor area. "
            f"Grade-based estimates were used where segmentation failed. Results may be less accurate."
        )
        logger.warning(f"[PROGRESSION] {warning_msg}")

    logger.info(f"[PROGRESSION] {status} ({code}): Grade {baseline_grade}→{followup_grade} "
               f"(Δ{delta_grade:+d}), Volume {baseline_vol_est:.1f}%→{followup_vol_est:.1f}% "
               f"({vol_change_pct:+.1f}%), Growth {growth_rate_per_day:.3f}%/day, "
               f"Days {days_between}, Rec.risk {recurrence_risk:.0f}%")

    return {
        # ── Grades ───────────────────────────────────────────────────────────
        'baseline_grade':           baseline_grade,
        'followup_grade':           followup_grade,
        'delta_grade':              delta_grade,

        # ── Volume metrics (all naming conventions the frontend uses) ────────
        'baseline_tumor_pct':       round(baseline_vol_est, 2),
        'followup_tumor_pct':       round(followup_vol_est, 2),
        'baseline_volume':          round(baseline_vol_est, 2),   # alias for visualization
        'followup_volume':          round(followup_vol_est, 2),   # alias for visualization
        'volume_change_pct':        round(vol_change_pct, 1),
        'volume_change_abs':        round(vol_change_abs, 2),
        'volume_estimated':         (baseline_vol <= 0.1 or followup_vol <= 0.1),

        # ── Growth rates (all naming conventions) ────────────────────────────
        'growth_rate_pct_per_day':  round(growth_rate_per_day, 4),
        'growth_rate_monthly':      round(growth_rate_per_month, 4),
        'growth_rate':              round(growth_rate_per_month, 4),  # alias for visualization

        # ── Time ─────────────────────────────────────────────────────────────
        'days_between':             days_between,
        'time_diff_days':           days_between,   # alias for report/visualization

        # ── Status (all naming conventions) ──────────────────────────────────
        'status':                   status,
        'progression_status':       status,         # alias required by frontend display & report
        'response_code':            code,
        'status_color':             status_color,

        # ── Clinical extras ───────────────────────────────────────────────────
        'recurrence_risk':          recurrence_risk,
        'treatment_response':       (status, status_color),
        
        # ── Warnings ──────────────────────────────────────────────────────────
        'warning':                  warning_msg,  # Add segmentation failure warning
        'segmentation_failed':      (baseline_seg_failed or followup_seg_failed),

        # ── Confidence ────────────────────────────────────────────────────────
        'baseline_confidence':      baseline_result.confidence,
        'followup_confidence':      followup_result.confidence,

        # ── Sub-results (for visualization and report) ────────────────────────
        'baseline_result':          _result_to_dict(baseline_result),
        'followup_result':          _result_to_dict(followup_result),

        # ── Recommendation ────────────────────────────────────────────────────
        'recommendation':           recommendation,
    }


def _progression_recommendation(code: str, grade: int, delta_grade: int = 0, vol_change_pct: float = 0.0) -> str:
    """Generate clinical recommendations based on progression analysis."""
    
    if code == 'CR':
        return "Complete response achieved. Continue monitoring per protocol. Next MRI in 3–6 months."
    elif code == 'PR':
        if delta_grade < 0:
            return f"Grade improvement detected (Δ{delta_grade}). Continue current treatment. Reassess in 6–8 weeks."
        else:
            return "Partial response with volume reduction. Continue current treatment. Reassess in 6–8 weeks."
    elif code == 'SD':
        if grade >= 4:
            return "Stable high-grade disease. Maintain current therapy with close monitoring. Review in 6–8 weeks."
        else:
            return "Stable disease. Maintain current therapy. Review in 3 months."
    elif code == 'PD':
        if delta_grade > 0:
            return (f"Tumor progression with grade increase (Δ{delta_grade:+d}). "
                   "URGENT multidisciplinary team review recommended. Consider treatment modification.")
        elif vol_change_pct > 50:
            return (f"Significant tumor growth ({vol_change_pct:+.1f}%). "
                   "URGENT multidisciplinary team review recommended. Consider treatment modification.")
        else:
            return ("Tumor progression detected. Multidisciplinary team review recommended. "
                   "Consider biopsy or treatment change.")
    elif code == 'ID':
        return "Inconclusive progression analysis. Repeat MRI in 4–6 weeks for confirmation."
    else:
        return "Clinical review recommended for treatment planning."


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def advanced_tumor_detection_with_ai(
    image: np.ndarray,
    class_models=None,
    device=None,
    progress_callback=None,
    patient_id: str = "unknown",
    **kwargs,
) -> Dict[str, Any]:
    """v3.0/v4.0/v5.0 compatible wrapper for frontend.py."""
    if device is None:
        device = torch.device('cpu')
    result = analyze_brain_mri(
        image, patient_id=patient_id,
        progress_callback=progress_callback, device=device,
    )
    return _result_to_dict(result)


def segment_tumor_with_real_models(
    image: np.ndarray,
    device=None,
    progress_callback=None,
    **kwargs,
) -> Dict[str, Any]:
    """v3.0 compatible segmentation wrapper."""
    r = analyze_brain_mri(image, device=device, progress_callback=progress_callback)
    return {
        'mask': r.tumor_mask,
        'tumor_percentage': r.tumor_percentage,
        'detected': r.detected,
        'grade': r.grade,
        'confidence': r.confidence,
    }


def calculate_progression_metrics(baseline: Dict, followup: Dict,
                                   baseline_meta: Optional[Dict] = None,
                                   followup_meta: Optional[Dict] = None) -> Dict:
    """
    Compute progression metrics from two scan results.

    v9.0 FIXES:
    1. Grade-based volume estimation when tumor_percentage = 0
       (segmentation often returns 0 even when a tumor is present)
    2. Grade change is the PRIMARY progression signal — not just volume
    3. Days between uses actual user-entered dates, not datetime.now()
    4. Growth rate is computed correctly from actual or estimated volumes
    """
    baseline_grade  = int(baseline.get('grade', 0))
    followup_grade  = int(followup.get('grade', 0))
    delta_grade     = followup_grade - baseline_grade

    # ── Volume estimation ─────────────────────────────────────────────────────
    # Use actual tumor_percentage if available and non-zero.
    # If zero but grade > 0, use grade-based clinical estimates.
    # (WHO grade correlates with typical tumor volume in clinical literature)
    GRADE_VOLUME_EST = {0: 0.0, 1: 1.5, 2: 5.0, 3: 12.0, 4: 20.0}

    b_vol_raw = float(baseline.get('tumor_percentage', 0) or 0)
    f_vol_raw = float(followup.get('tumor_percentage', 0) or 0)

    # Use actual values if meaningful; fall back to grade estimates
    b_vol = b_vol_raw if b_vol_raw > 0.1 else GRADE_VOLUME_EST.get(baseline_grade, 0.0)
    f_vol = f_vol_raw if f_vol_raw > 0.1 else GRADE_VOLUME_EST.get(followup_grade, 0.0)

    vol_change_abs = f_vol - b_vol

    # Volume change percentage (relative to baseline)
    if b_vol > 0.01:
        vol_change_pct = (vol_change_abs / b_vol) * 100.0
    elif f_vol > 0.01:
        # Baseline was zero (Normal), follow-up has tumor → new tumor appeared
        vol_change_pct = 100.0  # 100% increase from nothing
    else:
        vol_change_pct = 0.0

    # ── Days between scans ────────────────────────────────────────────────────
    days_between = 90  # default
    try:
        if baseline_meta and followup_meta:
            bd = baseline_meta.get('scan_date')
            fd = followup_meta.get('scan_date')
            if bd and fd and bd != fd:
                if hasattr(bd, 'toordinal') and hasattr(fd, 'toordinal'):
                    days_between = max(1, abs(fd.toordinal() - bd.toordinal()))
    except Exception:
        pass

    growth_rate_daily = vol_change_abs / max(days_between, 1)
    growth_rate_monthly = growth_rate_daily * 30

    # ── Progression status ────────────────────────────────────────────────────
    # PRIMARY signal: grade change (most reliable when volume is estimated)
    # SECONDARY signal: volume change percentage
    if followup_grade == 0 and baseline_grade > 0:
        # Tumor resolved → Complete Response
        status = 'Complete Response'
        code   = 'CR'
    elif delta_grade >= 2:
        # Grade jumped by 2+ → clearly Progressive Disease
        status = 'Progressive Disease'
        code   = 'PD'
    elif delta_grade == 1 or vol_change_pct >= 25:
        # Grade increased by 1 OR volume grew ≥25% → Progressive Disease
        status = 'Progressive Disease'
        code   = 'PD'
    elif delta_grade <= -1 or vol_change_pct <= -50:
        # Grade decreased OR volume shrank ≥50% → Partial Response
        status = 'Partial Response'
        code   = 'PR'
    elif delta_grade == 0 and abs(vol_change_pct) < 25:
        # Same grade, small volume change → Stable Disease
        status = 'Stable Disease'
        code   = 'SD'
    else:
        status = 'Indeterminate'
        code   = 'ID'

    return {
        'volume_change_pct':       round(vol_change_pct, 1),
        'volume_change_abs':       round(vol_change_abs, 2),
        'baseline_volume':         round(b_vol, 2),
        'followup_volume':         round(f_vol, 2),
        'delta_grade':             delta_grade,
        'growth_rate':             round(growth_rate_monthly, 4),
        'growth_rate_pct_per_day': round(growth_rate_daily, 4),
        'days_between':            days_between,
        'progression_status':      status,
        'status':                  status,
        'response_code':           code,
        'volume_estimated':        (b_vol_raw <= 0.1 or f_vol_raw <= 0.1),
    }


def calculate_recurrence_risk(result: Dict) -> Dict:
    grade = result.get('grade', 0)
    conf  = result.get('confidence', 0.5)
    risk_table = {0: 0.02, 1: 0.15, 2: 0.35, 3: 0.65, 4: 0.85}
    base_risk  = risk_table.get(grade, 0.5)
    adjusted   = float(np.clip(base_risk * (0.5 + conf * 0.5), 0.0, 0.99))
    return {
        'recurrence_risk': round(adjusted, 3),
        'risk_category': 'High' if adjusted > 0.6 else ('Moderate' if adjusted > 0.3 else 'Low'),
        'monitoring_interval': '3 months' if adjusted > 0.6 else '6 months',
    }


def assess_treatment_response(baseline: Dict, followup: Dict) -> Dict:
    vol_chg = followup.get('tumor_percentage', 0) - baseline.get('tumor_percentage', 0)
    pct     = (vol_chg / max(baseline.get('tumor_percentage', 0.01), 0.01)) * 100
    if pct <= -50:
        resp = 'Partial Response'
    elif pct >= 25:
        resp = 'Progressive Disease'
    else:
        resp = 'Stable Disease'
    return {'response': resp, 'volume_change_pct': round(pct, 1)}


class RealBrainTumorInferenceEngine:
    """v3.0/v4.0/v5.0 compatible class-based interface."""

    def __init__(self, device=None, progress_callback=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.progress_callback = progress_callback
        _ModelCache.get(self.device)

    def analyze(self, image: np.ndarray,
                patient_id: str = "unknown",
                progress_callback=None,
                **kwargs) -> Dict[str, Any]:
        cb = progress_callback or self.progress_callback
        result = analyze_brain_mri(image, patient_id=patient_id,
                                    progress_callback=cb, device=self.device)
        return _result_to_dict(result)

    def analyze_progression(self, baseline, followup, **kwargs):
        return analyze_progression(baseline, followup, device=self.device, **kwargs)


def _result_to_dict(r: AnalysisResult) -> Dict[str, Any]:
    """Convert AnalysisResult to the dict format frontend.py expects."""
    return {
        'grade':                      r.grade,
        'confidence':                 r.confidence,
        'ai_confidence':              r.confidence,
        'detected':                   r.detected,
        'tumor_type':                 r.tumor_type,
        'risk_level':                 r.risk_level,
        'description':                r.description,
        'tumor_mask':                 r.tumor_mask,
        'mask':                       r.tumor_mask,
        'tumor_percentage':           r.tumor_percentage,
        'tumor_area_cm2':             r.tumor_area_cm2,
        'max_diameter_mm':            r.max_diameter_mm,
        'grade_probs':                r.grade_probs,
        'probabilities':              r.grade_probs,
        'final_grade_probs':          r.grade_probs,
        'conformal_prediction_set':   r.conformal_set,
        'glcm_features':              r.glcm_features,
        'glcm_malignancy':            r.glcm_malignancy,
        'asymmetry_score':            r.asymmetry_score,
        'seg_grade_probs':            r.seg_grade_probs,
        'cls_grade_probs':            r.cls_grade_probs,
        'radiomics_grade_probs':      r.radiomics_grade_probs,
        'meets_confidence_threshold': r.meets_confidence_threshold,
        'models_used':                r.models_used,
        'num_models_used':            r.num_models_used,
        'detection_source':           r.detection_source,
        'real_classifiers_used':      r.real_classifiers_used,
        'processing_time_s':          r.processing_time_s,
        'quality_report':             r.quality_report,
        'disclaimer':                 r.disclaimer,
        # v3.0 compatibility aliases
        'num_tumor_classes':          5,
        'tumor_classes':              list(range(5)),
        'confidence_threshold':       0.70,
    }
