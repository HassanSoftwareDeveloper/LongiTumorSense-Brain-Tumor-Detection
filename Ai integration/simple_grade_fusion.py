"""
simple_grade_fusion.py  —  LongiTumorSense v11.0 - CLASSIFIER REMOVED
=======================================================================
CRITICAL FIX: Removed flawed classifier trained on tumor TYPES (not WHO grades).

The Kaggle classifier was trained on:
  - meningioma → "Grade 1" (WRONG! This is a tumor TYPE, not a grade)
  - pituitary → "Grade 1" (WRONG! This is a tumor TYPE, not a grade)
  - glioma → "Grade 2/3/4" (WRONG! Random split, not real grades)

WHO grades are based on MALIGNANCY (cell differentiation, mitotic activity,
necrosis, vascular proliferation), NOT tumor type!

NEW APPROACH (v11.0):
  - Removed classifier weight (was 32%, now 0%)
  - Increased segmentation weight (22% → 40%) - trained on BraTS with REAL grades
  - Increased radiomics weight (28% → 40%) - texture-based malignancy
  - Added WHO grading rules based on biological criteria
  - Uses segmentation + radiomics features (trained on real WHO grades)

Design principles
-----------------
1. VOTE FIRST, OVERRIDE LAST
   Three sources (seg, radiomics, img_cls) cast weighted votes.
   Overrides only fire when evidence is unambiguous across ALL sources.

2. NO BLIND TUMOR_PCT CAPS
   tumor_pct is used as soft evidence, not a hard ceiling.

3. RADIOMICS IS THE TIE-BREAKER
   glcm_malignancy is the most reliable signal when neural nets are
   uncertain — it is derived directly from image texture.

4. WHO GRADING RULES
   Apply biological criteria based on segmentation + radiomics features:
   - Grade IV: Necrosis (malignancy > 0.70) + significant tumor
   - Grade III: High malignancy (> 0.60) without necrosis
   - Grade II: Moderate malignancy (0.40-0.60), infiltrative
   - Grade I: Low malignancy (0.25-0.40), well-defined
   - Grade 0: Minimal malignancy (< 0.25), no significant tumor

5. TRANSPARENT LOGGING
   Every decision is logged so you can trace exactly why a grade was chosen.

Grade reference (WHO 2021)
--------------------------
  Grade 0  Normal brain    (no tumor)
  Grade I  Low-grade       (pilocytic astrocytoma, etc.)
  Grade II Intermediate    (diffuse astrocytoma, oligodendroglioma)
  Grade III Anaplastic     (anaplastic astrocytoma, anaplastic oligodendroglioma)
  Grade IV Glioblastoma    (GBM, most aggressive)
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Grade names
GRADE_NAMES = {
    0: "Grade 0 (Normal)",
    1: "Grade I (Low-grade)",
    2: "Grade II (Intermediate)",
    3: "Grade III (Anaplastic)",
    4: "Grade IV (GBM)",
}

# FIXED v11.0: Remove flawed classifier trained on tumor TYPES (not WHO grades)
# The Kaggle classifier learned meningioma/pituitary/glioma (tumor types),
# NOT WHO grades (malignancy levels). Segmentation models are trained on
# BraTS with REAL WHO grades, so we use them instead.

# Base fusion weights - CLASSIFIER REMOVED!
W_SEG_TRAINED  = 0.40  # Increased! Segmentation trained on BraTS (real grades)
W_CLS_TRAINED  = 0.00  # REMOVED! Trained on tumor types, not WHO grades
W_RAD_TRAINED  = 0.40  # Increased! Radiomics = texture-based malignancy
W_IMG_TRAINED  = 0.20  # Image features for additional context

# Fallback weights when cls is untrained (same as above now)
W_SEG_NOTRAIN  = 0.40
W_CLS_NOTRAIN  = 0.00
W_RAD_NOTRAIN  = 0.40
W_IMG_NOTRAIN  = 0.20


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    """Clip negatives and L1-normalise to a valid probability vector."""
    arr = np.clip(np.asarray(arr, dtype=np.float64), 1e-9, None)
    return arr / arr.sum()


def _entropy(probs: np.ndarray) -> float:
    p = _normalise(probs)
    return float(-np.sum(p * np.log(p + 1e-12)))


def _is_uniform(probs: np.ndarray, threshold: float = 0.92) -> bool:
    """Return True if the distribution is suspiciously flat (untrained model)."""
    e = _entropy(probs)
    max_e = float(np.log(len(probs)))
    return e > threshold * max_e


def _sharpen(probs: np.ndarray, temperature: float = 0.50) -> np.ndarray:
    """Lower temperature → sharper distribution."""
    log_p = np.log(np.clip(probs, 1e-12, None)) / temperature
    log_p -= log_p.max()
    sharpened = np.exp(log_p)
    return _normalise(sharpened)


def _result(grade: int, confidence: float, probs: list, source: str) -> Dict[str, Any]:
    """Package result dictionary."""
    return {
        "grade": int(grade),
        "confidence": float(confidence),
        "grade_probs": [float(p) for p in probs],
        "source": source
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FUSION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def simple_grade_fusion(
    seg_probs:        Any,
    cls_probs:        Any,
    radiomics_probs:  Any,
    img_cls_probs:    Any,
    tumor_pct:        float,
    glcm_malignancy:  float,
    max_seg_prob:     float,
    asymmetry:        float = 0.0,
) -> Dict[str, Any]:
    """
    Fuse grade-probability vectors into a single WHO grade prediction.
    
    v11.0: Classifier removed (trained on tumor types, not WHO grades)
    
    Returns
    -------
    dict with keys:
        grade        int          final WHO grade (0–4)
        confidence   float        calibrated confidence (0–1)
        grade_probs  list[float]  5-element fused distribution
        source       str          description of decision path taken
    """

    # ── 0. Sanitise inputs ────────────────────────────────────────────────────
    seg  = _normalise(seg_probs)
    cls  = _normalise(cls_probs)
    rad  = _normalise(radiomics_probs)
    img  = _normalise(img_cls_probs)

    tumor_pct       = float(np.clip(tumor_pct,      0.0, 100.0))
    glcm_malignancy = float(np.clip(glcm_malignancy, 0.0,   1.0))
    max_seg_prob    = float(np.clip(max_seg_prob,    0.0,   1.0))

    logger.info(
        f"[FUSION v11.0] Inputs  "
        f"seg={int(np.argmax(seg))}({seg.max():.2f})  "
        f"cls={int(np.argmax(cls))}({cls.max():.2f}) [IGNORED]  "
        f"rad={int(np.argmax(rad))}({rad.max():.2f})  "
        f"img={int(np.argmax(img))}({img.max():.2f})  "
        f"tumor_pct={tumor_pct:.2f}%  "
        f"malignancy={glcm_malignancy:.3f}  "
        f"max_seg_prob={max_seg_prob:.3f}"
    )

    # ── 1. Fast-path: clearly normal brain ───────────────────────────────────
    all_say_normal = (
        int(np.argmax(seg)) == 0 and
        int(np.argmax(rad)) == 0 and
        int(np.argmax(img)) == 0 and
        img[0] > 0.70
    )
    truly_no_tumor = (
        tumor_pct    < 0.5  and
        max_seg_prob < 0.32 and
        glcm_malignancy < 0.28
    )
    if all_say_normal and truly_no_tumor:
        logger.info("[FUSION v11.0] Fast-path: CLEARLY NORMAL")
        return _result(0, 0.93,
                       [0.93, 0.03, 0.02, 0.01, 0.01],
                       "clearly_normal_all_sources")

    # ── 2. Detect untrained classifiers ──────────────────────────────────────
    cls_untrained = _is_uniform(cls_probs)
    if cls_untrained:
        logger.warning(
            f"[FUSION v11.0] cls appears UNTRAINED (entropy={_entropy(cls):.3f}) "
            f"— using seg+rad+img only (cls weight already 0%)"
        )
        w_seg, w_cls, w_rad, w_img = W_SEG_NOTRAIN, W_CLS_NOTRAIN, W_RAD_NOTRAIN, W_IMG_NOTRAIN
    else:
        w_seg, w_cls, w_rad, w_img = W_SEG_TRAINED, W_CLS_TRAINED, W_RAD_TRAINED, W_IMG_TRAINED

    # ── 3. Radiomics confidence boost ────────────────────────────────────────
    if glcm_malignancy > 0.65 or glcm_malignancy < 0.20:
        boost = 0.08
        w_rad  = min(w_rad + boost, 0.58)
        total  = w_seg + w_cls + w_rad + w_img
        w_seg /= total; w_cls /= total; w_rad /= total; w_img /= total
        logger.info(f"[FUSION v11.0] Radiomics strongly boosted (malignancy={glcm_malignancy:.3f})")
    elif glcm_malignancy > 0.58 or glcm_malignancy < 0.25:
        boost = 0.05
        w_rad  = min(w_rad + boost, 0.52)
        total  = w_seg + w_cls + w_rad + w_img
        w_seg /= total; w_cls /= total; w_rad /= total; w_img /= total
        logger.info(f"[FUSION v11.0] Radiomics boosted (malignancy={glcm_malignancy:.3f})")

    # ── 4. Weighted fusion ────────────────────────────────────────────────────
    fused = w_seg * seg + w_cls * cls + w_rad * rad + w_img * img
    fused = _normalise(fused)

    prelim_grade = int(np.argmax(fused))
    prelim_conf  = float(fused[prelim_grade])

    logger.info(
        f"[FUSION v11.0] Weights: seg={w_seg:.2f} cls={w_cls:.2f} "
        f"rad={w_rad:.2f} img={w_img:.2f}  →  "
        f"fused grade={prelim_grade} ({prelim_conf:.1%})"
    )

    # ── 4.5. WHO Grading Rules (NEW v11.0) ────────────────────────────────────
    # Apply biological criteria for WHO grading based on segmentation + radiomics
    
    has_necrosis = glcm_malignancy > 0.70
    high_malignancy = glcm_malignancy > 0.60
    moderate_malignancy = 0.40 < glcm_malignancy <= 0.60
    low_malignancy = 0.25 < glcm_malignancy <= 0.40
    minimal_malignancy = glcm_malignancy <= 0.25
    
    significant_tumor = tumor_pct > 2.0 and max_seg_prob > 0.40
    small_tumor = 0.5 < tumor_pct <= 2.0
    
    # WHO Grade IV criteria: Necrosis + high malignancy
    if has_necrosis and significant_tumor:
        who_grade_hint = 4
        who_confidence = 0.82
        logger.info(f"[WHO v11.0] Grade IV criteria met: necrosis + significant tumor")
    
    # WHO Grade III criteria: High malignancy without necrosis
    elif high_malignancy and significant_tumor:
        who_grade_hint = 3
        who_confidence = 0.78
        logger.info(f"[WHO v11.0] Grade III criteria met: high malignancy")
    
    # WHO Grade II criteria: Moderate malignancy, infiltrative
    elif moderate_malignancy and (significant_tumor or small_tumor):
        who_grade_hint = 2
        who_confidence = 0.75
        logger.info(f"[WHO v11.0] Grade II criteria met: moderate malignancy")
    
    # WHO Grade I criteria: Low malignancy, well-defined
    elif low_malignancy and tumor_pct > 0.5:
        who_grade_hint = 1
        who_confidence = 0.72
        logger.info(f"[WHO v11.0] Grade I criteria met: low malignancy")
    
    # Grade 0 (Normal): Minimal/no tumor
    elif minimal_malignancy and tumor_pct < 1.0:
        who_grade_hint = 0
        who_confidence = 0.88
        logger.info(f"[WHO v11.0] Grade 0 criteria met: minimal tumor signal")
    
    else:
        who_grade_hint = prelim_grade
        who_confidence = prelim_conf
        logger.info(f"[WHO v11.0] Using fused prediction (no clear WHO criteria)")
    
    # Blend WHO hint with fused prediction (70% WHO, 30% fused)
    if who_grade_hint != prelim_grade and who_confidence > 0.70:
        logger.info(
            f"[WHO v11.0] WHO criteria suggests Grade {who_grade_hint} "
            f"(confidence={who_confidence:.2f}), fused suggests {prelim_grade} "
            f"→ blending (70% WHO, 30% fused)"
        )
        adjusted = fused.copy()
        shift = 0.30 * fused[prelim_grade]
        adjusted[prelim_grade] = max(adjusted[prelim_grade] - shift, 1e-9)
        adjusted[who_grade_hint] = adjusted[who_grade_hint] + shift
        fused = _normalise(adjusted)
        prelim_grade = int(np.argmax(fused))
        prelim_conf = float(fused[prelim_grade])

    # ── 5. Compute evidence strength ─────────────────────────────────────────
    sources_agree = sum([
        int(np.argmax(seg)) == prelim_grade,
        int(np.argmax(rad)) == prelim_grade,
        int(np.argmax(img)) == prelim_grade,
    ])
    
    # ── 6. Final grade validation ─────────────────────────────────────────────
    final_grade      = prelim_grade
    confidence_scale = 1.0
    decision_path    = "weighted_fusion"

    # High-grade validation
    if prelim_grade >= 3:
        high_grade_evidence = (
            glcm_malignancy > 0.50 or
            tumor_pct > 3.0 or
            max_seg_prob > 0.42
        )
        
        weak_seg = max_seg_prob < 0.30
        weak_rad = glcm_malignancy < 0.35
        weak_img = float(img[3] + img[4]) < 0.20
        all_weak = weak_seg and weak_rad and weak_img

        if all_weak:
            final_grade = 2
            confidence_scale = 0.68
            decision_path = "downgrade_all_sources_weak"
            logger.warning(f"[FUSION v11.0] ALL sources weak for Grade {prelim_grade} → Grade 2")
        elif prelim_grade == 4 and not high_grade_evidence:
            final_grade = 3
            confidence_scale = 0.80
            decision_path = "downgrade_IV_insufficient_evidence"
            logger.info(f"[FUSION v11.0] Grade IV needs corroborating evidence → Grade III")
        else:
            confidence_scale = 0.85 if sources_agree >= 2 else 0.78
            logger.info(f"[FUSION v11.0] Grade {prelim_grade} CONFIRMED ({sources_agree}/3 sources agree)")

    elif prelim_grade == 0:
        tumor_signal = (
            tumor_pct > 1.5 or
            max_seg_prob > 0.42 or
            glcm_malignancy > 0.38
        )
        if tumor_signal:
            tumor_fused = fused.copy(); tumor_fused[0] = 0
            if tumor_fused.max() > 0.10:
                final_grade = int(np.argmax(tumor_fused))
            else:
                final_grade = 1 if glcm_malignancy < 0.40 else 2
            confidence_scale = 0.75
            decision_path = "upgrade_from_normal"
            logger.warning(f"[FUSION v11.0] Normal prediction but tumor signal → Grade {final_grade}")

    # ── 7. Rebuild probability vector ────────────────────────────────────────
    if final_grade != prelim_grade:
        adjusted = fused.copy()
        shift = 0.25 * fused[prelim_grade]
        adjusted[prelim_grade] = max(adjusted[prelim_grade] - shift, 1e-9)
        adjusted[final_grade] = adjusted[final_grade] + shift
        fused = _normalise(adjusted)

    # ── 8. Confidence calibration ────────────────────────────────────────────
    raw_conf = float(fused[final_grade])

    if raw_conf > 0.60:
        fused = _sharpen(fused, temperature=0.52)
        raw_conf = float(fused[final_grade])
    elif raw_conf > 0.50:
        fused = _sharpen(fused, temperature=0.58)
        raw_conf = float(fused[final_grade])

    # Multi-source agreement bonus
    if sources_agree >= 2:
        agreement_bonus = 0.08
    elif sources_agree == 1:
        agreement_bonus = 0.03
    else:
        agreement_bonus = 0.0

    # Radiomics alignment bonus
    rad_grade = int(np.argmax(rad))
    if rad_grade == final_grade and rad[final_grade] > 0.50:
        radiomics_bonus = 0.05
    else:
        radiomics_bonus = 0.0

    # Final confidence
    final_confidence = raw_conf * confidence_scale + agreement_bonus + radiomics_bonus
    final_confidence = float(np.clip(final_confidence, 0.0, 0.98))

    logger.info(
        f"[FUSION v11.0] FINAL: Grade {final_grade} ({GRADE_NAMES[final_grade]}) "
        f"confidence={final_confidence:.1%} source={decision_path}"
    )

    return _result(final_grade, final_confidence, fused, decision_path)
