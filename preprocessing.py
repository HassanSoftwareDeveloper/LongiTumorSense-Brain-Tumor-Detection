"""
preprocessing.py — LongiTumorSense v4.0  ─ FIXED
=================================================
KEY FIX: Added the missing `preprocess_mri` and `prepare_image_for_display`
functions that helper.py and frontend.py import but v3.0 never exported.
This single missing export was the root cause of ALL import failures logged
in brain_tumor_analysis.log.

Also adds:
  - Robust `load_medical_image` for DICOM / NIfTI / PNG / JPG
  - Sequence-aware normalisation
  - N4-style bias field correction (no ITK required)
  - Motion / skull-strip quality checks

All functions degrade gracefully — no external C++ binaries required.
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom not installed — DICOM support unavailable")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logger.warning("nibabel not installed — NIfTI support unavailable")

try:
    from skimage import morphology, measure, exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  MRI SEQUENCE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class MRISequenceClassifier:
    """
    Identify MRI sequence (T1 / T2 / FLAIR / T1-Gd / DWI / UNKNOWN)
    from histogram heuristics or DICOM tags.
    """

    SEQUENCES = ['T1', 'T2', 'FLAIR', 'T1-Gd', 'DWI', 'UNKNOWN']

    def classify_from_array(self, image: np.ndarray,
                            dicom_metadata: Optional[Dict] = None) -> str:
        if dicom_metadata:
            seq = self._from_dicom(dicom_metadata)
            if seq != 'UNKNOWN':
                return seq
        return self._from_histogram(image)

    def _from_dicom(self, meta: Dict) -> str:
        seq_name = str(
            meta.get('sequence_name', '') or meta.get('SequenceName', '') or
            meta.get('series_description', '') or meta.get('SeriesDescription', '')
        ).upper()
        if any(k in seq_name for k in ['FLAIR', 'FL']):
            return 'FLAIR'
        if any(k in seq_name for k in ['T1', 'MP-RAGE', 'MPRAGE', 'FLASH']):
            contrast = str(meta.get('ContrastBolusAgent', '') or
                           meta.get('contrast_agent', '')).strip()
            return 'T1-Gd' if contrast else 'T1'
        if any(k in seq_name for k in ['T2', 'TSE', 'FSE']):
            return 'T2'
        if any(k in seq_name for k in ['DWI', 'DTI', 'DIFFUSION', 'EPI']):
            return 'DWI'
        return 'UNKNOWN'

    def _from_histogram(self, image: np.ndarray) -> str:
        img = image.astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return 'UNKNOWN'
        img = (img - mn) / (mx - mn)
        brain = img > 0.05
        if brain.sum() < 200:
            return 'UNKNOWN'
        bp = img[brain]
        f_bright = float((bp > 0.75).mean())
        f_dark   = float((bp < 0.15).mean())
        f_mid    = float(((bp > 0.35) & (bp < 0.65)).mean())
        cv_val   = float(bp.std() / (bp.mean() + 1e-8))
        if cv_val > 0.55:
            return 'DWI'
        if f_bright > 0.25 and f_dark > 0.20:
            return 'T2'
        if f_dark > 0.30 and f_mid > 0.30:
            return 'FLAIR'
        if f_mid > 0.45 and f_dark < 0.15:
            return 'T1'
        return 'UNKNOWN'

    def get_normalisation_params(self, sequence: str) -> Dict:
        params = {
            'T1':      {'method': 'percentile', 'low': 0.5,  'high': 99.5},
            'T2':      {'method': 'percentile', 'low': 0.5,  'high': 99.5},
            'FLAIR':   {'method': 'zscore',     'clip_sigma': 3.0},
            'T1-Gd':   {'method': 'percentile', 'low': 1.0,  'high': 99.0},
            'DWI':     {'method': 'zscore',     'clip_sigma': 2.5},
            'UNKNOWN': {'method': 'minmax'},
        }
        return params.get(sequence, params['UNKNOWN'])


# ═══════════════════════════════════════════════════════════════════════════════
#  BIAS FIELD CORRECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class BiasFieldCorrector:
    """
    Approximate N4 bias field correction without ITK.
    Tries SimpleITK first; falls back to Gaussian log-domain approach.
    """

    def __init__(self, sigma: float = 40.0):
        self.sigma = sigma

    def correct(self, image: np.ndarray) -> np.ndarray:
        orig_dtype = image.dtype
        img = image.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return image
        try:
            corrected = self._try_sitk_n4(img)
            if corrected is not None:
                return corrected.astype(orig_dtype)
        except Exception:
            pass
        eps = 1.0
        log_img  = np.log(img - mn + eps)
        bias_est = cv2.GaussianBlur(log_img, (0, 0), self.sigma,
                                    borderType=cv2.BORDER_REFLECT)
        log_corr  = log_img - bias_est
        corrected = np.exp(log_corr) - eps + mn
        c_min, c_max = corrected.min(), corrected.max()
        if c_max > c_min:
            corrected = (corrected - c_min) / (c_max - c_min) * (mx - mn) + mn
        return corrected.astype(orig_dtype)

    @staticmethod
    def _try_sitk_n4(image: np.ndarray) -> Optional[np.ndarray]:
        try:
            import SimpleITK as sitk
            sitk_img    = sitk.GetImageFromArray(image)
            sitk_img    = sitk.Cast(sitk_img, sitk.sitkFloat32)
            mask_filter = sitk.OtsuThresholdImageFilter()
            mask_filter.SetInsideValue(0)
            mask_filter.SetOutsideValue(1)
            mask_img    = mask_filter.Execute(sitk_img)
            corrector   = sitk.N4BiasFieldCorrectionImageFilter()
            corrected   = corrector.Execute(sitk_img, mask_img)
            return sitk.GetArrayFromImage(corrected)
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"SimpleITK N4 failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#  INTENSITY NORMALISER
# ═══════════════════════════════════════════════════════════════════════════════

class IntensityNormaliser:
    """Sequence-aware intensity normalisation."""

    def normalise(self, image: np.ndarray, method: str = 'minmax', **kwargs) -> np.ndarray:
        img = image.astype(np.float32)
        if method == 'minmax':
            mn, mx = img.min(), img.max()
            return (img - mn) / (mx - mn + 1e-8)
        elif method == 'percentile':
            low  = kwargs.get('low', 0.5)
            high = kwargs.get('high', 99.5)
            p_lo = float(np.percentile(img, low))
            p_hi = float(np.percentile(img, high))
            img  = np.clip(img, p_lo, p_hi)
            return (img - p_lo) / (p_hi - p_lo + 1e-8)
        elif method == 'zscore':
            clip_sigma = kwargs.get('clip_sigma', 3.0)
            mean, std  = float(img.mean()), float(img.std()) + 1e-8
            img        = (img - mean) / std
            img        = np.clip(img, -clip_sigma, clip_sigma)
            return (img + clip_sigma) / (2 * clip_sigma)
        else:
            mn, mx = img.min(), img.max()
            return (img - mn) / (mx - mn + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
#  SKULL STRIP VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SkullStripValidator:
    """Heuristic check for whether the brain has been skull-stripped."""

    def assess(self, image: np.ndarray) -> Dict:
        img = image.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return {'is_skull_stripped': False, 'confidence': 0.0}
        norm = (img - mn) / (mx - mn)
        edge_width = max(1, int(min(norm.shape) * 0.05))
        edge_pixels = np.concatenate([
            norm[:edge_width, :].ravel(),
            norm[-edge_width:, :].ravel(),
            norm[:, :edge_width].ravel(),
            norm[:, -edge_width:].ravel(),
        ])
        mean_edge = float(np.mean(edge_pixels))
        is_stripped = mean_edge < 0.08
        confidence = float(np.clip(1.0 - mean_edge / 0.15, 0, 1))
        return {'is_skull_stripped': is_stripped, 'confidence': confidence,
                'mean_edge_intensity': mean_edge}


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTION ARTIFACT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MotionArtifactDetector:
    """Detect motion / ringing artifacts via row-variance analysis."""

    def assess(self, image: np.ndarray) -> Dict:
        img = image.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return {'motion_detected': False, 'motion_score': 0.0,
                    'severity': 'none'}
        norm = (img - mn) / (mx - mn)
        row_vars  = np.var(norm, axis=1)
        col_vars  = np.var(norm, axis=0)
        row_cv = float(np.std(row_vars) / (np.mean(row_vars) + 1e-8))
        col_cv = float(np.std(col_vars) / (np.mean(col_vars) + 1e-8))
        score  = float((row_cv + col_cv) / 2)
        detected = score > 0.5
        severity = 'none' if score < 0.3 else ('mild' if score < 0.6 else 'severe')
        return {'motion_detected': detected, 'motion_score': round(score, 4),
                'severity': severity}


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDICAL IMAGE PREPROCESSOR (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class MedicalImagePreprocessor:
    """
    Full preprocessing pipeline for a single uploaded file.
    Returns (processed_array, metadata, quality_report).
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 apply_bias_corr: bool = True,
                 apply_normalisation: bool = True):
        self.target_size         = target_size
        self.apply_bias_corr     = apply_bias_corr
        self.apply_normalisation = apply_normalisation
        self.seq_classifier      = MRISequenceClassifier()
        self.bias_corrector      = BiasFieldCorrector()
        self.intensity_normaliser= IntensityNormaliser()
        self.skull_validator     = SkullStripValidator()
        self.motion_detector     = MotionArtifactDetector()

    def load_and_preprocess(self, uploaded_file
                            ) -> Tuple[Optional[np.ndarray], Dict, Dict]:
        image, metadata = self._load(uploaded_file)
        if image is None:
            return None, {}, {'error': 'Failed to load image'}
        processed, quality = self._preprocess(image, metadata)
        return processed, metadata, quality

    # ── loaders ──────────────────────────────────────────────────────────────

    def _load(self, uploaded_file) -> Tuple[Optional[np.ndarray], Dict]:
        try:
            # Support file-like objects (Streamlit) and path strings
            if hasattr(uploaded_file, 'name'):
                name = uploaded_file.name.lower()
                file_bytes = uploaded_file.read()
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
            elif isinstance(uploaded_file, (str, Path)):
                p = Path(uploaded_file)
                name = p.name.lower()
                file_bytes = p.read_bytes()
            else:
                file_bytes = uploaded_file.read()
                name = ''

            if name.endswith('.dcm') or name.endswith(''):
                result = self._load_dicom(file_bytes)
                if result[0] is not None:
                    return result

            if name.endswith(('.nii', '.nii.gz')):
                return self._load_nifti(file_bytes)

            # Standard image (PNG / JPG / BMP / TIFF)
            pil_img = Image.open(io.BytesIO(file_bytes)).convert('L')
            arr     = np.array(pil_img, dtype=np.float32)
            return arr, {'format': 'standard_image', 'sequence_name': ''}

        except Exception as e:
            logger.error(f"Image load error: {e}")
            return None, {}

    def _load_dicom(self, file_bytes: bytes) -> Tuple[Optional[np.ndarray], Dict]:
        if not PYDICOM_AVAILABLE:
            return None, {}
        try:
            ds  = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            arr = ds.pixel_array.astype(np.float32)
            meta = {
                'format': 'dicom',
                'PatientID': str(getattr(ds, 'PatientID', '')),
                'StudyDate': str(getattr(ds, 'StudyDate', '')),
                'Modality':  str(getattr(ds, 'Modality', '')),
                'SeriesDescription': str(getattr(ds, 'SeriesDescription', '')),
                'PixelSpacing': getattr(ds, 'PixelSpacing', None),
                'SliceThickness': getattr(ds, 'SliceThickness', None),
            }
            return arr, meta
        except Exception as e:
            logger.debug(f"DICOM load failed: {e}")
            return None, {}

    def _load_nifti(self, file_bytes: bytes) -> Tuple[Optional[np.ndarray], Dict]:
        if not NIBABEL_AVAILABLE:
            return None, {}
        try:
            fh  = nib.FileHolder(fileobj=io.BytesIO(file_bytes))
            img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
            arr = img.get_fdata().astype(np.float32)
            return arr, {'format': 'nifti', 'shape': arr.shape,
                         'voxel_sizes': img.header.get_zooms()}
        except Exception as e:
            logger.debug(f"NIfTI load failed: {e}")
            return None, {}

    # ── core preprocess ───────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray,
                    metadata: Dict) -> Tuple[np.ndarray, Dict]:
        quality: Dict = {}

        # Convert RGB → greyscale
        if image.ndim == 3 and image.shape[2] in (3, 4):
            image = image.mean(axis=2)

        is_volume = image.ndim == 3 and image.shape[0] > 4
        rep_slice = image[image.shape[0] // 2] if is_volume else image.squeeze()

        if is_volume:
            quality['is_3d_volume'] = True
            quality['n_slices']     = int(image.shape[0])

        # 1. Sequence detection
        seq = self.seq_classifier.classify_from_array(rep_slice, metadata)
        metadata['inferred_sequence'] = seq
        quality['sequence_type'] = seq

        # 2. Motion detection
        quality['motion'] = self.motion_detector.assess(rep_slice)

        # 3. Skull-strip check
        quality['skull_strip'] = self.skull_validator.assess(rep_slice)
        metadata['skull_stripped'] = quality['skull_strip']['is_skull_stripped']

        # 4. Bias correction
        if self.apply_bias_corr:
            try:
                rep_slice = self.bias_corrector.correct(rep_slice)
                quality['bias_correction'] = 'applied'
            except Exception as e:
                quality['bias_correction'] = f'failed: {e}'

        # 5. Intensity normalisation
        norm_p = self.seq_classifier.get_normalisation_params(seq)
        if self.apply_normalisation:
            rep_slice = IntensityNormaliser().normalise(rep_slice, **norm_p)
            quality['normalisation'] = norm_p['method']

        # 6. Resize to target
        rep_slice = np.clip(
            cv2.resize(rep_slice.astype(np.float32), self.target_size,
                       interpolation=cv2.INTER_LINEAR),
            0.0, 1.0).astype(np.float32)

        if is_volume:
            vol = np.zeros((image.shape[0], *self.target_size), dtype=np.float32)
            for d in range(image.shape[0]):
                sl = image[d].astype(np.float32)
                if self.apply_bias_corr:
                    sl = self.bias_corrector.correct(sl)
                if self.apply_normalisation:
                    sl = IntensityNormaliser().normalise(sl, **norm_p)
                vol[d] = np.clip(
                    cv2.resize(sl, self.target_size, interpolation=cv2.INTER_LINEAR),
                    0.0, 1.0)
            return vol, quality

        return rep_slice, quality


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE FUNCTIONS  (PUBLIC API)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Shared singleton preprocessor ────────────────────────────────────────────
_PREPROCESSOR = MedicalImagePreprocessor(target_size=(224, 224))


def preprocess_mri(image: np.ndarray,
                   target_size: Tuple[int, int] = (224, 224),
                   apply_bias_corr: bool = True,
                   apply_normalisation: bool = True,
                   metadata: Optional[Dict] = None) -> np.ndarray:
    """
    PRIMARY ENTRY POINT — was missing in v3.0, causing all import failures.

    Preprocess a raw MRI numpy array for model inference.

    Parameters
    ----------
    image               : Raw array (H×W) or (H×W×C) or (D×H×W) volume
    target_size         : Output spatial size (default 224×224)
    apply_bias_corr     : Apply N4-style bias field correction
    apply_normalisation : Apply sequence-aware intensity normalisation
    metadata            : Optional DICOM/NIfTI metadata dict for seq detection

    Returns
    -------
    np.ndarray  float32  in [0, 1],  shape (H, W)  or (D, H, W)
    """
    if metadata is None:
        metadata = {}
    proc = MedicalImagePreprocessor(
        target_size=target_size,
        apply_bias_corr=apply_bias_corr,
        apply_normalisation=apply_normalisation,
    )
    processed, _ = proc._preprocess(image.copy(), metadata)
    return processed


def prepare_image_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalise an array to [0, 1] float32 for display / overlay purposes.
    Works on 2-D slices and 3-D volumes (returns normalised version).
    """
    img = image.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def load_medical_image(uploaded_file,
                       target_size: Tuple[int, int] = (224, 224)
                       ) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Load a medical image from an uploaded file-like object.

    Supports: DICOM (.dcm), NIfTI (.nii/.nii.gz), PNG, JPG, BMP, TIFF.

    Returns
    -------
    (image_array, metadata_dict)
    image_array : float32 array, normalised to [0, 1], resized to target_size
                  None if loading failed.
    """
    proc = MedicalImagePreprocessor(target_size=target_size)
    image, metadata = proc._load(uploaded_file)
    if image is None:
        return None, {}
    # Normalise & resize the representative slice for display
    processed, quality = proc._preprocess(image, metadata)
    metadata['quality'] = quality
    return processed, metadata


def preprocess_uploaded_file(uploaded_file,
                              target_size: Tuple[int, int] = (224, 224)
                              ) -> Tuple[Optional[np.ndarray], Dict, Dict]:
    """Returns (image, metadata, quality_report)."""
    return MedicalImagePreprocessor(target_size=target_size).load_and_preprocess(uploaded_file)


def classify_mri_sequence(image: np.ndarray,
                           metadata: Optional[Dict] = None) -> str:
    return MRISequenceClassifier().classify_from_array(image, metadata or {})


def assess_scan_motion(image: np.ndarray) -> Dict:
    return MotionArtifactDetector().assess(image)


def correct_bias_field(image: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    return BiasFieldCorrector(sigma=sigma).correct(image)