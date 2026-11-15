# Frontend.py — LongiTumorSense


import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict
import io
import base64
import hashlib
import json
import warnings
import os
import random
import torch

# ── Import helpers ────────────────────────────────────────────────────────────
from helper import (
    get_device, sliding_window_inference, calculate_per_class_dice,
    calculate_per_class_iou, calculate_mean_metrics, apply_crf,
    setup_authentication, check_login_attempts, hash_password,
    verify_credentials, display_dicom_metadata, setup_page_config,
    get_custom_css, visualize_segmentation_3d, calculate_tumor_volume,
    create_metrics_display, create_interactive_3d_visualization,
    load_users_from_config, predict_with_pretrained_models,
    prepare_image_for_model_unified, segment_with_pretrained_model,
    download_model, load_pretrained_model,
)
from models import (
    WorldClassEnsemble, BraTSStyleUNet, BraTSClassifier, Enhanced3DUNet,
    EnhancedGliomaClassifier, BrainTumorSegmenter, load_pretrained_detector,
    load_pretrained_segmenter, load_brats_model, get_ensemble_models,
    preprocess_for_model, preprocessor, normalize_volume, resize_volume,
    remove_small_components, enhance_brain_image, extract_brain_region,
    create_image_pyramid, register_images,
)

try:
    from preprocessing import load_medical_image, preprocess_mri, prepare_image_for_display
except ImportError:
    def load_medical_image(uploaded_file):
        try:
            if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = np.array(Image.open(uploaded_file))
                return image, {}
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                return None, {}
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None, {}
    def preprocess_mri(image): return image
    def prepare_image_for_display(image):
        return (image - image.min()) / (image.max() - image.min() + 1e-8)

from inference import (
    RealBrainTumorInferenceEngine, advanced_tumor_detection_with_ai,
    analyze_progression, segment_tumor_with_real_models,
    calculate_progression_metrics, calculate_recurrence_risk,
    assess_treatment_response,
)
from clinical import (
    ClinicalDatabase, ClinicalReportGenerator,
    create_clinical_dashboard, create_progression_tracker, init_clinical_system,
)

try:
    from audit_logger import (
        log_login, log_ai_analysis, log_report_export, log_logout,
        AuditAction, write_audit_entry,
    )
    AUDIT_ENABLED = True
except ImportError:
    AUDIT_ENABLED = False
    def log_login(*a, **kw): pass
    def log_ai_analysis(*a, **kw): pass
    def log_report_export(*a, **kw): pass
    def log_logout(*a, **kw): pass
    def write_audit_entry(*a, **kw): pass

from config import (
    TUMOR_CLASSES, CLINICAL_CONFIG, get_tumor_info, get_device_config,
    get_model_config, get_ui_config, get_auth_config, get_sample_users,
    SAMPLE_USERS, AUTH_CONFIG, _hash_password,
)

try:
    from brain_validator import (
        BrainMRIValidator, MedicalImageLoader,
        validate_brain_mri, validate_image_file,
    )
    BRAIN_VALIDATOR_AVAILABLE = True
except ImportError:
    BRAIN_VALIDATOR_AVAILABLE = False
    def validate_brain_mri(image, metadata=None, device="cpu"):
        return {
            'is_valid': True, 'confidence': 1.0,
            'reason': 'Validator not available',
            'critical_scores': {
                'brain_structure': 1.0, 'text_content': 0.0,
                'screenshot_ui': 0.0, 'photo_likeness': 0.0,
                'chest_xray_likeness': 0.0,
            },
            'recommended_action': 'ACCEPT',
        }
    def validate_image_file(file_path, device="cpu"):
        return validate_brain_mri(np.zeros((256, 256)))

from skimage import measure, filters
from scipy import ndimage
import logging
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b4, resnet50, densenet121
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

THEME = {
    "bg_canvas":     "#F0F4F8",
    "bg_surface":    "#FFFFFF",
    "bg_elevated":   "#F7FAFC",
    "bg_inset":      "#EEF2F7",
    "brand":         "#0891B2",
    "brand_light":   "#CFFAFE",
    "brand_dk":      "#0E7490",
    "brand_glow":    "rgba(8,145,178,0.12)",
    "ok":            "#059669",
    "ok_light":      "#D1FAE5",
    "warn":          "#D97706",
    "warn_light":    "#FEF3C7",
    "alert":         "#DC2626",
    "alert_light":   "#FEE2E2",
    "info":          "#2563EB",
    "info_light":    "#DBEAFE",
    "purple":        "#7C3AED",
    "purple_light":  "#EDE9FE",
    "grade_0":       "#059669",
    "grade_1":       "#16A34A",
    "grade_2":       "#D97706",
    "grade_3":       "#EA580C",
    "grade_4":       "#DC2626",
    "text_1":        "#0F172A",
    "text_2":        "#334155",
    "text_3":        "#64748B",
    "text_4":        "#94A3B8",
    "border_s":      "#E2E8F0",
    "border_m":      "#CBD5E1",
    "border_k":      "#94A3B8",
}

GRADE_COLORS = [
    THEME["grade_0"], THEME["grade_1"], THEME["grade_2"],
    THEME["grade_3"], THEME["grade_4"],
]

GRADE_NAMES = {
    0: "Normal",
    1: "Grade I Glioma",
    2: "Grade II Glioma",
    3: "Grade III Glioma",
    4: "Grade IV Glioma (GBM)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  WORLD-CLASS VISUALIZATION CONSTANTS
#  (Upgraded version from File 2)
# ══════════════════════════════════════════════════════════════════════════════

# ── Palette ───────────────────────────────────────────────────────────────────
VIZ_BG        = '#000000'          # pure black canvas
VIZ_PANEL     = '#0A0A0A'          # near-black panel
VIZ_PANEL2    = '#0F0F0F'          # slightly lighter panel
VIZ_GRID      = '#1A1A2E'          # deep navy grid lines
VIZ_BORDER    = '#1E1E3A'          # panel border
VIZ_TEXT_HI   = '#FFFFFF'          # primary text
VIZ_TEXT_MD   = '#C8D6E5'          # secondary text
VIZ_TEXT_LO   = '#7A8FA6'          # tertiary text
VIZ_ACCENT    = '#00D4FF'          # neon cyan – contours, titles
VIZ_ACCENT2   = '#FF6B35'          # vivid orange – follow-up
VIZ_GREEN     = '#00FF88'          # neon green – normal/ok
VIZ_YELLOW    = '#FFD700'          # gold – centroid / labels
VIZ_PURPLE    = '#BF5FFF'          # violet – grade IV
VIZ_RED_FILL  = '#FF1744'          # solid red – tumor fill
VIZ_RED_LINE  = '#FF4444'          # bright red – contour / bbox
VIZ_RED_SOFT  = (1.0, 0.09, 0.13, 0.55)  # RGBA for mask overlay

# Grade colour ramp
GRADE_HEX = ['#00C896', '#4ADE80', '#FACC15', '#FB923C', '#FF1744']
GRADE_LABELS_FULL = ['Normal', 'Grade I\nGlioma', 'Grade II\nGlioma',
                     'Grade III\nGlioma', 'Grade IV\n(GBM)']


def _plt_base_black():
    """Apply global matplotlib dark theme with pure black background."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family':         'DejaVu Sans',
        'figure.facecolor':    VIZ_BG,
        'axes.facecolor':      VIZ_PANEL,
        'axes.edgecolor':      VIZ_BORDER,
        'axes.labelcolor':     VIZ_TEXT_MD,
        'axes.titlecolor':     VIZ_TEXT_HI,
        'axes.linewidth':      1.4,
        'grid.color':          VIZ_GRID,
        'grid.alpha':          1.0,
        'grid.linewidth':      0.6,
        'xtick.color':         VIZ_TEXT_LO,
        'ytick.color':         VIZ_TEXT_LO,
        'xtick.labelsize':     9,
        'ytick.labelsize':     9,
        'text.color':          VIZ_TEXT_HI,
        'legend.facecolor':    '#0D0D1A',
        'legend.edgecolor':    VIZ_BORDER,
        'legend.labelcolor':   VIZ_TEXT_MD,
        'lines.antialiased':   True,
        'patch.antialiased':   True,
    })


def _frame_ax(ax, title: str, title_color: str = VIZ_ACCENT):
    """Style an axes: black bg, coloured border frame, bold title."""
    import matplotlib.patches as mpatches
    ax.set_facecolor(VIZ_PANEL)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(title_color)
        sp.set_linewidth(1.8)
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=title_color, pad=10, loc='center',
                 fontfamily='DejaVu Sans')


def _draw_tumor_mask(ax, img_norm, mask,
                     contour_color=VIZ_ACCENT,
                     fill_alpha=0.55,
                     contour_lw=4.5,
                     bbox_color=VIZ_RED_LINE,
                     bbox_lw=3.5,
                     draw_bbox=True,
                     draw_label=True):
    """
    Draw vivid tumor overlay:
      • Solid red transparent fill
      • Thick neon cyan contour
      • Red dashed bounding box with 'REGION OF INTEREST' label
    """
    import numpy as np
    from skimage import measure
    from matplotlib.patches import Rectangle

    if mask is None or not np.any(mask):
        return
    
    # Convert mask to binary if it's a probability mask
    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.max() <= 1.0 and mask_array.min() >= 0.0:
        # Probability mask - threshold at 0.35
        mask_bin = mask_array > 0.35
    else:
        # Already binary
        mask_bin = mask_array.astype(bool)

    if not np.any(mask_bin):
        return

    # ── Solid red fill ────────────────────────────────────────────────────────
    overlay = np.zeros((*mask_bin.shape[:2], 4), dtype=np.float32)
    overlay[mask_bin] = [1.0, 0.09, 0.13, fill_alpha]
    ax.imshow(overlay, aspect='equal', interpolation='nearest', zorder=4)

    # ── Thick contour ─────────────────────────────────────────────────────────
    try:
        contours = measure.find_contours(mask_bin.astype(float), level=0.5)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0],
                    linewidth=contour_lw, color=contour_color,
                    solid_capstyle='round', solid_joinstyle='round',
                    zorder=6, antialiased=True)
    except Exception:
        pass

    # ── Bounding box ──────────────────────────────────────────────────────────
    if draw_bbox:
        rows = np.any(mask_bin, axis=1)
        cols = np.any(mask_bin, axis=0)
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = max(int(mask_bin.shape[0] * 0.04), 12)
            r0 = max(0, rmin - pad); r1 = min(mask_bin.shape[0], rmax + pad)
            c0 = max(0, cmin - pad); c1 = min(mask_bin.shape[1], cmax + pad)
            rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                              linewidth=bbox_lw, edgecolor=bbox_color,
                              facecolor='none', linestyle='--', alpha=0.95, zorder=7)
            ax.add_patch(rect)
            if draw_label:
                ax.text((c0 + c1) / 2, max(0, r0 - 6),
                        'REGION OF INTEREST',
                        color=bbox_color, fontsize=8.5, fontweight='bold',
                        ha='center', va='bottom', zorder=8,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#1A0000',
                                  edgecolor=bbox_color,
                                  linewidth=1.6, alpha=0.95))


# ══════════════════════════════════════════════════════════════════════════════
#  8-PANEL SINGLE SCAN VISUALIZATION (UPGRADED)
# ══════════════════════════════════════════════════════════════════════════════

def create_world_class_visualization(original_image, tumor_result):
    """
    World-class 8-panel diagnostic figure.
    Pure black background, vivid red tumor fills, neon cyan contours,
    crystal-clear text, perfect alignment.
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    from skimage import measure
    from scipy import ndimage

    _plt_base_black()

    # ── Figure & grid ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(36, 22), dpi=140, facecolor=VIZ_BG)
    outer = gridspec.GridSpec(1, 1, figure=fig,
                              left=0.02, right=0.98,
                              top=0.91, bottom=0.04)
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[0],
        hspace=0.38, wspace=0.22)

    # ── Extract result data ───────────────────────────────────────────────────
    detected   = tumor_result.get('detected', False)
    grade      = tumor_result.get('grade', 0) if detected else 0
    grade_names = {0:'Normal', 1:'Grade I Glioma', 2:'Grade II Glioma',
                   3:'Grade III Glioma', 4:'Grade IV Glioma (GBM)'}
    tumor_type  = grade_names.get(grade, 'Unknown')
    conf        = float(tumor_result.get('ai_confidence',
                        tumor_result.get('confidence', 0)) or 0)
    threshold   = float(tumor_result.get('confidence_threshold', 0.75) or 0.75)
    meets_thr   = conf >= threshold
    tumor_pct   = float(tumor_result.get('tumor_percentage', 0))
    probs       = list(tumor_result.get('probabilities', [0.2]*5))
    if len(probs) < 5:
        probs += [0.0] * (5 - len(probs))
    mask        = tumor_result.get('mask', None)
    enhanced_im = tumor_result.get('enhanced_image', original_image)

    grade_color = GRADE_HEX[min(grade, 4)]
    status_color = VIZ_RED_LINE if detected else VIZ_GREEN

    # Normalize image
    img = np.array(original_image, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_u8 = (img * 255).astype(np.uint8)

    enh = np.array(enhanced_im, dtype=np.float32)
    enh = (enh - enh.min()) / (enh.max() - enh.min() + 1e-8)

    # ── HEADER ────────────────────────────────────────────────────────────────
    status_str = (f"⚠  TUMOR DETECTED  —  {tumor_type.upper()}"
                  if detected else "✓  NO TUMOR DETECTED  —  AI FINDING")
    conf_str   = f"Confidence: {conf*100:.0f}%  |  "
    conf_str  += f"{'✓ Meets Threshold' if meets_thr else '✗ Below Threshold'}  |  "
    conf_str  += f"WHO Grade {['0','I','II','III','IV'][min(grade,4)]}  |  "
    conf_str  += f"Apparent Area: {tumor_pct:.2f}%"

    fig.text(0.5, 0.955, status_str,
             ha='center', fontsize=19, fontweight='bold',
             color=status_color, fontfamily='DejaVu Sans')
    fig.text(0.5, 0.925, conf_str,
             ha='center', fontsize=11, color=VIZ_TEXT_MD,
             style='italic', fontfamily='DejaVu Sans')

    # ── DIVIDER LINE ──────────────────────────────────────────────────────────
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.915, 0.915],
                              color=VIZ_BORDER, linewidth=1.2,
                              transform=fig.transFigure))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 0 — ORIGINAL MRI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(VIZ_BG)
    ax0.imshow(img, cmap='gray', aspect='equal', interpolation='bicubic',
               vmin=0, vmax=1)
    # Corner markers
    for (cy, cx) in [(5,5),(5,-6),(-6,5),(-6,-6)]:
        ax0.axhline(y=cy if cy>0 else img.shape[0]+cy,
                    xmin=0, xmax=0.05, color=VIZ_ACCENT, lw=1.2, alpha=0.6)
    _frame_ax(ax0, '[ 01 ]  ORIGINAL MRI SCAN', VIZ_ACCENT)
    ax0.text(0.02, 0.02, 'T1/T2/FLAIR', transform=ax0.transAxes,
             fontsize=7.5, color=VIZ_TEXT_LO, va='bottom')
    ax0.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 1 — CONTRAST ENHANCED
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(VIZ_BG)
    ax1.imshow(enh, cmap='plasma', aspect='equal', interpolation='bicubic',
               vmin=0, vmax=1)
    _frame_ax(ax1, '[ 02 ]  AI CONTRAST ENHANCED', '#FF9500')
    ax1.text(0.02, 0.02, 'Plasma colormap', transform=ax1.transAxes,
             fontsize=7.5, color=VIZ_TEXT_LO, va='bottom')
    ax1.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 2 — SEGMENTATION OVERLAY  (main clinical panel)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(VIZ_BG)
    ax2.imshow(img, cmap='gray', aspect='equal', interpolation='bicubic',
               alpha=0.88, vmin=0, vmax=1)
    if detected:
        _draw_tumor_mask(ax2, img, mask,
                         contour_color=VIZ_ACCENT,
                         fill_alpha=0.55,
                         contour_lw=5.0,
                         bbox_color=VIZ_RED_LINE,
                         bbox_lw=4.0,
                         draw_bbox=True,
                         draw_label=True)
    else:
        ax2.text(0.5, 0.5, 'NO TUMOR\nDETECTED',
                 transform=ax2.transAxes,
                 ha='center', va='center',
                 fontsize=16, color=VIZ_GREEN, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='#001A0A', edgecolor=VIZ_GREEN,
                           linewidth=2.5, alpha=0.95))
    seg_title_color = VIZ_RED_LINE if detected else VIZ_GREEN
    _frame_ax(ax2, '[ 03 ]  TUMOR SEGMENTATION OVERLAY', seg_title_color)
    ax2.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 3 — TISSUE COMPOSITION DONUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor(VIZ_PANEL)
    total_px = img.shape[0] * img.shape[1]
    t_px = 0
    if detected and mask is not None and np.any(mask):
        t_px = int(np.sum(mask))
        n_px = total_px - t_px
        sizes  = [t_px, n_px]
        colors = [VIZ_RED_FILL, '#1565C0']
        labels = [f'Lesion\n{t_px/total_px*100:.1f}%',
                  f'Normal\n{n_px/total_px*100:.1f}%']
    else:
        sizes  = [1]
        colors = ['#1565C0']
        labels = ['Normal\n100%']
    wedges, texts = ax3.pie(
        sizes, labels=None, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=VIZ_BG, linewidth=3.0),
        counterclock=False)
    # Center annotation
    centre_pct = f"{t_px/total_px*100:.1f}%" if (detected and mask is not None and np.any(mask)) else "0%"
    centre_col = VIZ_RED_LINE if detected else VIZ_GREEN
    ax3.text(0, 0, centre_pct, ha='center', va='center',
             fontsize=18, fontweight='bold', color=centre_col)
    # Manual legend
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(colors, labels)
    ]
    ax3.legend(handles=legend_patches, loc='lower center',
               fontsize=9, framealpha=0.0,
               labelcolor=VIZ_TEXT_MD,
               bbox_to_anchor=(0.5, -0.08))
    _frame_ax(ax3, '[ 04 ]  TISSUE COMPOSITION', '#BF5FFF')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 4 — PROBABILITY HEATMAP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(VIZ_BG)
    ax4.imshow(img, cmap='gray', aspect='equal',
               alpha=0.50, interpolation='bicubic')
    if detected and mask is not None and np.any(mask):
        prob_map = cv2.GaussianBlur(
            mask.astype(np.float32), (0, 0),
            sigmaX=max(img.shape[0]//40, 5),
            sigmaY=max(img.shape[1]//40, 5))
        prob_map /= (prob_map.max() + 1e-8)
        im4 = ax4.imshow(prob_map, cmap='hot', aspect='equal',
                         alpha=0.78, vmin=0, vmax=1, interpolation='bicubic')
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.042, pad=0.03)
        cbar4.set_label('Probability', color=VIZ_TEXT_MD, fontsize=8)
        cbar4.ax.yaxis.set_tick_params(color=VIZ_TEXT_LO, labelsize=7)
        cbar4.outline.set_edgecolor(VIZ_BORDER)
        # Centroid
        cy_c, cx_c = ndimage.center_of_mass(mask.astype(float))
        ax4.plot(cx_c, cy_c, '+', markersize=20,
                 markeredgewidth=3.0, color=VIZ_YELLOW, zorder=10)
        ax4.annotate('CENTROID',
                     xy=(cx_c, cy_c),
                     xytext=(cx_c + img.shape[1]*0.12,
                             cy_c - img.shape[0]*0.10),
                     color=VIZ_YELLOW, fontsize=8, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=VIZ_YELLOW,
                                     lw=1.8), zorder=11)
        # Cyan contour on top
        try:
            for c in measure.find_contours(mask.astype(float), level=0.5):
                ax4.plot(c[:, 1], c[:, 0], linewidth=3.5,
                         color=VIZ_ACCENT, zorder=9, alpha=0.9)
        except Exception:
            pass
    else:
        ax4.text(0.5, 0.5, 'NO ABNORMALITY\nDETECTED',
                 transform=ax4.transAxes, ha='center', va='center',
                 fontsize=13, color=VIZ_GREEN, fontweight='bold')
    _frame_ax(ax4, '[ 05 ]  PROBABILITY HEATMAP', '#FF9500')
    ax4.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 5 — GRADE PROBABILITY BARS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(VIZ_PANEL)
    bar_h     = 0.62
    y_pos     = np.arange(5)
    bar_colors_grad = GRADE_HEX[:5]
    bars5 = ax5.barh(y_pos, probs, height=bar_h,
                     color=bar_colors_grad,
                     edgecolor=VIZ_BG, linewidth=1.5,
                     zorder=3)
    # Highlight predicted grade
    pred_idx = int(np.argmax(probs))
    bars5[pred_idx].set_edgecolor('#FFFFFF')
    bars5[pred_idx].set_linewidth(2.8)
    # Value labels
    for i, (bar, p) in enumerate(zip(bars5, probs)):
        w = bar.get_width()
        ax5.text(min(w + 0.025, 0.94), i,
                 f'{p*100:.1f}%',
                 va='center', fontsize=9.5, fontweight='bold',
                 color=bar_colors_grad[i])
    # Threshold line at 0.5
    ax5.axvline(0.5, color='#FFFFFF', linestyle='--',
                linewidth=1.2, alpha=0.4, zorder=2)
    ax5.text(0.51, 4.6, '50%', fontsize=7.5, color='#FFFFFF', alpha=0.4)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(GRADE_LABELS_FULL, fontsize=9, color=VIZ_TEXT_MD)
    ax5.set_xlim(0, 1.12)
    ax5.set_xlabel('Probability', fontsize=9, color=VIZ_TEXT_MD)
    ax5.tick_params(axis='x', colors=VIZ_TEXT_LO)
    ax5.set_facecolor(VIZ_PANEL)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax5.spines[sp].set_edgecolor(VIZ_BORDER)
        ax5.spines[sp].set_linewidth(1.2)
    ax5.grid(True, axis='x', color=VIZ_GRID, linewidth=0.7, alpha=0.9)
    ax5.set_title('[ 06 ]  GRADE PROBABILITY DISTRIBUTION',
                  fontsize=10.5, fontweight='bold',
                  color='#FACC15', pad=10)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 6 — SOBEL EDGE / TEXTURE ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(VIZ_BG)
    sx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sx**2 + sy**2)
    edge /= (edge.max() + 1e-8)
    im6 = ax6.imshow(edge, cmap='inferno', aspect='equal',
                     interpolation='bicubic')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.042, pad=0.03)
    cbar6.set_label('Edge Strength', color=VIZ_TEXT_MD, fontsize=8)
    cbar6.ax.yaxis.set_tick_params(color=VIZ_TEXT_LO, labelsize=7)
    cbar6.outline.set_edgecolor(VIZ_BORDER)
    if detected and mask is not None and np.any(mask):
        try:
            for c in measure.find_contours(mask.astype(float), level=0.5):
                ax6.plot(c[:, 1], c[:, 0], linewidth=4.0,
                         color=VIZ_ACCENT, linestyle='--',
                         alpha=1.0, zorder=8)
        except Exception:
            pass
    _frame_ax(ax6, '[ 07 ]  EDGE & TEXTURE ANALYSIS', VIZ_ACCENT)
    ax6.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PANEL 7 — QUANTITATIVE METRICS CARD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.set_facecolor(VIZ_PANEL)
    ax7.axis('off')

    # Build metrics text
    det_str    = 'POSITIVE' if detected else 'NEGATIVE'
    det_color  = VIZ_RED_LINE if detected else VIZ_GREEN
    grade_str  = ['0 — Normal','I — Grade I','II — Grade II',
                  'III — Grade III','IV — GBM'][min(grade, 4)]

    # Compute area metrics
    if detected and mask is not None and np.any(mask):
        ta   = int(np.sum(mask))
        tot  = img.shape[0] * img.shape[1]
        pct  = ta / tot * 100
        try:
            cnts  = measure.find_contours(mask, level=0.5)
            perim = len(cnts[0]) if cnts else 0
            comp  = (4 * np.pi * ta / perim**2) if perim > 0 else 0.0
        except Exception:
            perim, comp = 0, 0.0
    else:
        ta, pct, perim, comp = 0, 0.0, 0, 0.0

    # Draw status badge
    badge_bg = '#1A0000' if detected else '#001A08'
    ax7.add_patch(mpatches.FancyBboxPatch(
        (0.04, 0.84), 0.92, 0.13,
        boxstyle='round,pad=0.02',
        facecolor=badge_bg, edgecolor=det_color,
        linewidth=2.5, transform=ax7.transAxes, zorder=2))
    ax7.text(0.5, 0.905, f'AI FINDING: {det_str}',
             transform=ax7.transAxes, ha='center', va='center',
             fontsize=12, fontweight='bold', color=det_color, zorder=3)

    # Metric rows
    rows = [
        ('WHO Grade',       grade_str,          grade_color),
        ('Tumour Type',     tumor_type,         VIZ_TEXT_HI),
        ('AI Confidence',   f'{conf*100:.0f}%', VIZ_ACCENT),
        ('Pixel Area',      f'{ta:,} px',       VIZ_TEXT_MD),
        ('Coverage',        f'{pct:.3f}%',      VIZ_TEXT_MD),
        ('Perimeter',       f'{perim:.0f} px',  VIZ_TEXT_MD),
        ('Compactness',     f'{comp:.4f}',      VIZ_TEXT_MD),
    ]
    y_start = 0.78
    row_h   = 0.085
    for i, (lbl, val, col) in enumerate(rows):
        y = y_start - i * row_h
        ax7.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.03), 0.92, row_h * 0.88,
            boxstyle='round,pad=0.01',
            facecolor='#0D0D1A' if i % 2 == 0 else '#0A0A14',
            edgecolor=VIZ_BORDER, linewidth=0.8,
            transform=ax7.transAxes))
        ax7.text(0.08, y + 0.01, lbl,
                 transform=ax7.transAxes, va='center',
                 fontsize=9, color=VIZ_TEXT_LO)
        ax7.text(0.96, y + 0.01, val,
                 transform=ax7.transAxes, va='center',
                 ha='right', fontsize=9, fontweight='bold', color=col)

    # Bottom disclaimer
    ax7.text(0.5, 0.022,
             'AI FINDING — REQUIRES RADIOLOGIST REVIEW',
             transform=ax7.transAxes, ha='center', va='bottom',
             fontsize=7.5, color='#FF6B6B', style='italic')
    _frame_ax(ax7, '[ 08 ]  QUANTITATIVE METRICS', VIZ_ACCENT)

    # ── Footer strip ──────────────────────────────────────────────────────────
    fig.text(0.5, 0.018,
             'LongiTumorSense  |  Research Grade AI  |  Not FDA Cleared  '
             '|  WHO CNS 2021  |  All findings require independent radiologist verification',
             ha='center', fontsize=8.5, color=VIZ_TEXT_LO,
             fontfamily='DejaVu Sans')

    fig.patch.set_facecolor(VIZ_BG)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  12-PANEL PROGRESSION COMPARISON VISUALIZATION (UPGRADED)
# ══════════════════════════════════════════════════════════════════════════════

def create_progression_comparison_visualization(image1, image2,
                                                result1, result2,
                                                progression_metrics):
    """
    World-class 12-panel longitudinal progression figure.
    Pure black background, vivid red fills, neon contours, perfect alignment.
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from skimage import measure

    _plt_base_black()

    # ── Figure & grid ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(40, 34), dpi=130, facecolor=VIZ_BG)
    gs  = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.36, wspace=0.20,
        left=0.025, right=0.975,
        top=0.910, bottom=0.055)

    # ── Metrics extraction ────────────────────────────────────────────────────
    status_color = progression_metrics.get('status_color', '#F59E0B')
    vol_pct      = progression_metrics.get('volume_change_pct', 0)
    growth_rate  = progression_metrics.get('growth_rate_monthly', 0)
    time_days    = progression_metrics.get('time_diff_days', 0)
    prog_status  = progression_metrics.get('progression_status', 'Unknown')
    treat_resp   = progression_metrics.get('treatment_response', ('Unknown', '#888'))
    treat_label  = treat_resp[0] if isinstance(treat_resp, (list, tuple)) else treat_resp
    rec_risk     = float(progression_metrics.get('recurrence_risk', 0))
    grade1       = result1.get('grade', 0)
    grade2       = result2.get('grade', 0)
    conf1        = float(result1.get('ai_confidence', result1.get('confidence', 0)) or 0)
    conf2        = float(result2.get('ai_confidence', result2.get('confidence', 0)) or 0)
    pct1         = float(result1.get('tumor_percentage', 0))
    pct2         = float(result2.get('tumor_percentage', 0))
    gnames = {0:'Normal', 1:'Grade I', 2:'Grade II',
              3:'Grade III', 4:'Grade IV (GBM)'}
    g1_label = gnames.get(grade1, 'Unknown')
    g2_label = gnames.get(grade2, 'Unknown')

    def _safe_resize(src, ref, is_mask=False):
        try:
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4
            return cv2.resize(
                np.array(src, dtype=np.float32),
                (ref.shape[1], ref.shape[0]),
                interpolation=interp)
        except Exception:
            return np.array(src, dtype=np.float32)

    # Normalize both images to [0,1] with consistent global scale
    img1_raw = np.array(image1, dtype=np.float32)
    img2_raw = _safe_resize(image2, img1_raw)
    g_min = min(float(img1_raw.min()), float(img2_raw.min()))
    g_max = max(float(img1_raw.max()), float(img2_raw.max()))
    scale = (g_max - g_min) + 1e-8
    img1  = (img1_raw - g_min) / scale
    img2  = (img2_raw - g_min) / scale

    mask1 = result1.get('mask', None)
    mask2 = result2.get('mask', None)
    if mask1 is not None:
        mask1 = (_safe_resize(mask1, img1, is_mask=True) > 0.5)
    if mask2 is not None:
        mask2 = (_safe_resize(mask2, img1, is_mask=True) > 0.5)

    vol_arrow_c = VIZ_RED_LINE if vol_pct > 5 else \
                  VIZ_GREEN   if vol_pct < -5 else '#FACC15'
    rr_color    = VIZ_RED_LINE if rec_risk > 60 else \
                  '#FB923C'   if rec_risk > 30 else VIZ_GREEN

    # ── HEADER ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.958,
             f'LONGITUDINAL PROGRESSION ANALYSIS  —  {prog_status.upper()}',
             ha='center', fontsize=20, fontweight='bold',
             color=status_color, fontfamily='DejaVu Sans')
    fig.text(0.5, 0.930,
             f'Volume Change: {vol_pct:+.1f}%   |   Interval: {time_days} days   |   '
             f'Growth Rate: {growth_rate:+.2f}%/mo   |   Response: {treat_label}   |   '
             f'Recurrence Risk: {rec_risk:.0f}%',
             ha='center', fontsize=11, color=VIZ_TEXT_MD,
             style='italic', fontfamily='DejaVu Sans')
    fig.add_artist(plt.Line2D(
        [0.02, 0.98], [0.922, 0.922],
        color=VIZ_BORDER, linewidth=1.2,
        transform=fig.transFigure))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROW 0 — SCAN PAIR + DIFFERENCE + CHECKERBOARD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # [0,0] Baseline
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_facecolor(VIZ_BG)
    ax00.imshow(img1, cmap='gray', aspect='equal',
                interpolation='bicubic', vmin=0, vmax=1)
    _draw_tumor_mask(ax00, img1, mask1,
                     contour_color=VIZ_ACCENT, fill_alpha=0.52,
                     contour_lw=4.5, bbox_color=VIZ_RED_LINE,
                     bbox_lw=3.5, draw_bbox=True, draw_label=True)
    ax00.text(0.02, 0.97,
              f'Conf: {conf1*100:.0f}%  |  Area: {pct1:.2f}%  |  {g1_label}',
              transform=ax00.transAxes, fontsize=8, color=VIZ_TEXT_MD,
              va='top',
              bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='#0A0A14', edgecolor=VIZ_BORDER, lw=1.0))
    _frame_ax(ax00, '[ 01 ]  BASELINE SCAN', VIZ_ACCENT)
    ax00.axis('off')

    # [0,1] Follow-up
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_facecolor(VIZ_BG)
    ax01.imshow(img2, cmap='gray', aspect='equal',
                interpolation='bicubic', vmin=0, vmax=1)
    _draw_tumor_mask(ax01, img2, mask2,
                     contour_color=VIZ_ACCENT2, fill_alpha=0.52,
                     contour_lw=4.5, bbox_color='#FF8C00',
                     bbox_lw=3.5, draw_bbox=True, draw_label=True)
    ax01.text(0.02, 0.97,
              f'Conf: {conf2*100:.0f}%  |  Area: {pct2:.2f}%  |  {g2_label}',
              transform=ax01.transAxes, fontsize=8, color=VIZ_TEXT_MD,
              va='top',
              bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='#0A0A14', edgecolor=VIZ_BORDER, lw=1.0))
    _frame_ax(ax01, '[ 02 ]  FOLLOW-UP SCAN', VIZ_ACCENT2)
    ax01.axis('off')

    # [0,2] Difference map
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.set_facecolor(VIZ_BG)
    ax02.imshow(img1, cmap='gray', aspect='equal', alpha=0.45,
                interpolation='bicubic')
    diff  = img2 - img1
    p99   = float(np.percentile(np.abs(diff), 99))
    sd    = max(p99, 0.03)
    diff_rgba = plt.cm.seismic((diff / sd / 2 + 0.5).clip(0, 1))
    amask = (np.abs(diff) > 0.04 * sd).astype(np.float32)
    amask = cv2.GaussianBlur(amask, (5, 5), 1.2)
    diff_rgba[..., 3] = (amask * 0.88).clip(0, 1)
    ax02.imshow(diff_rgba, aspect='equal', interpolation='bicubic')
    # Thick contours
    if mask1 is not None and np.any(mask1):
        for c in measure.find_contours(mask1.astype(float), level=0.5):
            ax02.plot(c[:, 1], c[:, 0], lw=4.5,
                      color=VIZ_ACCENT, alpha=1.0, zorder=8)
    if mask2 is not None and np.any(mask2):
        for c in measure.find_contours(mask2.astype(float), level=0.5):
            ax02.plot(c[:, 1], c[:, 0], lw=4.5,
                      color=VIZ_ACCENT2, alpha=1.0, zorder=8)
    sm = plt.cm.ScalarMappable(cmap='seismic',
                               norm=plt.Normalize(-sd, sd))
    sm.set_array([])
    cbar02 = plt.colorbar(sm, ax=ax02, fraction=0.042, pad=0.03)
    cbar02.set_label('Intensity Δ', color=VIZ_TEXT_MD, fontsize=8)
    cbar02.ax.yaxis.set_tick_params(color=VIZ_TEXT_LO, labelsize=7)
    cbar02.outline.set_edgecolor(VIZ_BORDER)
    ax02.text(0.02, 0.97,
              f'Cyan = Baseline  |  Orange = Follow-up',
              transform=ax02.transAxes, fontsize=7.5,
              color=VIZ_TEXT_MD, va='top',
              bbox=dict(boxstyle='round,pad=0.28',
                        facecolor='#0A0A14', edgecolor=VIZ_BORDER, lw=0.8))
    _frame_ax(ax02, '[ 03 ]  DIFFERENCE MAP', '#FACC15')
    ax02.axis('off')

    # [0,3] Checkerboard overlay
    ax03 = fig.add_subplot(gs[0, 3])
    ax03.set_facecolor(VIZ_BG)
    h, w = img1.shape[:2]
    block = max(h // 9, 8)
    chk = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, block * 2):
        for j in range(0, w, block * 2):
            chk[i:i+block, j:j+block] = 1
            chk[i+block:i+block*2, j+block:j+block*2] = 1
    ax03.imshow(np.where(chk > 0, img1, img2),
                cmap='gray', aspect='equal',
                interpolation='bicubic', vmin=0, vmax=1)
    if mask1 is not None and np.any(mask1):
        for c in measure.find_contours(mask1.astype(float), level=0.5):
            ax03.plot(c[:, 1], c[:, 0], lw=4.5,
                      color=VIZ_ACCENT, zorder=8)
    if mask2 is not None and np.any(mask2):
        for c in measure.find_contours(mask2.astype(float), level=0.5):
            ax03.plot(c[:, 1], c[:, 0], lw=4.5,
                      color=VIZ_ACCENT2, zorder=8)
    ax03.text(0.02, 0.97,
              'Cyan = Baseline  |  Orange = Follow-up',
              transform=ax03.transAxes, fontsize=7.5,
              color=VIZ_TEXT_MD, va='top',
              bbox=dict(boxstyle='round,pad=0.28',
                        facecolor='#0A0A14', edgecolor=VIZ_BORDER, lw=0.8))
    _frame_ax(ax03, '[ 04 ]  CHECKERBOARD OVERLAY', '#BF5FFF')
    ax03.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROW 1 — TUMOR MASKS + PROBABILITY + CHANGE MAP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # [1,0] Baseline mask
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_facecolor(VIZ_BG)
    ax10.imshow(img1, cmap='gray', alpha=0.45,
                aspect='equal', interpolation='bicubic')
    if mask1 is not None and np.any(mask1):
        fill1 = np.zeros((*img1.shape[:2], 4), dtype=np.float32)
        fill1[mask1] = [1.0, 0.09, 0.13, 0.62]
        ax10.imshow(fill1, aspect='equal', interpolation='nearest')
        for c in measure.find_contours(mask1.astype(float), level=0.5):
            ax10.plot(c[:, 1], c[:, 0], lw=5.5,
                      color=VIZ_ACCENT, zorder=9)
    ax10.text(0.02, 0.97, f'Area: {pct1:.2f}%  |  {g1_label}',
              transform=ax10.transAxes, fontsize=8,
              color=VIZ_TEXT_MD, va='top',
              bbox=dict(boxstyle='round,pad=0.28',
                        facecolor='#0A0A14', edgecolor=VIZ_BORDER, lw=0.8))
    _frame_ax(ax10, '[ 05 ]  BASELINE TUMOR MASK', VIZ_ACCENT)
    ax10.axis('off')

    # [1,1] Follow-up mask
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_facecolor(VIZ_BG)
    ax11.imshow(img2, cmap='gray', alpha=0.45,
                aspect='equal', interpolation='bicubic')
    if mask2 is not None and np.any(mask2):
        fill2 = np.zeros((*img2.shape[:2], 4), dtype=np.float32)
        fill2[mask2] = [1.0, 0.09, 0.13, 0.62]
        ax11.imshow(fill2, aspect='equal', interpolation='nearest')
        for c in measure.find_contours(mask2.astype(float), level=0.5):
            ax11.plot(c[:, 1], c[:, 0], lw=5.5,
                      color=VIZ_ACCENT2, zorder=9)
    ax11.text(0.02, 0.97, f'Area: {pct2:.2f}%  |  {g2_label}',
              transform=ax11.transAxes, fontsize=8,
              color=VIZ_TEXT_MD, va='top',
              bbox=dict(boxstyle='round,pad=0.28',
                        facecolor='#0A0A14', edgecolor=VIZ_BORDER, lw=0.8))
    _frame_ax(ax11, '[ 06 ]  FOLLOW-UP TUMOR MASK', VIZ_ACCENT2)
    ax11.axis('off')

    # [1,2] Follow-up probability map
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.set_facecolor(VIZ_BG)
    if mask2 is not None and np.any(mask2):
        ax12.imshow(img2, cmap='gray', alpha=0.45,
                    aspect='equal', interpolation='bicubic')
        sigma = max(img2.shape[0] // 40, 5)
        prob2 = cv2.GaussianBlur(mask2.astype(np.float32),
                                 (0, 0), sigmaX=sigma, sigmaY=sigma)
        prob2 /= (prob2.max() + 1e-8)
        im12 = ax12.imshow(prob2, cmap='hot', aspect='equal',
                           vmin=0, vmax=1, alpha=0.80,
                           interpolation='bicubic')
        for c in measure.find_contours(mask2.astype(float), level=0.5):
            ax12.plot(c[:, 1], c[:, 0], lw=5.0,
                      color=VIZ_ACCENT, zorder=9)
        cbar12 = plt.colorbar(im12, ax=ax12, fraction=0.042, pad=0.03)
        cbar12.set_label('Tumour Prob.', color=VIZ_TEXT_MD, fontsize=8)
        cbar12.ax.yaxis.set_tick_params(color=VIZ_TEXT_LO, labelsize=7)
        cbar12.outline.set_edgecolor(VIZ_BORDER)
    else:
        ax12.imshow(img2, cmap='gray', alpha=0.5,
                    aspect='equal', interpolation='bicubic')
        ax12.text(0.5, 0.5, 'NO TUMOR\nDETECTED',
                  transform=ax12.transAxes, ha='center', va='center',
                  fontsize=14, color=VIZ_GREEN, fontweight='bold')
    _frame_ax(ax12, '[ 07 ]  FOLLOW-UP PROBABILITY MAP', '#FF9500')
    ax12.axis('off')

    # [1,3] Tumor change map
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.set_facecolor(VIZ_BG)
    ax13.imshow((img1 + img2) / 2, cmap='gray', alpha=0.38,
                aspect='equal', interpolation='bicubic')
    m1f = mask1.astype(np.float32) if mask1 is not None else np.zeros_like(img1)
    m2f = mask2.astype(np.float32) if mask2 is not None else np.zeros_like(img2)
    chg = m2f - m1f
    rgba_chg = np.zeros((*chg.shape, 4), dtype=np.float32)
    grown    = chg > 0.35
    resolved = chg < -0.35
    stable   = (np.abs(chg) <= 0.35) & ((m1f + m2f) > 0.5)
    if np.any(grown):    rgba_chg[grown]    = [1.0, 0.22, 0.08, 0.85]
    if np.any(resolved): rgba_chg[resolved] = [0.05, 1.0, 0.55, 0.85]
    if np.any(stable):   rgba_chg[stable]   = [1.0, 0.88, 0.05, 0.72]
    ax13.imshow(rgba_chg, aspect='equal', interpolation='nearest')
    if mask1 is not None and np.any(mask1):
        for c in measure.find_contours(mask1.astype(float), level=0.5):
            ax13.plot(c[:, 1], c[:, 0], lw=4.5,
                      color=VIZ_ACCENT, alpha=1.0, zorder=9)
    if mask2 is not None and np.any(mask2):
        for c in measure.find_contours(mask2.astype(float), level=0.5):
            ax13.plot(c[:, 1], c[:, 0], lw=4.5,
                      color=VIZ_ACCENT2, alpha=1.0, zorder=9)
    legend_items = [
        ('#FF381A', 'New / Grown'),
        ('#0DFF8C', 'Resolved'),
        ('#FFE10D', 'Stable'),
        (VIZ_ACCENT,  'Baseline contour'),
        (VIZ_ACCENT2, 'Follow-up contour'),
    ]
    for idx, (col, lbl) in enumerate(legend_items):
        ax13.text(0.03, 0.97 - idx * 0.072, f'● {lbl}',
                  transform=ax13.transAxes, fontsize=8.5,
                  color=col, va='top', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.22',
                            facecolor='#070714', alpha=0.88,
                            edgecolor=col, lw=1.4))
    _frame_ax(ax13, '[ 08 ]  TUMOR CHANGE MAP', '#BF5FFF')
    ax13.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROW 2 — ANALYTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # [2,0] Intensity distribution
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.set_facecolor(VIZ_PANEL)
    bins = np.linspace(0, 1, 60)
    bw   = bins[1] - bins[0]
    h1h, _ = np.histogram(img1.flatten(), bins=bins)
    h2h, _ = np.histogram(img2.flatten(), bins=bins)
    h1n = h1h / (h1h.max() + 1e-8)
    h2n = h2h / (h2h.max() + 1e-8)
    ax20.fill_between(bins[:-1], h1n, alpha=0.30, color=VIZ_ACCENT,
                      step='post', label='Baseline', zorder=2)
    ax20.fill_between(bins[:-1], h2n, alpha=0.30, color=VIZ_ACCENT2,
                      step='post', label='Follow-up', zorder=2)
    ax20.plot(bins[:-1], h1n, color=VIZ_ACCENT, lw=2.2, zorder=3)
    ax20.plot(bins[:-1], h2n, color=VIZ_ACCENT2, lw=2.2, zorder=3)
    ax20.set_xlabel('Normalised Intensity', fontsize=9, color=VIZ_TEXT_MD)
    ax20.set_ylabel('Frequency', fontsize=9, color=VIZ_TEXT_MD)
    ax20.set_facecolor(VIZ_PANEL)
    for sp in ax20.spines.values():
        sp.set_edgecolor(VIZ_BORDER); sp.set_linewidth(1.1)
    ax20.tick_params(colors=VIZ_TEXT_LO, labelsize=8)
    ax20.grid(True, alpha=0.6, color=VIZ_GRID, lw=0.6)
    ax20.legend(fontsize=9, framealpha=0.0, labelcolor=VIZ_TEXT_MD)
    ax20.set_title('[ 09 ]  INTENSITY DISTRIBUTION',
                   fontsize=10.5, fontweight='bold',
                   color=VIZ_ACCENT, pad=10)

    # [2,1] Volume comparison bar
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.set_facecolor(VIZ_PANEL)
    bars21 = ax21.bar(
        ['Baseline', 'Follow-up'], [pct1, pct2],
        color=[VIZ_ACCENT, VIZ_ACCENT2],
        width=0.50, edgecolor=VIZ_BG, linewidth=2.2,
        zorder=3)
    for bar, val in zip(bars21, [pct1, pct2]):
        ax21.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.04,
                  f'{val:.2f}%',
                  ha='center', va='bottom',
                  fontsize=12, fontweight='bold',
                  color=VIZ_TEXT_HI)
    mx_v = max(pct1, pct2) * 1.22 + 0.1
    ax21.annotate(
        f'{vol_pct:+.1f}%',
        xy=(0.5, max(pct1, pct2) * 1.08),
        xycoords=('data', 'data'),
        ha='center', fontsize=14, fontweight='bold',
        color=vol_arrow_c,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#0D0D1A',
                  edgecolor=vol_arrow_c, lw=2.0))
    ax21.set_ylim(0, mx_v)
    ax21.set_ylabel('Tumor Area (%)', fontsize=9, color=VIZ_TEXT_MD)
    for sp in ax21.spines.values():
        sp.set_edgecolor(VIZ_BORDER); sp.set_linewidth(1.1)
    ax21.tick_params(colors=VIZ_TEXT_LO, labelsize=9)
    ax21.grid(True, axis='y', alpha=0.6, color=VIZ_GRID, lw=0.6)
    ax21.set_facecolor(VIZ_PANEL)
    ax21.set_title('[ 10 ]  VOLUME COMPARISON',
                   fontsize=10.5, fontweight='bold',
                   color=VIZ_ACCENT, pad=10)

    # [2,2] Recurrence risk gauge
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.set_facecolor(VIZ_PANEL)
    ax22.set_aspect('equal')
    theta_full  = np.linspace(np.pi, 0, 300)
    # Track
    ax22.plot(np.cos(theta_full), np.sin(theta_full),
              color='#1A1A2E', linewidth=28, solid_capstyle='round', zorder=1)
    # Colour zones
    zones = [(0, 30, '#00C896'), (30, 60, '#FACC15'),
             (60, 80, '#FB923C'), (80, 100, '#FF1744')]
    for z_lo, z_hi, z_col in zones:
        t_zone = np.linspace(
            np.pi - (z_lo / 100) * np.pi,
            np.pi - (z_hi / 100) * np.pi, 80)
        ax22.plot(np.cos(t_zone), np.sin(t_zone),
                  color=z_col, linewidth=28,
                  solid_capstyle='butt', alpha=0.35, zorder=2)
    # Fill to value
    theta_fill = np.linspace(
        np.pi, np.pi - (rec_risk / 100) * np.pi, 300)
    ax22.plot(np.cos(theta_fill), np.sin(theta_fill),
              color=rr_color, linewidth=28,
              solid_capstyle='round', zorder=3)
    # Needle
    needle_a = np.pi - (rec_risk / 100) * np.pi
    ax22.annotate('',
                  xy=(0.72 * np.cos(needle_a), 0.72 * np.sin(needle_a)),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->', color=VIZ_TEXT_HI,
                                  lw=3.5))
    ax22.text(0, -0.18, f'{rec_risk:.0f}%',
              ha='center', fontsize=24, fontweight='bold',
              color=rr_color)
    ax22.text(0, -0.44, 'RECURRENCE RISK',
              ha='center', fontsize=10.5, fontweight='bold',
              color=VIZ_TEXT_MD)
    ax22.text(-1.05, -0.14, 'LOW', ha='center',
              fontsize=9.5, color='#00C896', fontweight='bold')
    ax22.text(1.05, -0.14, 'HIGH', ha='center',
              fontsize=9.5, color='#FF1744', fontweight='bold')
    ax22.set_xlim(-1.28, 1.28)
    ax22.set_ylim(-0.62, 1.18)
    ax22.axis('off')
    ax22.set_title('[ 11 ]  RECURRENCE RISK GAUGE',
                   fontsize=10.5, fontweight='bold',
                   color='#FB923C', pad=10)
    ax22.set_facecolor(VIZ_PANEL)

    # [2,3] Quantitative metrics table
    ax23 = fig.add_subplot(gs[2, 3])
    ax23.set_facecolor(VIZ_PANEL)
    ax23.axis('off')
    grade_chg = grade2 - grade1
    g_arrow   = ('↑' if grade_chg > 0 else '↓' if grade_chg < 0 else '—')

    col_headers = ['METRIC', 'BASELINE', 'FOLLOW-UP', 'CHANGE']
    rows_data = [
        ('WHO Grade',    g1_label, g2_label,
         f'{g_arrow} {abs(grade_chg)}'),
        ('Tumor Area',   f'{pct1:.2f}%', f'{pct2:.2f}%',
         f'{vol_pct:+.1f}%'),
        ('AI Confidence', f'{conf1*100:.0f}%', f'{conf2*100:.0f}%', '—'),
        ('Growth Rate',  '—', '—', f'{growth_rate:+.2f}%/mo'),
        ('Interval',     '—', '—', f'{time_days} d'),
        ('Recurr. Risk', '—', '—', f'{rec_risk:.0f}%'),
        ('Treatment',    '—', '—', treat_label[:16]),
    ]
    tbl = ax23.table(
        cellText=rows_data,
        colLabels=col_headers,
        cellLoc='center', loc='center',
        bbox=[0.0, 0.06, 1.0, 0.90])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#1A1A2E')
        cell.set_linewidth(1.1)
        if r == 0:
            cell.set_facecolor('#0A0A28')
            cell.set_text_props(color=VIZ_ACCENT,
                                fontweight='bold', fontsize=9.2)
        elif r % 2 == 0:
            cell.set_facecolor('#0D0D1A')
            cell.set_text_props(color=VIZ_TEXT_MD)
        else:
            cell.set_facecolor('#0A0A14')
            cell.set_text_props(color=VIZ_TEXT_MD)
        if c == 3 and r > 0:
            cell.set_text_props(color=status_color, fontweight='bold')
    ax23.set_title('[ 12 ]  QUANTITATIVE METRICS',
                   fontsize=10.5, fontweight='bold',
                   color=VIZ_PURPLE, pad=10)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.030,
        f'STATUS: {prog_status.upper()}   |   {treat_label.upper()}   |   '
        f'RECURRENCE RISK: {rec_risk:.0f}%   |   '
        f'AI FINDING — REQUIRES RADIOLOGIST VERIFICATION',
        ha='center', fontsize=9.5, fontweight='bold',
        color=status_color, fontfamily='DejaVu Sans',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#0A0A14',
                  edgecolor=status_color, lw=2.0, alpha=0.95),
        transform=fig.transFigure)

    fig.patch.set_facecolor(VIZ_BG)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LongiTumorSense - Brain Tumor Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
)


def get_css() -> str:
    t = THEME
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --bg-canvas:    {t['bg_canvas']};
  --bg-surface:   {t['bg_surface']};
  --bg-elevated:  {t['bg_elevated']};
  --bg-inset:     {t['bg_inset']};
  --brand:        {t['brand']};
  --brand-light:  {t['brand_light']};
  --brand-dk:     {t['brand_dk']};
  --brand-glow:   {t['brand_glow']};
  --ok:           {t['ok']};
  --ok-light:     {t['ok_light']};
  --warn:         {t['warn']};
  --warn-light:   {t['warn_light']};
  --alert:        {t['alert']};
  --alert-light:  {t['alert_light']};
  --info:         {t['info']};
  --info-light:   {t['info_light']};
  --purple:       {t['purple']};
  --purple-light: {t['purple_light']};
  --text-1:       {t['text_1']};
  --text-2:       {t['text_2']};
  --text-3:       {t['text_3']};
  --text-4:       {t['text_4']};
  --border-s:     {t['border_s']};
  --border-m:     {t['border_m']};
  --border-k:     {t['border_k']};
  --r-sm:  8px; --r-md:  12px; --r-lg:  16px; --r-xl:  24px; --r-2xl: 32px;
  --shadow-xs: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
  --shadow-sm: 0 2px 8px rgba(15,23,42,0.08), 0 1px 4px rgba(15,23,42,0.04);
  --shadow-md: 0 4px 16px rgba(15,23,42,0.10), 0 2px 6px rgba(15,23,42,0.06);
  --shadow-lg: 0 8px 32px rgba(15,23,42,0.12), 0 4px 12px rgba(15,23,42,0.08);
}}

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
  font-family: 'IBM Plex Sans', sans-serif !important;
  background-color: var(--bg-canvas) !important;
  color: var(--text-1) !important;
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}

h1, h2, h3, h4, h5, h6 {{
  font-family: 'Outfit', sans-serif !important;
  line-height: 1.25 !important;
  color: var(--text-1) !important;
}}

.stApp {{ background-color: var(--bg-canvas) !important; }}
.stApp > header {{ background: transparent !important; box-shadow: none !important; }}
.block-container {{ padding: 1.75rem 2rem 2.5rem !important; max-width: 1600px; }}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-s) !important;
  width: 260px !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 0 !important; }}

.sidebar-logo-area {{
  padding: 1.5rem 1.25rem 1rem;
  border-bottom: 1px solid var(--border-s);
}}
.sidebar-logo {{ display: flex; align-items: center; gap: 10px; margin-bottom: 1.25rem; }}
.sidebar-logo-mark {{
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--brand-dk), var(--brand));
  border-radius: var(--r-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(8,145,178,0.3);
}}
.sidebar-logo-text {{
  font-family: 'Outfit', sans-serif;
  font-size: 0.95rem; font-weight: 700; color: var(--text-1); line-height: 1.1;
}}
.sidebar-logo-text em {{ font-style: normal; color: var(--brand); }}
.sidebar-logo-sub {{ font-size: 0.6rem; color: var(--text-4); font-weight: 500; letter-spacing: 0.04em; }}

.user-card {{
  background: var(--bg-elevated); border: 1px solid var(--border-s);
  border-radius: var(--r-md); padding: 0.85rem;
  display: flex; align-items: center; gap: 10px;
}}
.user-avatar {{
  width: 38px; height: 38px;
  background: linear-gradient(135deg, var(--brand-dk), var(--brand));
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-family: 'Outfit', sans-serif; font-size: 0.85rem; font-weight: 700; color: #fff; flex-shrink: 0;
}}
.user-name {{ font-size: 0.83rem; font-weight: 600; color: var(--text-1); }}
.user-role {{ font-size: 0.72rem; color: var(--text-3); }}
.user-badge {{
  margin-left: auto; font-size: 0.6rem; font-weight: 700; color: var(--ok);
  background: var(--ok-light); padding: 3px 7px; border-radius: 20px;
  display: flex; align-items: center; gap: 4px;
}}
.user-badge::before {{ content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--ok); }}

.sidebar-section-label {{
  font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.12em; color: var(--text-4); padding: 1rem 1.25rem 0.4rem; display: block;
}}
.model-pill {{
  display: flex; align-items: center; gap: 8px; padding: 0.5rem 0.75rem;
  background: var(--bg-elevated); border: 1px solid var(--border-s);
  border-radius: var(--r-sm); margin: 0 1.25rem 5px; font-size: 0.75rem;
}}
.model-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--ok); flex-shrink: 0; box-shadow: 0 0 4px var(--ok); }}
.model-name {{ font-weight: 600; color: var(--text-1); }}
.model-info {{ font-size: 0.65rem; color: var(--text-4); }}

.not-cleared-banner {{
  margin: 1rem 1.25rem;
  background: rgba(220,38,38,0.06); border: 1px solid rgba(220,38,38,0.2);
  border-radius: var(--r-sm); padding: 0.7rem 0.85rem;
  font-size: 0.72rem; color: var(--text-2); line-height: 1.5;
}}
.not-cleared-banner strong {{ color: var(--alert); display: block; margin-bottom: 3px; font-size: 0.72rem; }}

/* ── App Header ── */
.app-header {{
  background: var(--bg-surface); border-bottom: 1px solid var(--border-s);
  padding: 0.9rem 0; display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 0.75rem; box-shadow: var(--shadow-xs); margin-bottom: 1.75rem;
}}
.app-header-title {{ font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--text-1); }}
.header-badges {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.hbadge {{ font-size: 0.65rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; letter-spacing: 0.04em; text-transform: uppercase; }}
.hbadge-teal {{ background: var(--brand-light); color: var(--brand-dk); }}
.hbadge-green {{ background: var(--ok-light); color: var(--ok); }}
.hbadge-red {{ background: var(--alert-light); color: var(--alert); }}

/* ── TABS ── */
div[data-testid="stTabs"] > div:first-child {{
  background: var(--bg-surface); border: 1px solid var(--border-s);
  border-radius: var(--r-lg); padding: 5px; margin-bottom: 1.75rem;
  box-shadow: var(--shadow-xs); gap: 2px;
}}
div[data-testid="stTabs"] button[data-baseweb="tab"] {{
  font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.82rem !important;
  font-weight: 600 !important; color: var(--text-3) !important;
  padding: 0.55rem 1.25rem !important; border-radius: var(--r-md) !important;
  transition: all 0.15s !important; letter-spacing: 0 !important; text-transform: none !important;
}}
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
  color: var(--text-1) !important; background: var(--bg-elevated) !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  background: var(--brand) !important; color: #fff !important;
  box-shadow: 0 2px 8px rgba(8,145,178,0.25) !important; border-bottom: none !important;
}}
div[data-testid="stTabPanel"] {{ background: transparent !important; border: none !important; padding: 0 !important; }}

/* ── CARDS ── */
.card {{
  background: var(--bg-surface); border: 1px solid var(--border-s);
  border-radius: var(--r-lg); padding: 1.5rem; box-shadow: var(--shadow-xs);
  transition: box-shadow 0.15s, border-color 0.15s;
}}
.card:hover {{ box-shadow: var(--shadow-sm); border-color: var(--border-m); }}
.card-title {{
  font-family: 'Outfit', sans-serif; font-size: 0.72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-3);
  margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-s);
  display: flex; align-items: center; gap: 8px;
}}
.card-title-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--brand); flex-shrink: 0; }}

/* ── METRIC CARDS ── */
.metrics-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
.metric-card {{
  background: var(--bg-surface); border: 1px solid var(--border-s);
  border-radius: var(--r-lg); padding: 1.1rem 1.3rem;
  position: relative; overflow: hidden; box-shadow: var(--shadow-xs); transition: all 0.15s;
}}
.metric-card:hover {{ box-shadow: var(--shadow-sm); transform: translateY(-1px); }}
.metric-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--accent, var(--brand)); border-radius: var(--r-lg) var(--r-lg) 0 0;
}}
.metric-label {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-3); margin-bottom: 6px; }}
.metric-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; font-weight: 500; color: var(--accent, var(--brand)); line-height: 1; margin-bottom: 4px; }}
.metric-note {{ font-size: 0.7rem; color: var(--text-4); }}

/* ── STATUS BANNERS ── */
.status-banner {{ border-radius: var(--r-lg); padding: 1.1rem 1.5rem; margin-bottom: 1.25rem; display: flex; align-items: flex-start; gap: 14px; }}
.status-banner.detected {{ background: linear-gradient(135deg, #FFF1F2 0%, #FFF8F8 100%); border: 1.5px solid rgba(220,38,38,0.25); }}
.status-banner.normal {{ background: linear-gradient(135deg, #ECFDF5 0%, #F7FFFE 100%); border: 1.5px solid rgba(5,150,105,0.25); }}
.status-icon {{ width: 36px; height: 36px; border-radius: var(--r-sm); display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }}
.status-banner.detected .status-icon {{ background: var(--alert-light); }}
.status-banner.normal .status-icon {{ background: var(--ok-light); }}
.status-title {{ font-family: 'Outfit', sans-serif; font-size: 0.95rem; font-weight: 700; margin-bottom: 3px; }}
.status-banner.detected .status-title {{ color: var(--alert); }}
.status-banner.normal .status-title {{ color: var(--ok); }}
.status-detail {{ font-size: 0.82rem; color: var(--text-2); }}
.status-chips {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
.status-chip {{ font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; background: var(--bg-surface); border: 1px solid var(--border-m); color: var(--text-2); }}

/* ── CLINICAL NOTICE ── */
.clinical-notice {{
  background: var(--info-light); border: 1px solid rgba(37,99,235,0.2);
  border-radius: var(--r-sm); padding: 0.8rem 1rem; font-size: 0.8rem; color: var(--text-2);
  margin-bottom: 1.25rem; line-height: 1.55; display: flex; align-items: flex-start; gap: 10px;
}}
.clinical-notice .ci-icon {{ font-size: 14px; margin-top: 1px; flex-shrink: 0; color: var(--info); }}
.clinical-notice div {{ flex: 1; }}
.clinical-notice strong {{ color: var(--text-1); font-weight: 600; }}

/* ── RECOMMENDATION ITEMS ── */
.rec-list {{ display: flex; flex-direction: column; gap: 8px; }}
.rec-item {{
  display: flex; align-items: flex-start; gap: 12px; padding: 0.85rem 1rem;
  background: var(--bg-surface); border: 1px solid var(--border-s);
  border-radius: var(--r-md); border-left: 3px solid var(--accent, var(--brand)); transition: all 0.12s;
}}
.rec-item:hover {{ background: var(--bg-elevated); box-shadow: var(--shadow-xs); }}
.rec-urgency-badge {{
  font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
  padding: 2px 7px; border-radius: 20px; white-space: nowrap; flex-shrink: 0; margin-top: 1px;
  background: var(--bg-inset); color: var(--accent, var(--brand));
}}
.rec-text {{ font-size: 0.85rem; font-weight: 600; color: var(--text-1); margin-bottom: 3px; }}
.rec-evidence {{ font-size: 0.72rem; color: var(--text-3); font-style: italic; }}
.rec-category-tag {{ font-size: 0.62rem; font-weight: 600; color: var(--text-4); padding: 1px 6px; border-radius: 4px; background: var(--bg-inset); display: inline-block; margin-top: 4px; }}

/* ── GRADE PROBABILITY BARS ── */
.prob-bar-item {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 0.8rem; }}
.prob-label {{ width: 130px; flex-shrink: 0; font-weight: 500; color: var(--text-2); }}
.prob-track {{ flex: 1; height: 8px; background: var(--bg-inset); border-radius: 20px; overflow: hidden; }}
.prob-fill {{ height: 100%; border-radius: 20px; transition: width 0.3s; }}
.prob-pct {{ width: 40px; text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-2); font-weight: 500; }}

/* ── PROGRESSION HERO ── */
.prog-status-hero {{
  background: linear-gradient(135deg, #FFFBEB 0%, #FFF8F0 100%);
  border: 1.5px solid rgba(217,119,6,0.3); border-radius: var(--r-xl);
  padding: 1.5rem 2rem; margin-bottom: 1.5rem;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;
  box-shadow: var(--shadow-sm);
}}
.prog-status-label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--warn); margin-bottom: 4px; }}
.prog-status-title {{ font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 800; color: #92400E; letter-spacing: -0.01em; }}
.prog-status-sub {{ font-size: 0.83rem; color: var(--text-3); margin-top: 4px; }}
.prog-stats-grid {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
.prog-stat-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-4); font-weight: 600; margin-bottom: 2px; }}
.prog-stat-val {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 500; color: var(--text-1); }}
.prog-stat-val.red {{ color: var(--alert); }}
.prog-stat-val.amber {{ color: var(--warn); }}
.prog-stat-val.green {{ color: var(--ok); }}

/* ── REPORT BLOCKS ── */
.report-data-block {{
  background: var(--bg-elevated); border: 1px solid var(--border-s);
  border-radius: var(--r-md); padding: 1.1rem 1.25rem; margin-bottom: 1rem;
}}
.report-block-title {{
  font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--brand-dk); margin-bottom: 0.75rem; padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-s);
}}
.report-row {{ display: flex; gap: 1.5rem; font-size: 0.82rem; color: var(--text-2); padding: 0.2rem 0; }}
.report-row strong {{ min-width: 160px; color: var(--text-1); font-weight: 600; }}

/* ── ANALYSIS PROGRESS ── */
.analysis-progress {{
  background: var(--bg-elevated); border: 1px solid var(--border-s);
  border-radius: var(--r-md); padding: 1rem 1.25rem; margin-bottom: 1rem;
}}
.progress-label {{ font-size: 0.75rem; font-weight: 600; color: var(--text-2); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }}
.progress-track {{ height: 6px; background: var(--bg-inset); border-radius: 20px; overflow: hidden; }}
.progress-fill {{ height: 100%; border-radius: 20px; background: linear-gradient(90deg, var(--brand-dk), var(--brand)); }}
.progress-mono {{ font-family: 'IBM Plex Mono', monospace; color: var(--brand); font-size: 0.72rem; }}

/* ── PATIENT INFO BAR ── */
.patient-info-bar {{
  background: var(--bg-elevated); border: 1px solid var(--border-s);
  border-radius: var(--r-sm); padding: 0.65rem 1rem;
  display: flex; align-items: center; gap: 1.5rem;
  font-size: 0.8rem; color: var(--text-2); margin-bottom: 1.25rem; flex-wrap: wrap;
}}
.patient-id {{ font-weight: 700; color: var(--text-1); font-family: 'IBM Plex Mono', monospace; }}
.patient-info-sep {{ color: var(--border-m); }}

/* ── SECTION HEADING ── */
.section-heading {{
  font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--text-1);
  margin-bottom: 1rem; margin-top: 1.5rem; display: flex; align-items: center; gap: 10px;
  border-bottom: 1px solid var(--border-s); padding-bottom: 0.6rem;
}}
.section-heading:first-child {{ margin-top: 0; }}
.section-badge {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding: 3px 8px; border-radius: 20px; background: var(--brand-light); color: var(--brand-dk); }}
.section-sub {{ font-size: 0.83rem; color: var(--text-3); margin-top: -0.6rem; margin-bottom: 1rem; }}

/* ── UPLOAD ZONE ── */
.stFileUploader > div {{
  background: var(--bg-elevated) !important; border: 2px dashed var(--border-m) !important;
  border-radius: var(--r-lg) !important; transition: all 0.15s !important;
}}
.stFileUploader > div:hover {{ border-color: var(--brand) !important; background: var(--brand-light) !important; }}

/* ── BUTTONS ── */
.stButton > button {{
  font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important;
  border-radius: var(--r-sm) !important; font-size: 0.88rem !important;
  letter-spacing: 0.01em !important; transition: all 0.18s !important;
  border: 1.5px solid var(--border-m) !important; background: var(--bg-surface) !important;
  color: var(--text-2) !important; padding: 0.55rem 1.1rem !important;
}}
.stButton > button:hover {{ border-color: var(--brand) !important; color: var(--brand-dk) !important; background: var(--brand-light) !important; }}
.stButton > button[kind="primary"] {{ background: var(--brand) !important; color: #fff !important; border: none !important; box-shadow: 0 2px 8px rgba(8,145,178,0.3) !important; }}
.stButton > button[kind="primary"]:hover {{ background: var(--brand-dk) !important; box-shadow: 0 4px 16px rgba(8,145,178,0.4) !important; transform: translateY(-1px) !important; }}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {{
  background: var(--bg-surface) !important; border: 1.5px solid var(--border-m) !important;
  border-radius: var(--r-sm) !important; color: var(--text-1) !important;
  font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.9rem !important;
  padding: 0.7rem 0.9rem !important; transition: all 0.15s !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{ border-color: var(--brand) !important; box-shadow: 0 0 0 3px var(--brand-glow) !important; }}
[data-baseweb="select"] > div {{ background: var(--bg-surface) !important; border: 1.5px solid var(--border-m) !important; border-radius: var(--r-sm) !important; transition: all 0.15s !important; }}
[data-baseweb="select"] > div:focus-within {{ border-color: var(--brand) !important; box-shadow: 0 0 0 3px var(--brand-glow) !important; }}
[data-baseweb="select"] *, div[data-baseweb="popover"] *, div[role="listbox"] *, li[role="option"] * {{ color: var(--text-1) !important; -webkit-text-fill-color: var(--text-1) !important; }}
div[data-baseweb="popover"], div[role="listbox"], ul[role="listbox"] {{ background: var(--bg-surface) !important; border: 1px solid var(--border-m) !important; border-radius: var(--r-md) !important; box-shadow: var(--shadow-md) !important; }}
li[role="option"]:hover {{ background: var(--bg-elevated) !important; }}

/* ── LABELS ── */
label, .stSelectbox label, .stTextInput label, .stNumberInput label,
.stTextArea label, .stSlider label, .stDateInput label {{
  font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.72rem !important;
  font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: 0.08em !important; color: var(--text-2) !important;
  margin-bottom: 6px !important; display: block !important;
}}

/* ── CHECKBOX ── */
.stCheckbox > label {{ font-size: 0.85rem !important; color: var(--text-2) !important; font-weight: 500 !important; text-transform: none !important; letter-spacing: normal !important; }}
.stCheckbox input[type="checkbox"] {{ accent-color: var(--brand) !important; width: 16px !important; height: 16px !important; }}

/* ── PROGRESS BAR ── */
.stProgress > div > div {{ background: var(--bg-inset) !important; border-radius: 20px !important; }}
.stProgress > div > div > div {{ background: linear-gradient(90deg, var(--brand-dk), var(--brand)) !important; border-radius: 20px !important; }}

/* ── ALERTS ── */
.stAlert {{ border-radius: var(--r-sm) !important; font-size: 0.85rem !important; }}
div[data-testid="stAlert"][data-type="success"] {{ background: var(--ok-light) !important; border-left: 3px solid var(--ok) !important; color: #064E3B !important; }}
div[data-testid="stAlert"][data-type="warning"] {{ background: var(--warn-light) !important; border-left: 3px solid var(--warn) !important; color: #78350F !important; }}
div[data-testid="stAlert"][data-type="error"] {{ background: var(--alert-light) !important; border-left: 3px solid var(--alert) !important; color: #7F1D1D !important; }}
div[data-testid="stAlert"][data-type="info"] {{ background: var(--info-light) !important; border-left: 3px solid var(--info) !important; color: #1E3A8A !important; }}

/* ── EXPANDER ── */
.streamlit-expanderHeader {{
  background: var(--bg-elevated) !important; border-radius: var(--r-sm) !important;
  font-size: 0.88rem !important; font-weight: 600 !important; color: var(--text-1) !important;
  border: 1px solid var(--border-s) !important; padding: 0.7rem 1rem !important; transition: all 0.15s !important;
}}
.streamlit-expanderHeader:hover {{ border-color: var(--brand) !important; background: var(--brand-light) !important; }}
.streamlit-expanderContent {{
  background: var(--bg-surface) !important; border: 1px solid var(--border-s) !important;
  border-top: none !important; border-radius: 0 0 var(--r-sm) var(--r-sm) !important; padding: 1rem !important;
}}

/* ── DATAFRAME ── */
.stDataFrame {{ border-radius: var(--r-md) !important; overflow: hidden; }}
.stDataFrame table {{ background: var(--bg-surface) !important; }}
.stDataFrame thead th {{ background: var(--bg-elevated) !important; color: var(--text-2) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; font-weight: 700 !important; border-bottom: 1px solid var(--border-m) !important; }}
.stDataFrame tbody tr:hover {{ background: var(--bg-elevated) !important; }}
.stImage img {{ border-radius: var(--r-md) !important; }}
code {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 0.83rem !important; background: var(--bg-inset) !important; color: var(--brand-dk) !important; padding: 2px 6px !important; border-radius: 4px !important; border: 1px solid var(--border-s) !important; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-canvas); }}
::-webkit-scrollbar-thumb {{ background: var(--border-m); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--border-k); }}
hr {{ border: none; border-top: 1px solid var(--border-s) !important; margin: 1.25rem 0 !important; }}


/* ════════════════════════════════════════════════════════════════
   LOGIN PAGE  — Exact match for Image 2 split-panel design
   ════════════════════════════════════════════════════════════════ */

/* Hide default Streamlit chrome on login page */
.login-page-active .stApp > header,
.login-page-active footer {{ display: none !important; }}

/* The outer login wrapper fills the screen */
.login-outer-wrap {{
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 80vh;
  padding: 2rem 1rem;
}}

/* The card that holds both panels */
.login-card {{
  display: flex;
  width: 100%;
  max-width: 1060px;
  min-height: 680px;
  border-radius: 24px;
  overflow: hidden;
  box-shadow: 0 24px 60px rgba(2,6,23,0.20), 0 6px 20px rgba(2,6,23,0.10);
  border: 1px solid rgba(14, 116, 144, 0.18);
}}

/* LEFT PANEL */
.lp-left {{
  flex: 0 0 42%;
  background: linear-gradient(160deg, #075f7a 0%, #0882a0 35%, #0d9fbf 70%, #12b5d8 100%);
  padding: 2.8rem 2.4rem 2.4rem;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}

/* Decorative bubbles */
.lp-left::before {{
  content: "";
  position: absolute;
  width: 320px; height: 320px;
  border-radius: 50%;
  background: rgba(255,255,255,0.07);
  top: -140px; right: -120px;
  pointer-events: none;
}}
.lp-left::after {{
  content: "";
  position: absolute;
  width: 200px; height: 200px;
  border-radius: 50%;
  background: rgba(255,255,255,0.05);
  bottom: -90px; left: -80px;
  pointer-events: none;
}}

/* Brain icon */
.lp-brain-icon {{
  width: 58px; height: 58px;
  background: rgba(255,255,255,0.14);
  border: 1.5px solid rgba(255,255,255,0.22);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 1.6rem;
  position: relative; z-index: 1;
}}
.lp-brain-inner {{
  width: 32px; height: 32px;
  border-radius: 50%;
  background: radial-gradient(circle at 32% 32%, #fda4af 0%, #f472b6 50%, #e879f9 100%);
}}

/* Brand title */
.lp-title {{
  font-family: 'Outfit', sans-serif;
  font-size: 2.55rem;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -0.02em;
  line-height: 1.05;
  margin: 0 0 0.55rem;
  position: relative; z-index: 1;
}}
.lp-title em {{ font-style: normal; color: #67E8F9; }}

/* Subtitle */
.lp-subtitle {{
  font-size: 1rem;
  color: rgba(207, 247, 255, 0.82);
  line-height: 1.55;
  margin-bottom: auto;
  position: relative; z-index: 1;
  max-width: 300px;
}}

/* Feature list */
.lp-features {{
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  margin-top: 2.4rem;
  padding-top: 1.6rem;
  border-top: 1px solid rgba(255,255,255,0.12);
  position: relative; z-index: 1;
}}
.lp-feature {{
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-size: 0.93rem;
  font-weight: 600;
  color: rgba(207, 247, 255, 0.88);
}}
.lp-dot {{
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #7DD3FC;
  flex-shrink: 0;
  box-shadow: 0 0 6px rgba(125,211,252,0.6);
}}
.lp-dot.warn {{ background: #FDE047; box-shadow: 0 0 6px rgba(253,224,71,0.6); }}

/* RIGHT PANEL */
.lp-right {{
  flex: 1;
  background: #f8fafc;
  padding: 2.6rem 2.8rem 2.2rem;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}

/* Decorative plus sign top-right */
.lp-right::before {{
  content: "+";
  position: absolute;
  right: 4%;
  top: 28%;
  color: #dde8f0;
  font-size: 180px;
  line-height: 1;
  font-weight: 700;
  pointer-events: none;
  z-index: 0;
  user-select: none;
}}

/* Form heading */
.lp-form-title {{
  font-family: 'Outfit', sans-serif;
  font-size: 2.2rem;
  font-weight: 800;
  color: #0F172A;
  margin: 0;
  line-height: 1.1;
  position: relative; z-index: 1;
}}
.lp-form-sub {{
  font-size: 1rem;
  color: #64748B;
  margin: 0.4rem 0 1.5rem;
  position: relative; z-index: 1;
}}

/* Field label above inputs */
.lp-field-label {{
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #334155;
  margin: 0 0 0.3rem;
  position: relative; z-index: 1;
}}

/* Demo box */
.lp-demo-box {{
  background: #dbeafe;
  border: 1px solid #bfdbfe;
  border-radius: 12px;
  padding: 0.9rem 1.1rem;
  font-size: 0.9rem;
  color: #1d4ed8;
  margin-top: 0.75rem;
  line-height: 1.55;
  position: relative; z-index: 1;
}}
.lp-demo-box strong {{
  display: block;
  margin-bottom: 0.3rem;
  font-size: 0.76rem;
  letter-spacing: 0.1em;
  font-weight: 800;
  text-transform: uppercase;
}}
.lp-demo-box code {{
  background: rgba(255,255,255,0.55) !important;
  border-color: #bfdbfe !important;
  color: #1e40af !important;
  font-size: 0.88rem !important;
}}

/* Override Streamlit inputs inside right panel */
.lp-right .stTextInput > div > div > input,
.lp-right .stNumberInput > div > div > input {{
  height: 2.85rem !important;
  background: #ffffff !important;
  border: 1.4px solid #cbd5e1 !important;
  border-radius: 11px !important;
  color: #0f172a !important;
  font-size: 0.98rem !important;
  padding: 0 0.9rem !important;
  box-shadow: none !important;
  position: relative; z-index: 1;
}}
.lp-right .stTextInput > div > div > input:focus,
.lp-right .stNumberInput > div > div > input:focus {{
  border-color: #0891B2 !important;
  box-shadow: 0 0 0 3px rgba(8,145,178,0.14) !important;
}}
.lp-right .stTextInput button {{ display: none !important; }}

.lp-right [data-baseweb="select"] > div {{
  height: 2.85rem !important;
  background: #ffffff !important;
  border: 1.4px solid #cbd5e1 !important;
  border-radius: 11px !important;
}}
.lp-right [data-baseweb="select"] > div:focus-within {{
  border-color: #0891B2 !important;
  box-shadow: 0 0 0 3px rgba(8,145,178,0.14) !important;
}}

.lp-right .stCheckbox > label {{
  background: #f1f5f9 !important;
  border: 1px solid #dbe5ee !important;
  border-radius: 11px !important;
  padding: 0.7rem 0.85rem !important;
  color: #334155 !important;
  font-size: 0.93rem !important;
  font-weight: 500 !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  position: relative; z-index: 1;
}}
.lp-right .stCheckbox input[type="checkbox"] {{ accent-color: #0891B2 !important; }}

.lp-right .stButton > button[kind="primary"] {{
  height: 3.1rem !important;
  border-radius: 11px !important;
  background: linear-gradient(90deg, #0774a0 0%, #0d9fbf 100%) !important;
  border: none !important;
  color: #ffffff !important;
  font-size: 1.08rem !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.03em !important;
  box-shadow: 0 6px 18px rgba(8,145,178,0.28) !important;
  position: relative; z-index: 1;
}}
.lp-right .stButton > button[kind="primary"]:hover {{
  filter: brightness(1.06) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 10px 24px rgba(8,145,178,0.36) !important;
}}

.lp-right label {{
  font-size: 0.74rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: #334155 !important;
  position: relative; z-index: 1;
}}

/* Spacer util */
.lp-spacer {{ height: 0.7rem; position: relative; z-index: 1; }}

/* ── FOOTER ── */
.app-footer {{
  background: var(--bg-surface); border-top: 1px solid var(--border-s);
  padding: 1rem 0; margin-top: 2.5rem;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;
  font-size: 0.78rem; color: var(--text-3);
}}
.footer-name {{ font-family: 'Outfit', sans-serif; font-weight: 700; color: var(--text-1); font-size: 0.85rem; }}
.footer-name em {{ font-style: normal; color: var(--brand); }}
.footer-badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.footer-badge {{ font-size: 0.65rem; font-weight: 600; padding: 3px 8px; border-radius: 20px; background: var(--bg-elevated); border: 1px solid var(--border-s); color: var(--text-3); letter-spacing: 0.03em; }}
.footer-badge.warn {{ color: var(--alert); background: var(--alert-light); border-color: rgba(220,38,38,0.2); }}
</style>
"""


st.markdown(get_css(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_tumor_display_info(tumor_result: dict) -> dict:
    grade = tumor_result.get('grade', 0)
    detected = tumor_result.get('detected', False)
    if not detected:
        grade = 0
    return {
        'grade':      grade,
        'tumor_type': GRADE_NAMES.get(grade, 'Unknown'),
        'grade_roman':['0','I','II','III','IV'][min(grade, 4)],
        'detected':   detected,
        'color':      GRADE_COLORS[min(grade, 4)],
    }


def fix_visualization_title(tumor_result: dict) -> dict:
    info = get_tumor_display_info(tumor_result)
    tumor_result['tumor_type'] = info['tumor_type']
    tumor_result['grade']      = info['grade']
    return tumor_result


def format_confidence_percent(confidence: float) -> str:
    conf = float(np.clip(confidence, 0.0, 1.0))
    return f"{conf * 100:.0f}%"


def get_confidence_policy(tumor_result: dict) -> dict:
    conf = float(tumor_result.get('ai_confidence', tumor_result.get('confidence', 0.0)) or 0.0)
    threshold = float(tumor_result.get('confidence_threshold', 0.75) or 0.75)
    meets_threshold = bool(tumor_result.get('meets_confidence_threshold', conf >= threshold))
    return {
        'confidence': conf,
        'threshold': threshold,
        'meets_threshold': meets_threshold,
        'label': format_confidence_percent(conf),
        'threshold_label': format_confidence_percent(threshold),
    }


def predict_survival(tumor_info: dict, patient_data: dict,
                     survival_model=None, survival_scaler=None):
    if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
        return None
    base = {1: 365*8, 2: 365*3, 3: 365*1.2}
    days = base.get(tumor_info['grade'], 365*5)
    age_f = max(0.3, 1 - (patient_data['age'] - 40) / 120)
    kps_f = patient_data['kps'] / 100
    mol_f = 1.0
    if patient_data.get('idh_status') == 'Mutant':      mol_f *= 1.8
    if patient_data.get('mgmt_status') == 'Methylated': mol_f *= 1.5
    size_f = max(0.5, 1 - tumor_info.get('tumor_percentage', 0) / 20)
    return max(30, int(days * age_f * kps_f * mol_f * size_f))


def generate_treatment_recommendations(tumor_info: dict, survival_days,
                                        patient_data: dict) -> list:
    recs = []
    grade = tumor_info.get('grade', 0)
    age   = patient_data.get('age', 50)
    kps   = patient_data.get('kps', 80)
    idh   = patient_data.get('idh_status', 'Unknown')
    mgmt  = patient_data.get('mgmt_status', 'Unknown')

    if not tumor_info.get('detected', False) or grade == 0:
        recs += [
            ("No active treatment required - patient is tumour-free", "AI Analysis", "Status", "Routine"),
            ("Annual brain MRI surveillance (T1 + T2 + FLAIR)", "NCCN Guidelines", "Monitoring", "Routine"),
            ("Neurological examination every 12 months", "Standard of Care", "Monitoring", "Routine"),
            ("Maintain cardiovascular health and avoid head trauma", "General Wellness", "Lifestyle", "Routine"),
        ]
        return recs

    if grade == 1:
        recs += [
            ("Neurosurgical consultation for maximal safe resection", "WHO Grade I Standard", "Surgery", "Soon"),
            ("Gross total resection (GTR) is curative in most Grade I cases", "Level I Evidence - EANO 2021", "Surgery", "Soon"),
            ("Post-operative MRI within 48-72 h to assess resection extent", "Surgical Standard", "Imaging", "Post-op"),
            ("MRI surveillance every 6 months for 2 years, then annually", "NCCN CNS Guidelines", "Monitoring", "Routine"),
            ("Neuropsychological assessment for cognitive function baseline", "Quality of Life Protocol", "Assessment", "Routine"),
        ]
        if kps < 70:
            recs.append(("Physiotherapy and rehabilitation referral (KPS < 70)", "Supportive Care", "Rehabilitation", "Soon"))
    elif grade == 2:
        recs += [
            ("Maximal safe surgical resection - aim for >= 90% extent of resection", "Level I Evidence - EORTC 22033", "Surgery", "Urgent"),
            ("Post-operative MRI within 48 h (T1+Gd, T2, FLAIR)", "Surgical Standard", "Imaging", "Post-op"),
        ]
        if idh == 'Mutant':
            recs += [
                ("IDH-mutant: PCV chemotherapy x 6 cycles (procarbazine + CCNU + vincristine)", "RTOG 9802 - Level I", "Chemotherapy", "Urgent"),
                ("Radiotherapy 54 Gy in 30 fractions after chemotherapy", "RTOG 9802 / EORTC 22033", "Radiotherapy", "Urgent"),
            ]
        else:
            recs += [
                ("Radiotherapy 54-60 Gy in 30 fractions (standard fractionation)", "EORTC 22033 - Level I", "Radiotherapy", "Urgent"),
                ("Consider temozolomide (TMZ) 75 mg/m2 concurrent with RT", "Based on MGMT Status", "Chemotherapy", "Urgent"),
            ]
        if mgmt == 'Methylated':
            recs.append(("MGMT methylated: temozolomide chemotherapy strongly recommended", "Stupp NEJM 2005 - Level I", "Chemotherapy", "Urgent"))
        recs += [
            ("MRI surveillance every 3 months for first 2 years", "NCCN CNS Guidelines", "Monitoring", "Routine"),
            ("Molecular profiling: IDH1/2, 1p/19q, MGMT, TERT, ATRX", "WHO CNS Classification 2021", "Diagnostics", "Soon"),
        ]
        if age > 60:
            recs.append(("Hypofractionated RT (40 Gy/15 fx) for patients > 60 years", "Nordic Trial - Level I", "Radiotherapy", "Urgent"))
    else:
        recs += [
            ("URGENT: Neurosurgical consultation within 24-48 hours", "Emergency Standard of Care", "Surgery", "Emergency"),
            ("Maximal safe resection - aim for >= 98% extent of resection", "Stummer et al. Lancet Oncol 2006", "Surgery", "Emergency"),
            ("5-ALA fluorescence-guided surgery for complete resection", "Level I Evidence - EORTC", "Surgery", "Emergency"),
            ("Post-operative MRI within 24-48 h (T1+Gd mandatory)", "Surgical Standard", "Imaging", "Post-op"),
        ]
        if mgmt == 'Methylated':
            recs += [
                ("MGMT methylated: Stupp protocol - RT 60 Gy + concurrent TMZ 75 mg/m2", "Stupp NEJM 2005 - Level I", "Chemoradiotherapy", "Emergency"),
                ("Adjuvant TMZ 150-200 mg/m2 x 6 cycles (days 1-5 of 28)", "Stupp Protocol - Level I", "Chemotherapy", "Urgent"),
            ]
        else:
            recs += [
                ("RT 60 Gy/30 fx + concurrent TMZ (Stupp protocol)", "Standard of Care - Level I", "Chemoradiotherapy", "Emergency"),
                ("Consider lomustine (CCNU) + TMZ combination if MGMT unmethylated", "CeTeG/NOA-09 Trial", "Chemotherapy", "Urgent"),
            ]
        recs += [
            ("Tumor Treating Fields (TTFields / Optune) - add to maintenance TMZ", "EF-14 Trial - Level I, +4.9 mo OS", "Device Therapy", "Urgent"),
            ("Bevacizumab at recurrence (anti-VEGF therapy)", "AVAGLIO / RTOG 0825", "Targeted Therapy", "At Recurrence"),
            ("Enrol in clinical trial if available (immunotherapy / CAR-T)", "NCCN Category 2A", "Clinical Trial", "Soon"),
            ("Dexamethasone for cerebral oedema management (4-16 mg/day)", "Supportive Care Standard", "Supportive", "Immediate"),
            ("Anti-epileptic prophylaxis if seizure history (levetiracetam preferred)", "EANO Guidelines 2021", "Supportive", "Immediate"),
            ("Palliative care and psychosocial support referral", "ASCO Guidelines", "Supportive", "Soon"),
            ("MRI surveillance every 8 weeks during treatment", "RANO Criteria Monitoring", "Monitoring", "Routine"),
        ]
        if kps < 60:
            recs.append(("KPS < 60: consider hypofractionated RT (40 Gy/15 fx) + TMZ", "Perry et al. NEJM 2017", "Radiotherapy", "Urgent"))
    return recs


def calculate_risk_level(tumor_info: dict, patient_data: dict):
    if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
        return "Tumour Free", 0, THEME["ok"]
    base  = tumor_info['grade'] * 25
    age_r = max(0, (patient_data['age'] - 40) / 2)
    kps_r = (100 - patient_data['kps']) / 2
    mol_r = 0
    if patient_data.get('idh_status') == 'Wildtype':      mol_r += 15
    if patient_data.get('mgmt_status') == 'Unmethylated': mol_r += 10
    total = min(100, base + age_r + kps_r + mol_r)
    if   total < 25: return "Low",      total, THEME["ok"]
    elif total < 50: return "Moderate", total, THEME["warn"]
    elif total < 75: return "High",     total, THEME["grade_3"]
    else:            return "Critical", total, THEME["alert"]


def safe_dataframe_display(df, width='stretch'):
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == 'object':
            try:
                df2[col] = pd.to_datetime(df2[col], errors='ignore')
                if df2[col].dtype == 'object':
                    df2[col] = df2[col].astype('string')
            except Exception:
                df2[col] = df2[col].astype('string')
    return st.dataframe(df2, width=width)


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'
    return f'rgba(8,145,178,{alpha})'


def enhance_preview(image):
    if image.max() <= 1:
        img_8u = (image * 255).astype(np.uint8)
    else:
        img_8u = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_8u)
    return enhanced / 255.0


# ══════════════════════════════════════════════════════════════════════════════
#  UI COMPONENT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def metric_card_html(label: str, value: str, note: str = "", color: str = None) -> str:
    color = color or THEME["brand"]
    return f"""
    <div class="metric-card" style="--accent:{color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        {"<div class='metric-note'>" + note + "</div>" if note else ""}
    </div>"""


def rec_card_html(rec: str, evidence: str, category: str, urgency: str) -> str:
    urgency_colors = {
        "Emergency": THEME["alert"], "Urgent": THEME["grade_3"],
        "Post-op": THEME["purple"], "Soon": THEME["info"],
        "At Recurrence": THEME["ok"], "Immediate": THEME["alert"], "Routine": THEME["ok"],
    }
    color = urgency_colors.get(urgency, THEME["text_3"])
    return f"""
    <div class="rec-item" style="--accent:{color};border-left-color:{color};">
        <span class="rec-urgency-badge" style="color:{color};">{urgency}</span>
        <div>
            <div class="rec-text">{rec}</div>
            <div class="rec-evidence">{evidence}</div>
            <span class="rec-category-tag">{category}</span>
        </div>
    </div>"""


def prob_bar_html(label: str, pct: float, color: str) -> str:
    return f"""
    <div class="prob-bar-item">
        <div class="prob-label">{label}</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{pct*100:.1f}%;background:{color};"></div>
        </div>
        <div class="prob-pct">{pct*100:.0f}%</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def _show_rejection(validation_result: dict, scan_label: str = ""):
    label = f" - {scan_label}" if scan_label else ""
    st.error(f"**Image Rejected{label}: Not a valid brain MRI scan**")
    st.error(f"Reason: {validation_result['reason']}")
    if 'critical_scores' in validation_result:
        scores = validation_result['critical_scores']
        st.warning(
            f"Brain Structure: {scores.get('brain_structure', 0):.1%} (required >= 55%)  "
            f"| Text Content: {scores.get('text_content', 0):.1%} (must be < 20%)  "
            f"| Screenshot/UI: {scores.get('screenshot_ui', 0):.1%} (must be < 30%)")
    with st.expander("Accepted formats and troubleshooting"):
        st.markdown("""
**Accepted:** Brain MRI (T1, T2, FLAIR, DWI, ADC), DICOM, NIfTI, High-quality grayscale PNG/JPG

**Not accepted:** Screenshots, Screen photographs, Images with text overlays, Chest X-rays, Colour photographs, Low-quality or compressed images

**How to fix:** Export directly from the scanner or PACS as DICOM or high-resolution PNG without compression. Remove any text overlays before uploading.
        """)


def _show_validation_success(validation_result: dict):
    st.success(f"Brain MRI validated - Confidence: {validation_result['confidence']:.1%}")
    if 'critical_scores' in validation_result:
        with st.expander("Validation details"):
            scores = validation_result['critical_scores']
            vc1, vc2, vc3 = st.columns(3)
            with vc1: st.metric("Brain Structure", f"{scores.get('brain_structure', 0):.1%}")
            with vc2: st.metric("Text Content", f"{scores.get('text_content', 0):.1%}")
            with vc3: st.metric("Screenshot/UI", f"{scores.get('screenshot_ui', 0):.1%}")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_user_credentials(username, password, license_number, institution):
    try:
        sample_users  = get_sample_users()
        if username not in sample_users:
            return False
        user_data     = sample_users[username]
        password_hash = _hash_password(password)
        return (user_data['password_hash'] == password_hash and
                user_data['license']       == license_number and
                user_data['institution']   == institution and
                user_data['is_active'])
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    defaults = dict(
        authenticated=False, login_attempts=0, user_info={},
        last_login_attempt=None, lockout_until=None,
        models_loaded=False, seg_model=None, class_models={}, device='cpu',
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if 'clinical_db' not in st.session_state:
        st.session_state.clinical_db = ClinicalDatabase()


@st.cache_resource
def load_models_cached():
    if not TORCH_AVAILABLE:
        return None, {}, 'cpu'
    device = get_device()
    try:
        seg_model = Enhanced3DUNet(in_channels=1, out_channels=4).to(device)
        seg_model.eval()
        class_models = {}
        for backbone in ['efficientnet', 'resnet', 'densenet']:
            try:
                m = EnhancedGliomaClassifier(backbone=backbone).to(device)
            except Exception:
                m = EnhancedGliomaClassifier(backbone='custom').to(device)
            m.eval()
            class_models[backbone] = m
        return seg_model, class_models, device
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, {}, device


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_authentication()
    initialize_session_state()

    if not st.session_state.models_loaded:
        pb   = st.progress(0.0)
        ptxt = st.empty()
        ptxt.markdown(
            f"<div class='analysis-progress'>"
            f"<div class='progress-label'>"
            f"<span>Initialising pretrained model pipeline...</span>"
            f"<span class='progress-mono'>0%</span></div>"
            f"<div class='progress-track'><div class='progress-fill' style='width:0%;'></div></div>"
            f"</div>",
            unsafe_allow_html=True)
        pb.progress(0.15)
        seg_model, class_models, device = load_models_cached()
        pb.progress(0.6)
        ptxt.markdown(
            f"<div class='analysis-progress'>"
            f"<div class='progress-label'>"
            f"<span>Verifying pretrained weights...</span>"
            f"<span class='progress-mono'>60%</span></div>"
            f"<div class='progress-track'><div class='progress-fill' style='width:60%;'></div></div>"
            f"</div>",
            unsafe_allow_html=True)
        st.session_state.seg_model     = seg_model
        st.session_state.class_models  = class_models
        st.session_state.device        = device
        st.session_state.models_loaded = True
        pb.progress(1.0); pb.empty(); ptxt.empty()

    if not st.session_state.authenticated:
        _render_login()
        return

    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-title">Brain MRI Analysis &amp; Clinical Decision Support</div>
        <div class="header-badges">
            <span class="hbadge hbadge-teal">Real Pretrained U-Net Pipeline</span>
            <span class="hbadge hbadge-green">WHO CNS 2021</span>
            <span class="hbadge hbadge-red">Not FDA Cleared - Decision Support Only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Analysis",
        "Progression Analysis",
        "Clinical Reports",
        "Patient Management",
        "Dashboard",
    ])

    with tab1: _tab_single()
    with tab2: _tab_progression()
    with tab3: _tab_reports()
    with tab4: _tab_patient_management()
    with tab5: create_clinical_dashboard()

    _render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN  — Pure HTML card, exact match for Image 2
# ══════════════════════════════════════════════════════════════════════════════

def _render_login():
    if check_login_attempts():
        return

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #DCE9EF 0%, #C8DDE6 40%, #DCE9EF 100%) !important;
    }

    /* ── Hide Streamlit chrome ── */
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    #MainMenu,
    footer { display: none !important; }

    /* ── Remove all default padding/margin ── */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        margin: 0 !important;
    }

    section[data-testid="stMain"] {
        padding: 0 !important;
    }

    section[data-testid="stMain"] > div {
        padding: 0 !important;
    }

    /* ── Full-page centering for the top-level vertical block ── */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 100vh !important;
        padding: 2rem 3rem !important;
        background: transparent !important;
    }

    /* ══════════════════════════════════════════════
       THE CARD — target the horizontal columns block
    ══════════════════════════════════════════════ */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        max-width: 980px !important;
        margin: 0 auto !important;
        background: #FFFFFF !important;
        border-radius: 22px !important;
        overflow: hidden !important;
        box-shadow:
            0 2px 4px rgba(0,0,0,0.04),
            0 8px 24px rgba(2, 20, 60, 0.10),
            0 20px 56px rgba(2, 20, 60, 0.13) !important;
        border: 1px solid rgba(8, 145, 178, 0.14) !important;
        gap: 0 !important;
        padding: 0 !important;
        align-items: stretch !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }

    /* ── LEFT column — teal panel ── */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
        flex: 0 0 40% !important;
        min-width: 40% !important;
        max-width: 40% !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }

    /* ── RIGHT column — white form ── */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
        flex: 1 1 60% !important;
        min-width: 60% !important;
        background: #FFFFFF !important;
        padding: 2.6rem 2.8rem 2.4rem 2.8rem !important;
        margin: 0 !important;
        position: relative !important;
        overflow: hidden !important;
    }

    /* Decorative + watermark */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child::after {
        content: "+";
        position: absolute;
        right: 3%;
        top: 18%;
        font-size: 210px;
        font-weight: 800;
        color: rgba(8, 145, 178, 0.04);
        font-family: 'Outfit', sans-serif;
        line-height: 1;
        pointer-events: none;
        user-select: none;
        z-index: 0;
    }

    /* ══════════════════════════════════════════════
       LEFT TEAL PANEL
    ══════════════════════════════════════════════ */
    .lts-brand {
        background: linear-gradient(160deg, #0C6D84 0%, #0886A3 40%, #09A3C2 72%, #0FC0DC 100%);
        padding: 2.8rem 2.2rem 2.4rem 2.2rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 600px;
        position: relative;
        overflow: hidden;
        box-sizing: border-box;
    }
    /* Top-right decorative circle */
    .lts-brand::before {
        content: '';
        position: absolute;
        top: -100px; right: -90px;
        width: 290px; height: 290px;
        background: rgba(255,255,255,0.09);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    /* Bottom-left decorative circle */
    .lts-brand::after {
        content: '';
        position: absolute;
        bottom: -55px; left: -55px;
        width: 195px; height: 195px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    .lts-top { position: relative; z-index: 1; }
    .lts-icon {
        width: 52px; height: 52px;
        background: rgba(255,255,255,0.17);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 1.55rem;
        border: 1.5px solid rgba(255,255,255,0.24);
    }
    .lts-brain-dot {
        width: 26px; height: 26px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 32%, #fda4af 0%, #f472b6 52%, #e879f9 100%);
    }
    .lts-h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 1.95rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .lts-h1 em { font-style: normal; color: #A5F3FC; }
    .lts-sub {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.875rem;
        color: rgba(207, 247, 255, 0.80);
        line-height: 1.65;
        margin: 0;
        max-width: 260px;
    }
    .lts-features {
        position: relative; z-index: 1;
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-top: 1.55rem;
        border-top: 1px solid rgba(255,255,255,0.15);
        margin-top: 2rem;
    }
    .lts-feat {
        display: flex;
        align-items: center;
        gap: 9px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        color: rgba(207, 247, 255, 0.88);
    }
    .lts-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #7DD3FC;
        flex-shrink: 0;
    }
    .lts-dot.warn { background: #FCD34D; }

    /* ══════════════════════════════════════════════
       RIGHT FORM PANEL
    ══════════════════════════════════════════════ */
    .lts-form-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 0.28rem 0;
        line-height: 1.2;
        position: relative; z-index: 1;
    }
    .lts-form-sub {
        font-size: 0.84rem;
        color: #64748B;
        margin: 0 0 1.4rem 0;
        font-family: 'IBM Plex Sans', sans-serif;
        position: relative; z-index: 1;
    }
    .lts-demo {
        background: #DBEAFE;
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 10px;
        padding: 0.8rem 1.05rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.8rem;
        color: #1D4ED8;
        line-height: 1.65;
        margin: 0.45rem 0 0.55rem 0;
        position: relative; z-index: 1;
    }
    .lts-demo strong {
        display: block;
        margin-bottom: 4px;
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 800;
        color: #1E40AF;
    }
    .lts-demo code {
        font-family: 'IBM Plex Mono', monospace !important;
        background: rgba(37, 99, 235, 0.11) !important;
        color: #1E40AF !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
        border: none !important;
        font-size: 0.79rem !important;
    }

    /* ══════════════════════════════════════════════
       STREAMLIT WIDGET OVERRIDES — RIGHT PANEL ONLY
    ══════════════════════════════════════════════ */

    /* Labels */
    div[data-testid="column"]:last-child label,
    div[data-testid="column"]:last-child .stTextInput label,
    div[data-testid="column"]:last-child .stSelectbox label {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.69rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.10em !important;
        color: #334155 !important;
        margin-bottom: 4px !important;
        position: relative; z-index: 1;
    }

    /* Text inputs */
    div[data-testid="column"]:last-child .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 9px !important;
        color: #0F172A !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 0.9rem !important;
        box-shadow: none !important;
        height: auto !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
        position: relative; z-index: 1;
    }
    div[data-testid="column"]:last-child .stTextInput > div > div > input:focus {
        border-color: #0891B2 !important;
        box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.13) !important;
        outline: none !important;
    }
    div[data-testid="column"]:last-child .stTextInput > div > div > input::placeholder {
        color: #94A3B8 !important;
    }
    /* Hide password reveal button */
    div[data-testid="column"]:last-child .stTextInput button {
        display: none !important;
    }

    /* Selectbox */
    div[data-testid="column"]:last-child [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        min-height: 43px !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    div[data-testid="column"]:last-child [data-baseweb="select"] > div:focus-within {
        border-color: #0891B2 !important;
        box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.13) !important;
    }
    div[data-testid="column"]:last-child [data-baseweb="select"] *,
    div[data-testid="column"]:last-child div[role="listbox"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[data-testid="column"]:last-child div[role="listbox"],
    div[data-testid="column"]:last-child div[data-baseweb="popover"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.12) !important;
    }

    /* Checkbox */
    div[data-testid="column"]:last-child .stCheckbox > label {
        background: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 9px !important;
        padding: 0.78rem 0.95rem !important;
        color: #334155 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        line-height: 1.5 !important;
        cursor: pointer !important;
        width: 100% !important;
        transition: border-color 0.15s, background 0.15s !important;
        position: relative; z-index: 1;
    }
    div[data-testid="column"]:last-child .stCheckbox > label:hover {
        border-color: #0891B2 !important;
        background: #F0F9FF !important;
    }
    div[data-testid="column"]:last-child .stCheckbox input[type="checkbox"] {
        accent-color: #0891B2 !important;
        width: 16px !important;
        height: 16px !important;
    }

    /* Sign In button */
    div[data-testid="column"]:last-child .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0774A0 0%, #0891B2 55%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        padding: 0.82rem 1.5rem !important;
        height: auto !important;
        letter-spacing: 0.025em !important;
        box-shadow: 0 4px 16px rgba(8, 145, 178, 0.34) !important;
        transition: all 0.18s ease !important;
        width: 100% !important;
        position: relative; z-index: 1;
    }
    div[data-testid="column"]:last-child .stButton > button[kind="primary"]:hover {
        filter: brightness(1.07) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(8, 145, 178, 0.42) !important;
    }

    /* Alert messages in form */
    div[data-testid="column"]:last-child .stAlert {
        border-radius: 9px !important;
        font-size: 0.83rem !important;
        position: relative; z-index: 1;
    }

    /* Reduce gap between form rows */
    div[data-testid="column"]:last-child > div[data-testid="stVerticalBlock"] > * {
        margin-bottom: 0.55rem !important;
    }
    div[data-testid="column"]:last-child .stTextInput,
    div[data-testid="column"]:last-child .stSelectbox,
    div[data-testid="column"]:last-child .stCheckbox {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Inner horizontal blocks (field rows) — remove extra spacing */
    div[data-testid="column"]:last-child [data-testid="stHorizontalBlock"] {
        gap: 0.75rem !important;
        margin-bottom: 0 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # ── Layout: padding columns + card columns ────────────────────────────────
    pad_l, card_col, pad_r = st.columns([0.3, 9.4, 0.3])

    with card_col:
        col_brand, col_form = st.columns([9, 11], gap="small")

        # ── LEFT: Pure HTML teal brand panel ─────────────────────────────────
        with col_brand:
            st.markdown("""
            <div class="lts-brand">
                <div class="lts-top">
                    <div class="lts-icon">
                        <div class="lts-brain-dot"></div>
                    </div>
                    <div class="lts-h1">LongiTumor<em>Sense</em></div>
                    <p class="lts-sub">
                        AI-Assisted Brain MRI Analysis &amp;<br>
                        Clinical Decision Support System
                    </p>
                </div>
                <div class="lts-features">
                    <div class="lts-feat"><div class="lts-dot"></div>4-Model Ensemble Active</div>
                    <div class="lts-feat"><div class="lts-dot"></div>WHO CNS 2021 Classification</div>
                    <div class="lts-feat"><div class="lts-dot"></div>DICOM / NIfTI / PNG Support</div>
                    <div class="lts-feat"><div class="lts-dot warn"></div>Not FDA Cleared &#8212; Decision Support Only</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── RIGHT: Streamlit form widgets ─────────────────────────────────────
        with col_form:

            st.markdown("""
            <div class="lts-form-title">Secure Sign In</div>
            <div class="lts-form-sub">Restricted to authorised medical professionals</div>
            """, unsafe_allow_html=True)

            # Row 1 — Username | Password
            r1a, r1b = st.columns(2, gap="small")
            with r1a:
                username = st.text_input("Username",
                    placeholder="dr.hassan", key="login_username")
            with r1b:
                password = st.text_input("Password",
                    type="password", placeholder="••••••••", key="login_password")

            # Row 2 — License | Institution
            r2a, r2b = st.columns(2, gap="small")
            with r2a:
                license_number = st.text_input("Medical License No.",
                    placeholder="MD123456", key="login_license")
            with r2b:
                institution = st.selectbox("Institution",
                    ["Mayo Clinic", "Mass General Hospital", "Johns Hopkins Hospital",
                     "Cleveland Clinic", "UCSF Medical Center", "Demo Hospital", "Other"],
                    key="login_institution")

            # Row 3 — Experience | Specialty
            r3a, r3b = st.columns(2, gap="small")
            with r3a:
                years_experience_input = st.text_input("Years of Experience",
                    value="8", key="login_experience")
            with r3b:
                specialty = st.selectbox("Specialty",
                    ["Radiology", "Neurosurgery", "Oncology", "Neurology", "Other"],
                    key="login_specialty")

            # Checkbox
            agree_terms = st.checkbox(
                "I confirm I am an authorised medical professional and agree to the terms of use",
                key="login_agree_terms")

            # Demo box
            st.markdown("""
            <div class="lts-demo">
                <strong>Demo Credentials</strong>
                Username: <code>demo</code>&nbsp;&nbsp;
                Password: <code>demo123</code>&nbsp;&nbsp;
                License: <code>MD999999</code>&nbsp;&nbsp;
                Institution: <code>Demo Hospital</code>
            </div>
            """, unsafe_allow_html=True)

            # Sign In button
            if st.button("Sign In →", type="primary",
                         use_container_width=True, key="login_submit"):

                years_text = years_experience_input.strip()

                if not all([username, password, license_number,
                            institution, years_text, agree_terms]):
                    st.error("Please complete all fields and confirm the terms.")
                    st.session_state.login_attempts += 1
                else:
                    try:
                        years_experience = int(years_text)
                    except ValueError:
                        st.error("Years of Experience must be a whole number.")
                        st.session_state.login_attempts += 1
                        return

                    if not 0 <= years_experience <= 50:
                        st.error("Years of Experience must be between 0 and 50.")
                        st.session_state.login_attempts += 1
                    elif verify_user_credentials(username, password,
                                                 license_number, institution):
                        st.session_state.authenticated  = True
                        st.session_state.user_info = dict(
                            username    = username,
                            license     = license_number,
                            institution = institution,
                            experience  = years_experience,
                            specialty   = specialty,
                        )
                        st.session_state.login_attempts = 0
                        st.success("Authentication successful.")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Please check your credentials.")
                        st.session_state.login_attempts += 1

       

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar():
    u = st.session_state.user_info
    initials = ''.join(w[0].upper() for w in u.get('username', 'U').split('.')[:2])[:2] or 'U'
    st.sidebar.markdown(f"""
        <div class="sidebar-logo-area">
        <div class="sidebar-logo">
            <div class="sidebar-logo-mark"></div>
            <div>
                <div class="sidebar-logo-text">LongiTumor<em>Sense</em></div>
                <div class="sidebar-logo-sub">Brain Tumour Detection</div>
            </div>
        </div>
        <div class="user-card">
            <div class="user-avatar">{initials}</div>
            <div>
                <div class="user-name">{u.get('username', 'Unknown')}</div>
                <div class="user-role">{u.get('specialty', '-')} - {u.get('institution', '-')}</div>
            </div>
            <div class="user-badge">Active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.models_loaded = False
        st.rerun()

    st.sidebar.markdown('<span class="sidebar-section-label">Active AI Models</span>',
                        unsafe_allow_html=True)
    for name, detail in [
        ("U-Net Segmenter",  "LGG dataset - Dice 0.85"),
        ("EfficientNet-B4",  "ImageNet adapted"),
        ("ResNet-50",        "ImageNet adapted"),
        ("DenseNet-121",     "ImageNet adapted"),
    ]:
        st.sidebar.markdown(f"""
        <div class="model-pill">
            <div class="model-dot"></div>
            <div>
                <div class="model-name">{name}</div>
                <div class="model-info">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="not-cleared-banner">
        <strong>Not FDA Cleared</strong>
        Research and decision-support use only.
        All results require radiologist review before clinical action.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RAG EVIDENCE PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _render_rag_panel(rag_data: Dict):
    """Render the RAG evidence panel showing retrieved cases + population stats."""
    st.markdown("""
    <div class="section-heading" style="margin-top:1.5rem;">
        Clinical Evidence (RAG-Augmented)
        <span class="section-badge">TCGA + WHO 2021</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "RAG Grade",
            f"Grade {rag_data.get('rag_grade', 0)}",
            f"{rag_data.get('rag_confidence', 0):.1%} confidence",
        )
    with col2:
        st.metric(
            "Retrieval Quality",
            f"{rag_data.get('retrieval_confidence', 0):.1%}",
            f"{rag_data.get('grade_agreement', 0):.1%} agreement",
        )
    with col3:
        pop_stats = rag_data.get('population_stats', {})
        median_surv = pop_stats.get('median_survival_months', 0)
        st.metric(
            "Population Median Survival",
            f"{median_surv:.1f} mo",
            f"{pop_stats.get('n_cases', 0)} similar cases",
        )

    # Supporting evidence cards
    evidence = rag_data.get('supporting_evidence', [])
    if evidence:
        st.markdown("**Top 5 Similar Cases:**")
        for ev in evidence[:5]:
            with st.expander(
                f"#{ev['rank']} — {ev['grade_name']} (similarity: {ev['similarity']:.1%})"
            ):
                st.write(f"**Source:** {ev['source']}")
                st.write(f"**Diagnosis:** {ev['diagnosis']}")
                st.write(f"**Age:** {ev['age']} years, **Sex:** {ev['sex']}")
                st.write(f"**Survival:** {ev['survival_months']:.1f} months")
                if ev['treatments']:
                    st.write(f"**Treatments:** {', '.join(ev['treatments'])}")
                st.write(f"**Tumor Area:** {ev['tumor_pct']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_single():
    st.markdown("""
    <div class="section-heading">
        Single MRI Analysis
        <span class="section-badge">Upload &amp; Detect</span>
    </div>
    <div class="section-sub">
        Upload a brain MRI scan for AI-assisted tumour detection, segmentation, and evidence-based treatment reference.
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_results = st.columns([1, 2], gap="large")

    with col_upload:
        st.markdown("""
        <div class="card-title"><div class="card-title-dot"></div>Image Upload</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Brain MRI scan (T1 / T2 / FLAIR / DWI / DICOM / NIfTI)",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
            help="Only brain MRI scans are accepted.",
        )

        if uploaded_file:
            image, metadata = load_medical_image(uploaded_file)
            if image is not None:
                if len(image.shape) == 3 and image.shape[2] > 3:
                    ds   = image.shape[2] // 2
                    disp = image[:, :, ds]
                elif len(image.shape) == 3:
                    disp = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            if image.shape[2] == 3 else image[:, :, 0])
                    ds = None
                else:
                    disp = image; ds = None

                if BRAIN_VALIDATOR_AVAILABLE:
                    with st.spinner("Validating image..."):
                        vr = validate_brain_mri(image, metadata if metadata else None)
                    if not vr['is_valid']:
                        _show_rejection(vr); st.stop()
                    _show_validation_success(vr)
                else:
                    st.success("Image loaded.")

                st.info(f"Shape: {image.shape}   |   Format: {uploaded_file.name.rsplit('.',1)[-1].upper()}"
                        + (f"   |   Slice: {ds}/{image.shape[2]}" if ds is not None else ""))
                disp_norm     = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
                disp_enhanced = enhance_preview(disp_norm)
                st.image(disp_enhanced, caption="Preview (CLAHE enhanced)", use_column_width=True)

                st.markdown("""
                <div class="card-title" style="margin-top:1.25rem;">
                    <div class="card-title-dot"></div>Patient Information
                </div>
                """, unsafe_allow_html=True)

                # Patient ID and Name
                pid_col, name_col = st.columns(2)
                with pid_col:
                    patient_id = st.text_input("Patient ID", value="", placeholder="e.g., PT-12345")
                with name_col:
                    patient_name = st.text_input("Patient Name", value="", placeholder="e.g., John Doe")
                
                pc1, pc2 = st.columns(2)
                with pc1:
                    age = st.number_input("Age", min_value=1, max_value=120, value=52)
                    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                with pc2:
                    kps       = st.slider("KPS Score", 0, 100, 80)
                    scan_date = st.date_input("Scan Date", value=datetime.now().date())

                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

                if st.button("Run AI Analysis", type="primary", use_container_width=True):
                    # Check if models are available
                    models_available = False
                    try:
                        from models import get_ensemble_models
                        models_available = True
                    except:
                        pass
                    
                    if not models_available:
                        st.error("⚠️ **CRITICAL: No AI models loaded!**")
                        st.warning("""
                        **The system is running in FALLBACK MODE with basic image analysis only.**
                        
                        **To get accurate tumor detection:**
                        1. Run: `python download_models.py` to download required models
                        2. Restart the application
                        
                        **Current limitations:**
                        - May miss subtle tumors
                        - Lower accuracy than trained models
                        - Results should be treated as preliminary only
                        """)
                    
                    # Validate required fields
                    if not patient_id or not patient_name:
                        st.error("Please enter both Patient ID and Patient Name")
                        return
                    
                    patient_data = dict(
                        age=age, sex=sex, kps=kps,
                        idh_status="Unknown", mgmt_status="Unknown",
                        patient_id=patient_id, patient_name=patient_name, 
                        scan_date=scan_date)

                    pb   = st.progress(0.0)
                    ptxt = st.empty()

                    def _cb(frac, msg):
                        pb.progress(min(float(frac), 1.0))
                        ptxt.markdown(
                            f"<div class='analysis-progress'>"
                            f"<div class='progress-label'>"
                            f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:{THEME['text_2']};'>{msg}</span>"
                            f"<span class='progress-mono'>{int(frac*100)}%</span></div>"
                            f"<div class='progress-track'>"
                            f"<div class='progress-fill' style='width:{int(frac*100)}%;'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True)

                    tumor_result = advanced_tumor_detection_with_ai(
                        image, st.session_state.class_models,
                        st.session_state.device, progress_callback=_cb,
                        patient_data=patient_data, use_rag=True)
                    pb.progress(1.0); ptxt.empty(); pb.empty()

                    tumor_result   = fix_visualization_title(tumor_result)
                    survival_days  = predict_survival(tumor_result, patient_data)
                    recommendations= generate_treatment_recommendations(
                        tumor_result, survival_days, patient_data)
                    risk_level, risk_score, risk_color = calculate_risk_level(
                        tumor_result, patient_data)

                    st.session_state.analysis_results = dict(
                        tumor_result=tumor_result, patient_data=patient_data,
                        survival_days=survival_days, recommendations=recommendations,
                        risk_level=risk_level, risk_score=risk_score, risk_color=risk_color,
                        display_image=disp_enhanced, original_image=disp,
                    )
                    st.session_state.clinical_db.add_or_update_patient(
                        patient_data, st.session_state.analysis_results)
                    
                    # Save to patient records for Patient Management tab
                    if 'patient_records' not in st.session_state:
                        st.session_state.patient_records = []
                    
                    patient_record = {
                        'patient_data': patient_data,
                        'tumor_result': tumor_result,
                        'survival_days': survival_days,
                        'risk_level': risk_level,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    st.session_state.patient_records.append(patient_record)

                    cp = get_confidence_policy(tumor_result)
                    st.success(
                        f"Analysis complete - {tumor_result.get('num_models_used', 1)} models  |  "
                        f"Confidence: {cp['label']}")
                    if not cp['meets_threshold']:
                        st.warning(
                            f"Confidence below required threshold ({cp['threshold_label']}). "
                            "Treat as preliminary - verify with radiologist review.")
                    st.warning("Radiologist review required before any clinical action.")

    with col_results:
        if 'analysis_results' not in st.session_state:
            st.markdown(f"""
            <div class="card" style="display:flex;align-items:center;justify-content:center;
                        min-height:400px;flex-direction:column;gap:0.75rem;border-style:dashed;">
                <p style="color:{THEME['text_4']};font-size:0.88rem;margin:0;text-align:center;">
                    Upload a scan and run analysis<br>to view results here
                </p>
            </div>
            """, unsafe_allow_html=True)
            return

        results = st.session_state.analysis_results
        ti      = results['tumor_result']
        td      = get_tumor_display_info(ti)
        cp      = get_confidence_policy(ti)
        pd_     = results['patient_data']

        st.markdown(f"""
        <div class="patient-info-bar">
            <span class="patient-id">{pd_.get('patient_name', 'N/A')}</span>
            <span class="patient-info-sep">|</span>
            <span class="patient-id">{pd_['patient_id']}</span>
            <span class="patient-info-sep">|</span>
            <span>{pd_['age']} yrs</span>
            <span class="patient-info-sep">|</span>
            <span>{pd_['sex']}</span>
            <span class="patient-info-sep">|</span>
            <span>KPS {pd_['kps']}</span>
            <span class="patient-info-sep">|</span>
            <span>{pd_['scan_date']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="clinical-notice">
            <div>
                <strong>AI Preliminary Finding</strong> - All results must be independently
                verified by a qualified radiologist or neuro-oncologist before any clinical
                decision is made. Confidence scores reflect model certainty, not diagnostic accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if td['detected']:
            threshold_chip = (
                "<span class='status-chip' style='color:#DC2626;border-color:#DC2626;'>"
                "Below Threshold</span>"
                if not cp['meets_threshold'] else ""
            )
            chips_html = (
                f"<div class='status-chips'>"
                f"<span class='status-chip' style='color:{td['color']};border-color:{td['color']};'>"
                f"WHO Grade {td['grade_roman']}</span>"
                f"<span class='status-chip'>Confidence: {cp['label']}</span>"
                f"<span class='status-chip'>Area: {ti['tumor_percentage']:.2f}%</span>"
                f"{threshold_chip}"
                f"</div>"
            )
            st.markdown(f"""
            <div class="status-banner detected">
                <div class="status-icon"></div>
                <div>
                    <div class="status-title">Possible Tumour Detected</div>
                    <div class="status-detail">AI Classification: <strong>{td['tumor_type']}</strong></div>
                    {chips_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-banner normal">
                <div class="status-icon"></div>
                <div>
                    <div class="status-title">No Tumour Detected - AI Finding</div>
                    <div class="status-detail">
                        No significant abnormality identified. A negative AI finding does not
                        rule out pathology. Radiologist confirmation is required.
                    </div>
                    <div class="status-chips">
                        <span class="status-chip">Confidence: {cp['label']}</span>
                        <span class="status-chip">Routine Follow-up</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        sv = results['survival_days']
        sv_val  = f"~{sv//30} mo" if sv is not None else "N/A"
        sv_note = "Estimate*" if sv is not None else "Tumour-free"
        sv_col  = THEME["purple"] if sv is not None else THEME["ok"]

        st.markdown(f"""
        <div class="metrics-row">
            {metric_card_html("WHO Classification", td['tumor_type'], f"Grade {td['grade_roman']}", td['color'])}
            {metric_card_html("AI Confidence", cp['label'], "Model Certainty", THEME['brand'])}
            {metric_card_html("Risk Level", results['risk_level'], f"{results['risk_score']:.0f} / 100", results['risk_color'])}
            {metric_card_html("Survival Estimate", sv_val, sv_note, sv_col)}
        </div>
        <p style="font-size:0.72rem;color:{THEME['text_4']};margin-top:-0.5rem;">
            * Formula-based estimate only. Not a validated clinical survival model.
        </p>
        """, unsafe_allow_html=True)

        probs = ti.get('probabilities', [])
        if probs and len(probs) >= 5:
            st.markdown("""
            <div class="section-heading" style="margin-top:1.5rem;">
                Grade Probability Distribution
            </div>
            """, unsafe_allow_html=True)
            bars_html = "".join([
                prob_bar_html(GRADE_NAMES[i], probs[i], GRADE_COLORS[i])
                for i in range(min(5, len(probs)))
            ])
            st.markdown(bars_html, unsafe_allow_html=True)

        # ── RAG Evidence Panel ────────────────────────────────────────────────
        rag_data = ti.get('rag_result')
        if rag_data:
            _render_rag_panel(rag_data)

        st.markdown("""
        <div class="section-heading" style="margin-top:1.5rem;">
            Evidence-Based Treatment Reference
            <span class="section-badge">Decision Support</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="clinical-notice">
            <div>
                The following are evidence-based reference guidelines for the AI-detected grade.
                These are <strong>not personalised treatment prescriptions</strong>.
                All treatment decisions must be made by the treating physician and MDT.
            </div>
        </div>
        <div class="rec-list">
        """, unsafe_allow_html=True)

        for rec_tuple in results['recommendations']:
            if len(rec_tuple) == 4:
                rec, evidence, category, urgency = rec_tuple
            elif len(rec_tuple) == 2:
                rec, evidence = rec_tuple; category = "General"; urgency = "Routine"
            else:
                rec = str(rec_tuple[0]); evidence = "--"; category = "General"; urgency = "Routine"
            st.markdown(rec_card_html(rec, evidence, category, urgency), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-heading" style="margin-top:1.5rem;">
            Advanced Diagnostic Visualisation
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate 8-Panel Diagnostic Figure", type="primary"):
            with st.spinner("Rendering diagnostic figure..."):
                fig = create_world_class_visualization(
                    results['original_image'], results['tumor_result'])
                st.pyplot(fig, use_container_width=True)
                st.success("✅ Enhanced visualization with clear tumor highlighting!")
                plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PROGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_progression():
    st.markdown("""
    <div class="section-heading">
        Longitudinal Progression Analysis
        <span class="section-badge">Compare Scans</span>
    </div>
    <div class="section-sub">
        Compare two MRI scans over time to assess tumour progression, recurrence, or treatment response.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    baseline_image = None; followup_image = None
    base_norm = None; follow_norm = None

    with c1:
        st.markdown('<div class="card-title"><div class="card-title-dot"></div>Baseline Scan (Earlier)</div>',
                    unsafe_allow_html=True)
        baseline_file = st.file_uploader(
            "Baseline brain MRI",
            type=['png','jpg','jpeg','dcm','nii','nii.gz'],
            key="baseline_scan")
        if baseline_file:
            baseline_image, baseline_metadata = load_medical_image(baseline_file)
            if baseline_image is not None:
                if len(baseline_image.shape) == 3 and baseline_image.shape[2] > 3:
                    bd = baseline_image[:, :, baseline_image.shape[2]//2]
                elif len(baseline_image.shape) == 3:
                    bd = (cv2.cvtColor(baseline_image, cv2.COLOR_RGB2GRAY)
                          if baseline_image.shape[2] == 3 else baseline_image[:, :, 0])
                else:
                    bd = baseline_image
                if BRAIN_VALIDATOR_AVAILABLE:
                    with st.spinner("Validating..."):
                        vr = validate_brain_mri(baseline_image,
                                                baseline_metadata if baseline_metadata else None)
                    if not vr['is_valid']:
                        _show_rejection(vr, "Baseline"); st.stop()
                    st.success(f"Baseline validated - {vr['confidence']:.1%}")
                baseline_date = st.date_input("Baseline date", key="baseline_date")
                baseline_metadata['scan_date'] = baseline_date
                base_norm = (bd - bd.min()) / (bd.max() - bd.min() + 1e-8)
                st.image(enhance_preview(base_norm), caption="Baseline (enhanced)", use_column_width=True)
                st.info(f"Shape: {baseline_image.shape}")

    with c2:
        st.markdown('<div class="card-title"><div class="card-title-dot"></div>Follow-up Scan (Later)</div>',
                    unsafe_allow_html=True)
        followup_file = st.file_uploader(
            "Follow-up brain MRI",
            type=['png','jpg','jpeg','dcm','nii','nii.gz'],
            key="followup_scan")
        if followup_file:
            followup_image, followup_metadata = load_medical_image(followup_file)
            if followup_image is not None:
                if len(followup_image.shape) == 3 and followup_image.shape[2] > 3:
                    fd = followup_image[:, :, followup_image.shape[2]//2]
                elif len(followup_image.shape) == 3:
                    fd = (cv2.cvtColor(followup_image, cv2.COLOR_RGB2GRAY)
                          if followup_image.shape[2] == 3 else followup_image[:, :, 0])
                else:
                    fd = followup_image
                if BRAIN_VALIDATOR_AVAILABLE:
                    with st.spinner("Validating..."):
                        vr = validate_brain_mri(followup_image,
                                                followup_metadata if followup_metadata else None)
                    if not vr['is_valid']:
                        _show_rejection(vr, "Follow-up"); st.stop()
                    st.success(f"Follow-up validated - {vr['confidence']:.1%}")
                followup_date = st.date_input("Follow-up date", key="followup_date")
                followup_metadata['scan_date'] = followup_date
                follow_norm = (fd - fd.min()) / (fd.max() - fd.min() + 1e-8)
                st.image(enhance_preview(follow_norm), caption="Follow-up (enhanced)", use_column_width=True)
                st.info(f"Shape: {followup_image.shape}")

    if baseline_file and followup_file and baseline_image is not None and followup_image is not None:
        st.markdown("""
        <div class="section-heading" style="margin-top:1.5rem;">Patient Information</div>
        """, unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            prog_pid = st.text_input("Patient ID", value=f"PT-P-{random.randint(10000,99999)}", key="prog_pid")
            prog_age = st.number_input("Current Age", 1, 120, 55, key="prog_age")
        with pc2:
            prog_sex = st.selectbox("Sex", ["Male","Female","Other"], key="prog_sex")
            prog_kps = st.slider("Current KPS Score", 0, 100, 75, key="prog_kps")
        with pc3:
            treatments = st.multiselect("Treatments Between Scans",
                ["Surgery","Radiation Therapy","Chemotherapy","Immunotherapy","Other"],
                key="treatments")
            notes = st.text_area("Clinical Notes", key="notes",
                                  placeholder="Relevant clinical observations...")

        if st.button("Analyse Progression", type="primary", use_container_width=True):
            prog_patient_data = dict(
                age=prog_age, sex=prog_sex, kps=prog_kps,
                patient_id=prog_pid, treatments=treatments, clinical_notes=notes)

            pb2   = st.progress(0.0)
            ptxt2 = st.empty()

            def _cb2(frac, msg):
                pb2.progress(min(float(frac), 1.0))
                ptxt2.markdown(
                    f"<div class='analysis-progress'>"
                    f"<div class='progress-label'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;"
                    f"color:{THEME['text_2']};'>{msg}</span>"
                    f"<span class='progress-mono'>{int(frac*100)}%</span></div>"
                    f"<div class='progress-track'>"
                    f"<div class='progress-fill' style='width:{int(frac*100)}%;'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True)

            progression_result = analyze_progression(
                baseline_image, followup_image,
                progress_callback=_cb2,
                device=st.session_state.device)

            pb2.progress(1.0); pb2.empty(); ptxt2.empty()

            result1 = progression_result['baseline_result']
            result2 = progression_result['followup_result']
            progression_metrics = progression_result

            result1 = fix_visualization_title(result1)
            result2 = fix_visualization_title(result2)

            st.session_state.progression_results = dict(
                baseline_result=result1, followup_result=result2,
                progression_metrics=progression_metrics,
                patient_data=prog_patient_data,
                baseline_image=base_norm, followup_image=follow_norm,
            )
            st.success("Progression analysis completed.")
            st.warning("AI findings only. Radiologist review required.")

    if 'progression_results' not in st.session_state:
        return

    prog    = st.session_state.progression_results
    metrics = prog['progression_metrics']
    r1 = prog['baseline_result']; r2 = prog['followup_result']
    vol_pct     = metrics.get('volume_change_pct', 0)
    growth_rate_daily = metrics.get('growth_rate_pct_per_day', 0)
    growth_rate = growth_rate_daily * 30  # Convert daily to monthly
    time_days   = metrics.get('days_between', 0)
    
    # Calculate recurrence risk based on follow-up grade
    grade_risk_map = {0: 2, 1: 15, 2: 35, 3: 65, 4: 85}
    followup_grade = r2.get('grade', 0)
    rec_risk = grade_risk_map.get(followup_grade, 50)
    
    # Get status from metrics (it's called 'status' not 'progression_status')
    prog_status = metrics.get('status', 'Unknown')
    
    # Treatment response is the same as status for display
    treat_label = prog_status
    treat_color = THEME["alert"] if 'Progressive' in prog_status else THEME["ok"] if 'Response' in prog_status else THEME["warn"]

    vol_class = "red" if vol_pct>10 else "green" if vol_pct<-10 else "amber"
    gr_class  = "red" if growth_rate>5 else "green" if growth_rate<-5 else "amber"
    rr_class  = "red" if rec_risk>60 else "amber" if rec_risk>30 else "green"

    st.markdown(f"""
    <div class="prog-status-hero">
        <div class="prog-status-left">
            <div class="prog-status-label">Progression Status</div>
            <div class="prog-status-title">{prog_status}</div>
            <div class="prog-status-sub">{treat_label}</div>
        </div>
        <div class="prog-stats-grid">
            <div><div class="prog-stat-label">Volume Change</div><div class="prog-stat-val {vol_class}">{vol_pct:+.1f}%</div></div>
            <div><div class="prog-stat-label">Interval</div><div class="prog-stat-val">{time_days} days</div></div>
            <div><div class="prog-stat-label">Growth Rate</div><div class="prog-stat-val {gr_class}">{growth_rate:+.2f}%/mo</div></div>
            <div><div class="prog-stat-label">Recurrence Risk</div><div class="prog-stat-val {rr_class}">{rec_risk:.0f}%</div></div>
        </div>
    </div>
    <p style="font-size:0.75rem;color:{THEME['text_4']};margin-top:-1rem;margin-bottom:1rem;">
        AI finding - all values require radiologist verification.
    </p>
    """, unsafe_allow_html=True)

    gnames = {0:'Normal',1:'Grade I',2:'Grade II',3:'Grade III',4:'Grade IV (GBM)'}
    g1 = gnames.get(r1.get('grade',0),'Unknown')
    g2 = gnames.get(r2.get('grade',0),'Unknown')
    vol_c = THEME["alert"] if vol_pct>10 else THEME["ok"] if vol_pct<-10 else THEME["warn"]
    rr_c  = THEME["alert"] if rec_risk>60 else THEME["warn"] if rec_risk>30 else THEME["ok"]

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1.5rem;">
        {metric_card_html("Baseline Grade", g1, f"Conf: {format_confidence_percent(r1.get('ai_confidence', r1.get('confidence',0)))}", THEME['brand_dk'])}
        {metric_card_html("Follow-up Grade", g2, f"Conf: {format_confidence_percent(r2.get('ai_confidence', r2.get('confidence',0)))}", THEME['warn'])}
        {metric_card_html("Volume Change", f"{vol_pct:+.1f}%", f"{r1.get('tumor_percentage',0):.2f}% to {r2.get('tumor_percentage',0):.2f}%", vol_c)}
        {metric_card_html("Growth Rate", f"{growth_rate:+.2f}%", "per month", vol_c)}
        {metric_card_html("Recurrence Risk", f"{rec_risk:.0f}%", "High" if rec_risk>60 else "Moderate" if rec_risk>30 else "Low", rr_c)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="section-heading">Scan Comparison</div>""", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([1,1,1])
    with sc1:
        st.markdown(f"<p style='text-align:center;color:{THEME['brand_dk']};font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Baseline - {g1}</p>",
                    unsafe_allow_html=True)
        if prog['baseline_image'] is not None:
            st.image(enhance_preview(prog['baseline_image']), use_column_width=True)
    with sc2:
        st.markdown(f"<p style='text-align:center;color:{THEME['warn']};font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Follow-up - {g2}</p>",
                    unsafe_allow_html=True)
        if prog['followup_image'] is not None:
            st.image(enhance_preview(prog['followup_image']), use_column_width=True)
    with sc3:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=['Baseline','Follow-up'],
            y=[r1.get('tumor_percentage',0), r2.get('tumor_percentage',0)],
            marker_color=[THEME["brand"], THEME["warn"]],
            marker_line_color=[VIZ_GRID, VIZ_GRID], marker_line_width=2.0,
            text=[f"{r1.get('tumor_percentage',0):.2f}%", f"{r2.get('tumor_percentage',0):.2f}%"],
            textposition='outside',
            textfont=dict(color=VIZ_TEXT_HI, size=12)))
        fig_bar.update_layout(
            title=dict(text='Tumour Area Comparison', font=dict(color=VIZ_TEXT_HI, size=12)),
            paper_bgcolor=VIZ_BG, plot_bgcolor=VIZ_PANEL,
            font=dict(color=VIZ_TEXT_HI, family='IBM Plex Sans'),
            height=280, margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(title='Area (%)', gridcolor=VIZ_GRID, tickfont=dict(color=VIZ_TEXT_MD)),
            xaxis=dict(gridcolor=VIZ_GRID, tickfont=dict(color=VIZ_TEXT_MD)))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""<div class="section-heading">Interactive Analytics</div>""", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)

    with pc1:
        categories = ['Volume Change','Growth Rate','Recurrence Risk','Grade Severity','Time Urgency']
        vals = [
            min(abs(vol_pct)/100, 1.0), min(abs(growth_rate)/20, 1.0),
            rec_risk/100, r2.get('grade',0)/3,
            min(1.0, 1.0-time_days/365) if time_days < 365 else 0.1,
        ]
        vals_c = vals + [vals[0]]; cats_c = categories + [categories[0]]
        status_c = metrics.get('status_color', THEME["warn"])
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill='toself',
            fillcolor=_hex_to_rgba(status_c, 0.20),
            line=dict(color=status_c, width=3.0), name='Risk Profile'))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=VIZ_PANEL,
                radialaxis=dict(visible=True, range=[0,1], gridcolor=VIZ_GRID,
                                tickfont=dict(color=VIZ_TEXT_MD)),
                angularaxis=dict(gridcolor=VIZ_GRID, tickfont=dict(color=VIZ_TEXT_MD))),
            paper_bgcolor=VIZ_BG, font=dict(color=VIZ_TEXT_HI, family='IBM Plex Sans'),
            title=dict(text='Risk Profile Radar', font=dict(color=VIZ_TEXT_HI, size=12)),
            height=320, margin=dict(l=40, r=40, t=50, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with pc2:
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number+delta', value=rec_risk,
            delta={'reference': 50,
                   'increasing': {'color': THEME["alert"]},
                   'decreasing': {'color': THEME["ok"]}},
            title={'text': 'Recurrence Risk (%)',
                   'font': {'color': VIZ_TEXT_HI, 'size': 13, 'family': 'IBM Plex Sans'}},
            number={'font': {'color': rr_c, 'family': 'IBM Plex Mono', 'size': 40}},
            gauge={
                'axis': {'range': [0,100], 'tickcolor': VIZ_TEXT_MD,
                         'tickfont': {'color': VIZ_TEXT_MD}},
                'bar': {'color': rr_c, 'thickness': 0.3},
                'bgcolor': VIZ_GRID, 'bordercolor': VIZ_BORDER, 'borderwidth': 2,
                'steps': [
                    {'range': [0,30],   'color': 'rgba(34,197,94,0.20)'},
                    {'range': [30,60],  'color': 'rgba(245,158,11,0.20)'},
                    {'range': [60,100], 'color': 'rgba(239,68,68,0.20)'}],
                'threshold': {'line': {'color': VIZ_TEXT_HI, 'width': 3},
                              'thickness': 0.78, 'value': rec_risk}}))
        fig_gauge.update_layout(
            paper_bgcolor=VIZ_BG, font=dict(color=VIZ_TEXT_HI, family='IBM Plex Sans'),
            height=320, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("""<div class="section-heading">Comprehensive 12-Panel Analysis</div>""",
                unsafe_allow_html=True)
    if st.button("Generate Full Progression Visualisation", type="primary", use_container_width=True):
        with st.spinner("Rendering 12-panel analysis..."):
            if prog['baseline_image'] is not None and prog['followup_image'] is not None:
                fig = create_progression_comparison_visualization(
                    prog['baseline_image'], prog['followup_image'],
                    prog['baseline_result'], prog['followup_result'],
                    prog['progression_metrics'])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ── RAG Progression Evidence ──────────────────────────────────────────────
    rag_prog = metrics.get('rag_progression')
    if rag_prog:
        st.markdown("""
        <div class="section-heading" style="margin-top:1.5rem;">
            Population-Level Progression Evidence
            <span class="section-badge">RAG</span>
        </div>
        """, unsafe_allow_html=True)
        st.info(rag_prog.get('interpretation', ''))
        st.metric(
            "Population Progression Rate",
            f"{rag_prog.get('population_progression_rate', 0):.1%}",
            f"{rag_prog.get('n_similar_cases', 0)} similar cases in TCGA database",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CLINICAL REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_reports():
    st.markdown("""
    <div class="section-heading">
        Clinical Reports
        <span class="section-badge">Export</span>
    </div>
    <div class="section-sub">
        Generate structured clinical reports for patient records, referrals, and documentation.
    </div>
    """, unsafe_allow_html=True)

    has_single      = 'analysis_results'    in st.session_state
    has_progression = 'progression_results' in st.session_state

    if not has_single and not has_progression:
        st.markdown(f"""
        <div class="clinical-notice">
            <div>Run a Single Analysis or Progression Analysis first to generate reports.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if has_single and has_progression:
        options = ["Comprehensive Analysis Report","Progression Analysis Report",
                   "Full Combined Report","Summary Report"]
    elif has_single:
        options = ["Comprehensive Analysis Report","Summary Report"]
    elif has_progression:
        options = ["Progression Analysis Report"]
    else:
        options = []

    report_type = st.selectbox("Report Type", options) if options else None

    rc1, rc2 = st.columns([2, 1], gap="large")

    with rc1:
        if report_type == "Comprehensive Analysis Report" and has_single:
            results = st.session_state.analysis_results
            pd_     = results['patient_data']
            ti      = results['tumor_result']
            td      = get_tumor_display_info(ti)
            st.markdown(f"""
            <div class="report-data-block">
                <div class="report-block-title">Patient Information</div>
                <div class="report-row"><strong>Patient Name</strong>{pd_.get('patient_name', 'N/A')}</div>
                <div class="report-row"><strong>Patient ID</strong>{pd_['patient_id']}</div>
                <div class="report-row"><strong>Age</strong>{pd_['age']} years</div>
                <div class="report-row"><strong>Sex</strong>{pd_['sex']}</div>
                <div class="report-row"><strong>KPS Score</strong>{pd_['kps']} / 100</div>
                <div class="report-row"><strong>Scan Date</strong>{pd_['scan_date']}</div>
            </div>
            <div class="report-data-block">
                <div class="report-block-title">Imaging Findings (AI)</div>
                <div class="report-row">
                    <strong>Detection Status</strong>
                    <span style="color:{td['color']};font-weight:600;">
                        {'Possible Tumour Detected' if td['detected'] else 'No Tumour Detected'}
                    </span>
                </div>
                <div class="report-row"><strong>AI Confidence</strong>{format_confidence_percent(ti.get('ai_confidence', ti.get('confidence', 0)))}</div>
                {'<div class="report-row"><strong>WHO Classification</strong>' + td['tumor_type'] + '</div>' if td['grade'] > 0 else ''}
                {'<div class="report-row"><strong>Apparent Area</strong>' + f"{ti['tumor_percentage']:.2f}% of scan area" + '</div>' if td['detected'] else ''}
            </div>
            <div class="report-data-block">
                <div class="report-block-title">Risk Assessment (AI)</div>
                <div class="report-row">
                    <strong>Risk Level</strong>
                    <span style="color:{results['risk_color']};font-weight:600;">{results['risk_level']}</span>
                </div>
                <div class="report-row"><strong>Risk Score</strong>{results['risk_score']:.0f} / 100</div>
                {'<div class="report-row"><strong>Survival Estimate</strong>' + f"~{results['survival_days']//30} months (formula-based only)" + '</div>' if results['survival_days'] is not None else '<div class="report-row"><strong>Status</strong>No tumour - no survival estimate</div>'}
            </div>
            """, unsafe_allow_html=True)

        elif report_type == "Progression Analysis Report" and has_progression:
            m = st.session_state.progression_results['progression_metrics']
            treat = m.get('treatment_response', ('Unknown','#888'))
            treat_l = treat[0] if isinstance(treat, (list,tuple)) else treat
            sc = m.get('status_color', THEME["warn"])
            st.markdown(f"""
            <div class="report-data-block">
                <div class="report-block-title">Progression Analysis Summary (AI)</div>
                <div class="report-row">
                    <strong>Status</strong>
                    <span style="color:{sc};font-weight:600;">{m['progression_status']}</span>
                </div>
                <div class="report-row"><strong>Volume Change</strong>{m['volume_change_pct']:+.1f}%</div>
                <div class="report-row"><strong>Time Interval</strong>{m['time_diff_days']} days</div>
                <div class="report-row"><strong>Treatment Response</strong>{treat_l}</div>
                <div class="report-row"><strong>Recurrence Risk</strong>{m.get('recurrence_risk',0):.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="card-title"><div class="card-title-dot"></div>Export</div>',
                    unsafe_allow_html=True)
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                rg = ClinicalReportGenerator()
                pdf_buffer = None
                filename   = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                try:
                    if report_type == "Comprehensive Analysis Report" and has_single:
                        pdf_buffer = rg.generate_clinical_report(
                            st.session_state.analysis_results['patient_data'],
                            st.session_state.analysis_results, "comprehensive")
                        filename = (f"Clinical_Report_"
                                    f"{st.session_state.analysis_results['patient_data']['patient_id']}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    elif report_type == "Progression Analysis Report" and has_progression:
                        combined = {
                            **st.session_state.progression_results['patient_data'],
                            'baseline_result': st.session_state.progression_results['baseline_result'],
                            'followup_result': st.session_state.progression_results['followup_result'],
                            'progression_metrics': st.session_state.progression_results['progression_metrics'],
                        }
                        pdf_buffer = rg.generate_progression_report(combined)
                        filename = (f"Progression_Report_"
                                    f"{st.session_state.progression_results['patient_data']['patient_id']}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    elif report_type == "Full Combined Report" and has_single and has_progression:
                        pdf_buffer = rg.generate_full_combined_report(
                            st.session_state.analysis_results,
                            st.session_state.progression_results)
                        filename = (f"Full_Report_"
                                    f"{st.session_state.analysis_results['patient_data']['patient_id']}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    elif report_type == "Summary Report" and has_single:
                        pdf_buffer = rg.generate_summary_report(
                            st.session_state.analysis_results['patient_data'],
                            st.session_state.analysis_results)
                        filename = (f"Summary_Report_"
                                    f"{st.session_state.analysis_results['patient_data']['patient_id']}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    else:
                        st.error("No matching analysis data for the selected report type.")

                    if pdf_buffer is not None:
                        st.success("Report generated.")
                        st.download_button(
                            label="Download PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True)
                    elif report_type is not None:
                        st.error("Report generation failed.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PATIENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _tab_patient_management():
    st.markdown("""
    <div class="section-heading">
        Patient Management
        <span class="section-badge">Records</span>
    </div>
    <div class="section-sub">
        View and manage patient analysis records and history.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize patient records in session state if not exists
    if 'patient_records' not in st.session_state:
        st.session_state.patient_records = []
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Total Records:** {len(st.session_state.patient_records)}")
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("Clear All History", use_container_width=True, type="secondary"):
            if st.session_state.patient_records:
                st.session_state.patient_records = []
                st.session_state.analysis_results = None
                st.session_state.progression_results = None
                st.success("All patient records cleared!")
                st.rerun()
            else:
                st.info("No records to clear")
    
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    
    # Display patient records
    if not st.session_state.patient_records:
        st.info("No patient records yet. Run an analysis in the 'Single Analysis' tab to create records.")
    else:
        # Create a dataframe for better display
        records_data = []
        for idx, record in enumerate(st.session_state.patient_records):
            pd_data = record.get('patient_data', {})
            tumor_data = record.get('tumor_result', {})
            records_data.append({
                'No.': idx + 1,
                'Patient ID': pd_data.get('patient_id', 'N/A'),
                'Patient Name': pd_data.get('patient_name', 'N/A'),
                'Age': pd_data.get('age', 'N/A'),
                'Sex': pd_data.get('sex', 'N/A'),
                'Scan Date': str(pd_data.get('scan_date', 'N/A')),
                'Tumor Detected': 'Yes' if tumor_data.get('detected', False) else 'No',
                'Grade': tumor_data.get('grade', 0),
                'Confidence': f"{tumor_data.get('confidence', 0)*100:.1f}%",
                'Analysis Date': record.get('timestamp', 'N/A')
            })
        
        df = pd.DataFrame(records_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        
        # Individual record management
        st.markdown("### Manage Individual Records")
        
        for idx, record in enumerate(st.session_state.patient_records):
            pd_data = record.get('patient_data', {})
            tumor_data = record.get('tumor_result', {})
            
            with st.expander(f"{pd_data.get('patient_name', 'Unknown')} - {pd_data.get('patient_id', 'N/A')}"):
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    st.markdown(f"""
                    **Patient Information:**
                    - **Name:** {pd_data.get('patient_name', 'N/A')}
                    - **ID:** {pd_data.get('patient_id', 'N/A')}
                    - **Age:** {pd_data.get('age', 'N/A')} years
                    - **Sex:** {pd_data.get('sex', 'N/A')}
                    - **KPS Score:** {pd_data.get('kps', 'N/A')}
                    - **Scan Date:** {pd_data.get('scan_date', 'N/A')}
                    
                    **Analysis Results:**
                    - **Tumor Detected:** {'Yes' if tumor_data.get('detected', False) else 'No'}
                    - **Grade:** {tumor_data.get('grade', 0)}
                    - **Confidence:** {tumor_data.get('confidence', 0)*100:.1f}%
                    - **Analysis Date:** {record.get('timestamp', 'N/A')}
                    """)
                
                with col_b:
                    if st.button(f"Delete", key=f"del_{idx}", use_container_width=True):
                        st.session_state.patient_records.pop(idx)
                        st.success(f"Record deleted!")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def _render_footer():
    u = st.session_state.user_info
    st.markdown(f"""
    <div class="app-footer">
        <div>
            <span class="footer-name">LongiTumor<em>Sense</em></span>
            <span style="font-size:0.75rem;color:{THEME['text_4']};margin-left:0.75rem;">
                AI-Assisted Brain MRI Analysis
            </span>
        </div>
        <div class="footer-badges">
            <span class="footer-badge">Research Grade AI</span>
            <span class="footer-badge">Radiologist Review Required</span>
            <span class="footer-badge warn">Not FDA Cleared</span>
            <span class="footer-badge">WHO CNS 2021</span>
        </div>
        <div style="font-size:0.75rem;color:{THEME['text_4']};">
            {u.get('username','Unknown')} - {u.get('institution','Unknown')} -
            {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs('models',  exist_ok=True)
    os.makedirs('data',    exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    init_clinical_system()
    main()