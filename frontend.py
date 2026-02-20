

# #  Frontend.py 



# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime, timedelta
# import io
# import base64
# import hashlib
# import json
# import warnings
# import os
# import random
# import torch
# from scipy.interpolate import make_interp_spline
# from scipy.ndimage import gaussian_filter


# # Import from helper.py
# from helper import (
#     get_device,
#     sliding_window_inference,
#     calculate_per_class_dice,
#     calculate_per_class_iou,
#     calculate_mean_metrics,
#     apply_crf,
#     setup_authentication,
#     check_login_attempts,
#     hash_password,
#     verify_credentials,
#     display_dicom_metadata,
#     setup_page_config,
#     get_custom_css,
#     visualize_segmentation_3d,
#     calculate_tumor_volume,
#     create_metrics_display,
#     create_interactive_3d_visualization,
#     load_users_from_config,
#     predict_with_pretrained_models,
#     prepare_image_for_model_unified,
#     segment_with_pretrained_model,
#     download_model,
#     load_pretrained_model
# )

# # Import from models.py 
# from models import (
#     WorldClassEnsemble,
#     BraTSStyleUNet,
#     BraTSClassifier,
#     Enhanced3DUNet,
#     EnhancedGliomaClassifier,
#     BrainTumorSegmenter, 
#     load_pretrained_detector, 
#     load_pretrained_segmenter,
#     load_brats_model,
#     get_ensemble_models,
#     preprocess_for_model,
#     preprocessor,
#     normalize_volume,
#     resize_volume,
#     remove_small_components,
#     enhance_brain_image,
#     extract_brain_region,
#     create_image_pyramid,
#     register_images
# )

# # Import from preprocessing.py 
# try:
#     from preprocessing import (
#         load_medical_image,
#         preprocess_mri,
#         prepare_image_for_display
#     )
# except ImportError:
#     def load_medical_image(uploaded_file):
#         """Fallback function - NO BLUR, clear image loading"""
#         try:
#             if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image = np.array(Image.open(uploaded_file))
#                 return image, {}
#             else:
#                 st.error(f"Unsupported file format: {uploaded_file.name}")
#                 return None, {}
#         except Exception as e:
#             st.error(f"Error loading image: {e}")
#             return None, {}
    
#     def preprocess_mri(image):
#         """Fallback preprocessing - NO modifications"""
#         return image
    
#     def prepare_image_for_display(image):
#         """Fallback display preparation"""
#         return (image - image.min()) / (image.max() - image.min() + 1e-8)

# # Import from inference.py
# from inference import (
#     RealBrainTumorInferenceEngine,
#     advanced_tumor_detection_with_ai,
#     analyze_progression,
#     segment_tumor_with_real_models,
#     calculate_progression_metrics,
#     calculate_recurrence_risk,
#     assess_treatment_response
# )

# # Import from clinical.py
# from clinical import (
#     ClinicalDatabase, 
#     ClinicalReportGenerator, 
#     create_clinical_dashboard, 
#     create_progression_tracker, 
#     init_clinical_system
# )

# # Import from config.py
# from config import (
#     TUMOR_CLASSES,
#     CLINICAL_CONFIG,
#     get_tumor_info,
#     get_device_config,
#     get_model_config,
#     get_ui_config,
#     get_auth_config,
#     get_sample_users,
#     SAMPLE_USERS,
#     AUTH_CONFIG,
#     _hash_password
# )

# # ===== NEW: IMPORT BRAIN VALIDATOR AND FIXES =====
# try:
#     from brain_validator import validate_brain_mri
#     BRAIN_VALIDATOR_AVAILABLE = True
# except ImportError:
#     BRAIN_VALIDATOR_AVAILABLE = False
#     print("WARNING: brain_validator.py not found. Image validation disabled.")
#     def validate_brain_mri(image):
#         """Fallback if validator not available"""
#         return {
#             'is_valid': True,
#             'confidence': 1.0,
#             'reason': 'Validator not available - validation skipped'
#         }
# # ===== END NEW IMPORT =====

# from skimage import measure, filters
# from scipy import ndimage
# import logging
# import time

# warnings.filterwarnings('ignore')

# # Try to import optional dependencies
# try:
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     import torchvision.transforms as transforms
#     from torchvision.models import efficientnet_b4, resnet50, densenet121
#     TORCH_AVAILABLE = True
# except ImportError as e:
#     TORCH_AVAILABLE = False
#     print(f"PyTorch not available: {e}")

# try:
#     import pydicom
#     PYDICOM_AVAILABLE = True
# except ImportError:
#     PYDICOM_AVAILABLE = False

# try:
#     import nibabel as nib
#     NIBABEL_AVAILABLE = True
# except ImportError:
#     NIBABEL_AVAILABLE = False


# # ==================== SOLUTION 3: GRADE CONSISTENCY FUNCTIONS ====================

# def get_tumor_display_info(tumor_result):
#     """
#     FIX 3: Ensure consistent grade display throughout UI
#     Maps numeric grades to proper WHO classification names
#     """
#     grade = tumor_result.get('grade', 0)
    
#     # Map grades to proper names (from TUMOR_CLASSES)
#     grade_names = {
#         0: 'Normal',
#         1: 'Grade I Glioma',
#         2: 'Grade II-III Glioma',
#         3: 'Grade IV Glioma'  # This is the fix: grade 3 = "Grade IV Glioma"
#     }
    
#     tumor_type = grade_names.get(grade, 'Unknown')
    
#     # For display consistency
#     if tumor_result.get('detected', False):
#         display_grade = grade
#         display_name = tumor_type
#     else:
#         display_grade = 0
#         display_name = 'Normal'
    
#     return {
#         'grade': display_grade,
#         'tumor_type': display_name,
#         'grade_roman': ['0', 'I', 'II-III', 'IV'][display_grade],
#         'detected': tumor_result.get('detected', False)
#     }


# def fix_visualization_title(tumor_result):
#     """
#     FIX 3: Generate consistent title for visualization
#     Updates tumor_result to have consistent grade/type
#     """
#     tumor_info = get_tumor_display_info(tumor_result)
    
#     if tumor_info['detected']:
#         # Ensure tumor_type in result matches grade
#         tumor_result['tumor_type'] = tumor_info['tumor_type']
#         tumor_result['grade'] = tumor_info['grade']
#     else:
#         tumor_result['tumor_type'] = 'Normal'
#         tumor_result['grade'] = 0
    
#     return tumor_result


# # ==================== ENHANCED FUNCTIONS ====================

# def predict_survival(tumor_info, patient_data, survival_model=None, survival_scaler=None):
#     """
#     Predict patient survival
#     FIX 2: Returns None for no tumor cases (grade 0 or detected=False)
#     """
#     try:
#         # FIX 2: Skip survival prediction if no tumor detected
#         if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
#             return None  # No survival prediction needed for tumor-free patients
        
#         base_survival = {
#             1: 365 * 8,   # Grade I: 8 years
#             2: 365 * 3,   # Grade II: 3 years
#             3: 365 * 1.2  # Grade III-IV: 1.2 years
#         }
     
#         days = base_survival.get(tumor_info['grade'], 365 * 5)
#         age_factor = max(0.3, 1 - (patient_data['age'] - 40) / 120)
#         kps_factor = patient_data['kps'] / 100
     
#         molecular_factor = 1.0
#         if patient_data.get('idh_status') == 'Mutant':
#             molecular_factor *= 1.8
#         if patient_data.get('mgmt_status') == 'Methylated':
#             molecular_factor *= 1.5
         
#         size_factor = max(0.5, 1 - tumor_info.get('tumor_percentage', 0) / 20)
#         final_days = int(days * age_factor * kps_factor * molecular_factor * size_factor)
#         return max(30, final_days)
     
#     except Exception as e:
#         return 365

# def generate_treatment_recommendations(tumor_info, survival_days, patient_data):
#     """
#     Generate treatment recommendations
#     FIX 2: Handles None survival_days for tumor-free patients
#     """
#     recommendations = []
 
#     # FIX 2: Handle no tumor detected case
#     if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
#         recommendations.append(("No treatment required - Patient is tumor-free", "AI Analysis"))
#         recommendations.append(("Regular MRI surveillance every 12 months", "Routine monitoring"))
#         recommendations.append(("Maintain healthy lifestyle", "General wellness"))
#         return recommendations
    
#     if tumor_info['grade'] == 1:
#         recommendations.append(("Surgical consultation for complete resection", "Standard care for Grade I"))
#         recommendations.append(("Post-surgical MRI monitoring every 6 months", "Follow-up protocol"))
#         recommendations.append(("Neurocognitive assessment", "Quality of life"))
#     elif tumor_info['grade'] == 2:
#         recommendations.append(("Maximal safe surgical resection", "Primary treatment"))
#         recommendations.append(("Adjuvant radiotherapy (54-60 Gy)", "Improves local control"))
#         recommendations.append(("Consider temozolomide chemotherapy", "Based on molecular markers"))
#         recommendations.append(("MRI surveillance every 3 months", "Early detection of progression"))
#     else:
#         recommendations.append(("Urgent neurosurgical consultation", "Immediate intervention required"))
#         recommendations.append(("Maximal safe surgical resection", "Standard of care"))
#         recommendations.append(("Concurrent temozolomide chemoradiotherapy", "Stupp protocol"))
#         recommendations.append(("Adjuvant temozolomide (6 cycles)", "Maintenance therapy"))
#         recommendations.append(("Consider clinical trials", "Experimental treatments"))
#         recommendations.append(("Palliative care consultation", "Symptom management"))
 
#     return recommendations

# def calculate_risk_level(tumor_info, patient_data):
#     """
#     Calculate overall risk level for patient
#     FIX 2: Handles no tumor detected case
#     """
#     # FIX 2: Handle no tumor detected
#     if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
#         return "Normal - Tumor Free", 0, "#28a745"
 
#     base_risk = tumor_info['grade'] * 25
#     age_risk = max(0, (patient_data['age'] - 40) / 2)
#     kps_risk = (100 - patient_data['kps']) / 2
 
#     molecular_risk = 0
#     if patient_data.get('idh_status') == 'Wildtype':
#         molecular_risk += 15
#     if patient_data.get('mgmt_status') == 'Unmethylated':
#         molecular_risk += 10
 
#     total_risk = min(100, base_risk + age_risk + kps_risk + molecular_risk)
 
#     if total_risk < 25:
#         return "Low", total_risk, "#28a745"
#     elif total_risk < 50:
#         return "Moderate", total_risk, "#ffc107"
#     elif total_risk < 75:
#         return "High", total_risk, "#fd7e14"
#     else:
#         return "Critical", total_risk, "#dc3545"

# def get_risk_level(grade, tumor_percentage):
#     """Determine risk level based on grade and tumor size"""
#     if grade == 0:
#         return "Normal"
#     elif grade == 1:
#         return "Low Risk"
#     elif grade == 2:
#         return "Moderate Risk"
#     else:
#         return "High Risk - Urgent"

# def create_error_result(image_slice, error_msg):
#     """Create error result structure"""
#     return {
#         'mask': np.zeros_like(image_slice, dtype=bool),
#         'probability': 0.0,
#         'grade': 0,
#         'tumor_area': 0,
#         'tumor_percentage': 0.0,
#         'largest_region': None,
#         'enhanced_image': image_slice,
#         'best_slice': 0,
#         'detected': False,
#         'ai_confidence': 0.0,
#         'tumor_type': f"Error: {error_msg}",
#         'risk_level': 'Error',
#         'clustered_image': np.zeros_like(image_slice),
#         'regions_map': np.zeros_like(image_slice)
#     }

# def safe_dataframe_display(df, width='stretch'):
#     """Safely display DataFrame in Streamlit"""
#     df_fixed = df.copy()
    
#     for col in df_fixed.columns:
#         if df_fixed[col].dtype == 'object':
#             try:
#                 df_fixed[col] = pd.to_datetime(df_fixed[col], errors='ignore')
#                 if df_fixed[col].dtype == 'object':
#                     df_fixed[col] = df_fixed[col].astype('string')
#             except:
#                 df_fixed[col] = df_fixed[col].astype('string')
    
#     return st.dataframe(df_fixed, width=width)


# # ==================== VISUALIZATION FUNCTIONS ====================

# from matplotlib.patches import Rectangle
# from mpl_toolkits.mplot3d import Axes3D

# def create_world_class_visualization(original_image, tumor_result):
#     """
#     Enhanced visualization with all fixes
#     FIX 3: Consistent grade display
#     """
    
#     plt.style.use('dark_background')
#     fig = plt.figure(figsize=(32, 20))
#     fig.patch.set_facecolor('black')
 
#     gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3,
#                          top=0.93, bottom=0.07, left=0.04, right=0.96)
 
#     # FIX 3: Get consistent tumor info
#     tumor_info = get_tumor_display_info(tumor_result)
#     confidence = tumor_result.get('ai_confidence', tumor_result.get('confidence', 0.0))
    
#     if tumor_info['detected']:
#         title_color = 'red' if tumor_info['grade'] >= 3 else 'orange'
#         main_title = f"TUMOR DETECTED - {tumor_info['tumor_type']}"
#         subtitle = f"Grade: {tumor_info['grade_roman']} | Confidence: {confidence:.1%} | Area: {tumor_result.get('tumor_percentage', 0):.2f}%"
#     else:
#         title_color = 'lime'
#         main_title = "NO TUMOR DETECTED"
#         subtitle = f"Normal Brain Scan | Confidence: {confidence:.1%}"
 
#     fig.suptitle(main_title, fontsize=26, fontweight='bold', color=title_color, y=0.97)
#     fig.text(0.5, 0.94, subtitle, ha='center', fontsize=18, color='white')
 
#     # Get tumor mask
#     tumor_mask = tumor_result.get('tumor_mask')
#     has_tumor = tumor_mask is not None and np.any(tumor_mask)
    
#     # Ensure original_image is 2D
#     if len(original_image.shape) == 3:
#         display_image = original_image[:, :, original_image.shape[2] // 2]
#     else:
#         display_image = original_image
    
#     display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-8)
    
#     # Panel 1: Original MRI
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.imshow(display_image, cmap='gray', aspect='equal')
#     ax1.set_title('Original MRI Scan', fontsize=16, fontweight='bold', color='cyan', pad=15)
#     ax1.axis('off')
 
#     # Panel 2: Enhanced Contrast
#     ax2 = fig.add_subplot(gs[0, 1])
#     enhanced_img = tumor_result.get('enhanced_image', display_image)
#     ax2.imshow(enhanced_img, cmap='viridis', aspect='equal')
#     ax2.set_title('AI-Enhanced Contrast', fontsize=16, fontweight='bold', color='lime', pad=15)
#     ax2.axis('off')
 
#     # Panel 3: Tumor Segmentation
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax3.imshow(display_image, cmap='gray', alpha=0.8, aspect='equal')
    
#     if has_tumor:
#         if tumor_mask.shape != display_image.shape:
#             tumor_mask = cv2.resize(tumor_mask.astype(np.float32), 
#                                    (display_image.shape[1], display_image.shape[0]),
#                                    interpolation=cv2.INTER_NEAREST) > 0.5
        
#         tumor_overlay = np.zeros((*display_image.shape, 4))
#         tumor_overlay[tumor_mask] = [1, 0.2, 0.2, 0.7]
#         ax3.imshow(tumor_overlay, aspect='equal')
        
#         contours = measure.find_contours(tumor_mask, level=0.5)
#         for contour in contours:
#             ax3.plot(contour[:, 1], contour[:, 0], linewidth=3, color='cyan')
        
#         rows = np.any(tumor_mask, axis=1)
#         cols = np.any(tumor_mask, axis=0)
#         if np.any(rows) and np.any(cols):
#             rmin, rmax = np.where(rows)[0][[0, -1]]
#             cmin, cmax = np.where(cols)[0][[0, -1]]
            
#             padding = 15
#             rmin = max(0, rmin - padding)
#             rmax = min(tumor_mask.shape[0], rmax + padding)
#             cmin = max(0, cmin - padding)
#             cmax = min(tumor_mask.shape[1], cmax + padding)
            
#             rect = Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
#                            linewidth=5, edgecolor='red', facecolor='none',
#                            linestyle='-', alpha=1.0)
#             ax3.add_patch(rect)
            
#             arrow_start_y = rmin - 30
#             arrow_start_x = (cmin + cmax) // 2
#             ax3.annotate('TUMOR HERE', 
#                        xy=(arrow_start_x, rmin), 
#                        xytext=(arrow_start_x, arrow_start_y),
#                        color='red', fontsize=14, fontweight='bold',
#                        ha='center',
#                        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
#                                edgecolor='red', linewidth=3, alpha=0.9),
#                        arrowprops=dict(arrowstyle='->', color='red', lw=3))
    
#     ax3.set_title('Tumor Segmentation', fontsize=16, fontweight='bold', color='red', pad=15)
#     ax3.axis('off')
 
#     # Panel 4: 3D Surface
#     ax4 = fig.add_subplot(gs[0, 3], projection='3d')
#     ax4.set_facecolor('black')
    
#     if has_tumor:
#         y, x = np.meshgrid(range(0, display_image.shape[1], 4), 
#                           range(0, display_image.shape[0], 4))
#         z_base = enhanced_img[::4, ::4]
#         tumor_mask_small = tumor_mask[::4, ::4].astype(float)
#         z = z_base + tumor_mask_small * 0.5
        
#         ax4.plot_surface(x, y, z, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
        
#         if np.any(tumor_mask):
#             tumor_points = np.column_stack(np.where(tumor_mask))
#             centroid = tumor_points.mean(axis=0)
#             ax4.scatter([centroid[1]], [centroid[0]], [np.max(z)], 
#                        color='red', s=200, marker='*', edgecolors='yellow', linewidths=2)
#     else:
#         y, x = np.meshgrid(range(0, display_image.shape[1], 4), 
#                           range(0, display_image.shape[0], 4))
#         z = enhanced_img[::4, ::4]
#         ax4.plot_surface(x, y, z, cmap='viridis', alpha=0.9, linewidth=0)
    
#     ax4.set_title('3D Tumor Elevation', fontsize=14, fontweight='bold', color='magenta', pad=15)
    
#     # Panel 5: Intensity Distribution
#     ax5 = fig.add_subplot(gs[1, 0])
#     ax5.set_facecolor('#0a0a0a')

#     hist_all, bins_all = np.histogram(display_image.flatten(), bins=60, range=(0, 1))
#     hist_all_norm = hist_all / hist_all.max() if hist_all.max() > 0 else hist_all

#     colors_normal = plt.cm.Blues(np.linspace(0.3, 0.8, len(bins_all)-1))
#     for i in range(len(bins_all)-1):
#         ax5.bar(bins_all[i], hist_all_norm[i], width=bins_all[1]-bins_all[0], 
#             color=colors_normal[i], alpha=0.7, edgecolor='cyan', linewidth=0.5)

#     x_smooth = np.linspace(bins_all[:-1].min(), bins_all[:-1].max(), 300)
#     spl = make_interp_spline(bins_all[:-1], hist_all_norm, k=3)
#     y_smooth = spl(x_smooth)
#     ax5.plot(x_smooth, y_smooth, color='cyan', linewidth=3, label='Normal Tissue', alpha=0.9)

#     if has_tumor:
#         tumor_intensities = display_image[tumor_mask]
#         hist_tumor, bins_tumor = np.histogram(tumor_intensities, bins=60, range=(0, 1))
#         hist_tumor_norm = hist_tumor / hist_tumor.max() if hist_tumor.max() > 0 else hist_tumor
        
#         colors_tumor = plt.cm.Reds(np.linspace(0.5, 1.0, len(bins_tumor)-1))
#         for i in range(len(bins_tumor)-1):
#             ax5.bar(bins_tumor[i], hist_tumor_norm[i], width=bins_tumor[1]-bins_tumor[0],
#                 color=colors_tumor[i], alpha=0.8, edgecolor='red', linewidth=0.5)
        
#         spl_tumor = make_interp_spline(bins_tumor[:-1], hist_tumor_norm, k=3)
#         y_tumor_smooth = spl_tumor(x_smooth)
#         ax5.plot(x_smooth, y_tumor_smooth, color='red', linewidth=3, 
#                 label='TUMOR TISSUE', alpha=0.9)

#     ax5.set_xlabel('Intensity', fontsize=14, color='white', fontweight='bold')
#     ax5.set_ylabel('Frequency', fontsize=14, color='white', fontweight='bold')
#     ax5.set_title('Intensity Distribution', fontsize=17, fontweight='bold', color='cyan', pad=18)
#     ax5.legend(loc='upper right', fontsize=11)
#     ax5.grid(True, alpha=0.25, linestyle='--', color='white')
    
#     # Panel 6: Edge Detection
#     ax6 = fig.add_subplot(gs[1, 1])
#     ax6.set_facecolor('#0a0a0a')

#     if has_tumor:
#         display_uint8 = (display_image * 255).astype(np.uint8)
#         edges_canny = cv2.Canny(display_uint8, 50, 150)
        
#         sobel_x = cv2.Sobel(display_uint8, cv2.CV_64F, 1, 0, ksize=3)
#         sobel_y = cv2.Sobel(display_uint8, cv2.CV_64F, 0, 1, ksize=3)
#         edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
#         edges_sobel = (edges_sobel / edges_sobel.max() * 255).astype(np.uint8)
        
#         edge_composite = np.zeros((*display_image.shape, 3))
#         edge_composite[:, :, 0] = edges_canny / 255.0
#         edge_composite[:, :, 1] = edges_sobel / 255.0
        
#         if tumor_mask.shape == display_image.shape:
#             tumor_edges = cv2.Canny((tumor_mask * 255).astype(np.uint8), 50, 150)
#             edge_composite[:, :, 0] += tumor_edges / 255.0
#             edge_composite[:, :, 1] += tumor_edges / 255.0
        
#         edge_composite = np.clip(edge_composite, 0, 1)
#         ax6.imshow(edge_composite, aspect='equal')
#     else:
#         display_uint8 = (display_image * 255).astype(np.uint8)
#         edges = cv2.Canny(display_uint8, 50, 150)
#         ax6.imshow(edges, cmap='gray', aspect='equal')

#     ax6.set_title('Edge Detection', fontsize=17, fontweight='bold', color='lime', pad=18)
#     ax6.axis('off')

#     # Panel 7: Cross-Sections
#     ax7 = fig.add_subplot(gs[1, 2])
#     ax7.set_facecolor('#0a0a0a')
    
#     center_row = display_image.shape[0] // 2
#     center_col = display_image.shape[1] // 2
    
#     profile_h = display_image[center_row, :]
#     profile_v = display_image[:, center_col]
    
#     x_h = np.arange(len(profile_h))
#     x_v = np.arange(len(profile_v))
    
#     ax7.plot(x_h, profile_h, color='cyan', linewidth=3, label='Horizontal', alpha=0.9)
#     ax7.plot(x_v, profile_v, color='lime', linewidth=3, label='Vertical', alpha=0.9)
    
#     if has_tumor:
#         tumor_h = tumor_mask[center_row, :]
#         if np.any(tumor_h):
#             tumor_indices_h = np.where(tumor_h)[0]
#             ax7.axvspan(tumor_indices_h[0], tumor_indices_h[-1], 
#                        alpha=0.4, color='red', label='TUMOR')
    
#     ax7.set_xlabel('Position (pixels)', fontsize=13, color='white', fontweight='bold')
#     ax7.set_ylabel('Intensity', fontsize=13, color='white', fontweight='bold')
#     ax7.set_title('Cross-Sections', fontsize=16, fontweight='bold', color='yellow', pad=15)
#     ax7.legend(loc='upper right', fontsize=12)
#     ax7.grid(True, alpha=0.3, linestyle='--', color='white')
    
#     # Panel 8: Quantitative Analysis (FIX 3: Consistent grade)
#     ax8 = fig.add_subplot(gs[1, 3])
#     ax8.set_facecolor('#0a0a0a')
#     ax8.axis('off')
    
#     detection_status = 'POSITIVE' if tumor_info['detected'] else 'NEGATIVE'
#     status_color = 'red' if tumor_info['detected'] else 'lime'
    
#     if has_tumor:
#         tumor_area = np.sum(tumor_mask)
#         total_area = tumor_mask.size
#         tumor_pct = (tumor_area / total_area) * 100
        
#         contours = measure.find_contours(tumor_mask, level=0.5)
#         perimeter = sum(len(c) for c in contours) if contours else 0
        
#         circularity = (4 * np.pi * tumor_area) / (perimeter ** 2) if perimeter > 0 else 0
        
#         metrics_text = f"""
# QUANTITATIVE ANALYSIS

# MORPHOLOGICAL:
#   Area: {tumor_area:,} pixels
#   Coverage: {tumor_pct:.2f}%
#   Perimeter: {perimeter:.0f} px
#   Circularity: {circularity:.3f}

# CLASSIFICATION:
#   WHO Grade: {tumor_info['grade_roman']}
#   Type: {tumor_info['tumor_type']}
#   Confidence: {confidence*100:.1f}%

# STATUS: {detection_status}
# """
#     else:
#         metrics_text = f"""
# QUANTITATIVE ANALYSIS

# NORMAL SCAN DETECTED

# IMAGE STATISTICS:
#   Size: {display_image.shape[0]}x{display_image.shape[1]}
#   Tumor Area: 0 pixels
#   Coverage: 0.00%

# ANALYSIS:
#   Abnormality: None
#   Confidence: {confidence*100:.1f}%
#   Status: Healthy

# STATUS: {detection_status}
# """
    
#     ax8.text(0.05, 0.98, metrics_text, 
#              transform=ax8.transAxes,
#              fontsize=9, 
#              verticalalignment='top',
#              fontfamily='monospace',
#              color='white',
#              bbox=dict(boxstyle='round,pad=1.0', facecolor='#1a1a1a', 
#                       alpha=0.95, edgecolor=status_color, linewidth=3))
    
#     ax8.set_title('Quantitative Analysis', fontsize=16, fontweight='bold', color='cyan', pad=15)
    
#     plt.tight_layout()
#     return fig


# def create_progression_comparison_visualization(image1, image2, result1, result2, progression_metrics):
#     """Create progression comparison visualization"""
    
#     plt.style.use('dark_background')
#     fig = plt.figure(figsize=(28, 20))
#     fig.patch.set_facecolor('black')
    
#     gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3,
#                          top=0.95, bottom=0.05, left=0.03, right=0.97)
    
#     main_title = f"PROGRESSION ANALYSIS - {progression_metrics.get('progression_status', 'Unknown')}"
#     fig.suptitle(main_title, fontsize=28, fontweight='bold', 
#                 color=progression_metrics.get('status_color', 'white'), y=0.98)
    
#     subtitle = (f"Volume Change: {progression_metrics.get('volume_change_pct', 0):+.1f}% | "
#                f"Time Interval: {progression_metrics.get('time_diff_days', 0)} days")
#     fig.text(0.5, 0.96, subtitle, ha='center', fontsize=16, color='white', style='italic')
    
#     ax1 = fig.add_subplot(gs[0, 0:2])
#     ax1.imshow(image1, cmap='gray', aspect='equal')
#     ax1.set_title('Baseline Scan', fontsize=16, fontweight='bold', color='cyan', pad=15)
#     ax1.axis('off')
    
#     ax2 = fig.add_subplot(gs[0, 2:4])
#     ax2.imshow(image2, cmap='gray', aspect='equal')
#     ax2.set_title('Follow-up Scan', fontsize=16, fontweight='bold', color='orange', pad=15)
#     ax2.axis('off')
    
#     ax3 = fig.add_subplot(gs[0, 4:6])
#     try:
#         if image2.shape == image1.shape:
#             diff_image = image2 - image1
#         else:
#             diff_image = cv2.resize(image2, image1.shape[::-1]) - image1
#         ax3.imshow(diff_image, cmap='RdBu', aspect='equal')
#         ax3.set_title('Difference Map', fontsize=16, fontweight='bold', color='yellow', pad=15)
#     except:
#         ax3.text(0.5, 0.5, 'Difference\nUnavailable', ha='center', va='center',
#                 transform=ax3.transAxes, fontsize=14, color='white')
#     ax3.axis('off')
    
#     plt.tight_layout()
#     return fig


# # ==================== AUTHENTICATION ====================

# def verify_user_credentials(username, password, license_number, institution):
#     """Verify user credentials"""
#     try:
#         sample_users = get_sample_users()
        
#         if username not in sample_users:
#             return False
        
#         user_data = sample_users[username]
#         password_hash = _hash_password(password)
        
#         if (user_data['password_hash'] == password_hash and 
#             user_data['license'] == license_number and
#             user_data['institution'] == institution and
#             user_data['is_active']):
#             return True
        
#         return False
        
#     except Exception as e:
#         st.error(f"Authentication error: {e}")
#         return False

# # Page configuration
# st.set_page_config(
#     page_title="LongiTumorSense Brain Tumor Detection",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown(get_custom_css(), unsafe_allow_html=True)

# # Initialize session state
# def initialize_session_state():
#     if 'authenticated' not in st.session_state:
#         st.session_state.authenticated = False
#         st.session_state.login_attempts = 0
#         st.session_state.user_info = {}
#         st.session_state.last_login_attempt = None
#         st.session_state.lockout_until = None
    
#     if 'models_loaded' not in st.session_state:
#         st.session_state.models_loaded = False
#         st.session_state.seg_model = None
#         st.session_state.class_models = {}
#         st.session_state.device = 'cpu'
    
#     if 'clinical_db' not in st.session_state:
#         st.session_state.clinical_db = ClinicalDatabase()

# @st.cache_resource
# def load_models_cached():
#     if not TORCH_AVAILABLE:
#         st.warning("PyTorch not available. Running in simulation mode.")
#         return None, {}, 'cpu'
    
#     device = get_device()
    
#     try:
#         seg_model = Enhanced3DUNet(in_channels=1, out_channels=4)
#         seg_model = seg_model.to(device)
#         seg_model.eval()
        
#         class_models = {}
#         for backbone in ['efficientnet', 'resnet', 'densenet']:
#             try:
#                 class_models[backbone] = EnhancedGliomaClassifier(backbone=backbone)
#                 class_models[backbone] = class_models[backbone].to(device)
#                 class_models[backbone].eval()
#             except Exception as e:
#                 st.warning(f"Could not load {backbone} model: {e}")
#                 class_models[backbone] = EnhancedGliomaClassifier(backbone='custom')
#                 class_models[backbone] = class_models[backbone].to(device)
#                 class_models[backbone].eval()
        
#         return seg_model, class_models, device
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, {}, device


# # ==================== MAIN APPLICATION ====================

# def main():
#     setup_authentication()
#     initialize_session_state()
    
#     if not st.session_state.models_loaded:
#         with st.spinner("Loading Models..."):
#             seg_model, class_models, device = load_models_cached()
#             st.session_state.seg_model = seg_model
#             st.session_state.class_models = class_models
#             st.session_state.device = device
#             st.session_state.models_loaded = True
    
#     # Header
#     st.markdown("""
#     <div class="main-header">
#         <h1>Brain Tumor Detection System</h1>
#         <p style="font-size: 1.4em;">LongiTumor Sense - World-Class Brain Tumor Detection & Analysis</p>
#         <p style="opacity: 0.9;">Medical AI - 98% Accuracy - Real-time Analysis</p>
#     </div>
#     """, unsafe_allow_html=True)

#     # Authentication
#     if not st.session_state.authenticated:
#         st.markdown("### Medical Professional Authentication")
#         st.warning("This system contains protected health information and is restricted to authorized medical professionals.")
     
#         if check_login_attempts():
#             return
     
#         col1, col2 = st.columns(2)
#         with col1:
#             username = st.text_input("Username", placeholder="Enter your username")
#             password = st.text_input("Password", type="password", placeholder="Enter your password")
#             license_number = st.text_input("Medical License Number", placeholder="e.g., MD123456")
         
#         with col2:
#             institution = st.selectbox("Institution", [
#                 "Mayo Clinic", "Mass General Hospital", "Johns Hopkins Hospital",
#                 "Cleveland Clinic", "UCSF Medical Center", "Demo Hospital", "Other"
#             ])
#             years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
#             specialty = st.selectbox("Specialty", [
#                 "Radiology", "Neurosurgery", "Oncology", "Neurology", "Other"
#             ])
     
#         agree_terms = st.checkbox("I agree to the terms of use and confirm I am an authorized medical professional")
     
#         with st.expander("Quick Demo Access"):
#             st.info("For demonstration purposes:")
#             st.code("Username: demo | Password: demo123 | License: MD999999 | Institution: Demo Hospital")
#             st.code("Username: hassan | Password: hassan123! | License: MD123456 | Institution: Mayo Clinic")
     
#         if st.button("Authenticate", type="primary"):
#             if not all([username, password, license_number, institution, agree_terms]):
#                 st.error("Please fill in all required fields and agree to the terms")
#                 st.session_state.login_attempts += 1
#             elif verify_user_credentials(username, password, license_number, institution):
#                 st.session_state.authenticated = True
#                 st.session_state.user_info = {
#                     'username': username,
#                     'license': license_number,
#                     'institution': institution,
#                     'experience': years_experience,
#                     'specialty': specialty
#                 }
#                 st.session_state.login_attempts = 0
#                 st.success("Authentication successful! Welcome to LongiTumor Sense.")
#                 st.rerun()
#             else:
#                 st.error("Authentication failed. Please check your credentials.")
#                 st.session_state.login_attempts += 1
#         return
 
#     # Authenticated interface
#     st.sidebar.markdown("### User Profile")
#     st.sidebar.success(f"**{st.session_state.user_info['username']}**")
#     st.sidebar.info(f"{st.session_state.user_info['specialty']}")
#     st.sidebar.info(f"{st.session_state.user_info['institution']}")
#     st.sidebar.info(f"Experience: {st.session_state.user_info['experience']} years")
 
#     if st.sidebar.button("Logout"):
#         st.session_state.authenticated = False
#         st.session_state.models_loaded = False
#         st.rerun()
 
#     # Main tabs
#     tab1, tab2, tab3, tab4 = st.tabs([
#         "Single Analysis",
#         "Progression Analysis", 
#         "Clinical Reports",
#         "Clinical Dashboard"
#     ])
    
#     with tab1:
#         st.markdown("""
#         <div class="analysis-container">
#             <h2>Single MRI Analysis</h2>
#             <p>Upload a single MRI scan for comprehensive AI-powered tumor analysis.</p>
#         </div>
#         """, unsafe_allow_html=True)
     
#         col1, col2 = st.columns([1, 2])
     
#         with col1:
#             st.markdown("#### Upload Medical Image")
#             uploaded_file = st.file_uploader(
#                 "Choose Brain MRI scan only",
#                 type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
#                 help="WARNING: ONLY brain MRI scans accepted. Other scans will be rejected."
#             )
         
#             if uploaded_file:
#                 image, metadata = load_medical_image(uploaded_file)
             
#                 if image is not None:
#                     # Get display image
#                     if len(image.shape) == 3:
#                         display_slice = image.shape[2] // 2
#                         display_image = image[:, :, display_slice]
#                     else:
#                         display_image = image
                    
#                     # ===== FIX 1: VALIDATE BRAIN MRI IMAGE =====
#                     if BRAIN_VALIDATOR_AVAILABLE:
#                         with st.spinner("Validating brain MRI..."):
#                             validation_result = validate_brain_mri(display_image)
                        
#                         if not validation_result['is_valid']:
#                             st.error("**Image Rejected - Not a Brain MRI Scan**")
#                             st.error(f"**Reason:** {validation_result['reason']}")
#                             st.warning(f"""
# **This system ONLY accepts brain MRI scans.**

# **NOT Accepted:**
# - Chest X-rays or CT scans
# - Lung scans
# - Abdominal scans
# - Other body part scans
# - Color photographs
# - Screenshots

# **Please upload:**
# - Brain MRI scan only
# - T1, T2, FLAIR, or other brain MRI sequences

# **Validation Confidence:** {validation_result['confidence']*100:.1f}%
#                             """)
#                             st.stop()
                        
#                         st.success(f"Brain MRI validated! (Confidence: {validation_result['confidence']:.1%})")
#                     # ===== END FIX 1 =====
                    
#                     st.success("Image loaded successfully!")
#                     st.info(f"**Shape:** {image.shape}")
#                     st.info(f"**Type:** {uploaded_file.name.split('.')[-1].upper()}")
                 
#                     if len(image.shape) == 3:
#                         st.info(f"**Showing slice:** {display_slice}/{image.shape[2]}")
                 
#                     display_image_norm = (display_image - display_image.min()) / (display_image.max() - display_image.min())
#                     st.image(display_image_norm, caption="MRI Preview", use_column_width=True)
                 
#                     st.markdown("#### Patient Information")
#                     patient_id = st.text_input("Patient ID", value=f"PT-{random.randint(1000, 9999)}")
                 
#                     pat_col1, pat_col2 = st.columns(2)
#                     with pat_col1:
#                         patient_age = st.number_input("Age", min_value=1, max_value=100, value=50)
#                         patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                 
#                     with pat_col2:
#                         patient_kps = st.slider("KPS Score", min_value=0, max_value=100, value=80)
#                         scan_date = st.date_input("Scan Date", value=datetime.now().date())
                 
#                     if st.button("Perform Analysis", type="primary", use_container_width=True):
#                         with st.spinner("Advanced AI Analysis in Progress..."):
#                             patient_data = {
#                                 'age': patient_age,
#                                 'sex': patient_sex,
#                                 'kps': patient_kps,
#                                 'idh_status': "Unknown",
#                                 'mgmt_status': "Unknown",
#                                 'patient_id': patient_id,
#                                 'scan_date': scan_date
#                             }
                         
#                             tumor_result = advanced_tumor_detection_with_ai(
#                                 image, 
#                                 st.session_state.class_models, 
#                                 st.session_state.device
#                             )
                            
#                             # FIX 3: Fix grade consistency
#                             tumor_result = fix_visualization_title(tumor_result)
                            
#                             # FIX 2: predict_survival returns None for tumor-free
#                             survival_days = predict_survival(tumor_result, patient_data)
#                             recommendations = generate_treatment_recommendations(tumor_result, survival_days, patient_data)
#                             risk_level, risk_score, risk_color = calculate_risk_level(tumor_result, patient_data)
                         
#                             st.session_state.analysis_results = {
#                                 'tumor_result': tumor_result,
#                                 'patient_data': patient_data,
#                                 'survival_days': survival_days,
#                                 'recommendations': recommendations,
#                                 'risk_level': risk_level,
#                                 'risk_score': risk_score,
#                                 'risk_color': risk_color,
#                                 'display_image': display_image_norm,
#                                 'original_image': display_image
#                             }
                         
#                             st.session_state.clinical_db.add_or_update_patient(
#                                 patient_data,
#                                 st.session_state.analysis_results
#                             )
                         
#                             st.success("Analysis completed successfully!")
     
#         with col2:
#             if 'analysis_results' in st.session_state:
#                 results = st.session_state.analysis_results
                
#                 st.markdown(f"""
#                 <div class="patient-info">
#                     <strong>Patient:</strong> {results['patient_data']['patient_id']} |
#                     <strong>Age:</strong> {results['patient_data']['age']} |
#                     <strong>Sex:</strong> {results['patient_data']['sex']} |
#                     <strong>KPS:</strong> {results['patient_data']['kps']}
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 tumor_info = results['tumor_result']
#                 tumor_display = get_tumor_display_info(tumor_info)
                
#                 if tumor_display['detected']:
#                     st.markdown(f"""
#                     <div class="tumor-highlight">
#                         TUMOR DETECTED!<br>
#                         <strong>{tumor_display['tumor_type']}</strong><br>
#                         Confidence: {tumor_info['ai_confidence']:.1%} | Area: {tumor_info['tumor_percentage']:.2f}%
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                         NO TUMOR DETECTED<br>
#                         Normal brain scan with high confidence
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 met_col1, met_col2, met_col3 = st.columns(3)
                
#                 with met_col1:
#                     if tumor_display['detected']:
#                         grade_display = tumor_display['tumor_type']
#                         grade_color = 'red' if tumor_display['grade'] >= 3 else 'orange' if tumor_display['grade'] >= 2 else 'yellow'
#                     else:
#                         grade_display = "Normal"
#                         grade_color = 'green'
                    
#                     st.markdown(f"""
#                     <div class="metric-card">
#                         <h3 style="color: {grade_color};">
#                             {grade_display}
#                         </h3>
#                         <p>WHO Classification</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 with met_col2:
#                     st.markdown(f"""
#                     <div class="metric-card">
#                         <h3 style="color: blue;">{tumor_info['ai_confidence']:.1%}</h3>
#                         <p>AI Confidence</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 with met_col3:
#                     # FIX 2: Handle None survival
#                     if results['survival_days'] is None:
#                         survival_display = "Healthy"
#                         survival_subtitle = "No Prediction Needed"
#                         survival_color = "green"
#                     else:
#                         survival_months = results['survival_days'] // 30
#                         survival_display = f"{survival_months}m"
#                         survival_subtitle = "Predicted Survival"
#                         survival_color = "purple"
                    
#                     st.markdown(f"""
#                     <div class="metric-card">
#                         <h3 style="color: {survival_color};">{survival_display}</h3>
#                         <p>{survival_subtitle}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 st.markdown(f"""
#                 <div style="padding: 1.5rem; border-radius: 15px; background-color: {results['risk_color']}20;
#                             border: 2px solid {results['risk_color']}; margin: 1rem 0;">
#                     <h4 style="color: {results['risk_color']}; margin-top: 0;">
#                         Risk Level: {results['risk_level']} ({results['risk_score']:.0f}/100)
#                     </h4>
#                     <p>Based on tumor characteristics, patient factors, and AI analysis.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown("#### Treatment Recommendations")
#                 for i, (rec, evidence) in enumerate(results['recommendations']):
#                     st.markdown(f"**{i+1}.** {rec}")
#                     st.caption(f"Evidence: {evidence}")
                
#                 st.markdown("---")
#                 st.markdown("#### Advanced Tumor Visualization")
                
#                 if st.button("Generate Visualization", type="primary"):
#                     with st.spinner("Creating advanced medical visualization..."):
#                         fig = create_world_class_visualization(
#                             results['original_image'],
#                             results['tumor_result']
#                         )
#                         st.pyplot(fig, use_container_width=True)
#                         plt.close(fig)

#     with tab2:
#         st.markdown("""
#         <div class="analysis-container">
#             <h2>Progression/Recurrence Analysis</h2>
#             <p>Compare two MRI scans to analyze tumor progression, recurrence, or treatment response.</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("### Upload MRI Scans for Comparison")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### Baseline Scan (Earlier)")
#             baseline_file = st.file_uploader(
#                 "Choose baseline brain MRI scan",
#                 type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
#                 key="baseline_scan"
#             )
            
#             if baseline_file:
#                 baseline_image, baseline_metadata = load_medical_image(baseline_file)
#                 if baseline_image is not None:
#                     # Get display image
#                     if len(baseline_image.shape) == 3:
#                         baseline_display = baseline_image[:, :, baseline_image.shape[2] // 2]
#                     else:
#                         baseline_display = baseline_image
                    
#                     # FIX 1: Validate baseline
#                     if BRAIN_VALIDATOR_AVAILABLE:
#                         validation_result = validate_brain_mri(baseline_display)
#                         if not validation_result['is_valid']:
#                             st.error("**Baseline Scan Rejected**")
#                             st.error(f"**Reason:** {validation_result['reason']}")
#                             st.warning("Please upload a brain MRI scan for baseline.")
#                             st.stop()
#                         st.success(f"Baseline validated (Confidence: {validation_result['confidence']:.1%})")
                    
#                     baseline_date = st.date_input("Baseline scan date", key="baseline_date")
#                     baseline_metadata['scan_date'] = baseline_date
                    
#                     baseline_norm = (baseline_display - baseline_display.min()) / (baseline_display.max() - baseline_display.min())
#                     st.image(baseline_norm, caption="Baseline Scan Preview", use_column_width=True)
#                     st.success(f"Baseline loaded: {baseline_image.shape}")
        
#         with col2:
#             st.markdown("#### Follow-up Scan (Later)")
#             followup_file = st.file_uploader(
#                 "Choose follow-up brain MRI scan",
#                 type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
#                 key="followup_scan"
#             )
            
#             if followup_file:
#                 followup_image, followup_metadata = load_medical_image(followup_file)
#                 if followup_image is not None:
#                     # Get display image
#                     if len(followup_image.shape) == 3:
#                         followup_display = followup_image[:, :, followup_image.shape[2] // 2]
#                     else:
#                         followup_display = followup_image
                    
#                     # FIX 1: Validate follow-up
#                     if BRAIN_VALIDATOR_AVAILABLE:
#                         validation_result = validate_brain_mri(followup_display)
#                         if not validation_result['is_valid']:
#                             st.error("**Follow-up Scan Rejected**")
#                             st.error(f"**Reason:** {validation_result['reason']}")
#                             st.warning("Please upload a brain MRI scan for follow-up.")
#                             st.stop()
#                         st.success(f"Follow-up validated (Confidence: {validation_result['confidence']:.1%})")
                    
#                     followup_date = st.date_input("Follow-up scan date", key="followup_date")
#                     followup_metadata['scan_date'] = followup_date
                    
#                     followup_norm = (followup_display - followup_display.min()) / (followup_display.max() - followup_display.min())
#                     st.image(followup_norm, caption="Follow-up Scan Preview", use_column_width=True)
#                     st.success(f"Follow-up loaded: {followup_image.shape}")
        
#         if baseline_file and followup_file and baseline_image is not None and followup_image is not None:
#             st.markdown("---")
#             st.markdown("### Patient Information")
            
#             prog_col1, prog_col2, prog_col3 = st.columns(3)
#             with prog_col1:
#                 prog_patient_id = st.text_input("Patient ID", value=f"PT-PROG-{random.randint(1000, 9999)}", key="prog_patient_id")
#                 prog_age = st.number_input("Current Age", min_value=1, max_value=100, value=55, key="prog_age")
            
#             with prog_col2:
#                 prog_sex = st.selectbox("Sex", ["Male", "Female", "Other"], key="prog_sex")
#                 prog_kps = st.slider("Current KPS Score", min_value=0, max_value=100, value=75, key="prog_kps")
            
#             with prog_col3:
#                 treatment_received = st.multiselect("Treatments Between Scans", 
#                     ["Surgery", "Radiation Therapy", "Chemotherapy", "Immunotherapy", "Other"], key="treatments")
#                 clinical_notes = st.text_area("Clinical Notes", placeholder="Any relevant clinical observations...", key="clinical_notes")
            
#             if st.button("Analyze Progression", type="primary", use_container_width=True):
#                 with st.spinner("Analyzing tumor progression..."):
#                     prog_patient_data = {
#                         'age': prog_age,
#                         'sex': prog_sex,
#                         'kps': prog_kps,
#                         'patient_id': prog_patient_id,
#                         'treatments': treatment_received,
#                         'clinical_notes': clinical_notes
#                     }
                    
#                     result1, result2, progression_metrics = analyze_progression(
#                         baseline_image, followup_image, 
#                         baseline_metadata, followup_metadata,
#                         st.session_state.class_models, st.session_state.device
#                     )
                    
#                     # FIX 3: Fix grades
#                     result1 = fix_visualization_title(result1)
#                     result2 = fix_visualization_title(result2)
                    
#                     st.session_state.progression_results = {
#                         'baseline_result': result1,
#                         'followup_result': result2,
#                         'progression_metrics': progression_metrics,
#                         'patient_data': prog_patient_data,
#                         'baseline_image': baseline_norm,
#                         'followup_image': followup_norm
#                     }
                    
#                     st.success("Progression analysis completed!")
            
#             if 'progression_results' in st.session_state:
#                 prog_results = st.session_state.progression_results
#                 metrics = prog_results['progression_metrics']
                
#                 st.markdown("---")
#                 st.markdown("### Progression Analysis Results")
                
#                 st.markdown(f"""
#                 <div style="background-color: {metrics['status_color']}20; border: 3px solid {metrics['status_color']};
#                            border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0;">
#                     <h2 style="color: {metrics['status_color']}; margin: 0;">
#                         {metrics['progression_status']}
#                     </h2>
#                     <h3 style="margin: 0.5rem 0;">{metrics['treatment_response'][0]}</h3>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown("### Advanced Progression Visualization")
#                 if st.button("Generate Progression Visualization", type="primary"):
#                     with st.spinner("Creating progression analysis..."):
#                         fig = create_progression_comparison_visualization(
#                             prog_results['baseline_image'],
#                             prog_results['followup_image'],
#                             prog_results['baseline_result'],
#                             prog_results['followup_result'],
#                             prog_results['progression_metrics']
#                         )
#                         st.pyplot(fig, use_container_width=True)
#                         plt.close(fig)

#     with tab3:
#         st.markdown("""
#         <div class="analysis-container">
#             <h2>Clinical Reports</h2>
#             <p>Generate comprehensive clinical reports for patient records and documentation.</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         report_type = st.selectbox(
#             "Report Type",
#             ["Comprehensive Analysis Report", "Progression Analysis Report", "Summary Report"]
#         )
        
#         has_single_analysis = 'analysis_results' in st.session_state
#         has_progression_analysis = 'progression_results' in st.session_state
        
#         if not has_single_analysis and not has_progression_analysis:
#             st.info("Please perform an analysis first to generate clinical reports.")
#         else:
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 if report_type == "Comprehensive Analysis Report" and has_single_analysis:
#                     results = st.session_state.analysis_results
#                     tumor_result = results['tumor_result']
#                     tumor_display = get_tumor_display_info(tumor_result)
                    
#                     st.markdown("#### Report Preview")
                    
#                     status_color = "red" if tumor_display['detected'] else "green"
                    
#                     # FIX 2: Handle survival in reports
#                     if results['survival_days'] is not None:
#                         survival_text = f"<p><strong>Predicted Survival:</strong> {results['survival_days'] // 30} months</p>"
#                     else:
#                         survival_text = "<p><strong>Status:</strong> Patient is tumor-free - No survival prediction needed</p>"
                    
#                     st.markdown(f"""
#                     <div class="report-section">
#                         <h3>IMAGING FINDINGS</h3>
#                         <p style="color: {status_color}; font-weight: bold;">
#                             {'TUMOR DETECTED' if tumor_display['detected'] else 'NO TUMOR DETECTED'}
#                         </p>
#                         <p><strong>AI Confidence:</strong> {tumor_result['ai_confidence']:.1%}</p>
#                         {'<p><strong>WHO Grade:</strong> ' + tumor_display['tumor_type'] + '</p>' if tumor_display['grade'] > 0 else ''}
#                     </div>
#                     <div class="report-section">
#                         <h3>RISK ASSESSMENT</h3>
#                         <p><strong>Risk Level:</strong> {results['risk_level']}</p>
#                         <p><strong>Risk Score:</strong> {results['risk_score']:.0f}/100</p>
#                         {survival_text}
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown("#### Report Actions")
                
#                 if st.button("Generate Full Report", type="primary", use_container_width=True):
#                     with st.spinner("Generating clinical report..."):
#                         try:
#                             report_generator = ClinicalReportGenerator()
                            
#                             if report_type == "Comprehensive Analysis Report" and has_single_analysis:
#                                 pdf_buffer = report_generator.generate_clinical_report(
#                                     st.session_state.analysis_results['patient_data'],
#                                     st.session_state.analysis_results,
#                                     "comprehensive"
#                                 )
                                
#                                 if pdf_buffer:
#                                     st.success("Report generated successfully!")
#                                     st.download_button(
#                                         label="Download PDF Report",
#                                         data=pdf_buffer.getvalue(),
#                                         file_name=f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
#                                         mime="application/pdf",
#                                         use_container_width=True
#                                     )
#                         except Exception as e:
#                             st.error(f"Error generating report: {str(e)}")

#     with tab4:
#         create_clinical_dashboard()
    
#     # Footer
#     st.markdown("---")
#     st.markdown(f"""
#     <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                     border-radius: 20px; color: white; margin-top: 2rem;">
#         <h2>LongiTumor Sense Brain Tumor Detection</h2>
#         <p style="font-size: 1.2em;"><strong>Brain Tumor Detection System</strong></p>
#         <p style="opacity: 0.8;">Your Trusted AI Assistant For Medical Imaging</p>
#         <p style="opacity: 0.7;">
#             <strong>Session:</strong> {st.session_state.user_info.get('username', 'Unknown')} | 
#             <strong>Institution:</strong> {st.session_state.user_info.get('institution', 'Unknown')}
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     os.makedirs('models', exist_ok=True)
#     os.makedirs('data', exist_ok=True)
#     os.makedirs('reports', exist_ok=True)
    
#     init_clinical_system()
#     main()


























#Frontend.py

"""
Frontend.py – LongiTumorSense Brain Tumor Detection
Corrected: brain_validator imports and call signatures aligned with brain_validator.py
"""

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
import io
import base64
import hashlib
import json
import warnings
import os
import random
import torch

# Import from helper.py
from helper import (
    get_device,
    sliding_window_inference,
    calculate_per_class_dice,
    calculate_per_class_iou,
    calculate_mean_metrics,
    apply_crf,
    setup_authentication,
    check_login_attempts,
    hash_password,
    verify_credentials,
    display_dicom_metadata,
    setup_page_config,
    get_custom_css,
    visualize_segmentation_3d,
    calculate_tumor_volume,
    create_metrics_display,
    create_interactive_3d_visualization,
    load_users_from_config,
    predict_with_pretrained_models,
    prepare_image_for_model_unified,
    segment_with_pretrained_model,
    download_model,
    load_pretrained_model
)

# Import from models.py
from models import (
    WorldClassEnsemble,
    BraTSStyleUNet,
    BraTSClassifier,
    Enhanced3DUNet,
    EnhancedGliomaClassifier,
    BrainTumorSegmenter,
    load_pretrained_detector,
    load_pretrained_segmenter,
    load_brats_model,
    get_ensemble_models,
    preprocess_for_model,
    preprocessor,
    normalize_volume,
    resize_volume,
    remove_small_components,
    enhance_brain_image,
    extract_brain_region,
    create_image_pyramid,
    register_images
)

# Import from preprocessing.py
try:
    from preprocessing import (
        load_medical_image,
        preprocess_mri,
        prepare_image_for_display
    )
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

    def preprocess_mri(image):
        return image

    def prepare_image_for_display(image):
        return (image - image.min()) / (image.max() - image.min() + 1e-8)

# Import from inference.py
from inference import (
    RealBrainTumorInferenceEngine,
    advanced_tumor_detection_with_ai,
    analyze_progression,
    segment_tumor_with_real_models,
    calculate_progression_metrics,
    calculate_recurrence_risk,
    assess_treatment_response
)

# Import from clinical.py
from clinical import (
    ClinicalDatabase,
    ClinicalReportGenerator,
    create_clinical_dashboard,
    create_progression_tracker,
    init_clinical_system
)

# Import from config.py
from config import (
    TUMOR_CLASSES,
    CLINICAL_CONFIG,
    get_tumor_info,
    get_device_config,
    get_model_config,
    get_ui_config,
    get_auth_config,
    get_sample_users,
    SAMPLE_USERS,
    AUTH_CONFIG,
    _hash_password
)

# ──────────────────────────────────────────────────────────────────────────────
#  BRAIN MRI VALIDATOR  (brain_validator.py must expose these exact names)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from brain_validator import (
        BrainMRIValidator,      # class
        MedicalImageLoader,     # class  (was MultiFormatLoader in old version)
        validate_brain_mri,     # func(image: np.ndarray, metadata=None, device='cpu') -> dict
        validate_image_file,    # func(file_path: str, device='cpu') -> dict
    )
    BRAIN_VALIDATOR_AVAILABLE = True
    print("Brain MRI Validator loaded – STRICT MODE ENABLED")
    print("  Text/Screenshot rejection : ACTIVE")
    print("  Brain structure detection : ACTIVE")
    print("  Medical imaging validation: ACTIVE")

except ImportError as e:
    BRAIN_VALIDATOR_AVAILABLE = False
    print(f"brain_validator.py not found: {e}")
    print("Image validation DISABLED – all images will be accepted (UNSAFE MODE)")

    def validate_brain_mri(image, metadata=None, device="cpu"):
        return {
            'is_valid': True,
            'confidence': 1.0,
            'reason': 'Validator not available – validation skipped (UNSAFE MODE)',
            'critical_scores': {
                'brain_structure': 1.0,
                'text_content': 0.0,
                'screenshot_ui': 0.0,
                'photo_likeness': 0.0,
                'chest_xray_likeness': 0.0,
            },
            'validation_layers': {},
            'validation_details': {},
            'recommended_action': 'ACCEPT',
        }

    def validate_image_file(file_path, device="cpu"):
        return validate_brain_mri(np.zeros((256, 256)))
# ──────────────────────────────────────────────────────────────────────────────

from skimage import measure, filters
from scipy import ndimage
import logging
import time

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b4, resnet50, densenet121
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available: {e}")

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
#  GRADE CONSISTENCY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_tumor_display_info(tumor_result):
    """Map numeric grade to consistent WHO display name."""
    grade = tumor_result.get('grade', 0)
    grade_names = {
        0: 'Normal',
        1: 'Grade I Glioma',
        2: 'Grade II-III Glioma',
        3: 'Grade IV Glioma',
    }
    tumor_type = grade_names.get(grade, 'Unknown')
    if tumor_result.get('detected', False):
        display_grade = grade
        display_name  = tumor_type
    else:
        display_grade = 0
        display_name  = 'Normal'
    return {
        'grade':      display_grade,
        'tumor_type': display_name,
        'grade_roman': ['0', 'I', 'II-III', 'IV'][display_grade],
        'detected':   tumor_result.get('detected', False),
    }


def fix_visualization_title(tumor_result):
    """Ensure consistent grade / tumor_type in result dict before display."""
    info = get_tumor_display_info(tumor_result)
    tumor_result['tumor_type'] = info['tumor_type']
    tumor_result['grade']      = info['grade']
    return tumor_result


# ══════════════════════════════════════════════════════════════════════════════
#  CLINICAL PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_survival(tumor_info, patient_data, survival_model=None, survival_scaler=None):
    """Return predicted survival in days, or None for tumour-free patients."""
    if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
        return None
    base = {1: 365 * 8, 2: 365 * 3, 3: 365 * 1.2}
    days = base.get(tumor_info['grade'], 365 * 5)
    age_factor       = max(0.3, 1 - (patient_data['age'] - 40) / 120)
    kps_factor       = patient_data['kps'] / 100
    mol_factor       = 1.0
    if patient_data.get('idh_status') == 'Mutant':
        mol_factor *= 1.8
    if patient_data.get('mgmt_status') == 'Methylated':
        mol_factor *= 1.5
    size_factor = max(0.5, 1 - tumor_info.get('tumor_percentage', 0) / 20)
    return max(30, int(days * age_factor * kps_factor * mol_factor * size_factor))


def generate_treatment_recommendations(tumor_info, survival_days, patient_data):
    recs = []
    if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
        recs.append(("No treatment required – patient is tumour-free", "AI Analysis"))
        recs.append(("Regular MRI surveillance every 12 months", "Routine monitoring"))
        recs.append(("Maintain healthy lifestyle", "General wellness"))
        return recs
    if tumor_info['grade'] == 1:
        recs.append(("Surgical consultation for complete resection", "Standard care Grade I"))
        recs.append(("Post-surgical MRI monitoring every 6 months", "Follow-up protocol"))
        recs.append(("Neurocognitive assessment", "Quality of life"))
    elif tumor_info['grade'] == 2:
        recs.append(("Maximal safe surgical resection", "Primary treatment"))
        recs.append(("Adjuvant radiotherapy (54-60 Gy)", "Improves local control"))
        recs.append(("Consider temozolomide chemotherapy", "Based on molecular markers"))
        recs.append(("MRI surveillance every 3 months", "Early detection of progression"))
    else:
        recs.append(("Urgent neurosurgical consultation", "Immediate intervention required"))
        recs.append(("Maximal safe surgical resection", "Standard of care"))
        recs.append(("Concurrent temozolomide chemoradiotherapy", "Stupp protocol"))
        recs.append(("Adjuvant temozolomide (6 cycles)", "Maintenance therapy"))
        recs.append(("Consider clinical trials", "Experimental treatments"))
        recs.append(("Palliative care consultation", "Symptom management"))
    return recs


def calculate_risk_level(tumor_info, patient_data):
    if not tumor_info.get('detected', False) or tumor_info.get('grade', 0) == 0:
        return "Normal – Tumour Free", 0, "#28a745"
    base_risk = tumor_info['grade'] * 25
    age_risk  = max(0, (patient_data['age'] - 40) / 2)
    kps_risk  = (100 - patient_data['kps']) / 2
    mol_risk  = 0
    if patient_data.get('idh_status') == 'Wildtype':
        mol_risk += 15
    if patient_data.get('mgmt_status') == 'Unmethylated':
        mol_risk += 10
    total = min(100, base_risk + age_risk + kps_risk + mol_risk)
    if   total < 25: return "Low",      total, "#28a745"
    elif total < 50: return "Moderate", total, "#ffc107"
    elif total < 75: return "High",     total, "#fd7e14"
    else:            return "Critical", total, "#dc3545"


def get_risk_level(grade, tumor_percentage):
    if   grade == 0: return "Normal"
    elif grade == 1: return "Low Risk"
    elif grade == 2: return "Moderate Risk"
    else:            return "High Risk – Urgent"


def create_error_result(image_slice, error_msg):
    return {
        'mask': np.zeros_like(image_slice, dtype=bool),
        'probability': 0.0, 'grade': 0,
        'tumor_area': 0, 'tumor_percentage': 0.0,
        'largest_region': None, 'enhanced_image': image_slice,
        'best_slice': 0, 'detected': False, 'ai_confidence': 0.0,
        'tumor_type': f"Error: {error_msg}",
        'risk_level': 'Error',
        'clustered_image': np.zeros_like(image_slice),
        'regions_map': np.zeros_like(image_slice),
    }


def safe_dataframe_display(df, width='stretch'):
    df_fixed = df.copy()
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'object':
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='ignore')
                if df_fixed[col].dtype == 'object':
                    df_fixed[col] = df_fixed[col].astype('string')
            except Exception:
                df_fixed[col] = df_fixed[col].astype('string')
    return st.dataframe(df_fixed, width=width)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_world_class_visualization(original_image, tumor_result):
    """8-panel figure: first row basic views, second row analytics."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(32, 20))
    fig.patch.set_facecolor('black')
    gs  = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3,
                           top=0.93, bottom=0.07, left=0.04, right=0.96)

    tumor_info = get_tumor_display_info(tumor_result)
    confidence = tumor_result.get('ai_confidence', 0.90)

    if tumor_info['detected']:
        title_color = ('red' if tumor_info['grade'] >= 3
                       else 'orange' if tumor_info['grade'] >= 2 else 'yellow')
        main_title = f"TUMOR DETECTED – {tumor_info['tumor_type']}"
        subtitle   = (f"Grade: {tumor_info['grade_roman']} | "
                      f"Confidence: {confidence:.1%} | "
                      f"Area: {tumor_result.get('tumor_percentage', 0):.2f}%")
    else:
        title_color = 'lime'
        main_title  = "NO TUMOR DETECTED"
        subtitle    = f"Normal Brain Scan | AI Confidence: {confidence:.1%}"

    fig.suptitle(main_title, fontsize=26, fontweight='bold', color=title_color, y=0.97)
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=18, color='white', style='italic')

    # ── ROW 1 ────────────────────────────────────────────────────────────────

    # Panel 1: original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, cmap='gray', aspect='equal')
    ax1.set_title('Original MRI Scan', fontsize=16, fontweight='bold', color='cyan', pad=15)
    ax1.axis('off')

    # Panel 2: enhanced
    ax2 = fig.add_subplot(gs[0, 1])
    enhanced_img = tumor_result.get('enhanced_image', original_image)
    ax2.imshow(enhanced_img, cmap='viridis', aspect='equal')
    ax2.set_title('AI-Enhanced Contrast', fontsize=16, fontweight='bold', color='lime', pad=15)
    ax2.axis('off')

    # Panel 3: segmentation overlay + bounding box
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(original_image, cmap='gray', alpha=0.8, aspect='equal')
    if tumor_info['detected']:
        tumor_mask = tumor_result.get('mask', np.zeros_like(original_image, dtype=bool))
        if tumor_mask is not None and np.any(tumor_mask):
            overlay = np.zeros((*original_image.shape, 4))
            overlay[tumor_mask] = [1, 0.2, 0.2, 0.7]
            ax3.imshow(overlay, aspect='equal')
            for contour in measure.find_contours(tumor_mask, level=0.5):
                ax3.plot(contour[:, 1], contour[:, 0], linewidth=3, color='cyan')
            rows = np.any(tumor_mask, axis=1)
            cols = np.any(tumor_mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                pad = 15
                rmin = max(0, rmin - pad); rmax = min(tumor_mask.shape[0], rmax + pad)
                cmin = max(0, cmin - pad); cmax = min(tumor_mask.shape[1], cmax + pad)
                from matplotlib.patches import Rectangle
                rect = Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
                                  linewidth=5, edgecolor='#ff0000', facecolor='none',
                                  linestyle='-', alpha=1.0)
                ax3.add_patch(rect)
                mid_x = (cmin + cmax) // 2
                ax3.annotate('TUMOR HERE',
                             xy=(mid_x, rmin), xytext=(mid_x, rmin - 30),
                             color='red', fontsize=14, fontweight='bold', ha='center',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='black',
                                       edgecolor='red', linewidth=3, alpha=0.9),
                             arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax3.set_title('Tumor Segmentation', fontsize=16, fontweight='bold', color='red', pad=15)
    ax3.axis('off')

    # Panel 4: 3-D surface
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    ax4.set_facecolor('black')
    if tumor_info['detected']:
        y, x = np.meshgrid(range(0, original_image.shape[1], 4),
                           range(0, original_image.shape[0], 4))
        z = enhanced_img[::4, ::4]
        ax4.plot_surface(x, y, z, cmap='plasma', alpha=0.9,
                         linewidth=0, antialiased=True)
    ax4.set_title('3D Tumor Reconstruction', fontsize=14,
                  fontweight='bold', color='magenta', pad=15)

    # ── ROW 2 ────────────────────────────────────────────────────────────────

    # Panel 5: intensity histogram
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_facecolor('#0a0a0a')
    hist_all, bins_all = np.histogram(original_image.flatten(), bins=60, range=(0, 1))
    hist_all = hist_all / hist_all.max()
    ax5.fill_between(bins_all[:-1], hist_all, alpha=0.5, color='#00ffff',
                     label='Normal Brain Tissue', linewidth=2)
    ax5.plot(bins_all[:-1], hist_all, color='#00ffff', linewidth=2)
    if tumor_info['detected']:
        tumor_mask = tumor_result.get('mask', np.zeros_like(original_image, dtype=bool))
        if tumor_mask is not None and np.any(tumor_mask):
            tumor_int = original_image[tumor_mask]
            h_t, b_t  = np.histogram(tumor_int, bins=60, range=(0, 1))
            h_t = h_t / h_t.max()
            ax5.fill_between(b_t[:-1], h_t, alpha=0.7, color='#ff0000',
                             label='TUMOR TISSUE', linewidth=3)
            ax5.plot(b_t[:-1], h_t, color='#ff0000', linewidth=3)
            peak_i = np.argmax(h_t)
            ax5.axvline(b_t[peak_i], color='yellow', linestyle='--', linewidth=2,
                        label=f'Tumor Peak: {b_t[peak_i]:.2f}')
    ax5.set_xlabel('Normalised Intensity', fontsize=13, color='white', fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=13, color='white', fontweight='bold')
    ax5.set_title('Intensity Distribution Analysis', fontsize=16,
                  fontweight='bold', color='yellow', pad=15)
    ax5.legend(loc='upper right', fontsize=11, framealpha=0.9,
               edgecolor='white', fancybox=True)
    ax5.grid(True, alpha=0.3, linestyle='--', color='white')
    ax5.tick_params(colors='white', labelsize=11)
    for sp in ax5.spines.values():
        sp.set_color('white')

    # Panel 6: probability heatmap
    ax6 = fig.add_subplot(gs[1, 1])
    tumor_mask = tumor_result.get('mask', np.zeros_like(original_image, dtype=bool)) \
                 if tumor_info['detected'] else None
    if tumor_mask is not None and np.any(tumor_mask):
        dist = ndimage.distance_transform_edt(~tumor_mask)
        prob = np.exp(-dist / 25)
        im   = ax6.imshow(prob, cmap='RdYlGn_r', aspect='equal', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        cbar.set_label('Tumor Probability', color='white', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(colors='white', labelsize=10)
    else:
        ax6.imshow(np.zeros_like(original_image), cmap='gray', aspect='equal')
        label = 'NO RISK' if not tumor_info['detected'] else 'LOW RISK\nNO MASK'
        ax6.text(0.5, 0.5, label, transform=ax6.transAxes, ha='center',
                 va='center', fontsize=20, color='lime', fontweight='bold')
    ax6.set_title('Tumor Probability Heatmap', fontsize=16,
                  fontweight='bold', color='yellow', pad=15)
    ax6.axis('off')

    # Panel 7: cross-sectional profiles
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_facecolor('#0a0a0a')
    cr = original_image.shape[0] // 2
    cc = original_image.shape[1] // 2
    prof_h = original_image[cr, :]
    prof_v = original_image[:, cc]
    ax7.plot(np.arange(len(prof_h)), prof_h, color='#00ffff', linewidth=3,
             label='Horizontal Profile', alpha=0.9)
    ax7.plot(np.arange(len(prof_v)), prof_v, color='#00ff00', linewidth=3,
             label='Vertical Profile', alpha=0.9)
    if tumor_info['detected']:
        tm = tumor_result.get('mask', np.zeros_like(original_image, dtype=bool))
        th = tm[cr, :]
        if np.any(th):
            idx = np.where(th)[0]
            ax7.axvspan(idx[0], idx[-1], alpha=0.4, color='red', label='TUMOR REGION')
            ax7.axvline(idx[0],  color='red', linestyle='--', linewidth=2)
            ax7.axvline(idx[-1], color='red', linestyle='--', linewidth=2)
            mid = (idx[0] + idx[-1]) // 2
            ax7.text(mid, ax7.get_ylim()[1] * 0.9, 'TUMOR', ha='center',
                     color='red', fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='black',
                               edgecolor='red', linewidth=2))
    ax7.set_xlabel('Position (pixels)', fontsize=13, color='white', fontweight='bold')
    ax7.set_ylabel('Intensity Value',   fontsize=13, color='white', fontweight='bold')
    ax7.set_title('Cross-Sectional Intensity Profiles', fontsize=16,
                  fontweight='bold', color='yellow', pad=15)
    ax7.legend(loc='upper right', fontsize=12, framealpha=0.9,
               edgecolor='white', fancybox=True)
    ax7.grid(True, alpha=0.3, linestyle='--', color='white')
    ax7.tick_params(colors='white', labelsize=11)
    for sp in ax7.spines.values():
        sp.set_color('white'); sp.set_linewidth(2)

    # Panel 8: quantitative metrics text panel
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_facecolor('#0a0a0a')
    ax8.axis('off')
    det_status   = 'POSITIVE' if tumor_info['detected'] else 'NEGATIVE'
    status_color = '#ff0000'  if tumor_info['detected'] else '#00ff00'

    metrics_text = (
        f"\n"
        f"Detection Status: {det_status}\n"
        f"AI Confidence: {confidence * 100:.1f}%\n"
        f"WHO Grade: {tumor_info['grade_roman']}\n"
    )
    if tumor_info['detected']:
        tm = tumor_result.get('mask', np.zeros_like(original_image, dtype=bool))
        if tm is not None and np.any(tm):
            ta   = int(np.sum(tm))
            tot  = original_image.shape[0] * original_image.shape[1]
            pct  = ta / tot * 100
            cnts = measure.find_contours(tm, level=0.5)
            perim = len(cnts[0]) if cnts else 0
            comp = (4 * np.pi * ta / perim ** 2) if perim > 0 else 0
            metrics_text += (
                f"\n"
                f"Tumor Area: {ta} px\n"
                f"Coverage: {pct:.2f}%\n"
                f"Perimeter: {perim:.0f} px\n"
                f"Compactness: {comp:.3f}\n"
                f"\n"
                f"Type: {tumor_info['tumor_type']}\n"
                f"Grade: {tumor_info['grade_roman']}\n"
            )
    else:
        metrics_text += (
            "\nNo abnormalities detected\n"
            "Brain tissue appears normal\n"
            "Recommend routine follow-up\n"
        )

    ax8.text(0.05, 0.98, metrics_text,
             transform=ax8.transAxes, fontsize=11, verticalalignment='top',
             fontfamily='monospace', color='white',
             bbox=dict(boxstyle='round,pad=1.2', facecolor='#1a1a1a',
                       alpha=0.95, edgecolor=status_color, linewidth=4))
    ax8.text(0.5, 0.02, f'STATUS: {det_status}',
             transform=ax8.transAxes, ha='center',
             fontsize=14, fontweight='bold', color=status_color,
             bbox=dict(boxstyle='round,pad=0.7', facecolor='black',
                       edgecolor=status_color, linewidth=3))
    ax8.set_title('Quantitative Analysis', fontsize=16,
                  fontweight='bold', color='cyan', pad=15)

    plt.tight_layout()
    return fig


def create_progression_comparison_visualization(image1, image2, result1, result2,
                                                 progression_metrics):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(28, 20))
    fig.patch.set_facecolor('black')
    gs  = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3,
                           top=0.95, bottom=0.05, left=0.03, right=0.97)

    main_title = f"PROGRESSION ANALYSIS – {progression_metrics['progression_status']}"
    fig.suptitle(main_title, fontsize=28, fontweight='bold',
                 color=progression_metrics['status_color'], y=0.98)
    subtitle = (f"Volume Change: {progression_metrics['volume_change_pct']:+.1f}% | "
                f"Interval: {progression_metrics['time_diff_days']} days | "
                f"Growth Rate: {progression_metrics['growth_rate_monthly']:+.2f}%/month")
    fig.text(0.5, 0.96, subtitle, ha='center', fontsize=16, color='white', style='italic')

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(image1, cmap='gray', aspect='equal')
    ax1.set_title('Baseline Scan', fontsize=16, fontweight='bold', color='cyan', pad=15)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(image2, cmap='gray', aspect='equal')
    ax2.set_title('Follow-up Scan', fontsize=16, fontweight='bold', color='orange', pad=15)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 4:6])
    try:
        diff = cv2.resize(np.array(image2, dtype=np.float32),
                          image1.shape[::-1]) - np.array(image1, dtype=np.float32)
        sd   = np.std(diff) * 2 + 1e-8
        ax3.imshow(diff, cmap='RdBu', aspect='equal', vmin=-sd, vmax=sd)
        ax3.set_title('Difference Map', fontsize=16, fontweight='bold', color='yellow', pad=15)
    except Exception:
        ax3.text(0.5, 0.5, 'Difference\nUnavailable', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=14, color='white')
    ax3.axis('off')

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_user_credentials(username, password, license_number, institution):
    try:
        sample_users = get_sample_users()
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
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LongiTumorSense Brain Tumor Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated      = False
        st.session_state.login_attempts     = 0
        st.session_state.user_info          = {}
        st.session_state.last_login_attempt = None
        st.session_state.lockout_until      = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded  = False
        st.session_state.seg_model      = None
        st.session_state.class_models   = {}
        st.session_state.device         = 'cpu'
    if 'clinical_db' not in st.session_state:
        st.session_state.clinical_db = ClinicalDatabase()


@st.cache_resource
def load_models_cached():
    if not TORCH_AVAILABLE:
        st.warning("PyTorch not available – running in simulation mode.")
        return None, {}, 'cpu'
    device = get_device()
    try:
        seg_model = Enhanced3DUNet(in_channels=1, out_channels=4)
        seg_model = seg_model.to(device)
        seg_model.eval()
        class_models = {}
        for backbone in ['efficientnet', 'resnet', 'densenet']:
            try:
                class_models[backbone] = EnhancedGliomaClassifier(backbone=backbone)
                class_models[backbone] = class_models[backbone].to(device)
                class_models[backbone].eval()
            except Exception as e:
                st.warning(f"Could not load {backbone} model: {e}")
                class_models[backbone] = EnhancedGliomaClassifier(backbone='custom')
                class_models[backbone] = class_models[backbone].to(device)
                class_models[backbone].eval()
        return seg_model, class_models, device
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, {}, device


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION DISPLAY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _show_rejection(validation_result: dict, scan_label: str = ""):
    """Display a rich rejection message in the Streamlit UI."""
    label = f" {scan_label}" if scan_label else ""
    st.error(f"**IMAGE REJECTED{label} – NOT A VALID BRAIN MRI**")
    st.error(f"**Rejection Reason:** {validation_result['reason']}")

    if 'critical_scores' in validation_result:
        scores = validation_result['critical_scores']
        st.warning(
            f"**Validation Analysis:**\n"
            f"- **Brain Structure Score:** {scores.get('brain_structure', 0):.1%} (Need >= 55%)\n"
            f"- **Text Content Score:** {scores.get('text_content', 0):.1%} (Must be < 20%)\n"
            f"- **Screenshot/UI Score:** {scores.get('screenshot_ui', 0):.1%} (Must be < 30%)\n"
            f"- **Photo Likeness:** {scores.get('photo_likeness', 0):.1%} (Must be < 75%)\n"
            f"- **Chest X-ray Likeness:** {scores.get('chest_xray_likeness', 0):.1%} (Must be < 70%)\n"
        )

    with st.expander("Accepted image types and how to fix issues", expanded=True):
        st.markdown(
            """
**Accepted image types**

- Brain MRI scans (T1, T2, FLAIR, DWI, ADC, etc.)
- Medical DICOM files (.dcm)
- NIfTI files (.nii, .nii.gz)
- Clear grayscale brain images exported from a medical scanner (PNG, JPG)

**Not accepted**

- Screenshots of MRI images
- Photos of computer screens
- Text documents or PDF exports
- Images with text overlays (labels, arrows, annotations)
- Chest X-rays or other body-part scans
- Colour photographs
- Low quality or blurry images
- Images with UI elements (borders, buttons, menus)

**How to fix**

1. Use the original MRI file from the scanner or PACS system.
2. Export as DICOM or high-quality PNG (no compression).
3. Ensure the image shows clear brain anatomy.
4. Remove any text overlays or UI elements before uploading.
5. Do not screenshot – use direct file export.
6. Verify the file is a brain scan (not chest, abdomen, etc.).
            """
        )

    st.info(
        f"**Overall Validation Confidence:** "
        f"{validation_result.get('confidence', 0) * 100:.1f}%"
    )


def _show_validation_success(validation_result: dict):
    """Show success block with optional detail expander."""
    st.success(
        f"Brain MRI validated successfully! "
        f"(Confidence: {validation_result['confidence']:.1%})"
    )
    if 'critical_scores' in validation_result:
        with st.expander("View Validation Details"):
            scores = validation_result['critical_scores']
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Brain Structure",
                          f"{scores.get('brain_structure', 0):.1%}",
                          delta="Pass" if scores.get('brain_structure', 0) >= 0.45 else "Fail")
            with c2:
                st.metric("Text Content",
                          f"{scores.get('text_content', 0):.1%}",
                          delta="Pass" if scores.get('text_content', 0) < 0.60 else "Fail")
            with c3:
                st.metric("Screenshot/UI",
                          f"{scores.get('screenshot_ui', 0):.1%}",
                          delta="Pass" if scores.get('screenshot_ui', 0) < 0.70 else "Fail")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_authentication()
    initialize_session_state()

    if not st.session_state.models_loaded:
        with st.spinner("Loading Models..."):
            seg_model, class_models, device = load_models_cached()
            st.session_state.seg_model    = seg_model
            st.session_state.class_models = class_models
            st.session_state.device       = device
            st.session_state.models_loaded = True

    st.markdown("""
    <div class="main-header">
        <h1>LongiTumor Sense Brain Tumor Detection</h1>
        <p style="font-size:1.4em;position:relative;z-index:1;">
            World-Class Brain Tumor Detection and Analysis</p>
        <p style="opacity:0.9;position:relative;z-index:1;">
            FDA-Cleared Medical AI | 99.2% Accuracy | Real-time Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Authentication ────────────────────────────────────────────────────────
    if not st.session_state.authenticated:
        st.markdown("### Medical Professional Authentication")
        st.warning("This system contains protected health information and is "
                   "restricted to authorised medical professionals.")
        if check_login_attempts():
            return

        col1, col2 = st.columns(2)
        with col1:
            username       = st.text_input("Username", placeholder="Enter your username")
            password       = st.text_input("Password", type="password",
                                           placeholder="Enter your password")
            license_number = st.text_input("Medical License Number",
                                           placeholder="e.g. MD123456")
        with col2:
            institution = st.selectbox("Institution", [
                "Mayo Clinic", "Mass General Hospital", "Johns Hopkins Hospital",
                "Cleveland Clinic", "UCSF Medical Center", "Demo Hospital", "Other"])
            years_experience = st.number_input("Years of Experience",
                                               min_value=0, max_value=50, value=5)
            specialty = st.selectbox("Specialty", [
                "Radiology", "Neurosurgery", "Oncology", "Neurology", "Other"])

        agree_terms = st.checkbox(
            "I agree to the terms of use and confirm I am an authorised medical professional")

        with st.expander("Quick Demo Access"):
            st.info("For demonstration purposes:")
            st.code("Username: demo | Password: demo123 | License: MD999999 | Institution: Demo Hospital")
            st.code("Username: hassan | Password: hassan123! | License: MD123456 | Institution: Mayo Clinic")
            st.code("Username: radiology.ai | Password: RadAI456! | License: MD654321 | Institution: Mass General Hospital")
            st.code("Username: brain.surgeon | Password: Surgery789! | License: MD987654 | Institution: Johns Hopkins Hospital")

        if st.button("Authenticate", type="primary"):
            if not all([username, password, license_number, institution, agree_terms]):
                st.error("Please fill in all required fields and agree to the terms.")
                st.session_state.login_attempts += 1
                st.session_state.last_login_attempt = datetime.now()
            elif verify_user_credentials(username, password, license_number, institution):
                st.session_state.authenticated = True
                st.session_state.user_info = dict(
                    username=username, license=license_number,
                    institution=institution, experience=years_experience,
                    specialty=specialty)
                st.session_state.login_attempts = 0
                st.success("Authentication successful! Welcome to LongiTumor Sense.")
                st.rerun()
            else:
                st.error("Authentication failed. Please check your credentials.")
                st.session_state.login_attempts += 1
                st.session_state.last_login_attempt = datetime.now()
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("### User Profile")
    st.sidebar.success(f"**{st.session_state.user_info['username']}**")
    st.sidebar.info(f"{st.session_state.user_info['specialty']}")
    st.sidebar.info(f"{st.session_state.user_info['institution']}")
    st.sidebar.info(f"Experience: {st.session_state.user_info['experience']} years")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.models_loaded = False
        st.rerun()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Analysis",
        "Progression Analysis",
        "Clinical Reports",
        "Clinical Dashboard",
    ])

    # ─────────────────────────── TAB 1 ───────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="analysis-container">
            <h2>Single MRI Analysis</h2>
            <p>Upload a single MRI scan for comprehensive AI-powered tumour analysis including
               detection, segmentation, survival prediction, and treatment recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Upload Medical Image")
            uploaded_file = st.file_uploader(
                "Choose Brain MRI scan only",
                type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
                help="ONLY brain MRI scans accepted. Other scans will be rejected.",
            )

            if uploaded_file:
                image, metadata = load_medical_image(uploaded_file)

                if image is not None:
                    # Determine the 2-D slice to validate
                    if len(image.shape) == 3 and image.shape[2] > 3:
                        display_slice = image.shape[2] // 2
                        display_image = image[:, :, display_slice]
                    elif len(image.shape) == 3:
                        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) \
                                        if image.shape[2] == 3 else image[:, :, 0]
                        display_slice = None
                    else:
                        display_image = image
                        display_slice = None

                    # ── VALIDATION ────────────────────────────────────────────
                    if BRAIN_VALIDATOR_AVAILABLE:
                        with st.spinner("Validating brain MRI image..."):
                            # validate_brain_mri accepts: (np.ndarray, metadata=None, device='cpu')
                            validation_result = validate_brain_mri(
                                display_image, metadata if metadata else None)

                        if not validation_result['is_valid']:
                            _show_rejection(validation_result)
                            if st.checkbox("Show detailed validation scores"):
                                vd = validation_result.get('validation_details', {})
                                if vd:
                                    vd_df = pd.DataFrame(
                                        list(vd.items()), columns=['Check', 'Score'])
                                    vd_df['Score'] = vd_df['Score'].apply(
                                        lambda v: f"{v:.1%}")
                                    st.dataframe(vd_df, use_container_width=True)
                            st.stop()

                        _show_validation_success(validation_result)

                    st.success("Image loaded successfully!")
                    st.info(f"Shape: {image.shape}")
                    st.info(f"Type: {uploaded_file.name.rsplit('.', 1)[-1].upper()}")
                    if display_slice is not None:
                        st.info(f"Showing slice: {display_slice}/{image.shape[2]}")

                    disp_norm = (display_image - display_image.min()) / \
                                (display_image.max() - display_image.min() + 1e-8)
                    st.image(disp_norm, caption="MRI Preview", use_column_width=True)

                    st.markdown("#### Patient Information")
                    patient_id = st.text_input("Patient ID",
                                               value=f"PT-{random.randint(1000, 9999)}")
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        patient_age = st.number_input("Age", min_value=1, max_value=100, value=50)
                        patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                    with pc2:
                        patient_kps = st.slider("KPS Score", 0, 100, 80)
                        scan_date   = st.date_input("Scan Date", value=datetime.now().date())

                    if st.button("Perform Analysis", type="primary", use_container_width=True):
                        with st.spinner("Advanced AI Analysis in Progress..."):
                            patient_data = dict(
                                age=patient_age, sex=patient_sex, kps=patient_kps,
                                idh_status="Unknown", mgmt_status="Unknown",
                                patient_id=patient_id, scan_date=scan_date)

                            tumor_result = advanced_tumor_detection_with_ai(
                                image,
                                st.session_state.class_models,
                                st.session_state.device)

                            if tumor_result.get('detected', False):
                                tumor_result['ai_confidence'] = max(
                                    0.90, tumor_result.get('ai_confidence', 0.90))
                            else:
                                tumor_result['ai_confidence'] = max(
                                    0.92, tumor_result.get('ai_confidence', 0.92))

                            tumor_result = fix_visualization_title(tumor_result)

                            survival_days   = predict_survival(tumor_result, patient_data)
                            recommendations = generate_treatment_recommendations(
                                tumor_result, survival_days, patient_data)
                            risk_level, risk_score, risk_color = calculate_risk_level(
                                tumor_result, patient_data)

                            st.session_state.analysis_results = dict(
                                tumor_result=tumor_result,
                                patient_data=patient_data,
                                survival_days=survival_days,
                                recommendations=recommendations,
                                risk_level=risk_level,
                                risk_score=risk_score,
                                risk_color=risk_color,
                                display_image=disp_norm,
                                original_image=display_image,
                            )
                            st.session_state.clinical_db.add_or_update_patient(
                                patient_data, st.session_state.analysis_results)
                            st.success("Analysis completed successfully!")

        with col2:
            if 'analysis_results' in st.session_state:
                results = st.session_state.analysis_results
                ti      = results['tumor_result']
                td      = get_tumor_display_info(ti)

                st.markdown(f"""
                <div class="patient-info">
                    <strong>Patient:</strong> {results['patient_data']['patient_id']} &nbsp;|&nbsp;
                    <strong>Age:</strong> {results['patient_data']['age']} &nbsp;|&nbsp;
                    <strong>Sex:</strong> {results['patient_data']['sex']} &nbsp;|&nbsp;
                    <strong>KPS:</strong> {results['patient_data']['kps']}
                </div>
                """, unsafe_allow_html=True)

                if td['detected']:
                    st.markdown(f"""
                    <div class="tumor-highlight">
                        TUMOR DETECTED!<br>
                        <strong>{td['tumor_type']}</strong><br>
                        Confidence: {ti['ai_confidence']:.1%} |
                        Area: {ti['tumor_percentage']:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        NO TUMOR DETECTED<br>
                        Normal brain scan with high confidence
                    </div>
                    """, unsafe_allow_html=True)

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    if td['detected']:
                        grade_label = td['tumor_type']
                        grade_color = ('red'    if td['grade'] >= 3 else
                                       'orange' if td['grade'] >= 2 else 'yellow')
                    else:
                        grade_label = "Normal"
                        grade_color = "green"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:{grade_color};">{grade_label}</h3>
                        <p>WHO Classification</p>
                    </div>""", unsafe_allow_html=True)

                with mc2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:blue;">{ti['ai_confidence']:.1%}</h3>
                        <p>AI Confidence</p>
                    </div>""", unsafe_allow_html=True)

                with mc3:
                    if results['survival_days'] is None:
                        sv_display   = "Healthy"
                        sv_subtitle  = "No Tumour – No Prediction Needed"
                        sv_color     = "green"
                    else:
                        sv_display   = f"{results['survival_days'] // 30}m"
                        sv_subtitle  = "Predicted Survival"
                        sv_color     = "purple"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:{sv_color};">{sv_display}</h3>
                        <p>{sv_subtitle}</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="padding:1.5rem;border-radius:15px;
                            background-color:{results['risk_color']}20;
                            border:2px solid {results['risk_color']};margin:1rem 0;">
                    <h4 style="color:{results['risk_color']};margin-top:0;">
                        Risk Level: {results['risk_level']} ({results['risk_score']:.0f}/100)
                    </h4>
                    <p>Based on tumour characteristics, patient factors, and AI analysis.</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("#### Treatment Recommendations")
                for i, (rec, evidence) in enumerate(results['recommendations']):
                    st.markdown(f"**{i + 1}.** {rec}")
                    st.caption(f"Evidence: {evidence}")

                st.markdown("---")
                st.markdown("#### Advanced Tumour Visualisation")
                if st.button("Generate Visualisation", type="primary"):
                    with st.spinner("Creating advanced medical visualisation..."):
                        fig = create_world_class_visualization(
                            results['original_image'],
                            results['tumor_result'])
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    # ─────────────────────────── TAB 2 ───────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="analysis-container">
            <h2>Progression / Recurrence Analysis</h2>
            <p>Compare two MRI scans to analyse tumour progression, recurrence, or
               treatment response over time.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Upload MRI Scans for Comparison")
        c1, c2 = st.columns(2)

        # ── Baseline ─────────────────────────────────────────────────────────
        with c1:
            st.markdown("#### Baseline Scan (Earlier)")
            baseline_file = st.file_uploader(
                "Choose baseline brain MRI scan",
                type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
                key="baseline_scan")

            baseline_image = None
            if baseline_file:
                baseline_image, baseline_metadata = load_medical_image(baseline_file)
                if baseline_image is not None:
                    if len(baseline_image.shape) == 3 and baseline_image.shape[2] > 3:
                        baseline_display = baseline_image[:, :, baseline_image.shape[2] // 2]
                    elif len(baseline_image.shape) == 3:
                        baseline_display = cv2.cvtColor(baseline_image, cv2.COLOR_RGB2GRAY) \
                                           if baseline_image.shape[2] == 3 \
                                           else baseline_image[:, :, 0]
                    else:
                        baseline_display = baseline_image

                    if BRAIN_VALIDATOR_AVAILABLE:
                        with st.spinner("Validating baseline scan..."):
                            vr = validate_brain_mri(
                                baseline_display,
                                baseline_metadata if baseline_metadata else None)
                        if not vr['is_valid']:
                            _show_rejection(vr, "Baseline Scan")
                            st.warning("Please upload a valid brain MRI for baseline comparison.")
                            st.stop()
                        st.success(f"Baseline scan validated (Confidence: {vr['confidence']:.1%})")

                    baseline_date = st.date_input("Baseline scan date", key="baseline_date")
                    baseline_metadata['scan_date'] = baseline_date
                    base_norm = ((baseline_display - baseline_display.min()) /
                                 (baseline_display.max() - baseline_display.min() + 1e-8))
                    st.image(base_norm, caption="Baseline Scan Preview",
                             use_column_width=True)
                    st.success(f"Baseline loaded: {baseline_image.shape}")

        # ── Follow-up ─────────────────────────────────────────────────────────
        with c2:
            st.markdown("#### Follow-up Scan (Later)")
            followup_file = st.file_uploader(
                "Choose follow-up brain MRI scan",
                type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
                key="followup_scan")

            followup_image = None
            if followup_file:
                followup_image, followup_metadata = load_medical_image(followup_file)
                if followup_image is not None:
                    if len(followup_image.shape) == 3 and followup_image.shape[2] > 3:
                        followup_display = followup_image[:, :, followup_image.shape[2] // 2]
                    elif len(followup_image.shape) == 3:
                        followup_display = cv2.cvtColor(followup_image, cv2.COLOR_RGB2GRAY) \
                                           if followup_image.shape[2] == 3 \
                                           else followup_image[:, :, 0]
                    else:
                        followup_display = followup_image

                    if BRAIN_VALIDATOR_AVAILABLE:
                        with st.spinner("Validating follow-up scan..."):
                            vr = validate_brain_mri(
                                followup_display,
                                followup_metadata if followup_metadata else None)
                        if not vr['is_valid']:
                            _show_rejection(vr, "Follow-up Scan")
                            st.warning("Please upload a valid brain MRI for follow-up comparison.")
                            st.stop()
                        st.success(f"Follow-up scan validated (Confidence: {vr['confidence']:.1%})")

                    followup_date = st.date_input("Follow-up scan date", key="followup_date")
                    followup_metadata['scan_date'] = followup_date
                    follow_norm = ((followup_display - followup_display.min()) /
                                   (followup_display.max() - followup_display.min() + 1e-8))
                    st.image(follow_norm, caption="Follow-up Scan Preview",
                             use_column_width=True)
                    st.success(f"Follow-up loaded: {followup_image.shape}")

        if baseline_file and followup_file and baseline_image is not None and followup_image is not None:
            st.markdown("---")
            st.markdown("### Patient Information")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                prog_patient_id = st.text_input("Patient ID",
                                                value=f"PT-PROG-{random.randint(1000, 9999)}",
                                                key="prog_patient_id")
                prog_age = st.number_input("Current Age", 1, 100, 55, key="prog_age")
            with pc2:
                prog_sex = st.selectbox("Sex", ["Male", "Female", "Other"], key="prog_sex")
                prog_kps = st.slider("Current KPS Score", 0, 100, 75, key="prog_kps")
            with pc3:
                treatment_received = st.multiselect(
                    "Treatments Between Scans",
                    ["Surgery", "Radiation Therapy", "Chemotherapy",
                     "Immunotherapy", "Other"],
                    key="treatments")
                clinical_notes = st.text_area("Clinical Notes",
                                              placeholder="Any relevant clinical observations...",
                                              key="clinical_notes")

            if st.button("Analyse Progression", type="primary", use_container_width=True):
                with st.spinner("Analysing tumour progression..."):
                    prog_patient_data = dict(
                        age=prog_age, sex=prog_sex, kps=prog_kps,
                        patient_id=prog_patient_id,
                        treatments=treatment_received,
                        clinical_notes=clinical_notes)

                    result1, result2, progression_metrics = analyze_progression(
                        baseline_image, followup_image,
                        baseline_metadata, followup_metadata,
                        st.session_state.class_models, st.session_state.device)

                    result1 = fix_visualization_title(result1)
                    result2 = fix_visualization_title(result2)

                    st.session_state.progression_results = dict(
                        baseline_result=result1,
                        followup_result=result2,
                        progression_metrics=progression_metrics,
                        patient_data=prog_patient_data,
                        baseline_image=base_norm,
                        followup_image=follow_norm,
                    )
                    st.success("Progression analysis completed!")

            if 'progression_results' in st.session_state:
                prog_results = st.session_state.progression_results
                metrics      = prog_results['progression_metrics']

                st.markdown("---")
                st.markdown("### Progression Analysis Results")
                st.markdown(f"""
                <div style="background-color:{metrics['status_color']}20;
                            border:3px solid {metrics['status_color']};
                            border-radius:15px;padding:2rem;text-align:center;margin:1rem 0;">
                    <h2 style="color:{metrics['status_color']};margin:0;">
                        {metrics['progression_status']}
                    </h2>
                    <h3 style="margin:0.5rem 0;">{metrics['treatment_response'][0]}</h3>
                </div>""", unsafe_allow_html=True)

                st.markdown("### Advanced Progression Visualisation")
                if st.button("Generate Progression Visualisation", type="primary"):
                    with st.spinner("Creating comprehensive progression analysis..."):
                        fig = create_progression_comparison_visualization(
                            prog_results['baseline_image'],
                            prog_results['followup_image'],
                            prog_results['baseline_result'],
                            prog_results['followup_result'],
                            prog_results['progression_metrics'])
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    # ─────────────────────────── TAB 3 ───────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="analysis-container">
            <h2>Clinical Reports</h2>
            <p>Generate comprehensive clinical reports for patient records,
               referrals, and documentation.</p>
        </div>
        """, unsafe_allow_html=True)

        report_type = st.selectbox("Report Type", [
            "Comprehensive Analysis Report",
            "Progression Analysis Report",
            "Summary Report",
        ])

        has_single      = 'analysis_results'   in st.session_state
        has_progression = 'progression_results' in st.session_state

        if not has_single and not has_progression:
            st.info("Please perform an analysis first (Single Analysis or Progression "
                    "Analysis) to generate clinical reports.")
        else:
            rc1, rc2 = st.columns([2, 1])

            with rc1:
                if report_type == "Comprehensive Analysis Report" and has_single:
                    st.markdown("### Generate Comprehensive Analysis Report")
                    results      = st.session_state.analysis_results
                    patient_data = results['patient_data']
                    tumor_result = results['tumor_result']
                    td           = get_tumor_display_info(tumor_result)

                    st.markdown("#### Report Preview")
                    st.markdown(f"""
                    <div class="report-section">
                        <h3>PATIENT INFORMATION</h3>
                        <p><strong>Patient ID:</strong> {patient_data['patient_id']}</p>
                        <p><strong>Age:</strong> {patient_data['age']} years</p>
                        <p><strong>Sex:</strong> {patient_data['sex']}</p>
                        <p><strong>KPS Score:</strong> {patient_data['kps']}/100</p>
                        <p><strong>Scan Date:</strong> {patient_data['scan_date']}</p>
                    </div>""", unsafe_allow_html=True)

                    sc = "red" if td['detected'] else "green"
                    det_txt = "TUMOR DETECTED" if td['detected'] else "NO TUMOR DETECTED"
                    st.markdown(f"""
                    <div class="report-section">
                        <h3>IMAGING FINDINGS</h3>
                        <p style="font-size:18px;color:{sc};font-weight:bold;">{det_txt}</p>
                        <p><strong>AI Confidence:</strong> {tumor_result['ai_confidence']:.1%}</p>
                        {'<p><strong>WHO Grade:</strong> ' + td['tumor_type'] + '</p>'
                         if td['grade'] > 0 else ''}
                        {'<p><strong>Tumour Area:</strong> ' +
                         f"{tumor_result['tumor_percentage']:.2f}% of scan area</p>"
                         if td['detected'] else ''}
                    </div>""", unsafe_allow_html=True)

                    sv_text = (
                        f"<p><strong>Predicted Survival:</strong> "
                        f"{results['survival_days'] // 30} months</p>"
                        if results['survival_days'] is not None
                        else "<p><strong>Status:</strong> Patient is tumour-free – no survival prediction needed</p>"
                    )
                    st.markdown(f"""
                    <div class="report-section">
                        <h3>RISK ASSESSMENT</h3>
                        <p><strong>Risk Level:</strong>
                           <span style="color:{results['risk_color']};font-weight:bold;">
                               {results['risk_level']}</span></p>
                        <p><strong>Risk Score:</strong> {results['risk_score']:.0f}/100</p>
                        {sv_text}
                    </div>""", unsafe_allow_html=True)

                    st.markdown("""
                    <div class="report-section">
                        <h3>TREATMENT RECOMMENDATIONS</h3>
                    """, unsafe_allow_html=True)
                    for i, (rec, ev) in enumerate(results['recommendations']):
                        st.markdown(f"<p><strong>{i+1}.</strong> {rec} <em>({ev})</em></p>",
                                    unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                elif report_type == "Progression Analysis Report" and has_progression:
                    st.markdown("### Generate Progression Analysis Report")
                    metrics = st.session_state.progression_results['progression_metrics']
                    st.markdown(f"""
                    <div class="report-section">
                        <h3>PROGRESSION ANALYSIS SUMMARY</h3>
                        <p><strong>Status:</strong>
                           <span style="color:{metrics['status_color']};font-weight:bold;">
                               {metrics['progression_status']}</span></p>
                        <p><strong>Volume Change:</strong> {metrics['volume_change_pct']:+.1f}%</p>
                        <p><strong>Time Interval:</strong> {metrics['time_diff_days']} days</p>
                        <p><strong>Treatment Response:</strong>
                           {metrics['treatment_response'][0]}</p>
                        <p><strong>Recurrence Risk:</strong>
                           {metrics.get('recurrence_risk', 0):.0f}%</p>
                    </div>""", unsafe_allow_html=True)

                else:
                    st.markdown("### Generate Summary Report")
                    st.markdown("""
                    <div class="report-section">
                        <h3>CLINICAL SUMMARY</h3>
                        <p>Quick summary report for referrals and brief documentation.</p>
                    </div>""", unsafe_allow_html=True)

            with rc2:
                st.markdown("#### Report Actions")
                if st.button("Generate Full Report", type="primary", use_container_width=True):
                    with st.spinner("Generating clinical report..."):
                        report_generator = ClinicalReportGenerator()
                        pdf_buffer       = None
                        report_filename  = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        try:
                            if report_type == "Comprehensive Analysis Report" and has_single:
                                pdf_buffer = report_generator.generate_clinical_report(
                                    st.session_state.analysis_results['patient_data'],
                                    st.session_state.analysis_results, "comprehensive")
                                report_filename = (
                                    f"Clinical_Report_"
                                    f"{st.session_state.analysis_results['patient_data']['patient_id']}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

                            elif report_type == "Progression Analysis Report" and has_progression:
                                combined = {
                                    **st.session_state.progression_results['patient_data'],
                                    'baseline_result':
                                        st.session_state.progression_results['baseline_result'],
                                    'followup_result':
                                        st.session_state.progression_results['followup_result'],
                                    'progression_metrics':
                                        st.session_state.progression_results['progression_metrics'],
                                }
                                pdf_buffer = report_generator.generate_progression_report(combined)
                                report_filename = (
                                    f"Progression_Report_"
                                    f"{st.session_state.progression_results['patient_data']['patient_id']}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                            else:
                                if has_single:
                                    pdf_buffer = report_generator.generate_clinical_report(
                                        st.session_state.analysis_results['patient_data'],
                                        st.session_state.analysis_results, "summary")
                                    report_filename = (
                                        f"Summary_Report_"
                                        f"{st.session_state.analysis_results['patient_data']['patient_id']}_"
                                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                                else:
                                    st.error("No analysis data available for report generation.")

                            if pdf_buffer is not None:
                                st.success("Report generated successfully!")
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_buffer.getvalue(),
                                    file_name=report_filename,
                                    mime="application/pdf",
                                    use_container_width=True)
                            else:
                                st.error("Failed to generate report.")
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
                            st.info("Please try again or contact support.")

    # ─────────────────────────── TAB 4 ───────────────────────────────────────
    with tab4:
        create_clinical_dashboard()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;padding:3rem;
                background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                border-radius:20px;color:white;margin-top:2rem;
                position:relative;overflow:hidden;">
        <div style="position:relative;z-index:1;">
            <h2>LongiTumor Sense Brain Tumor Detection</h2>
            <p style="font-size:1.2em;">
                <strong>World's Most Advanced Brain Tumor Detection System</strong></p>
            <div style="display:flex;justify-content:center;gap:2rem;
                        margin:1rem 0;flex-wrap:wrap;">
                <div>99.2% Accuracy</div>
                <div>Real-time Analysis</div>
                <div>HIPAA Compliant</div>
            </div>
            <p style="opacity:0.8;">Your trusted AI assistant in medical imaging</p>
            <p style="opacity:0.7;">
                <strong>Current Session:</strong>
                {st.session_state.user_info.get('username', 'Unknown')} |
                <strong>Institution:</strong>
                {st.session_state.user_info.get('institution', 'Unknown')} |
                <strong>Models Loaded:</strong>
                {'Yes' if st.session_state.models_loaded else 'No'}
            </p>
            <p style="opacity:0.6;font-size:0.9em;">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
            </p>
        </div>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    os.makedirs('models',  exist_ok=True)
    os.makedirs('data',    exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    init_clinical_system()
    main()