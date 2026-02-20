# <<<<<<< HEAD
# # config.py

# """
# NeuroVision AI Configuration File
# Fully compatible with models.py, inference.py, preprocessing.py, and helper.py
# """
# import os
# import json
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Any
# import torch
# import hashlib
# import sys


# def fix_windows_logging():
#     """Fix Unicode logging issues on Windows"""
#     if sys.platform == "win32":
#         # Force UTF-8 encoding
#         import io
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
#         sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
#     # Configure logging WITHOUT emoji characters
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(sys.stdout),
#             logging.FileHandler('app.log', encoding='utf-8', errors='replace')
#         ]
#     )


# # Call this at module level
# fix_windows_logging()


# # Set up logging
# logger = logging.getLogger(__name__)


# def setup_windows_logging():
#     """Fix Unicode logging issues on Windows"""
#     if sys.platform == "win32":
#         # Force UTF-8 encoding in Windows console
#         if hasattr(sys.stdout, 'reconfigure'):
#             sys.stdout.reconfigure(encoding='utf-8')
#         if hasattr(sys.stderr, 'reconfigure'):
#             sys.stderr.reconfigure(encoding='utf-8')
    
#     # Configure logging to handle Unicode
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(sys.stdout),
#             logging.FileHandler('app.log', encoding='utf-8')
#         ]
#     )


# def get_safe_logger(name):
#     """Get a logger that safely handles Unicode characters"""
#     return logging.getLogger(name)


# # ========================= SYSTEM CONFIGURATION =========================
# # Application Settings
# APP_NAME = "LongiTumorSense Brain Tumor Detection"
# APP_VERSION = "5.0"
# DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# # Directory Structure - Compatible with all modules
# BASE_DIR = Path(__file__).parent
# DATA_DIR = BASE_DIR / "data"
# MODELS_DIR = BASE_DIR / "models"
# LOGS_DIR = BASE_DIR / "logs"
# REPORTS_DIR = BASE_DIR / "reports"
# TEMP_DIR = BASE_DIR / "temp"

# # Ensure directories exist
# for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, TEMP_DIR]:
#     directory.mkdir(exist_ok=True)

# # Database Configuration
# DATABASE_CONFIG = {
#     'clinical_db_path': DATA_DIR / "clinical_database.json",
#     'user_sessions_path': DATA_DIR / "user_sessions.json",
#     'system_logs_path': LOGS_DIR / "system.log",
#     'backup_interval_hours': 24,
#     'max_backup_files': 7
# }

# # ========================= UI CONFIGURATION =========================
# UI_CONFIG = {
#     'page_config': {
#         'page_title': 'LongiTumorSense Brain Tumor Detection',
#         'page_icon': '',  # Removed icon
#         'layout': 'wide',
#         'initial_sidebar_state': 'expanded'
#     },
#     'theme': {
#         'primary_color': '#1E88E5',
#         'secondary_color': '#FF5252',
#         'background_color': '#F5F5F5',
#         'text_color': '#212121',
#         'success_color': '#4CAF50',
#         'warning_color': '#FFC107',
#         'error_color': '#F44336',
#         'font_family': 'Arial, sans-serif',
#         'font_size': '14px'
#     },
#     'layout': {
#         'sidebar_width': '280px',
#         'main_content_max_width': '1200px',
#         'spacing_unit': '8px',
#         'border_radius': '8px',
#         'shadow': '0 2px 10px rgba(0,0,0,0.1)'
#     },
#     'components': {
#         'header_height': '60px',
#         'footer_height': '40px',
#         'button_height': '36px',
#         'input_height': '40px',
#         'card_padding': '16px'
#     },
#     'animations': {
#         'transition_speed': '0.3s',
#         'hover_effect': True,
#         'loading_animation': True
#     },
#     'features': {
#         'dark_mode': True,
#         'high_contrast_mode': False,
#         'font_scaling': True,
#         'voice_commands': False
#     }
# }

# # ========================= SAMPLE USERS CONFIGURATION =========================
# # Helper function to hash passwords - Compatible with helper.py
# def _hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()


# SAMPLE_USERS = {
#     'hassan': {
#         'username': 'hassan',
#         'password': 'hassan123!',
#         'password_hash': _hash_password('hassan123!'),
#         'role': 'radiologist',
#         'full_name': 'Dr. Hassan',
#         'email': 'hassan@hospital.com',
#         'department': 'Radiology',
#         'license': 'MD123456',
#         'institution': 'Mayo Clinic',
#         'permissions': ['view:all', 'analyze:all', 'report:all', 'export:all'],
#         'is_active': True,
#         'last_login': None
#     },
#     'radiology.ai': {
#         'username': 'radiology.ai',
#         'password': 'RadAI456!',
#         'password_hash': _hash_password('RadAI456!'),
#         'role': 'radiologist',
#         'full_name': 'Radiology AI System',
#         'email': 'ai@radiology.com',
#         'department': 'Radiology',
#         'license': 'MD654321',
#         'institution': 'Mass General Hospital',
#         'permissions': ['view:all', 'analyze:all', 'report:all', 'export:all'],
#         'is_active': True,
#         'last_login': None
#     },
#     'brain.surgeon': {
#         'username': 'brain.surgeon',
#         'password': 'Surgery789!',
#         'password_hash': _hash_password('Surgery789!'),
#         'role': 'neurosurgeon',
#         'full_name': 'Brain Surgeon',
#         'email': 'surgeon@hospital.com',
#         'department': 'Neurosurgery',
#         'license': 'MD987654',
#         'institution': 'Johns Hopkins Hospital',
#         'permissions': ['view:all', 'analyze:all', 'report:view'],
#         'is_active': True,
#         'last_login': None
#     },
#     'demo': {
#         'username': 'demo',
#         'password': 'demo123',
#         'password_hash': _hash_password('demo123'),
#         'role': 'clinician',
#         'full_name': 'Demo User',
#         'email': 'demo@neurovision.ai',
#         'department': 'Demo',
#         'license': 'MD999999',
#         'institution': 'Demo Hospital',
#         'permissions': ['view:all', 'analyze:all', 'report:view'],
#         'is_active': True,
#         'last_login': None
#     }
# }

# # ========================= AUTHENTICATION CONFIGURATION =========================
# AUTHENTICATION_CONFIG = {
#     'secret_key': os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
#     'algorithm': 'HS256',
#     'access_token_expire_minutes': 120,
#     'password_hashing': {
#         'algorithm': 'sha256',
#         'rounds': 12
#     },
#     'user_roles': {
#         'radiologist': ['view:all', 'analyze:all', 'report:all', 'export:all'],
#         'clinician': ['view:all', 'analyze:all', 'report:view'],
#         'technician': ['upload', 'view:own', 'analyze:own'],
#         'admin': ['manage:users', 'manage:system', 'view:audit', 'manage:models'],
#         'neurosurgeon': ['view:all', 'analyze:all', 'report:view']
#     },
#     'session_timeout_minutes': 30,
#     'max_login_attempts': 5,
#     'lockout_duration_minutes': 15,
#     'authorized_institutions': [
#         'Mayo Clinic',
#         'Johns Hopkins Hospital',
#         'Massachusetts General Hospital',
#         'Mass General Hospital',
#         'Cleveland Clinic',
#         'UCLA Medical Center',
#         'NYU Langone Health',
#         'Stanford Health Care',
#         'Mount Sinai Hospital',
#         'Northwestern Memorial Hospital',
#         'Demo Hospital'
#     ],
#     'medical_specialties': [
#         'Radiology',
#         'Neurology',
#         'Neurosurgery',
#         'Oncology',
#         'Radiation Oncology',
#         'Pathology',
#         'Emergency Medicine',
#         'Internal Medicine',
#         'Family Medicine',
#         'Research'
#     ]
# }

# # For backward compatibility with existing code
# AUTH_CONFIG = AUTHENTICATION_CONFIG

# # ========================= MODEL CONFIGURATION =========================
# # Device Configuration - Compatible with models.py and helper.py
# DEVICE_CONFIG = {
#     'use_cuda': torch.cuda.is_available(),
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     'mixed_precision': True,
#     'num_workers': min(8, os.cpu_count() or 1),
#     'default_device': 'cuda' if torch.cuda.is_available() else 'cpu'  # For helper.py compatibility
# }

# # Model Architecture Settings - Compatible with unified models.py
# MODEL_CONFIG = {
#     'segmentation': {
#         'in_channels': 1,
#         'out_channels': 4,
#         'use_attention': True,
#         'dropout_rate': 0.3,
#         'use_deep_supervision': True,
#         'base_channels': 64,
#         'encoder_name': 'efficientnet-b4',  # For BrainTumorSegmenter
#         'num_classes': 4
#     },
#     'classification': {
#         'num_classes': 4,
#         'backbones': ['efficientnet', 'resnet', 'densenet'],
#         'default_backbone': 'efficientnet',
#         'clinical_feature_size': 10,
#         'dropout_rate': 0.4,
#         'freeze_backbone': False,
#         'pretrained': True,
#         'base_filters': 32  # For BraTS models
#     },
#     'ensemble': {
#         'weights': {
#             'brats_segmentation': 0.35,
#             'brats_classification': 0.25,
#             'detector': 0.20,
#             'glioma_classifier': 0.15,
#             'segmenter': 0.05
#         }
#     },
#     'checkpoint_paths': {
#         'segmentation': MODELS_DIR / "brain_segmenter.pth",
#         'detector': MODELS_DIR / "brain_tumor_detector.pth",
#         'brats_segmentation': MODELS_DIR / "brats_segmentation.pth",
#         'brats_classification': MODELS_DIR / "brats_classification.pth"
#     }
# }

# # Model Paths for pre-trained models - Compatible with helper.py
# MODEL_PATHS = {
#     'segmentation_model': MODELS_DIR / "unet_brain_tumor.pth",
#     'classification_efficientnet': MODELS_DIR / "efficientnet_brain_tumor.pth",
#     'classification_resnet': MODELS_DIR / "resnet_brain_tumor.pth",
#     'classification_densenet': MODELS_DIR / "densenet_brain_tumor.pth",
#     # Unified model paths
#     'brain_tumor_detector': MODELS_DIR / "brain_tumor_detector.pth",
#     'brain_segmenter': MODELS_DIR / "brain_segmenter.pth"
# }

# # Training Configuration (for future model training)
# TRAINING_CONFIG = {
#     'batch_size': 8,
#     'learning_rate': 1e-4,
#     'weight_decay': 1e-5,
#     'max_epochs': 100,
#     'patience': 15,
#     'scheduler': 'cosine',
#     'optimizer': 'adamw',
#     'loss_functions': {
#         'segmentation': 'dice_focal',
#         'classification': 'cross_entropy',
#         'survival': 'concordance'
#     }
# }

# # ========================= IMAGE PROCESSING CONFIGURATION =========================
# # Image Processing Parameters - Compatible with preprocessing.py
# IMAGE_CONFIG = {
#     'supported_formats': ['.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz'],
#     'target_size': (224, 224),  # For preprocess_for_model
#     'voxel_size': (1.0, 1.0, 1.0),
#     'normalize_intensity': True,
#     'clip_percentile': (1, 99),
#     'normalization_methods': ['zscore', 'percentile', 'minmax'],  # For normalize_volume
#     'enhancement_methods': {
#         'clahe': {
#             'clip_limit': 2.0,
#             'tile_grid_size': (8, 8)
#         },
#         'histogram_equalization': True,
#         'gamma_correction': 1.2,
#         'noise_reduction': True
#     }
# }

# # Preprocessing Pipeline - Compatible with preprocessing.py
# PREPROCESSING_CONFIG = {
#     'skull_stripping': False,
#     'bias_field_correction': False,
#     'registration_template': None,
#     'resampling_method': 'linear',
#     'augmentation': {
#         'rotation_range': 10,
#         'zoom_range': 0.1,
#         'shift_range': 0.1,
#         'flip_probability': 0.5
#     },
#     'brain_extraction_threshold': 0.1,  # For extract_brain_region
#     'small_component_min_size': 100    # For remove_small_components
# }

# # ========================= CLINICAL CONFIGURATION =========================
# # Tumor Classification - Compatible with inference.py and helper.py
# TUMOR_CLASSES = {
#     0: {
#         'name': 'Normal',
#         'description': 'No tumor detected',
#         'color': 'green',
#         'risk_level': 'Normal',
#         'follow_up_months': 12,
#         'survival_estimate_days': 3650
#     },
#     1: {
#         'name': 'Grade I Glioma',
#         'description': 'Low-grade glioma',
#         'color': 'yellow',
#         'risk_level': 'Low',
#         'follow_up_months': 6,
#         'survival_estimate_days': 2555
#     },
#     2: {
#         'name': 'Grade II-III Glioma',
#         'description': 'Intermediate-grade glioma',
#         'color': 'orange',
#         'risk_level': 'Intermediate',
#         'follow_up_months': 3,
#         'survival_estimate_days': 1825
#     },
#     3: {
#         'name': 'Grade IV Glioma',
#         'description': 'High-grade glioma (GBM)',
#         'color': 'red',
#         'risk_level': 'High',
#         'follow_up_months': 1,
#         'survival_estimate_days': 450
#     }
# }

# # Function for inference.py compatibility
# def get_tumor_info(grade: int) -> Dict:
#     """Get tumor information for a given grade - Used in inference.py"""
#     return TUMOR_CLASSES.get(grade, TUMOR_CLASSES[0])


# # For backward compatibility with existing inference code
# CLINICAL_CONFIG = TUMOR_CLASSES

# # RANO Criteria (separate variable for easy import)
# RANO_CRITERIA = {
#     'complete_response': {
#         'description': 'Disappearance of all enhancing disease, stable/improved non-enhancing FLAIR lesions, off corticosteroids, clinically stable/improved.',
#         'requirements': ['no_enhancing_lesion', 'stable_flair', 'no_steroids', 'stable_neurology']
#     },
#     'partial_response': {
#         'description': '>=50% decrease in sum of enhancing lesion products compared to baseline, stable/improved non-enhancing FLAIR lesions, on stable/decreasing corticosteroids, clinically stable/improved.',
#         'requirements': ['50_percent_reduction_enhancing', 'stable_flair', 'stable_steroids', 'stable_neurology']
#     },
#     'stable_disease': {
#         'description': 'Does not qualify for CR, PR, or PD.',
#         'requirements': ['not_cr_pr_pd']
#     },
#     'progressive_disease': {
#         'description': '>=25% increase in sum of enhancing lesion products compared to nadir, significant increase in non-enhancing FLAIR lesions, new lesions, clinical deterioration, or increased corticosteroid dose.',
#         'requirements': ['25_percent_increase_enhancing', 'or', 'significant_increase_flair', 'or', 'new_lesion', 'or', 'clinical_deterioration']
#     }
# }

# # Clinical Protocols and RANO Criteria
# CLINICAL_PROTOCOLS = {
#     'rano_criteria': RANO_CRITERIA,
#     'treatment_guidelines': {
#         'Grade I Glioma': ['Surgical resection', 'Observation'],
#         'Grade II-III Glioma': ['Maximal safe resection', 'Radiotherapy', 'Chemotherapy (PCV or Temozolomide)'],
#         'Grade IV Glioma': ['Maximal safe resection', 'Radiotherapy with concurrent and adjuvant Temozolomide', 'Tumor Treating Fields (TTFields)']
#     },
#     'reporting_metrics': ['volume_enhancing', 'volume_non_enhancing', 'volume_necrosis', 'largest_diameter_axial', 'rankscale_performance']
# }

# # Survival estimates by grade - Compatible with inference.py
# SURVIVAL_ESTIMATES = {
#     'grade_0': 3650,
#     'grade_1': 2555,
#     'grade_2': 1825,
#     'grade_3': 450
# }

# # ========================= INFERENCE CONFIGURATION =========================
# # Inference settings compatible with inference.py
# INFERENCE_CONFIG = {
#     'ensemble_weights': {
#         'brats_segmentation': 0.35,
#         'brats_classification': 0.25,
#         'detector': 0.20,
#         'glioma_classifier': 0.15,
#         'segmenter': 0.05
#     },
#     'preprocessing': {
#         'target_sizes': {
#             'standard': (224, 224),
#             'highres': (384, 384),
#             'segmentation': (256, 256),
#             'brats_2d': (224, 224)
#         }
#     },
#     'clinical_feature_mapping': {
#         'idh_status': {'Unknown': 0, 'Mutant': 1, 'Wildtype': 2},
#         'mgmt_status': {'Unknown': 0, 'Methylated': 1, 'Unmethylated': 2}
#     }
# }

# # ========================= PERFORMANCE CONFIGURATION =========================
# PERFORMANCE_CONFIG = {
#     'inference': {
#         'batch_size': 4,
#         'max_cache_size_gb': 2,
#         'preload_models': True,
#         'gpu_memory_fraction': 0.8,
#         'sliding_window': {  # For helper.py compatibility
#             'patch_size': (64, 64, 64),
#             'overlap': 0.5,
#             'batch_size': 1
#         }
#     },
#     'memory_management': {
#         'clear_cache_after_analysis': True,
#         'max_concurrent_analyses': 2,
#         'temp_file_cleanup_hours': 1
#     },
#     'optimization': {
#         'use_tensorrt': False,
#         'precision': 'fp16',
#         'threads': 4
#     },
#     'monitoring': {
#         'log_latency': True,
#         'log_memory_usage': True,
#         'performance_report_interval': 24
#     }
# }

# # ========================= EXPORT & REPORTING CONFIGURATION =========================
# REPORTING_CONFIG = {
#     'default_format': 'pdf',
#     'include_confidence_scores': True,
#     'anonymize_data': True,
#     'dicom_sr_template': 'tumor_measurement_report.xml',
#     'hospital_logo_path': BASE_DIR / "assets" / "hospital_logo.png",
#     'signature_required': True
# }

# # ========================= ANALYSIS THRESHOLDS =========================
# ANALYSIS_THRESHOLDS = {
#     'min_confidence_score': 0.6,
#     'max_uncertainty_score': 0.4,
#     'min_tumor_volume_mm3': 50,
#     'significant_growth_percentage': 25,
#     'significant_shrinkage_percentage': 50,
#     'ensemble_agreement_threshold': 0.7  # For inference.py
# }

# # ========================= EXPORT CONFIGURATION =========================
# EXPORT_CONFIG = {
#     'dicom_export': {
#         'include_original_images': True,
#         'include_segmentation_masks': True,
#         'include_measurements': True,
#         'compression_quality': 90
#     },
#     'pdf_export': {
#         'include_all_slices': False,
#         'key_slices_count': 12,
#         'include_statistics': True,
#         'include_recommendations': True
#     },
#     'csv_export': {
#         'include_volumetrics': True,
#         'include_radiomics': True,
#         'include_confidence_scores': True
#     }
# }

# # ========================= HELPER.PY COMPATIBILITY =========================
# # Pre-trained model URLs for helper.py
# PRETRAINED_MODELS = {
#     'segmentation': {
#         'model_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/unet_brain_tumor.pth',
#         'input_size': (128, 128, 128),
#         'num_classes': 4
#     },
#     'classification': {
#         'efficientnet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/efficientnet_brain_tumor.pth',
#         'resnet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/resnet_brain_tumor.pth',
#         'densenet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/densenet_brain_tumor.pth'
#     }
# }

# # ========================= UTILITY FUNCTIONS =========================
# def get_rano_criteria() -> Dict:
#     """Get RANO criteria"""
#     return RANO_CRITERIA


# def get_clinical_config() -> Dict:
#     """Get clinical configuration"""
#     return CLINICAL_CONFIG


# def get_device_config() -> Dict:
#     """Get device configuration"""
#     return DEVICE_CONFIG


# def get_model_config() -> Dict:
#     """Get model configuration"""
#     return MODEL_CONFIG


# def get_ui_config() -> Dict:
#     """Get UI configuration"""
#     return UI_CONFIG


# def get_auth_config() -> Dict:
#     """Get authentication configuration"""
#     return AUTH_CONFIG


# def get_sample_users() -> Dict:
#     """Get sample users configuration"""
#     return SAMPLE_USERS


# def get_inference_config() -> Dict:
#     """Get inference configuration"""
#     return INFERENCE_CONFIG


# def validate_config() -> List[str]:
#     """
#     Validate the configuration settings
#     Returns: List of error messages (empty if valid)
#     """
#     errors = []
    
#     # Check if required directories exist
#     required_dirs = [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, TEMP_DIR]
#     for directory in required_dirs:
#         if not directory.exists():
#             try:
#                 directory.mkdir(parents=True, exist_ok=True)
#             except Exception as e:
#                 errors.append(f"Failed to create directory {directory}: {e}")
    
#     # Validate tumor classes
#     if not TUMOR_CLASSES:
#         errors.append("TUMOR_CLASSES configuration is empty")
    
#     # Validate device configuration
#     if not DEVICE_CONFIG.get('device'):
#         errors.append("Device configuration is missing")
    
#     # Validate authentication configuration
#     if not AUTH_CONFIG.get('secret_key'):
#         errors.append("Authentication secret key is missing")
    
#     # Validate UI configuration
#     if not UI_CONFIG:
#         errors.append("UI configuration is missing")
    
#     # Validate sample users
#     if not SAMPLE_USERS:
#         errors.append("SAMPLE_USERS configuration is empty")
    
#     return errors


# def save_config(file_path: Path = None) -> bool:
#     """Save current configuration to a file"""
#     if file_path is None:
#         file_path = BASE_DIR / "config_backup.json"
    
#     try:
#         config_dict = {
#             'APP_NAME': APP_NAME,
#             'APP_VERSION': APP_VERSION,
#             'DEBUG_MODE': DEBUG_MODE,
#             'DATABASE_CONFIG': {k: str(v) if isinstance(v, Path) else v for k, v in DATABASE_CONFIG.items()},
#             'UI_CONFIG': UI_CONFIG,
#             'AUTH_CONFIG': AUTH_CONFIG,
#             'DEVICE_CONFIG': DEVICE_CONFIG,
#             'MODEL_CONFIG': MODEL_CONFIG,
#             'MODEL_PATHS': {k: str(v) for k, v in MODEL_PATHS.items()},
#             'IMAGE_CONFIG': IMAGE_CONFIG,
#             'TUMOR_CLASSES': TUMOR_CLASSES,
#             'CLINICAL_PROTOCOLS': CLINICAL_PROTOCOLS,
#             'PERFORMANCE_CONFIG': PERFORMANCE_CONFIG,
#             'REPORTING_CONFIG': {k: str(v) if isinstance(v, Path) else v for k, v in REPORTING_CONFIG.items()}
#         }
        
#         with open(file_path, 'w') as f:
#             json.dump(config_dict, f, indent=2)
        
#         logger.info(f"Configuration saved to {file_path}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Failed to save configuration: {e}")
#         return False


# def load_config(file_path: Path) -> bool:
#     """Load configuration from a file"""
#     try:
#         if not file_path.exists():
#             logger.error(f"Config file {file_path} does not exist")
#             return False
        
#         with open(file_path, 'r') as f:
#             config_dict = json.load(f)
        
#         # Update global variables (this is a simplified version)
#         # In a real implementation, you'd want to be more careful about this
#         for key, value in config_dict.items():
#             globals()[key] = value
        
#         logger.info(f"Configuration loaded from {file_path}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Failed to load configuration: {e}")
#         return False


# def get_config_summary() -> Dict[str, Any]:
#     """Get a summary of the current configuration"""
#     return {
#         'app_info': {
#             'name': APP_NAME,
#             'version': APP_VERSION,
#             'debug_mode': DEBUG_MODE
#         },
#         'system': {
#             'device': DEVICE_CONFIG.get('device', 'unknown'),
#             'cuda_available': torch.cuda.is_available(),
#             'model_count': len(MODEL_PATHS)
#         },
#         'paths': {
#             'data_dir': str(DATA_DIR),
#             'models_dir': str(MODELS_DIR),
#             'logs_dir': str(LOGS_DIR)
#         },
#         'models': list(MODEL_PATHS.keys()),
#         'validation_errors': validate_config()
#     }


# # Auto-validate configuration on import
# config_errors = validate_config()
# if config_errors:
#     logger.warning(f"Configuration validation failed with {len(config_errors)} errors:")
#     for error in config_errors:
#         logger.warning(f" - {error}")
# else:
#     logger.info("Configuration validation passed")

# # Initialize Windows logging if needed
# if sys.platform == "win32":
#     setup_windows_logging()

# config.py

"""
NeuroVision AI Configuration File
Fully compatible with models.py, inference.py, preprocessing.py, and helper.py
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import hashlib
import sys


def fix_windows_logging():
    """Fix Unicode logging issues on Windows"""
    if sys.platform == "win32":
        # Force UTF-8 encoding
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Configure logging WITHOUT emoji characters
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', encoding='utf-8', errors='replace')
        ]
    )


# Call this at module level
fix_windows_logging()


# Set up logging
logger = logging.getLogger(__name__)


def setup_windows_logging():
    """Fix Unicode logging issues on Windows"""
    if sys.platform == "win32":
        # Force UTF-8 encoding in Windows console
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    
    # Configure logging to handle Unicode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', encoding='utf-8')
        ]
    )


def get_safe_logger(name):
    """Get a logger that safely handles Unicode characters"""
    return logging.getLogger(name)


# ========================= SYSTEM CONFIGURATION =========================
# Application Settings
APP_NAME = "LongiTumorSense Brain Tumor Detection"
APP_VERSION = "5.0"
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Directory Structure - Compatible with all modules
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Database Configuration
DATABASE_CONFIG = {
    'clinical_db_path': DATA_DIR / "clinical_database.json",
    'user_sessions_path': DATA_DIR / "user_sessions.json",
    'system_logs_path': LOGS_DIR / "system.log",
    'backup_interval_hours': 24,
    'max_backup_files': 7
}

# ========================= UI CONFIGURATION =========================
UI_CONFIG = {
    'page_config': {
        'page_title': 'LongiTumorSense Brain Tumor Detection',
        'page_icon': '',  # Removed icon
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    },
    'theme': {
        'primary_color': '#1E88E5',
        'secondary_color': '#FF5252',
        'background_color': '#F5F5F5',
        'text_color': '#212121',
        'success_color': '#4CAF50',
        'warning_color': '#FFC107',
        'error_color': '#F44336',
        'font_family': 'Arial, sans-serif',
        'font_size': '14px'
    },
    'layout': {
        'sidebar_width': '280px',
        'main_content_max_width': '1200px',
        'spacing_unit': '8px',
        'border_radius': '8px',
        'shadow': '0 2px 10px rgba(0,0,0,0.1)'
    },
    'components': {
        'header_height': '60px',
        'footer_height': '40px',
        'button_height': '36px',
        'input_height': '40px',
        'card_padding': '16px'
    },
    'animations': {
        'transition_speed': '0.3s',
        'hover_effect': True,
        'loading_animation': True
    },
    'features': {
        'dark_mode': True,
        'high_contrast_mode': False,
        'font_scaling': True,
        'voice_commands': False
    }
}

# ========================= SAMPLE USERS CONFIGURATION =========================
# Helper function to hash passwords - Compatible with helper.py
def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


SAMPLE_USERS = {
    'hassan': {
        'username': 'hassan',
        'password': 'hassan123!',
        'password_hash': _hash_password('hassan123!'),
        'role': 'radiologist',
        'full_name': 'Dr. Hassan',
        'email': 'hassan@hospital.com',
        'department': 'Radiology',
        'license': 'MD123456',
        'institution': 'Mayo Clinic',
        'permissions': ['view:all', 'analyze:all', 'report:all', 'export:all'],
        'is_active': True,
        'last_login': None
    },
    'radiology.ai': {
        'username': 'radiology.ai',
        'password': 'RadAI456!',
        'password_hash': _hash_password('RadAI456!'),
        'role': 'radiologist',
        'full_name': 'Radiology AI System',
        'email': 'ai@radiology.com',
        'department': 'Radiology',
        'license': 'MD654321',
        'institution': 'Mass General Hospital',
        'permissions': ['view:all', 'analyze:all', 'report:all', 'export:all'],
        'is_active': True,
        'last_login': None
    },
    'brain.surgeon': {
        'username': 'brain.surgeon',
        'password': 'Surgery789!',
        'password_hash': _hash_password('Surgery789!'),
        'role': 'neurosurgeon',
        'full_name': 'Brain Surgeon',
        'email': 'surgeon@hospital.com',
        'department': 'Neurosurgery',
        'license': 'MD987654',
        'institution': 'Johns Hopkins Hospital',
        'permissions': ['view:all', 'analyze:all', 'report:view'],
        'is_active': True,
        'last_login': None
    },
    'demo': {
        'username': 'demo',
        'password': 'demo123',
        'password_hash': _hash_password('demo123'),
        'role': 'clinician',
        'full_name': 'Demo User',
        'email': 'demo@neurovision.ai',
        'department': 'Demo',
        'license': 'MD999999',
        'institution': 'Demo Hospital',
        'permissions': ['view:all', 'analyze:all', 'report:view'],
        'is_active': True,
        'last_login': None
    }
}

# ========================= AUTHENTICATION CONFIGURATION =========================
AUTHENTICATION_CONFIG = {
    'secret_key': os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
    'algorithm': 'HS256',
    'access_token_expire_minutes': 120,
    'password_hashing': {
        'algorithm': 'sha256',
        'rounds': 12
    },
    'user_roles': {
        'radiologist': ['view:all', 'analyze:all', 'report:all', 'export:all'],
        'clinician': ['view:all', 'analyze:all', 'report:view'],
        'technician': ['upload', 'view:own', 'analyze:own'],
        'admin': ['manage:users', 'manage:system', 'view:audit', 'manage:models'],
        'neurosurgeon': ['view:all', 'analyze:all', 'report:view']
    },
    'session_timeout_minutes': 30,
    'max_login_attempts': 5,
    'lockout_duration_minutes': 15,
    'authorized_institutions': [
        'Mayo Clinic',
        'Johns Hopkins Hospital',
        'Massachusetts General Hospital',
        'Mass General Hospital',
        'Cleveland Clinic',
        'UCLA Medical Center',
        'NYU Langone Health',
        'Stanford Health Care',
        'Mount Sinai Hospital',
        'Northwestern Memorial Hospital',
        'Demo Hospital'
    ],
    'medical_specialties': [
        'Radiology',
        'Neurology',
        'Neurosurgery',
        'Oncology',
        'Radiation Oncology',
        'Pathology',
        'Emergency Medicine',
        'Internal Medicine',
        'Family Medicine',
        'Research'
    ]
}

# For backward compatibility with existing code
AUTH_CONFIG = AUTHENTICATION_CONFIG

# ========================= MODEL CONFIGURATION =========================
# Device Configuration - Compatible with models.py and helper.py
DEVICE_CONFIG = {
    'use_cuda': torch.cuda.is_available(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    'num_workers': min(8, os.cpu_count() or 1),
    'default_device': 'cuda' if torch.cuda.is_available() else 'cpu'  # For helper.py compatibility
}

# Model Architecture Settings - Compatible with unified models.py
MODEL_CONFIG = {
    'segmentation': {
        'in_channels': 1,
        'out_channels': 4,
        'use_attention': True,
        'dropout_rate': 0.3,
        'use_deep_supervision': True,
        'base_channels': 64,
        'encoder_name': 'efficientnet-b4',  # For BrainTumorSegmenter
        'num_classes': 4
    },
    'classification': {
        'num_classes': 4,
        'backbones': ['efficientnet', 'resnet', 'densenet'],
        'default_backbone': 'efficientnet',
        'clinical_feature_size': 10,
        'dropout_rate': 0.4,
        'freeze_backbone': False,
        'pretrained': True,
        'base_filters': 32  # For BraTS models
    },
    'ensemble': {
        'weights': {
            'brats_segmentation': 0.35,
            'brats_classification': 0.25,
            'detector': 0.20,
            'glioma_classifier': 0.15,
            'segmenter': 0.05
        }
    },
    'checkpoint_paths': {
        'segmentation': MODELS_DIR / "brain_segmenter.pth",
        'detector': MODELS_DIR / "brain_tumor_detector.pth",
        'brats_segmentation': MODELS_DIR / "brats_segmentation.pth",
        'brats_classification': MODELS_DIR / "brats_classification.pth"
    }
}

# Model Paths for pre-trained models - Compatible with helper.py
MODEL_PATHS = {
    'segmentation_model': MODELS_DIR / "unet_brain_tumor.pth",
    'classification_efficientnet': MODELS_DIR / "efficientnet_brain_tumor.pth",
    'classification_resnet': MODELS_DIR / "resnet_brain_tumor.pth",
    'classification_densenet': MODELS_DIR / "densenet_brain_tumor.pth",
    # Unified model paths
    'brain_tumor_detector': MODELS_DIR / "brain_tumor_detector.pth",
    'brain_segmenter': MODELS_DIR / "brain_segmenter.pth"
}

# Training Configuration (for future model training)
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'max_epochs': 100,
    'patience': 15,
    'scheduler': 'cosine',
    'optimizer': 'adamw',
    'loss_functions': {
        'segmentation': 'dice_focal',
        'classification': 'cross_entropy',
        'survival': 'concordance'
    }
}

# ========================= IMAGE PROCESSING CONFIGURATION =========================
# Image Processing Parameters - Compatible with preprocessing.py
IMAGE_CONFIG = {
    'supported_formats': ['.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz'],
    'target_size': (224, 224),  # For preprocess_for_model
    'voxel_size': (1.0, 1.0, 1.0),
    'normalize_intensity': True,
    'clip_percentile': (1, 99),
    'normalization_methods': ['zscore', 'percentile', 'minmax'],  # For normalize_volume
    'enhancement_methods': {
        'clahe': {
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        },
        'histogram_equalization': True,
        'gamma_correction': 1.2,
        'noise_reduction': True
    }
}

# Preprocessing Pipeline - Compatible with preprocessing.py
PREPROCESSING_CONFIG = {
    'skull_stripping': False,
    'bias_field_correction': False,
    'registration_template': None,
    'resampling_method': 'linear',
    'augmentation': {
        'rotation_range': 10,
        'zoom_range': 0.1,
        'shift_range': 0.1,
        'flip_probability': 0.5
    },
    'brain_extraction_threshold': 0.1,  # For extract_brain_region
    'small_component_min_size': 100    # For remove_small_components
}

# ========================= CLINICAL CONFIGURATION =========================
# Tumor Classification - Compatible with inference.py and helper.py
TUMOR_CLASSES = {
    0: {
        'name': 'Normal',
        'description': 'No tumor detected',
        'color': 'green',
        'risk_level': 'Normal',
        'follow_up_months': 12,
        'survival_estimate_days': 3650
    },
    1: {
        'name': 'Grade I Glioma',
        'description': 'Low-grade glioma',
        'color': 'yellow',
        'risk_level': 'Low',
        'follow_up_months': 6,
        'survival_estimate_days': 2555
    },
    2: {
        'name': 'Grade II-III Glioma',
        'description': 'Intermediate-grade glioma',
        'color': 'orange',
        'risk_level': 'Intermediate',
        'follow_up_months': 3,
        'survival_estimate_days': 1825
    },
    3: {
        'name': 'Grade IV Glioma',
        'description': 'High-grade glioma (GBM)',
        'color': 'red',
        'risk_level': 'High',
        'follow_up_months': 1,
        'survival_estimate_days': 450
    }
}

# Function for inference.py compatibility
def get_tumor_info(grade: int) -> Dict:
    """Get tumor information for a given grade - Used in inference.py"""
    return TUMOR_CLASSES.get(grade, TUMOR_CLASSES[0])


# For backward compatibility with existing inference code
CLINICAL_CONFIG = TUMOR_CLASSES

# RANO Criteria (separate variable for easy import)
RANO_CRITERIA = {
    'complete_response': {
        'description': 'Disappearance of all enhancing disease, stable/improved non-enhancing FLAIR lesions, off corticosteroids, clinically stable/improved.',
        'requirements': ['no_enhancing_lesion', 'stable_flair', 'no_steroids', 'stable_neurology']
    },
    'partial_response': {
        'description': '>=50% decrease in sum of enhancing lesion products compared to baseline, stable/improved non-enhancing FLAIR lesions, on stable/decreasing corticosteroids, clinically stable/improved.',
        'requirements': ['50_percent_reduction_enhancing', 'stable_flair', 'stable_steroids', 'stable_neurology']
    },
    'stable_disease': {
        'description': 'Does not qualify for CR, PR, or PD.',
        'requirements': ['not_cr_pr_pd']
    },
    'progressive_disease': {
        'description': '>=25% increase in sum of enhancing lesion products compared to nadir, significant increase in non-enhancing FLAIR lesions, new lesions, clinical deterioration, or increased corticosteroid dose.',
        'requirements': ['25_percent_increase_enhancing', 'or', 'significant_increase_flair', 'or', 'new_lesion', 'or', 'clinical_deterioration']
    }
}

# Clinical Protocols and RANO Criteria
CLINICAL_PROTOCOLS = {
    'rano_criteria': RANO_CRITERIA,
    'treatment_guidelines': {
        'Grade I Glioma': ['Surgical resection', 'Observation'],
        'Grade II-III Glioma': ['Maximal safe resection', 'Radiotherapy', 'Chemotherapy (PCV or Temozolomide)'],
        'Grade IV Glioma': ['Maximal safe resection', 'Radiotherapy with concurrent and adjuvant Temozolomide', 'Tumor Treating Fields (TTFields)']
    },
    'reporting_metrics': ['volume_enhancing', 'volume_non_enhancing', 'volume_necrosis', 'largest_diameter_axial', 'rankscale_performance']
}

# Survival estimates by grade - Compatible with inference.py
SURVIVAL_ESTIMATES = {
    'grade_0': 3650,
    'grade_1': 2555,
    'grade_2': 1825,
    'grade_3': 450
}

# ========================= INFERENCE CONFIGURATION =========================
# Inference settings compatible with inference.py
INFERENCE_CONFIG = {
    'ensemble_weights': {
        'brats_segmentation': 0.35,
        'brats_classification': 0.25,
        'detector': 0.20,
        'glioma_classifier': 0.15,
        'segmenter': 0.05
    },
    'preprocessing': {
        'target_sizes': {
            'standard': (224, 224),
            'highres': (384, 384),
            'segmentation': (256, 256),
            'brats_2d': (224, 224)
        }
    },
    'clinical_feature_mapping': {
        'idh_status': {'Unknown': 0, 'Mutant': 1, 'Wildtype': 2},
        'mgmt_status': {'Unknown': 0, 'Methylated': 1, 'Unmethylated': 2}
    }
}

# ========================= PERFORMANCE CONFIGURATION =========================
PERFORMANCE_CONFIG = {
    'inference': {
        'batch_size': 4,
        'max_cache_size_gb': 2,
        'preload_models': True,
        'gpu_memory_fraction': 0.8,
        'sliding_window': {  # For helper.py compatibility
            'patch_size': (64, 64, 64),
            'overlap': 0.5,
            'batch_size': 1
        }
    },
    'memory_management': {
        'clear_cache_after_analysis': True,
        'max_concurrent_analyses': 2,
        'temp_file_cleanup_hours': 1
    },
    'optimization': {
        'use_tensorrt': False,
        'precision': 'fp16',
        'threads': 4
    },
    'monitoring': {
        'log_latency': True,
        'log_memory_usage': True,
        'performance_report_interval': 24
    }
}

# ========================= EXPORT & REPORTING CONFIGURATION =========================
REPORTING_CONFIG = {
    'default_format': 'pdf',
    'include_confidence_scores': True,
    'anonymize_data': True,
    'dicom_sr_template': 'tumor_measurement_report.xml',
    'hospital_logo_path': BASE_DIR / "assets" / "hospital_logo.png",
    'signature_required': True
}

# ========================= ANALYSIS THRESHOLDS =========================
ANALYSIS_THRESHOLDS = {
    'min_confidence_score': 0.6,
    'max_uncertainty_score': 0.4,
    'min_tumor_volume_mm3': 50,
    'significant_growth_percentage': 25,
    'significant_shrinkage_percentage': 50,
    'ensemble_agreement_threshold': 0.7  # For inference.py
}

# ========================= EXPORT CONFIGURATION =========================
EXPORT_CONFIG = {
    'dicom_export': {
        'include_original_images': True,
        'include_segmentation_masks': True,
        'include_measurements': True,
        'compression_quality': 90
    },
    'pdf_export': {
        'include_all_slices': False,
        'key_slices_count': 12,
        'include_statistics': True,
        'include_recommendations': True
    },
    'csv_export': {
        'include_volumetrics': True,
        'include_radiomics': True,
        'include_confidence_scores': True
    }
}

# ========================= HELPER.PY COMPATIBILITY =========================
# Pre-trained model URLs for helper.py
PRETRAINED_MODELS = {
    'segmentation': {
        'model_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/unet_brain_tumor.pth',
        'input_size': (128, 128, 128),
        'num_classes': 4
    },
    'classification': {
        'efficientnet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/efficientnet_brain_tumor.pth',
        'resnet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/resnet_brain_tumor.pth',
        'densenet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/densenet_brain_tumor.pth'
    }
}

# ========================= UTILITY FUNCTIONS =========================
def get_rano_criteria() -> Dict:
    """Get RANO criteria"""
    return RANO_CRITERIA


def get_clinical_config() -> Dict:
    """Get clinical configuration"""
    return CLINICAL_CONFIG


def get_device_config() -> Dict:
    """Get device configuration"""
    return DEVICE_CONFIG


def get_model_config() -> Dict:
    """Get model configuration"""
    return MODEL_CONFIG


def get_ui_config() -> Dict:
    """Get UI configuration"""
    return UI_CONFIG


def get_auth_config() -> Dict:
    """Get authentication configuration"""
    return AUTH_CONFIG


def get_sample_users() -> Dict:
    """Get sample users configuration"""
    return SAMPLE_USERS


def get_inference_config() -> Dict:
    """Get inference configuration"""
    return INFERENCE_CONFIG


def validate_config() -> List[str]:
    """
    Validate the configuration settings
    Returns: List of error messages (empty if valid)
    """
    errors = []
    
    # Check if required directories exist
    required_dirs = [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, TEMP_DIR]
    for directory in required_dirs:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to create directory {directory}: {e}")
    
    # Validate tumor classes
    if not TUMOR_CLASSES:
        errors.append("TUMOR_CLASSES configuration is empty")
    
    # Validate device configuration
    if not DEVICE_CONFIG.get('device'):
        errors.append("Device configuration is missing")
    
    # Validate authentication configuration
    if not AUTH_CONFIG.get('secret_key'):
        errors.append("Authentication secret key is missing")
    
    # Validate UI configuration
    if not UI_CONFIG:
        errors.append("UI configuration is missing")
    
    # Validate sample users
    if not SAMPLE_USERS:
        errors.append("SAMPLE_USERS configuration is empty")
    
    return errors


def save_config(file_path: Path = None) -> bool:
    """Save current configuration to a file"""
    if file_path is None:
        file_path = BASE_DIR / "config_backup.json"
    
    try:
        config_dict = {
            'APP_NAME': APP_NAME,
            'APP_VERSION': APP_VERSION,
            'DEBUG_MODE': DEBUG_MODE,
            'DATABASE_CONFIG': {k: str(v) if isinstance(v, Path) else v for k, v in DATABASE_CONFIG.items()},
            'UI_CONFIG': UI_CONFIG,
            'AUTH_CONFIG': AUTH_CONFIG,
            'DEVICE_CONFIG': DEVICE_CONFIG,
            'MODEL_CONFIG': MODEL_CONFIG,
            'MODEL_PATHS': {k: str(v) for k, v in MODEL_PATHS.items()},
            'IMAGE_CONFIG': IMAGE_CONFIG,
            'TUMOR_CLASSES': TUMOR_CLASSES,
            'CLINICAL_PROTOCOLS': CLINICAL_PROTOCOLS,
            'PERFORMANCE_CONFIG': PERFORMANCE_CONFIG,
            'REPORTING_CONFIG': {k: str(v) if isinstance(v, Path) else v for k, v in REPORTING_CONFIG.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False


def load_config(file_path: Path) -> bool:
    """Load configuration from a file"""
    try:
        if not file_path.exists():
            logger.error(f"Config file {file_path} does not exist")
            return False
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update global variables (this is a simplified version)
        # In a real implementation, you'd want to be more careful about this
        for key, value in config_dict.items():
            globals()[key] = value
        
        logger.info(f"Configuration loaded from {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of the current configuration"""
    return {
        'app_info': {
            'name': APP_NAME,
            'version': APP_VERSION,
            'debug_mode': DEBUG_MODE
        },
        'system': {
            'device': DEVICE_CONFIG.get('device', 'unknown'),
            'cuda_available': torch.cuda.is_available(),
            'model_count': len(MODEL_PATHS)
        },
        'paths': {
            'data_dir': str(DATA_DIR),
            'models_dir': str(MODELS_DIR),
            'logs_dir': str(LOGS_DIR)
        },
        'models': list(MODEL_PATHS.keys()),
        'validation_errors': validate_config()
    }


# Auto-validate configuration on import
config_errors = validate_config()
if config_errors:
    logger.warning(f"Configuration validation failed with {len(config_errors)} errors:")
    for error in config_errors:
        logger.warning(f" - {error}")
else:
    logger.info("Configuration validation passed")

# Initialize Windows logging if needed
if sys.platform == "win32":
    setup_windows_logging()

