

# helper.py
import torch
import numpy as np
from typing import Tuple, Callable, List, Dict, Optional, Union
import streamlit as st
from datetime import datetime, timedelta
import logging
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure, exposure
import cv2
import warnings
import json
import hashlib
import requests
import tempfile
import os
from huggingface_hub import hf_hub_download

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_tumor_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pre-trained model URLs and configuration
PRETRAINED_MODELS = {
    'segmentation': {
        'model_url': 'https://huggingface.co/matei-raluca/brain-tumor-segmentation/resolve/main/unet_brain_tumor.pth',
        'input_size': (128, 128, 128),
        'num_classes': 4
    },
    'classification': {
        'efficientnet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/efficientnet_brain_tumor.pth',
        'resnet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/resnet_brain_tumor.pth',
        'densenet_url': 'https://huggingface.co/rzcoder/brain-tumor-classification/resolve/main/densenet_brain_tumor.pth'
    }
}

def download_model(model_url, model_name):
    """Download pre-trained model weights"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', model_name)
        
        # Download from Hugging Face Hub or direct URL
        if 'huggingface.co' in model_url:
            repo_id = '/'.join(model_url.split('/')[3:5])
            filename = model_url.split('/')[-1]
            model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir='models')
        else:
            # Direct download
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info(f"Downloaded model to {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None

def load_pretrained_model(model_type, device='cpu'):
    """Load a pre-trained model based on type"""
    try:
        if model_type == 'segmentation':
            from monai.networks.nets import UNet
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=4,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2
            )
            
            # Download and load weights
            model_path = download_model(PRETRAINED_MODELS['segmentation']['model_url'], 'unet_brain_tumor.pth')
            if model_path:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                logger.info("Loaded pre-trained segmentation model")
            
        elif model_type in ['efficientnet', 'resnet', 'densenet']:
            if model_type == 'efficientnet':
                from efficientnet_pytorch import EfficientNet
                model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
                model_url = PRETRAINED_MODELS['classification']['efficientnet_url']
                model_name = 'efficientnet_brain_tumor.pth'
            elif model_type == 'resnet':
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 4)
                model_url = PRETRAINED_MODELS['classification']['resnet_url']
                model_name = 'resnet_brain_tumor.pth'
            else:  # densenet
                model = models.densenet121(pretrained=False)
                model.classifier = nn.Linear(model.classifier.in_features, 4)
                model_url = PRETRAINED_MODELS['classification']['densenet_url']
                model_name = 'densenet_brain_tumor.pth'
            
            # Download and load weights
            model_path = download_model(model_url, model_name)
            if model_path:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                logger.info(f"Loaded pre-trained {model_type} classification model")
        
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"Error loading pre-trained model: {e}")
        return None

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return device (CUDA if available and preferred, else CPU)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def sliding_window_inference(
    inputs: torch.Tensor,
    predictor: Callable,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    num_classes: int = 4,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    progress_update_interval: float = 0.25
) -> torch.Tensor:
    """
    Sliding window inference for large 3D volumes with optimized batch processing.
    """
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    try:
        if inputs.dim() != 5:
            raise ValueError(f"Input must be 5D tensor (1,C,D,H,W), got {inputs.dim()}D")

        inputs = inputs.to(device)
        _, C, D, H, W = inputs.shape
        patch_d, patch_h, patch_w = patch_size

        if patch_d <= 0 or patch_h <= 0 or patch_w <= 0:
            raise ValueError("patch_size elements must be > 0")

        step_d = max(1, int(patch_d * (1 - overlap)))
        step_h = max(1, int(patch_h * (1 - overlap)))
        step_w = max(1, int(patch_w * (1 - overlap)))

        d_positions = list(range(0, max(1, D - patch_d + 1), step_d))
        if d_positions[-1] != max(0, D - patch_d):
            d_positions.append(max(0, D - patch_d))

        h_positions = list(range(0, max(1, H - patch_h + 1), step_h))
        if h_positions[-1] != max(0, H - patch_h):
            h_positions.append(max(0, H - patch_h))

        w_positions = list(range(0, max(1, W - patch_w + 1), step_w))
        if w_positions[-1] != max(0, W - patch_w):
            w_positions.append(max(0, W - patch_w))

        total_patches = len(d_positions) * len(h_positions) * len(w_positions)
        if total_patches == 0:
            raise ValueError("Volume smaller than patch size in at least one dimension")

        output = torch.zeros((num_classes, D, H, W), device=device, dtype=torch.float32)
        count_map = torch.zeros((D, H, W), device=device, dtype=torch.float32)

        current_patch = 0
        progress_bar = st.progress(0.0)
        last_progress_update = time.time()

        batch_patches: List[torch.Tensor] = []
        batch_coords: List[Tuple[int, int, int]] = []

        for d in d_positions:
            for h in h_positions:
                for w in w_positions:
                    patch = inputs[:, :, d:d + patch_d, h:h + patch_h, w:w + patch_w]
                    if patch.shape[2:] != (patch_d, patch_h, patch_w):
                        pad_d = patch_d - patch.shape[2]
                        pad_h = patch_h - patch.shape[3]
                        pad_w = patch_w - patch.shape[4]
                        pad = (0, pad_w, 0, pad_h, 0, pad_d)
                        patch = torch.nn.functional.pad(patch, pad, mode='constant', value=0.0)

                    batch_patches.append(patch)
                    batch_coords.append((d, h, w))

                    if len(batch_patches) == batch_size:
                        batch_tensor = torch.cat(batch_patches, dim=0).to(device)

                        with torch.no_grad():
                            preds = predictor(batch_tensor)

                        if isinstance(preds, list) or isinstance(preds, tuple):
                            preds = preds[0]

                        if preds.dim() < 5:
                            raise ValueError(f"Predictor output must be 5D tensor (B,C,pd,ph,pw), got {preds.shape}")

                        if preds.shape[1] == num_classes:
                            preds = torch.softmax(preds, dim=1)

                        for i, (ds, hs, ws) in enumerate(batch_coords):
                            pred_patch = preds[i].to(device)
                            pd, ph, pw = pred_patch.shape[1], pred_patch.shape[2], pred_patch.shape[3]
                            d_end = ds + pd
                            h_end = hs + ph
                            w_end = ws + pw
                            output[:, ds:d_end, hs:h_end, ws:w_end] += pred_patch
                            count_map[ds:d_end, hs:h_end, ws:w_end] += 1.0

                        current_patch += len(batch_patches)

                        if time.time() - last_progress_update >= progress_update_interval:
                            progress = min(current_patch / total_patches, 1.0)
                            progress_bar.progress(progress)
                            last_progress_update = time.time()

                        batch_patches = []
                        batch_coords = []

        if batch_patches:
            batch_tensor = torch.cat(batch_patches, dim=0).to(device)
            with torch.no_grad():
                preds = predictor(batch_tensor)
            if isinstance(preds, list) or isinstance(preds, tuple):
                preds = preds[0]
            if preds.shape[1] == num_classes:
                preds = torch.softmax(preds, dim=1)

            for i, (ds, hs, ws) in enumerate(batch_coords):
                pred_patch = preds[i].to(device)
                pd, ph, pw = pred_patch.shape[1], pred_patch.shape[2], pred_patch.shape[3]
                d_end = ds + pd
                h_end = hs + ph
                w_end = ws + pw
                output[:, ds:d_end, hs:h_end, ws:w_end] += pred_patch
                count_map[ds:d_end, hs:h_end, ws:w_end] += 1.0
            current_patch += len(batch_patches)

        denom = count_map.clone()
        denom[denom == 0] = 1.0
        output = output / denom.unsqueeze(0)

        progress_bar.progress(1.0)
        logger.info(f"Sliding-window inference done. Processed {current_patch}/{total_patches} patches. Volume=({D},{H},{W})")
        return output.cpu()

    except Exception as e:
        logger.exception(f"Error in sliding_window_inference: {e}")
        raise

def calculate_per_class_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> List[float]:
    """Calculate Dice score for each class separately."""
    try:
        if pred.dim() == 4:
            pred_idx = torch.argmax(pred, dim=0)
        else:
            pred_idx = pred.long()

        if target.dim() == 4:
            target_idx = torch.argmax(target, dim=0)
        else:
            target_idx = target.long()

        dice_scores = []
        for cls in range(num_classes):
            pred_cls = (pred_idx == cls).float()
            target_cls = (target_idx == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(float(np.clip(dice.item(), 0.0, 1.0)))
        return dice_scores
    except Exception as e:
        logger.exception(f"Error calculating per-class dice: {e}")
        return [0.0] * num_classes

def calculate_per_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> List[float]:
    """Calculate IoU per class with clipping."""
    try:
        if pred.dim() == 4:
            pred_idx = torch.argmax(pred, dim=0)
        else:
            pred_idx = pred.long()

        if target.dim() == 4:
            target_idx = torch.argmax(target, dim=0)
        else:
            target_idx = target.long()

        iou_scores = []
        for cls in range(num_classes):
            pred_cls = (pred_idx == cls).float()
            target_cls = (target_idx == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(float(np.clip(iou.item(), 0.0, 1.0)))
        return iou_scores
    except Exception as e:
        logger.exception(f"Error calculating per-class IoU: {e}")
        return [0.0] * num_classes

def calculate_mean_metrics(dice_scores: List[float], iou_scores: List[float]) -> Dict[str, float]:
    """Return mean and std of Dice / IoU lists."""
    return {
        'mean_dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
        'mean_iou': float(np.mean(iou_scores)) if iou_scores else 0.0,
        'std_dice': float(np.std(dice_scores)) if dice_scores else 0.0,
        'std_iou': float(np.std(iou_scores)) if iou_scores else 0.0
    }

def apply_crf(image: np.ndarray, mask: np.ndarray, max_iterations: int = 5, pos_weight: float = 3, bilateral_weight: float = 5) -> np.ndarray:
    """Apply DenseCRF refinement to 2D segmentation mask."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        if image.ndim == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image

        if mask.ndim == 3 and mask.shape[2] > 1:
            mask_onehot = mask
        elif mask.ndim == 2:
            labels = np.unique(mask)
            num_classes = int(labels.max()) + 1 if len(labels) > 0 else 1
            mask_onehot = np.eye(num_classes)[mask.astype(int)]
        else:
            raise ValueError("Unsupported mask shape for CRF")

        if mask_onehot.shape[-1] == image_rgb.shape[-1]:
            mask_onehot = mask_onehot.transpose(2, 0, 1)
        else:
            mask_onehot = mask_onehot.transpose(2, 0, 1)

        d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], mask_onehot.shape[0])
        U = unary_from_softmax(mask_onehot)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=pos_weight)
        d.addPairwiseBilateral(sxy=20, srgb=10, rgbim=image_rgb, compat=bilateral_weight)

        Q = d.inference(max_iterations)
        refined = np.array(Q).reshape((-1, image_rgb.shape[0], image_rgb.shape[1]))
        refined_labels = np.argmax(refined, axis=0)
        logger.info(f"Applied CRF refinement ({max_iterations} iter).")
        return refined_labels

    except ImportError:
        logger.warning("pydensecrf not installed - skipping CRF. Install via: pip install pydensecrf")
        try:
            st.info("For better segmentation refinement install `pydensecrf`: pip install pydensecrf")
        except Exception:
            pass
        return mask
    except Exception as e:
        logger.exception(f"Error applying CRF: {e}")
        return mask

def setup_authentication():
    """Set up a more realistic authentication system"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.login_attempts = 0
        st.session_state.user_info = {}
        st.session_state.last_login_attempt = None
        st.session_state.lockout_until = None

def check_login_attempts():
    """Check if too many login attempts have been made"""
    if st.session_state.get('lockout_until') and datetime.now() < st.session_state.lockout_until:
        st.error(f"Account temporarily locked. Try again after {st.session_state.lockout_until}")
        return True
        
    if (st.session_state.login_attempts >= 3 and 
        st.session_state.last_login_attempt and 
        (datetime.now() - st.session_state.last_login_attempt).seconds < 300):
        st.session_state.lockout_until = datetime.now() + timedelta(minutes=5)
        st.error("Too many failed login attempts. Please try again in 5 minutes.")
        return True
    return False

def hash_password(password: str) -> str:
    """Simple password hashing for demonstration"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username: str, password: str, license_number: str, institution: str, users_config: Optional[Dict] = None) -> bool:
    """
    Verify credentials with EXACT matching for medical authentication security.
    """
    try:
        if users_config is None:
            # Fixed: Removed trailing spaces and made matching case-sensitive
            approved_users = {
                "hassan": {"password": hash_password("hassan123!"), "license": "MD123456", "institution": "Mayo Clinic"},
                "radiology.ai": {"password": hash_password("RadAI456!"), "license": "MD654321", "institution": "Mass General Hospital"},
                "brain.surgeon": {"password": hash_password("Surgery789!"), "license": "MD987654", "institution": "Johns Hopkins Hospital"}
            }
        else:
            approved_users = users_config

        user = approved_users.get(username.strip())
        if not user:
            logger.warning(f"User {username} not found in approved users")
            return False
        
        # Get hashed input password
        input_password_hashed = hash_password(password.strip())
        user_password = user.get("password", "").strip()
        
        # EXACT matching for all fields (case-sensitive for security)
        username_match = username.strip() in approved_users
        password_match = user_password == input_password_hashed
        license_match = user.get("license", "").strip() == license_number.strip()
        institution_match = user.get("institution", "").strip() == institution.strip()
        
        # Debug logging
        logger.info(f"Auth check: User='{username}', User_exists={username_match}, "
                   f"Pass_match={password_match}, License_match={license_match}, "
                   f"Institution_match={institution_match}")
        
        if username_match and password_match and license_match and institution_match:
            logger.info(f"Authentication successful for user: {username}")
            return True
        
        logger.warning(f"Authentication failed for user: {username}")
        return False
        
    except Exception as e:
        logger.exception(f"Authentication error: {e}")
        return False

def display_dicom_metadata(metadata):
    """Display DICOM metadata in an organized way"""
    if not metadata:
        return
    
    st.markdown("#### 📋 DICOM Metadata")
    with st.expander("View DICOM Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'patient_id' in metadata:
                st.text_input("Patient ID", metadata['patient_id'], disabled=True)
            if 'patient_name' in metadata:
                st.text_input("Patient Name", str(metadata['patient_name']), disabled=True)
        
        with col2:
            if 'study_date' in metadata:
                st.text_input("Study Date", metadata['study_date'], disabled=True)
            if 'modality' in metadata:
                st.text_input("Modality", metadata['modality'], disabled=True)
        
        if 'study_description' in metadata:
            st.text_area("Study Description", metadata['study_description'], disabled=True)

def setup_page_config():
    """Set up Streamlit page configuration"""
    st.set_page_config(
        page_title="NeuroVision AI - Advanced Brain Tumor Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def get_custom_css():
    """Return custom CSS for the application"""
    return """
    <style>
        .main-header {
            background: linear-gradient(135deg, #1a6db3 0%, #103a61 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
            border-left: 4px solid #1a6db3;
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12);
        }
        .result-container {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            border: 1px solid #e9ecef;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-weight: 500;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-weight: 500;
        }
        .critical-box {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-weight: 500;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #1a6db3 0%, #0d6efd 100%);
        }
        .sidebar .sidebar-content {
            background: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
        .patient-info {
            background: #e9f7fe;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1a6db3;
            margin-bottom: 1.5rem;
        }
        .section-header {
            border-bottom: 2px solid #1a6db3;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            color: #1a6db3;
            font-weight: 600;
        }
        .stButton button {
            background: linear-gradient(135deg, #1a6db3 0%, #0d6efd 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(13, 110, 253, 0.3);
        }
        .risk-high {
            background: #f8d7da !important;
            border-left: 4px solid #dc3545 !important;
        }
        .risk-medium {
            background: #fff3cd !important;
            border-left: 4px solid #ffc107 !important;
        }
        .risk-low {
            background: #d4edda !important;
            border-left: 4px solid #28a745 !important;
        }
    </style>
    """

def visualize_segmentation_3d(volume: np.ndarray, mask: np.ndarray, alpha: float = 0.3, colormap: str = 'jet'):
    """Create 3D visualization of segmentation"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        sx = max(0, volume.shape[0] // 2)
        sy = max(0, volume.shape[1] // 2)
        sz = max(0, volume.shape[2] // 2)

        axes[0, 0].imshow(volume[sx], cmap='gray')
        axes[0, 0].imshow(mask[sx], alpha=alpha, cmap=colormap)
        axes[0, 0].set_title('Axial View'); axes[0, 0].axis('off')

        axes[0, 1].imshow(volume[:, sy, :], cmap='gray')
        axes[0, 1].imshow(mask[:, sy, :], alpha=alpha, cmap=colormap)
        axes[0, 1].set_title('Coronal View'); axes[0, 1].axis('off')

        axes[0, 2].imshow(volume[:, :, sz], cmap='gray')
        axes[0, 2].imshow(mask[:, :, sz], alpha=alpha, cmap=colormap)
        axes[0, 2].set_title('Sagittal View'); axes[0, 2].axis('off')

        axes[1, 0].text(0.5, 0.5, '3D Volume\nRendering', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
        axes[1, 1].axis('off'); axes[1, 2].axis('off')

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.exception(f"Error in visualize_segmentation_3d: {e}")
        return None

def calculate_tumor_volume(mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
    """Calculate tumor volume in cm^3"""
    try:
        if len(voxel_size) != 3 or any(v <= 0 for v in voxel_size):
            raise ValueError("voxel_size must be a 3-tuple of positive numbers (mm).")
        voxel_volume_mm3 = float(voxel_size[0]) * float(voxel_size[1]) * float(voxel_size[2])
        tumor_voxels = int(np.sum(mask > 0))
        volume_mm3 = tumor_voxels * voxel_volume_mm3
        return float(volume_mm3) / 1000.0
    except Exception as e:
        logger.exception(f"Error calculating tumor volume: {e}")
        return 0.0

def create_metrics_display(dice_scores: List[float], iou_scores: List[float], class_names: Optional[List[str]] = None) -> Dict:
    """Create a dictionary summarizing per-class and average metrics."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(dice_scores))]
    metrics = {}
    for i, (d, iou) in enumerate(zip(dice_scores, iou_scores)):
        metrics[class_names[i]] = {
            'dice': float(d),
            'iou': float(iou),
            'quality': 'Excellent' if d > 0.8 else 'Good' if d > 0.6 else 'Fair' if d > 0.4 else 'Poor'
        }
    mean_metrics = calculate_mean_metrics(dice_scores, iou_scores)
    metrics['average'] = {
        'dice': mean_metrics['mean_dice'],
        'iou': mean_metrics['mean_iou'],
        'quality': 'Excellent' if mean_metrics['mean_dice'] > 0.8 else 'Good' if mean_metrics['mean_dice'] > 0.6 else 'Fair' if mean_metrics['mean_dice'] > 0.4 else 'Poor'
    }
    return metrics

def normalize_volume(volume: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize volume using 'zscore' (percentile clipping + z-score), 'percentile' (0-1), or 'minmax'.
    """
    volume = volume.astype(np.float32)
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

def create_interactive_3d_visualization(volume: np.ndarray, mask: np.ndarray):
    """
    Create interactive 3D visualization using Plotly (if available).
    Returns a Plotly Figure or None.
    """
    try:
        import plotly.graph_objects as go
        z, y, x = np.where(mask > 0)
        if x.size == 0:
            logger.info("Mask empty - no points for 3D visualization")
            return None

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=z, colorscale='Viridis', opacity=0.6)
        )])
        fig.update_layout(title='3D Tumor Visualization', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        return fig
    except ImportError:
        logger.info("Plotly not installed. For interactive 3D visualization: pip install plotly")
        return None
    except Exception as e:
        logger.exception(f"Error creating interactive 3D visualization: {e}")
        return None

def load_users_from_config(config_path: str = "users_config.json") -> Dict:
    """
    Load users from configuration (secure) file. Returns dict or empty on failure.
    """
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        logger.warning(f"User config file {config_path} not found. Using fallback users if any.")
        return {}
    except Exception as e:
        logger.exception(f"Error loading user config: {e}")
        return {}

# Additional functions for pre-trained model integration
def predict_with_pretrained_models(image, clinical_features=None):
    """Make predictions using pre-trained models"""
    device = get_device()
    
    # Load models if not already loaded
    if 'pretrained_models' not in st.session_state:
        with st.spinner("Loading pre-trained models..."):
            st.session_state.pretrained_models = {
                'segmentation': load_pretrained_model('segmentation', device),
                'efficientnet': load_pretrained_model('efficientnet', device),
                'resnet': load_pretrained_model('resnet', device),
                'densenet': load_pretrained_model('densenet', device)
            }
    
    models = st.session_state.pretrained_models
    
    # Run ensemble prediction
    all_predictions = []
    model_confidences = []
    
    for model_name in ['efficientnet', 'resnet', 'densenet']:
        model = models[model_name]
        if model is not None:
            try:
                # Prepare input
                input_tensor = prepare_image_for_model(image, model_name)
                
                with torch.no_grad():
                    if clinical_features is not None and hasattr(model, 'fc'):
                        # For models that support clinical features
                        clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0).to(device)
                        outputs = model(input_tensor, clinical_tensor)
                    else:
                        outputs = model(input_tensor)
                    
                    # Apply softmax to get probabilities
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    all_predictions.append(probs)
                    
                    # Calculate model confidence (max probability)
                    model_confidences.append(np.max(probs))
                    
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
                # Use uniform distribution as fallback
                all_predictions.append(np.ones(4) / 4)
                model_confidences.append(0.5)
    
    # Weighted ensemble voting
    ensemble_weights = {'efficientnet': 0.4, 'resnet': 0.3, 'densenet': 0.3}
    ensemble_probs = np.zeros(4)
    total_weight = 0
    
    for i, (probs, model_name) in enumerate(zip(all_predictions, ['efficientnet', 'resnet', 'densenet'])):
        weight = ensemble_weights.get(model_name, 1.0) * model_confidences[i]
        ensemble_probs += probs * weight
        total_weight += weight
    
    if total_weight > 0:
        ensemble_probs /= total_weight
    
    # Determine final prediction
    predicted_class = np.argmax(ensemble_probs)
    confidence = float(ensemble_probs[predicted_class])
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_predictions': all_predictions,
        'ensemble_probs': ensemble_probs.tolist()
    }

def prepare_image_for_model(image, model_type):
    """Prepare image for model input"""
    device = get_device()
    
    if len(image.shape) == 3:
        # Use the slice with highest variance (most information)
        slice_variances = np.var(image, axis=(0, 1))
        best_slice = np.argmax(slice_variances)
        image_2d = image[:, :, best_slice]
    else:
        image_2d = image
    
    # Resize to expected input size
    image_2d = cv2.resize(image_2d, (224, 224))
    
    # Normalize and convert to tensor
    image_2d = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-8)
    image_2d = (image_2d * 255).astype(np.uint8)
    
    # Convert to RGB (3 channels)
    if len(image_2d.shape) == 2:
        image_rgb = np.stack([image_2d] * 3, axis=-1)
    else:
        image_rgb = image_2d
    
    # Normalize for pre-trained models
    image_rgb = image_rgb.astype(np.float32) / 255.0
    image_rgb = (image_rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    return tensor

def segment_with_pretrained_model(image):
    """Perform segmentation using pre-trained model"""
    device = get_device()
    
    # Load model if not already loaded
    if 'pretrained_models' not in st.session_state or 'segmentation' not in st.session_state.pretrained_models:
        st.session_state.pretrained_models = {
            'segmentation': load_pretrained_model('segmentation', device)
        }
    
    model = st.session_state.pretrained_models['segmentation']
    
    if model is None:
        return None
    
    try:
        # Preprocess volume
        processed_volume = normalize_volume(image, method='zscore')
        
        # Add batch and channel dimensions
        input_tensor = torch.tensor(processed_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Perform sliding window inference
        with torch.no_grad():
            output = sliding_window_inference(
                input_tensor,
                lambda x: model(x),
                patch_size=(64, 64, 64),
                overlap=0.5,
                num_classes=4,
                device=device
            )
        
        # Get segmentation mask
        seg_mask = torch.argmax(output, dim=0).cpu().numpy()
        
        return seg_mask
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return None










