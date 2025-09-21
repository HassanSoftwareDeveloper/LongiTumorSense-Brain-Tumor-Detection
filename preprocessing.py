
import numpy as np
import cv2
from PIL import Image
import pydicom
import nibabel as nib
from skimage import exposure, filters
import logging
import streamlit as st
from scipy import ndimage

# Set up logging
logger = logging.getLogger(__name__)

def load_medical_image(file):
    """Load DICOM, NIfTI, or standard image formats with enhanced processing"""
    try:
        image = None
        metadata = {}
        
        # Read file content once
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        if file.name.lower().endswith('.dcm'):
            # Load DICOM with metadata extraction
            import pydicom
            from io import BytesIO
            
            dicom_data = pydicom.dcmread(BytesIO(file_content))
            image = dicom_data.pixel_array.astype(np.float32)
            
            # Extract DICOM metadata
            try:
                metadata['patient_id'] = str(getattr(dicom_data, 'PatientID', 'Unknown'))
                metadata['patient_name'] = str(getattr(dicom_data, 'PatientName', 'Unknown'))
                metadata['study_date'] = str(getattr(dicom_data, 'StudyDate', 'Unknown'))
                metadata['modality'] = str(getattr(dicom_data, 'Modality', 'Unknown'))
                metadata['study_description'] = str(getattr(dicom_data, 'StudyDescription', 'Unknown'))
                
                # Extract voxel size for volume calculations
                if hasattr(dicom_data, 'PixelSpacing'):
                    pixel_spacing = dicom_data.PixelSpacing
                    slice_thickness = getattr(dicom_data, 'SliceThickness', 1.0)
                    metadata['voxel_size'] = (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
                else:
                    metadata['voxel_size'] = (1.0, 1.0, 1.0)
                    
            except Exception as meta_error:
                logger.warning(f"Metadata extraction failed: {meta_error}")
                metadata = {}
            
            # Apply DICOM windowing if available
            center = getattr(dicom_data, 'WindowCenter', None)
            width = getattr(dicom_data, 'WindowWidth', None)
            if center is not None and width is not None:
                try:
                    if hasattr(center, '__iter__'):
                        center = float(center[0])
                    if hasattr(width, '__iter__'):
                        width = float(width[0])
                    
                    window_min = center - width/2
                    window_max = center + width/2
                    image = np.clip(image, window_min, window_max)
                    image = (image - window_min) / (window_max - window_min + 1e-8)
                except:
                    # Fallback normalization
                    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
            else:
                # Default normalization for medical images
                p1, p99 = np.percentile(image, [1, 99])
                image = np.clip(image, p1, p99)
                image = (image - p1) / (p99 - p1 + 1e-8)
                
        elif file.name.lower().endswith(('.nii', '.nii.gz')):
            # Load NIfTI with enhanced processing
            import nibabel as nib
            from io import BytesIO
            
            nifti_img = nib.load(BytesIO(file_content))
            image_data = nifti_img.get_fdata()
            
            # Extract NIfTI metadata
            try:
                header = nifti_img.header
                metadata['voxel_size'] = header.get_zooms()[:3]
                metadata['dimensions'] = header.get_data_shape()
            except:
                metadata['voxel_size'] = (1.0, 1.0, 1.0)
            
            # Take middle slice for 2D visualization, but keep 3D for processing
            if len(image_data.shape) == 3:
                image = image_data.astype(np.float32)
                # Normalize each slice separately
                for z in range(image.shape[2]):
                    slice_data = image[:, :, z]
                    p1, p99 = np.percentile(slice_data[slice_data > 0], [1, 99]) if np.any(slice_data > 0) else (0, 1)
                    slice_data = np.clip(slice_data, p1, p99)
                    if p99 > p1:
                        image[:, :, z] = (slice_data - p1) / (p99 - p1 + 1e-8)
                    else:
                        image[:, :, z] = slice_data
            else:
                image = image_data.astype(np.float32)
                p1, p99 = np.percentile(image[image > 0], [1, 99]) if np.any(image > 0) else (0, 1)
                image = np.clip(image, p1, p99)
                if p99 > p1:
                    image = (image - p1) / (p99 - p1 + 1e-8)
                    
        else:
            # Standard image formats with enhanced processing
            from io import BytesIO
            pil_image = Image.open(BytesIO(file_content))
            
            # Convert to grayscale if needed
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
                
            image = np.array(pil_image).astype(np.float32)
            
            # Advanced normalization for standard images
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1 + 1e-8)

        return image, metadata
        
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        st.error(f"Error loading medical image: {str(e)}")
        return None, {}

def preprocess_mri(image, target_size=(224, 224), is_3d=False, enhance=True, for_model=True):
    """
    Advanced MRI preprocessing optimized for pre-trained models
    Returns:
    - model_ready: Preprocessed image for model input (normalized appropriately)
    - display_ready: Image prepared for display
    """
    if image is None:
        return None, None
    
    # Handle 3D volumes
    if is_3d and len(image.shape) == 3:
        model_slices, display_slices = [], []
        for i in range(image.shape[2]):
            slice_img = image[:, :, i]
            m_ready, d_ready = preprocess_mri(slice_img, target_size, is_3d=False, 
                                            enhance=enhance, for_model=for_model)
            model_slices.append(m_ready)
            display_slices.append(d_ready)
        return np.stack(model_slices, axis=2), np.stack(display_slices, axis=2)

    # Convert to 2D if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, 0]  # Take first channel

    # Ensure proper data type
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Enhancement steps (only for display, models prefer normalized raw data)
    display_image = image.copy()
    
    if enhance:
        # Convert to uint8 for OpenCV operations
        if display_image.max() <= 1.0:
            display_uint8 = (display_image * 255).astype(np.uint8)
        else:
            display_uint8 = display_image.astype(np.uint8)
        
        # CLAHE contrast enhancement for display
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        display_uint8 = clahe.apply(display_uint8)
        
        # Mild denoising for display
        display_uint8 = cv2.fastNlMeansDenoising(display_uint8)
        
        # Convert back to float for consistency
        display_image = display_uint8.astype(np.float32) / 255.0
    else:
        if display_image.max() > 1.0:
            display_image = display_image / 255.0

    # Resize both model and display images
    model_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    display_image = cv2.resize(display_image, target_size, interpolation=cv2.INTER_CUBIC)

    # --------- MODEL-READY PROCESSING ---------
    if for_model:
        # Different normalization for pre-trained models
        if model_image.max() > 1.0:
            model_image = model_image / 255.0
        
        # Standard normalization for pre-trained models (mean 0, std 1)
        model_mean = np.mean(model_image)
        model_std = np.std(model_image)
        if model_std > 0:
            model_image = (model_image - model_mean) / model_std
        else:
            model_image = model_image - model_mean
            
        # For models expecting 3 channels, replicate grayscale to RGB
        if len(model_image.shape) == 2:
            model_image = np.stack([model_image] * 3, axis=-1)
    
    # --------- DISPLAY-READY PROCESSING ---------
    # Ensure display image is in [0, 1] range
    display_image = np.clip(display_image, 0, 1)
    if len(display_image.shape) == 2:
        display_image = np.stack([display_image] * 3, axis=-1)

    return model_image, display_image

def prepare_image_for_display(image):
    """Prepare image for display by ensuring it's in uint8 format [0,255]."""
    if image is None:
        return None
    
    # Handle different input types
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to float for processing
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Normalize to [0, 1] range
    min_val, max_val = np.min(image), np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)
    
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Ensure 3 channels for color display
    if len(image_uint8.shape) == 2:
        image_uint8 = np.stack([image_uint8] * 3, axis=-1)
    
    return image_uint8

def normalize_volume(volume, method='zscore'):
    """
    Normalize 3D volume using different methods
    Methods: 'zscore', 'minmax', 'percentile'
    """
    if volume is None:
        return None
    
    volume = volume.astype(np.float32)
    
    if method == 'zscore':
        # Remove outliers using percentile clipping, then z-score
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
        
    else:  # minmax
        # Simple min-max normalization
        mn, mx = np.min(volume), np.max(volume)
        return (volume - mn) / (mx - mn + 1e-8)

def resize_volume(volume, target_shape):
    """Resize 3D volume to target shape with linear interpolation"""
    try:
        if volume.shape == target_shape:
            return volume
            
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        resized = ndimage.zoom(volume, zoom=factors, order=1, mode='nearest')
        return resized
    except Exception as e:
        logger.error(f"Volume resize failed: {e}")
        return volume

def remove_small_components(mask, min_size=100):
    """Remove small connected components from binary mask"""
    try:
        from skimage import morphology
        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
        return cleaned.astype(mask.dtype)
    except ImportError:
        # Fallback implementation
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            small_components = np.where(component_sizes < min_size)[0]
            for comp_id in small_components:
                if comp_id > 0:  # Skip background
                    mask[labeled == comp_id] = 0
        return mask
    except Exception as e:
        logger.error(f"Error in remove_small_components: {e}")
        return mask

def enhance_brain_image(image, clip_limit=0.03, kernel_size=5):
    """
    Enhanced brain image processing for better visualization
    """
    if image is None:
        return None
    
    # Convert to float if needed
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Normalize to 0-1 range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply adaptive histogram equalization
    enhanced = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    
    # Apply mild Gaussian blur for noise reduction
    enhanced = filters.gaussian(enhanced, sigma=0.5)
    
    return enhanced

def extract_brain_region(image, threshold_method='otsu'):
    """
    Extract brain region from MRI using various thresholding methods
    """
    if image is None:
        return None, None
    
    # Normalize image
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Apply thresholding
    if threshold_method == 'otsu':
        thresh = filters.threshold_otsu(normalized)
        mask = normalized > thresh * 0.8
    elif threshold_method == 'multiotsu':
        thresholds = filters.threshold_multiotsu(normalized, classes=3)
        mask = normalized > thresholds[1]
    else:  # adaptive
        mask = normalized > filters.threshold_local(normalized, block_size=51)
    
    # Clean up mask
    mask = cv2.morphologyEx.binary_closing(mask, cv2.morphologyEx.disk(3))
    mask = cv2.morphologyEx.binary_opening(mask, cv2.morphologyEx.disk(2))
    mask = cv2.morphologyEx.remove_small_holes(mask, area_threshold=100)
    mask = cv2.morphologyEx.remove_small_objects(mask, min_size=500)
    
    # Apply mask to original image
    brain_only = image.copy()
    brain_only[~mask] = 0
    
    return brain_only, mask

def create_image_pyramid(image, levels=3):
    """
    Create image pyramid for multi-scale processing
    """
    pyramid = [image]
    current = image.copy()
    
    for i in range(1, levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid

def register_images(fixed_image, moving_image):
    """
    Simple image registration using feature matching
    """
    try:
        # Convert to uint8 for OpenCV
        fixed_8bit = (fixed_image * 255).astype(np.uint8)
        moving_8bit = (moving_image * 255).astype(np.uint8)
        
        # Use ORB feature detector
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(fixed_8bit, None)
        kp2, des2 = orb.detectAndCompute(moving_8bit, None)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Apply transformation
        registered = cv2.warpPerspective(moving_image, M, (fixed_image.shape[1], fixed_image.shape[0]))
        
        return registered, M
        
    except Exception as e:
        logger.error(f"Image registration failed: {e}")
        return moving_image, None

# Example usage
if __name__ == "__main__":
    # Test the preprocessing functions
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    # Test preprocessing
    model_img, display_img = preprocess_mri(test_image)
    print(f"Model image shape: {model_img.shape}, Display image shape: {display_img.shape}")
    
    # Test enhancement
    enhanced = enhance_brain_image(test_image)
    print(f"Enhanced image shape: {enhanced.shape}")
    
    # Test brain extraction
    brain, mask = extract_brain_region(test_image)
    print(f"Brain extracted: {brain is not None}, Mask shape: {mask.shape if mask is not None else 'None'}")
    
    print("Preprocessing functions tested successfully")
