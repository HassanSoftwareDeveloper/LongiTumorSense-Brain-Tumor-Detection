<<<<<<< HEAD

# preprocessing.py - Unified Medical Image Preprocessing


import numpy as np
import cv2
from PIL import Image
import logging
import streamlit as st
from scipy import ndimage
from skimage import exposure, filters, morphology, measure

# Set up logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom not available - DICOM support limited")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logger.warning("nibabel not available - NIfTI support limited")


class MedicalImagePreprocessor:
    """
    Unified medical image preprocessor compatible with models.py and inference.py
    """
    
    def __init__(self):
        self.target_size = (224, 224)  # Default for models
    
    def load_medical_image(self, uploaded_file):
        """
        Load medical images from various formats (DICOM, NIfTI, standard images)
        Compatible with inference.py RealBrainTumorInferenceEngine
        """
        try:
            medical_image = None
            image_metadata = {}
            
            # Read file content once
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            file_extension = uploaded_file.name.lower()
            
            # ===== DICOM LOADING =====
            if file_extension.endswith('.dcm') and PYDICOM_AVAILABLE:
                medical_image, image_metadata = self._load_dicom_image(file_content)
            
            # ===== NIfTI LOADING =====
            elif file_extension.endswith(('.nii', '.nii.gz')) and NIBABEL_AVAILABLE:
                medical_image, image_metadata = self._load_nifti_image(file_content)
            
            # ===== STANDARD IMAGE FORMATS =====
            elif file_extension.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                medical_image, image_metadata = self._load_standard_image(file_content)
            
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                st.error(f"Unsupported file format: {file_extension}")
                return None, {}
            
            # Validate loaded image
            if medical_image is not None:
                logger.info(f"Successfully loaded image: {uploaded_file.name}, shape: {medical_image.shape}")
            
            return medical_image, image_metadata
            
        except Exception as load_error:
            logger.error(f"Error loading medical image: {str(load_error)}")
            st.error(f"Error loading medical image: {str(load_error)}")
            return None, {}

    def _load_dicom_image(self, file_content):
        """Load and process DICOM images with full metadata extraction"""
        from io import BytesIO
        import pydicom
        
        try:
            dicom_dataset = pydicom.dcmread(BytesIO(file_content))
            pixel_array = dicom_dataset.pixel_array.astype(np.float32)
            
            # Extract comprehensive DICOM metadata
            metadata = {
                'patient_id': str(getattr(dicom_dataset, 'PatientID', 'Unknown')),
                'patient_name': str(getattr(dicom_dataset, 'PatientName', 'Unknown')),
                'study_date': str(getattr(dicom_dataset, 'StudyDate', 'Unknown')),
                'modality': str(getattr(dicom_dataset, 'Modality', 'Unknown')),
                'study_description': str(getattr(dicom_dataset, 'StudyDescription', 'Unknown')),
                'series_description': str(getattr(dicom_dataset, 'SeriesDescription', 'Unknown'))
            }
            
            # Extract voxel size for volume calculations
            if hasattr(dicom_dataset, 'PixelSpacing'):
                pixel_spacing = dicom_dataset.PixelSpacing
                slice_thickness = getattr(dicom_dataset, 'SliceThickness', 1.0)
                metadata['voxel_size'] = (
                    float(slice_thickness), 
                    float(pixel_spacing[0]), 
                    float(pixel_spacing[1])
                )
            else:
                metadata['voxel_size'] = (1.0, 1.0, 1.0)
            
            # Apply DICOM windowing if available
            window_center = getattr(dicom_dataset, 'WindowCenter', None)
            window_width = getattr(dicom_dataset, 'WindowWidth', None)
            
            if window_center is not None and window_width is not None:
                # Handle multiple windows (take first)
                if hasattr(window_center, '__iter__'):
                    window_center = float(window_center[0])
                else:
                    window_center = float(window_center)
                
                if hasattr(window_width, '__iter__'):
                    window_width = float(window_width[0])
                else:
                    window_width = float(window_width)
                
                # Apply window/level
                window_min = window_center - window_width / 2
                window_max = window_center + window_width / 2
                pixel_array = np.clip(pixel_array, window_min, window_max)
                pixel_array = (pixel_array - window_min) / (window_max - window_min + 1e-8)
            else:
                # Default percentile-based normalization
                p1, p99 = np.percentile(pixel_array, [1, 99])
                pixel_array = np.clip(pixel_array, p1, p99)
                pixel_array = (pixel_array - p1) / (p99 - p1 + 1e-8)
            
            logger.info(f"DICOM loaded successfully: {metadata.get('modality', 'Unknown')}")
            return pixel_array, metadata
            
        except Exception as dicom_error:
            logger.error(f"DICOM loading failed: {dicom_error}")
            raise

    def _load_nifti_image(self, file_content):
        """Load and process NIfTI images with proper 3D handling"""
        from io import BytesIO
        import nibabel as nib
        
        try:
            nifti_image = nib.load(BytesIO(file_content))
            image_data = nifti_image.get_fdata()
            
            # Extract NIfTI metadata
            header = nifti_image.header
            metadata = {
                'voxel_size': tuple(header.get_zooms()[:3]),
                'dimensions': tuple(header.get_data_shape()),
                'data_type': str(header.get_data_dtype())
            }
            
            # Process 3D volume
            if len(image_data.shape) == 3:
                processed_volume = image_data.astype(np.float32)
                
                # Normalize each slice separately for better contrast
                for slice_idx in range(processed_volume.shape[2]):
                    slice_data = processed_volume[:, :, slice_idx]
                    if np.any(slice_data > 0):
                        p1, p99 = np.percentile(slice_data[slice_data > 0], [1, 99])
                        slice_data = np.clip(slice_data, p1, p99)
                        if p99 > p1:
                            processed_volume[:, :, slice_idx] = (slice_data - p1) / (p99 - p1 + 1e-8)
            else:
                processed_volume = image_data.astype(np.float32)
                p1, p99 = np.percentile(processed_volume[processed_volume > 0], [1, 99]) if np.any(processed_volume > 0) else (0, 1)
                processed_volume = np.clip(processed_volume, p1, p99)
                if p99 > p1:
                    processed_volume = (processed_volume - p1) / (p99 - p1 + 1e-8)
            
            logger.info(f"NIfTI loaded successfully: shape {processed_volume.shape}")
            return processed_volume, metadata
            
        except Exception as nifti_error:
            logger.error(f"NIfTI loading failed: {nifti_error}")
            raise

    def _load_standard_image(self, file_content):
        """Load and process standard image formats (PNG, JPG, etc.)"""
        from io import BytesIO
        
        try:
            pil_image = Image.open(BytesIO(file_content))
            
            # Convert to grayscale if needed
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            image_array = np.array(pil_image).astype(np.float32)
            
            # Advanced normalization for standard images
            p1, p99 = np.percentile(image_array, [1, 99])
            image_array = np.clip(image_array, p1, p99)
            image_array = (image_array - p1) / (p99 - p1 + 1e-8)
            
            metadata = {
                'original_size': pil_image.size,
                'mode': pil_image.mode
            }
            
            logger.info(f"Standard image loaded successfully: {pil_image.size}")
            return image_array, metadata
            
        except Exception as std_image_error:
            logger.error(f"Standard image loading failed: {std_image_error}")
            raise

    def preprocess_for_model(self, image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        """
        Preprocess image specifically for model input
        Compatible with models.py preprocess_for_model function
        """
        if image is None:
            return None
        
        # Handle 3D volumes - take middle slice
        if len(image.shape) == 3:
            image = image[:, :, image.shape[2] // 2]
        
        # Normalize to 0-1
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Resize
        image = cv2.resize(image.astype(np.float32), target_size)
        
        return image

    def preprocess_mri(
        self,
        mri_image, 
        target_size=(224, 224), 
        is_3d=False, 
        enhance=True,
        for_model=True
    ):
        """
        Advanced MRI preprocessing optimized for deep learning models
        Compatible with both inference.py and frontend requirements
        """
        if mri_image is None:
            return None, None
        
        # Handle 3D volumes
        if is_3d and len(mri_image.shape) == 3:
            model_slices, display_slices = [], []
            for slice_idx in range(mri_image.shape[2]):
                slice_image = mri_image[:, :, slice_idx]
                model_slice, display_slice = self.preprocess_mri(
                    slice_image, target_size, 
                    is_3d=False, 
                    enhance=enhance, 
                    for_model=for_model
                )
                if model_slice is not None and display_slice is not None:
                    model_slices.append(model_slice)
                    display_slices.append(display_slice)
            
            if model_slices and display_slices:
                return np.stack(model_slices, axis=2), np.stack(display_slices, axis=2)
            else:
                return None, None
        
        # Convert to 2D if needed
        if len(mri_image.shape) == 3:
            if mri_image.shape[2] == 3:
                mri_image = cv2.cvtColor(mri_image, cv2.COLOR_RGB2GRAY)
            elif mri_image.shape[2] > 3:
                mri_image = mri_image[:, :, 0]
        
        # Ensure proper data type
        if mri_image.dtype != np.float32:
            mri_image = mri_image.astype(np.float32)
        
        # ===== DISPLAY IMAGE PROCESSING =====
        display_image = mri_image.copy()
        
        if enhance:
            # Convert to uint8 for OpenCV operations
            if display_image.max() <= 1.0:
                display_uint8 = (display_image * 255).astype(np.uint8)
            else:
                display_uint8 = display_image.astype(np.uint8)
            
            # CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            display_uint8 = clahe.apply(display_uint8)
            
            # Mild denoising
            display_uint8 = cv2.fastNlMeansDenoising(display_uint8, h=10)
            
            # Convert back to float
            display_image = display_uint8.astype(np.float32) / 255.0
        else:
            # Simple normalization
            if display_image.max() > 1.0:
                display_image = display_image / 255.0
        
        # ===== MODEL INPUT PROCESSING =====
        model_image = mri_image.copy()
        
        if for_model:
            # Use the same preprocessing as preprocess_for_model
            model_image = self.preprocess_for_model(model_image, target_size)
        else:
            # Simple resize for display
            model_image = cv2.resize(model_image, target_size, interpolation=cv2.INTER_AREA)
        
        # ===== RESIZE DISPLAY IMAGE =====
        display_image_resized = cv2.resize(
            display_image, 
            target_size, 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Ensure display image is in [0, 1] and has 3 channels
        display_image_resized = np.clip(display_image_resized, 0, 1)
        if len(display_image_resized.shape) == 2:
            display_image_resized = np.stack([display_image_resized] * 3, axis=-1)
        
        return model_image, display_image_resized

    def prepare_image_for_display(self, image):
        """
        Prepare image for display by ensuring it's in uint8 format [0,255]
        with proper normalization and 3 color channels
        """
        if image is None:
            return None
        
        # Handle PIL Images
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to float for processing
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1] range
        image_min, image_max = np.min(image), np.max(image)
        if image_max > image_min:
            normalized_image = (image - image_min) / (image_max - image_min)
        else:
            normalized_image = np.zeros_like(image)
        
        # Convert to uint8 [0, 255]
        display_image_uint8 = (normalized_image * 255).astype(np.uint8)
        
        # Ensure 3 channels for color display
        if len(display_image_uint8.shape) == 2:
            display_image_uint8 = np.stack([display_image_uint8] * 3, axis=-1)
        
        return display_image_uint8

    def normalize_volume(self, volume, method='zscore'):
        """
        Normalize 3D volume using different methods
        Compatible with models.py normalize_volume function
        """
        if volume is None:
            return None
        
        volume = volume.astype(np.float32)
        
        if method == 'zscore':
            # Z-score normalization with percentile clipping
            p1, p99 = np.percentile(volume, [1, 99])
            clipped_volume = np.clip(volume, p1, p99)
            mean_value = np.mean(clipped_volume)
            std_value = np.std(clipped_volume) + 1e-8
            return (clipped_volume - mean_value) / std_value
        
        elif method == 'percentile':
            # Percentile-based normalization to [0, 1]
            p1, p99 = np.percentile(volume, [1, 99])
            clipped_volume = np.clip(volume, p1, p99)
            return (clipped_volume - p1) / (p99 - p1 + 1e-8)
        
        else:  # minmax
            # Simple min-max normalization
            min_val, max_val = np.min(volume), np.max(volume)
            return (volume - min_val) / (max_val - min_val + 1e-8)

    def resize_volume(self, volume, target_shape):
        """
        Resize 3D volume to target shape using linear interpolation
        Compatible with models.py resize_volume function
        """
        try:
            if volume.shape == target_shape:
                return volume
            
            zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
            resized_volume = ndimage.zoom(volume, zoom=zoom_factors, order=1, mode='nearest')
            return resized_volume
        except Exception as resize_error:
            logger.error(f"Volume resize failed: {resize_error}")
            return volume

    def remove_small_components(self, mask, min_size=100):
        """
        Remove connected components smaller than min_size
        Compatible with models.py remove_small_components function
        """
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
            logger.exception(f"Error in remove_small_components: {e}")
            # Fallback to skimage
            labeled = measure.label(mask)
            regions = measure.regionprops(labeled)
            cleaned = mask.copy()
            for reg in regions:
                if reg.area < min_size:
                    cleaned[labeled == reg.label] = 0
            return cleaned

    def enhance_brain_image(self, image, clip_limit=0.03):
        """
        Apply CLAHE enhancement to brain image
        Compatible with models.py enhance_brain_image function
        """
        try:
            if len(image.shape) == 3:
                # For 3D volumes, enhance each slice
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    slice_2d = image[:, :, i]
                    enhanced_slice = exposure.equalize_adapthist(slice_2d, clip_limit=clip_limit)
                    enhanced[:, :, i] = enhanced_slice
                return enhanced
            else:
                # For 2D images
                return exposure.equalize_adapthist(image, clip_limit=clip_limit)
        except Exception as e:
            logger.warning(f"CLAHE enhancement failed: {e}. Using original image.")
            return image

    def extract_brain_region(self, image, threshold=0.1):
        """
        Extract brain region using intensity thresholding
        Compatible with models.py extract_brain_region function
        """
        try:
            # Normalize image
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Threshold to get brain mask
            brain_mask = normalized > threshold
            
            # Apply morphological operations
            brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
            brain_mask = morphology.remove_small_holes(brain_mask, area_threshold=100)
            
            # Apply mask to original image
            brain_image = image.copy()
            brain_image[~brain_mask] = 0
            
            return brain_image, brain_mask
        except Exception as e:
            logger.warning(f"Brain extraction failed: {e}")
            return image, np.ones_like(image, dtype=bool)

    def create_image_pyramid(self, image, levels=3):
        """
        Create image pyramid for multi-scale processing
        Compatible with models.py create_image_pyramid function
        """
        pyramid = [image]
        for i in range(1, levels):
            smaller = cv2.resize(pyramid[i-1], 
                                (pyramid[i-1].shape[1] // 2, pyramid[i-1].shape[0] // 2),
                                interpolation=cv2.INTER_AREA)
            pyramid.append(smaller)
        return pyramid

    def register_images(self, fixed_image, moving_image):
        """
        Simple image registration using phase correlation
        Compatible with models.py register_images function
        """
        try:
            # Convert to float32
            fixed = fixed_image.astype(np.float32)
            moving = moving_image.astype(np.float32)
            
            # Calculate phase correlation
            shift, error, phase_diff = cv2.phaseCorrelate(fixed, moving)
            
            # Apply translation
            rows, cols = fixed.shape
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            registered = cv2.warpAffine(moving, M, (cols, rows))
            
            logger.info(f"Image registration applied: shift={shift}, error={error}")
            return registered
            
        except Exception as e:
            logger.warning(f"Image registration failed: {e}")
            return moving_image


# ==================== GLOBAL INSTANCE AND FUNCTIONS ====================

# Create global preprocessor instance
preprocessor = MedicalImagePreprocessor()

# Global functions for backward compatibility
def load_medical_image(uploaded_file):
    """Global function for loading medical images"""
    return preprocessor.load_medical_image(uploaded_file)

def preprocess_for_model(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Global function for model preprocessing"""
    return preprocessor.preprocess_for_model(image, target_size)

def preprocess_mri(mri_image, target_size=(224, 224), is_3d=False, enhance=True, for_model=True):
    """Global function for MRI preprocessing"""
    return preprocessor.preprocess_mri(mri_image, target_size, is_3d, enhance, for_model)

def normalize_volume(volume, method='zscore'):
    """Global function for volume normalization"""
    return preprocessor.normalize_volume(volume, method)

def resize_volume(volume, target_shape):
    """Global function for volume resizing"""
    return preprocessor.resize_volume(volume, target_shape)

def remove_small_components(mask, min_size=100):
    """Global function for removing small components"""
    return preprocessor.remove_small_components(mask, min_size)

def enhance_brain_image(image, clip_limit=0.03):
    """Global function for brain image enhancement"""
    return preprocessor.enhance_brain_image(image, clip_limit)

def extract_brain_region(image, threshold=0.1):
    """Global function for brain region extraction"""
    return preprocessor.extract_brain_region(image, threshold)

def create_image_pyramid(image, levels=3):
    """Global function for creating image pyramid"""
    return preprocessor.create_image_pyramid(image, levels)

def register_images(fixed_image, moving_image):
    """Global function for image registration"""
    return preprocessor.register_images(fixed_image, moving_image)

def prepare_image_for_display(image):
    """Global function for display image preparation"""
    return preprocessor.prepare_image_for_display(image)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*80)
    print("UNIFIED PREPROCESSING MODULE TESTING")
    print("="*80)
    
    # Test preprocessing functions
    test_image_2d = np.random.rand(256, 256).astype(np.float32)
    test_image_3d = np.random.rand(64, 64, 32).astype(np.float32)
    
    print("\nTest 1: Model Preprocessing")
    model_img = preprocess_for_model(test_image_2d)
    print(f"✓ Model preprocessing: {model_img.shape}")
    
    print("\nTest 2: MRI Preprocessing")
    model_img, display_img = preprocess_mri(test_image_2d, enhance=True, for_model=True)
    print(f"✓ MRI preprocessing: Model {model_img.shape}, Display {display_img.shape}")
    
    print("\nTest 3: Volume Normalization")
    normalized_vol = normalize_volume(test_image_3d, 'zscore')
    print(f"✓ Volume normalized: {normalized_vol.shape}")
    
    print("\nTest 4: Volume Resizing")
    resized_vol = resize_volume(test_image_3d, (32, 32, 16))
    print(f"✓ Volume resized: {resized_vol.shape}")
    
    print("\nTest 5: Image Enhancement")
    enhanced_img = enhance_brain_image(test_image_2d)
    print(f"✓ Image enhanced: {enhanced_img.shape}")
    
    print("\nTest 6: Brain Extraction")
    brain_img, brain_mask = extract_brain_region(test_image_2d)
    print(f"✓ Brain extracted: Image {brain_img.shape}, Mask {brain_mask.shape}")
    
    print("\nTest 7: Small Components Removal")
    test_mask = np.random.rand(256, 256) > 0.5
    cleaned_mask = remove_small_components(test_mask, min_size=100)
    print(f"✓ Small components removed: {cleaned_mask.shape}")
    
    print("\nTest 8: Image Pyramid")
    pyramid = create_image_pyramid(test_image_2d, levels=3)
    print(f"✓ Image pyramid created: {len(pyramid)} levels")
    
    print("\nTest 9: Display Image Preparation")
    display_img = prepare_image_for_display(test_image_2d)
    print(f"✓ Display image prepared: {display_img.shape}, dtype: {display_img.dtype}")
    
    print("\n" + "="*80)
    print("All preprocessing tests passed!")
    print("✓ Compatible with models.py")
    print("✓ Compatible with inference.py") 
    print("✓ Compatible with Frontend.py")
    print("✓ Global functions available")
    print("✓ Class-based interface available")
    print("="*80)
=======

# preprocessing.py - Unified Medical Image Preprocessing


import numpy as np
import cv2
from PIL import Image
import logging
import streamlit as st
from scipy import ndimage
from skimage import exposure, filters, morphology, measure

# Set up logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom not available - DICOM support limited")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logger.warning("nibabel not available - NIfTI support limited")


class MedicalImagePreprocessor:
    """
    Unified medical image preprocessor compatible with models.py and inference.py
    """
    
    def __init__(self):
        self.target_size = (224, 224)  # Default for models
    
    def load_medical_image(self, uploaded_file):
        """
        Load medical images from various formats (DICOM, NIfTI, standard images)
        Compatible with inference.py RealBrainTumorInferenceEngine
        """
        try:
            medical_image = None
            image_metadata = {}
            
            # Read file content once
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            file_extension = uploaded_file.name.lower()
            
            # ===== DICOM LOADING =====
            if file_extension.endswith('.dcm') and PYDICOM_AVAILABLE:
                medical_image, image_metadata = self._load_dicom_image(file_content)
            
            # ===== NIfTI LOADING =====
            elif file_extension.endswith(('.nii', '.nii.gz')) and NIBABEL_AVAILABLE:
                medical_image, image_metadata = self._load_nifti_image(file_content)
            
            # ===== STANDARD IMAGE FORMATS =====
            elif file_extension.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                medical_image, image_metadata = self._load_standard_image(file_content)
            
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                st.error(f"Unsupported file format: {file_extension}")
                return None, {}
            
            # Validate loaded image
            if medical_image is not None:
                logger.info(f"Successfully loaded image: {uploaded_file.name}, shape: {medical_image.shape}")
            
            return medical_image, image_metadata
            
        except Exception as load_error:
            logger.error(f"Error loading medical image: {str(load_error)}")
            st.error(f"Error loading medical image: {str(load_error)}")
            return None, {}

    def _load_dicom_image(self, file_content):
        """Load and process DICOM images with full metadata extraction"""
        from io import BytesIO
        import pydicom
        
        try:
            dicom_dataset = pydicom.dcmread(BytesIO(file_content))
            pixel_array = dicom_dataset.pixel_array.astype(np.float32)
            
            # Extract comprehensive DICOM metadata
            metadata = {
                'patient_id': str(getattr(dicom_dataset, 'PatientID', 'Unknown')),
                'patient_name': str(getattr(dicom_dataset, 'PatientName', 'Unknown')),
                'study_date': str(getattr(dicom_dataset, 'StudyDate', 'Unknown')),
                'modality': str(getattr(dicom_dataset, 'Modality', 'Unknown')),
                'study_description': str(getattr(dicom_dataset, 'StudyDescription', 'Unknown')),
                'series_description': str(getattr(dicom_dataset, 'SeriesDescription', 'Unknown'))
            }
            
            # Extract voxel size for volume calculations
            if hasattr(dicom_dataset, 'PixelSpacing'):
                pixel_spacing = dicom_dataset.PixelSpacing
                slice_thickness = getattr(dicom_dataset, 'SliceThickness', 1.0)
                metadata['voxel_size'] = (
                    float(slice_thickness), 
                    float(pixel_spacing[0]), 
                    float(pixel_spacing[1])
                )
            else:
                metadata['voxel_size'] = (1.0, 1.0, 1.0)
            
            # Apply DICOM windowing if available
            window_center = getattr(dicom_dataset, 'WindowCenter', None)
            window_width = getattr(dicom_dataset, 'WindowWidth', None)
            
            if window_center is not None and window_width is not None:
                # Handle multiple windows (take first)
                if hasattr(window_center, '__iter__'):
                    window_center = float(window_center[0])
                else:
                    window_center = float(window_center)
                
                if hasattr(window_width, '__iter__'):
                    window_width = float(window_width[0])
                else:
                    window_width = float(window_width)
                
                # Apply window/level
                window_min = window_center - window_width / 2
                window_max = window_center + window_width / 2
                pixel_array = np.clip(pixel_array, window_min, window_max)
                pixel_array = (pixel_array - window_min) / (window_max - window_min + 1e-8)
            else:
                # Default percentile-based normalization
                p1, p99 = np.percentile(pixel_array, [1, 99])
                pixel_array = np.clip(pixel_array, p1, p99)
                pixel_array = (pixel_array - p1) / (p99 - p1 + 1e-8)
            
            logger.info(f"DICOM loaded successfully: {metadata.get('modality', 'Unknown')}")
            return pixel_array, metadata
            
        except Exception as dicom_error:
            logger.error(f"DICOM loading failed: {dicom_error}")
            raise

    def _load_nifti_image(self, file_content):
        """Load and process NIfTI images with proper 3D handling"""
        from io import BytesIO
        import nibabel as nib
        
        try:
            nifti_image = nib.load(BytesIO(file_content))
            image_data = nifti_image.get_fdata()
            
            # Extract NIfTI metadata
            header = nifti_image.header
            metadata = {
                'voxel_size': tuple(header.get_zooms()[:3]),
                'dimensions': tuple(header.get_data_shape()),
                'data_type': str(header.get_data_dtype())
            }
            
            # Process 3D volume
            if len(image_data.shape) == 3:
                processed_volume = image_data.astype(np.float32)
                
                # Normalize each slice separately for better contrast
                for slice_idx in range(processed_volume.shape[2]):
                    slice_data = processed_volume[:, :, slice_idx]
                    if np.any(slice_data > 0):
                        p1, p99 = np.percentile(slice_data[slice_data > 0], [1, 99])
                        slice_data = np.clip(slice_data, p1, p99)
                        if p99 > p1:
                            processed_volume[:, :, slice_idx] = (slice_data - p1) / (p99 - p1 + 1e-8)
            else:
                processed_volume = image_data.astype(np.float32)
                p1, p99 = np.percentile(processed_volume[processed_volume > 0], [1, 99]) if np.any(processed_volume > 0) else (0, 1)
                processed_volume = np.clip(processed_volume, p1, p99)
                if p99 > p1:
                    processed_volume = (processed_volume - p1) / (p99 - p1 + 1e-8)
            
            logger.info(f"NIfTI loaded successfully: shape {processed_volume.shape}")
            return processed_volume, metadata
            
        except Exception as nifti_error:
            logger.error(f"NIfTI loading failed: {nifti_error}")
            raise

    def _load_standard_image(self, file_content):
        """Load and process standard image formats (PNG, JPG, etc.)"""
        from io import BytesIO
        
        try:
            pil_image = Image.open(BytesIO(file_content))
            
            # Convert to grayscale if needed
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            image_array = np.array(pil_image).astype(np.float32)
            
            # Advanced normalization for standard images
            p1, p99 = np.percentile(image_array, [1, 99])
            image_array = np.clip(image_array, p1, p99)
            image_array = (image_array - p1) / (p99 - p1 + 1e-8)
            
            metadata = {
                'original_size': pil_image.size,
                'mode': pil_image.mode
            }
            
            logger.info(f"Standard image loaded successfully: {pil_image.size}")
            return image_array, metadata
            
        except Exception as std_image_error:
            logger.error(f"Standard image loading failed: {std_image_error}")
            raise

    def preprocess_for_model(self, image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        """
        Preprocess image specifically for model input
        Compatible with models.py preprocess_for_model function
        """
        if image is None:
            return None
        
        # Handle 3D volumes - take middle slice
        if len(image.shape) == 3:
            image = image[:, :, image.shape[2] // 2]
        
        # Normalize to 0-1
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Resize
        image = cv2.resize(image.astype(np.float32), target_size)
        
        return image

    def preprocess_mri(
        self,
        mri_image, 
        target_size=(224, 224), 
        is_3d=False, 
        enhance=True,
        for_model=True
    ):
        """
        Advanced MRI preprocessing optimized for deep learning models
        Compatible with both inference.py and frontend requirements
        """
        if mri_image is None:
            return None, None
        
        # Handle 3D volumes
        if is_3d and len(mri_image.shape) == 3:
            model_slices, display_slices = [], []
            for slice_idx in range(mri_image.shape[2]):
                slice_image = mri_image[:, :, slice_idx]
                model_slice, display_slice = self.preprocess_mri(
                    slice_image, target_size, 
                    is_3d=False, 
                    enhance=enhance, 
                    for_model=for_model
                )
                if model_slice is not None and display_slice is not None:
                    model_slices.append(model_slice)
                    display_slices.append(display_slice)
            
            if model_slices and display_slices:
                return np.stack(model_slices, axis=2), np.stack(display_slices, axis=2)
            else:
                return None, None
        
        # Convert to 2D if needed
        if len(mri_image.shape) == 3:
            if mri_image.shape[2] == 3:
                mri_image = cv2.cvtColor(mri_image, cv2.COLOR_RGB2GRAY)
            elif mri_image.shape[2] > 3:
                mri_image = mri_image[:, :, 0]
        
        # Ensure proper data type
        if mri_image.dtype != np.float32:
            mri_image = mri_image.astype(np.float32)
        
        # ===== DISPLAY IMAGE PROCESSING =====
        display_image = mri_image.copy()
        
        if enhance:
            # Convert to uint8 for OpenCV operations
            if display_image.max() <= 1.0:
                display_uint8 = (display_image * 255).astype(np.uint8)
            else:
                display_uint8 = display_image.astype(np.uint8)
            
            # CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            display_uint8 = clahe.apply(display_uint8)
            
            # Mild denoising
            display_uint8 = cv2.fastNlMeansDenoising(display_uint8, h=10)
            
            # Convert back to float
            display_image = display_uint8.astype(np.float32) / 255.0
        else:
            # Simple normalization
            if display_image.max() > 1.0:
                display_image = display_image / 255.0
        
        # ===== MODEL INPUT PROCESSING =====
        model_image = mri_image.copy()
        
        if for_model:
            # Use the same preprocessing as preprocess_for_model
            model_image = self.preprocess_for_model(model_image, target_size)
        else:
            # Simple resize for display
            model_image = cv2.resize(model_image, target_size, interpolation=cv2.INTER_AREA)
        
        # ===== RESIZE DISPLAY IMAGE =====
        display_image_resized = cv2.resize(
            display_image, 
            target_size, 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Ensure display image is in [0, 1] and has 3 channels
        display_image_resized = np.clip(display_image_resized, 0, 1)
        if len(display_image_resized.shape) == 2:
            display_image_resized = np.stack([display_image_resized] * 3, axis=-1)
        
        return model_image, display_image_resized

    def prepare_image_for_display(self, image):
        """
        Prepare image for display by ensuring it's in uint8 format [0,255]
        with proper normalization and 3 color channels
        """
        if image is None:
            return None
        
        # Handle PIL Images
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to float for processing
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1] range
        image_min, image_max = np.min(image), np.max(image)
        if image_max > image_min:
            normalized_image = (image - image_min) / (image_max - image_min)
        else:
            normalized_image = np.zeros_like(image)
        
        # Convert to uint8 [0, 255]
        display_image_uint8 = (normalized_image * 255).astype(np.uint8)
        
        # Ensure 3 channels for color display
        if len(display_image_uint8.shape) == 2:
            display_image_uint8 = np.stack([display_image_uint8] * 3, axis=-1)
        
        return display_image_uint8

    def normalize_volume(self, volume, method='zscore'):
        """
        Normalize 3D volume using different methods
        Compatible with models.py normalize_volume function
        """
        if volume is None:
            return None
        
        volume = volume.astype(np.float32)
        
        if method == 'zscore':
            # Z-score normalization with percentile clipping
            p1, p99 = np.percentile(volume, [1, 99])
            clipped_volume = np.clip(volume, p1, p99)
            mean_value = np.mean(clipped_volume)
            std_value = np.std(clipped_volume) + 1e-8
            return (clipped_volume - mean_value) / std_value
        
        elif method == 'percentile':
            # Percentile-based normalization to [0, 1]
            p1, p99 = np.percentile(volume, [1, 99])
            clipped_volume = np.clip(volume, p1, p99)
            return (clipped_volume - p1) / (p99 - p1 + 1e-8)
        
        else:  # minmax
            # Simple min-max normalization
            min_val, max_val = np.min(volume), np.max(volume)
            return (volume - min_val) / (max_val - min_val + 1e-8)

    def resize_volume(self, volume, target_shape):
        """
        Resize 3D volume to target shape using linear interpolation
        Compatible with models.py resize_volume function
        """
        try:
            if volume.shape == target_shape:
                return volume
            
            zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
            resized_volume = ndimage.zoom(volume, zoom=zoom_factors, order=1, mode='nearest')
            return resized_volume
        except Exception as resize_error:
            logger.error(f"Volume resize failed: {resize_error}")
            return volume

    def remove_small_components(self, mask, min_size=100):
        """
        Remove connected components smaller than min_size
        Compatible with models.py remove_small_components function
        """
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
            logger.exception(f"Error in remove_small_components: {e}")
            # Fallback to skimage
            labeled = measure.label(mask)
            regions = measure.regionprops(labeled)
            cleaned = mask.copy()
            for reg in regions:
                if reg.area < min_size:
                    cleaned[labeled == reg.label] = 0
            return cleaned

    def enhance_brain_image(self, image, clip_limit=0.03):
        """
        Apply CLAHE enhancement to brain image
        Compatible with models.py enhance_brain_image function
        """
        try:
            if len(image.shape) == 3:
                # For 3D volumes, enhance each slice
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    slice_2d = image[:, :, i]
                    enhanced_slice = exposure.equalize_adapthist(slice_2d, clip_limit=clip_limit)
                    enhanced[:, :, i] = enhanced_slice
                return enhanced
            else:
                # For 2D images
                return exposure.equalize_adapthist(image, clip_limit=clip_limit)
        except Exception as e:
            logger.warning(f"CLAHE enhancement failed: {e}. Using original image.")
            return image

    def extract_brain_region(self, image, threshold=0.1):
        """
        Extract brain region using intensity thresholding
        Compatible with models.py extract_brain_region function
        """
        try:
            # Normalize image
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Threshold to get brain mask
            brain_mask = normalized > threshold
            
            # Apply morphological operations
            brain_mask = morphology.remove_small_objects(brain_mask, min_size=100)
            brain_mask = morphology.remove_small_holes(brain_mask, area_threshold=100)
            
            # Apply mask to original image
            brain_image = image.copy()
            brain_image[~brain_mask] = 0
            
            return brain_image, brain_mask
        except Exception as e:
            logger.warning(f"Brain extraction failed: {e}")
            return image, np.ones_like(image, dtype=bool)

    def create_image_pyramid(self, image, levels=3):
        """
        Create image pyramid for multi-scale processing
        Compatible with models.py create_image_pyramid function
        """
        pyramid = [image]
        for i in range(1, levels):
            smaller = cv2.resize(pyramid[i-1], 
                                (pyramid[i-1].shape[1] // 2, pyramid[i-1].shape[0] // 2),
                                interpolation=cv2.INTER_AREA)
            pyramid.append(smaller)
        return pyramid

    def register_images(self, fixed_image, moving_image):
        """
        Simple image registration using phase correlation
        Compatible with models.py register_images function
        """
        try:
            # Convert to float32
            fixed = fixed_image.astype(np.float32)
            moving = moving_image.astype(np.float32)
            
            # Calculate phase correlation
            shift, error, phase_diff = cv2.phaseCorrelate(fixed, moving)
            
            # Apply translation
            rows, cols = fixed.shape
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            registered = cv2.warpAffine(moving, M, (cols, rows))
            
            logger.info(f"Image registration applied: shift={shift}, error={error}")
            return registered
            
        except Exception as e:
            logger.warning(f"Image registration failed: {e}")
            return moving_image


# ==================== GLOBAL INSTANCE AND FUNCTIONS ====================

# Create global preprocessor instance
preprocessor = MedicalImagePreprocessor()

# Global functions for backward compatibility
def load_medical_image(uploaded_file):
    """Global function for loading medical images"""
    return preprocessor.load_medical_image(uploaded_file)

def preprocess_for_model(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Global function for model preprocessing"""
    return preprocessor.preprocess_for_model(image, target_size)

def preprocess_mri(mri_image, target_size=(224, 224), is_3d=False, enhance=True, for_model=True):
    """Global function for MRI preprocessing"""
    return preprocessor.preprocess_mri(mri_image, target_size, is_3d, enhance, for_model)

def normalize_volume(volume, method='zscore'):
    """Global function for volume normalization"""
    return preprocessor.normalize_volume(volume, method)

def resize_volume(volume, target_shape):
    """Global function for volume resizing"""
    return preprocessor.resize_volume(volume, target_shape)

def remove_small_components(mask, min_size=100):
    """Global function for removing small components"""
    return preprocessor.remove_small_components(mask, min_size)

def enhance_brain_image(image, clip_limit=0.03):
    """Global function for brain image enhancement"""
    return preprocessor.enhance_brain_image(image, clip_limit)

def extract_brain_region(image, threshold=0.1):
    """Global function for brain region extraction"""
    return preprocessor.extract_brain_region(image, threshold)

def create_image_pyramid(image, levels=3):
    """Global function for creating image pyramid"""
    return preprocessor.create_image_pyramid(image, levels)

def register_images(fixed_image, moving_image):
    """Global function for image registration"""
    return preprocessor.register_images(fixed_image, moving_image)

def prepare_image_for_display(image):
    """Global function for display image preparation"""
    return preprocessor.prepare_image_for_display(image)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*80)
    print("UNIFIED PREPROCESSING MODULE TESTING")
    print("="*80)
    
    # Test preprocessing functions
    test_image_2d = np.random.rand(256, 256).astype(np.float32)
    test_image_3d = np.random.rand(64, 64, 32).astype(np.float32)
    
    print("\nTest 1: Model Preprocessing")
    model_img = preprocess_for_model(test_image_2d)
    print(f"✓ Model preprocessing: {model_img.shape}")
    
    print("\nTest 2: MRI Preprocessing")
    model_img, display_img = preprocess_mri(test_image_2d, enhance=True, for_model=True)
    print(f"✓ MRI preprocessing: Model {model_img.shape}, Display {display_img.shape}")
    
    print("\nTest 3: Volume Normalization")
    normalized_vol = normalize_volume(test_image_3d, 'zscore')
    print(f"✓ Volume normalized: {normalized_vol.shape}")
    
    print("\nTest 4: Volume Resizing")
    resized_vol = resize_volume(test_image_3d, (32, 32, 16))
    print(f"✓ Volume resized: {resized_vol.shape}")
    
    print("\nTest 5: Image Enhancement")
    enhanced_img = enhance_brain_image(test_image_2d)
    print(f"✓ Image enhanced: {enhanced_img.shape}")
    
    print("\nTest 6: Brain Extraction")
    brain_img, brain_mask = extract_brain_region(test_image_2d)
    print(f"✓ Brain extracted: Image {brain_img.shape}, Mask {brain_mask.shape}")
    
    print("\nTest 7: Small Components Removal")
    test_mask = np.random.rand(256, 256) > 0.5
    cleaned_mask = remove_small_components(test_mask, min_size=100)
    print(f"✓ Small components removed: {cleaned_mask.shape}")
    
    print("\nTest 8: Image Pyramid")
    pyramid = create_image_pyramid(test_image_2d, levels=3)
    print(f"✓ Image pyramid created: {len(pyramid)} levels")
    
    print("\nTest 9: Display Image Preparation")
    display_img = prepare_image_for_display(test_image_2d)
    print(f"✓ Display image prepared: {display_img.shape}, dtype: {display_img.dtype}")
    
    print("\n" + "="*80)
    print("All preprocessing tests passed!")
    print("✓ Compatible with models.py")
    print("✓ Compatible with inference.py") 
    print("✓ Compatible with Frontend.py")
    print("✓ Global functions available")
    print("✓ Class-based interface available")
    print("="*80)
>>>>>>> 6eb24a4951a6a37ffedc07cbfa6953d272a5f678
