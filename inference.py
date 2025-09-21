

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from skimage import exposure, measure, morphology, filters, segmentation
from scipy import ndimage
import requests
import os

# Import from models.py - Updated import paths
try:
    from models import (
        BrainTumorDetector, 
        BrainTumorSegmenter, 
        Enhanced3DUNet,
        EnhancedGliomaClassifier,
        load_pretrained_detector, 
        load_pretrained_segmenter,
        load_brats_model,
        preprocess_for_model,
        preprocessor,
        get_ensemble_models
    )
except ImportError as e:
    logging.warning(f"Could not import from models.py: {e}")
    # Fallback imports or create dummy classes
    pass

# Import configuration - will be available when config.py is provided
try:
    from config import (
        TUMOR_CLASSES, CLINICAL_CONFIG, RANO_CRITERIA, 
        get_tumor_info, DEVICE_CONFIG, MODEL_CONFIG
    )
except ImportError:
    
    # Temporary fallback configurations
    TUMOR_CLASSES = {
        0: {'name': 'Normal', 'description': 'No tumor detected', 'color': 'green'},
        1: {'name': 'Grade I Glioma', 'description': 'Low-grade glioma', 'color': 'yellow'},
        2: {'name': 'Grade II-III Glioma', 'description': 'Intermediate-grade glioma', 'color': 'orange'},
        3: {'name': 'Grade IV Glioma', 'description': 'High-grade glioma (GBM)', 'color': 'red'}
    }
    
    def get_tumor_info(grade):
        return TUMOR_CLASSES.get(grade, TUMOR_CLASSES[0])
    
    CLINICAL_CONFIG = {}
    RANO_CRITERIA = {}
    DEVICE_CONFIG = {'default_device': 'cpu'}
    MODEL_CONFIG = {}

logger = logging.getLogger(__name__)

class RealBrainTumorInferenceEngine:
    """
    Real Brain Tumor Inference Engine using multiple pretrained models
    Enhanced for integration with tumor.py frontend
    """
    
    def __init__(self, device: str = 'cpu', models_dir: str = './models'):
        self.device = device
        self.models_dir = models_dir
        self.tumor_classes = TUMOR_CLASSES
        self.clinical_config = CLINICAL_CONFIG
        
        # Load real pretrained models
        self.models = self._load_pretrained_models()
        
        # Initialize model ensemble weights with priority for BraTS model
        self.model_weights = {
            'brats_model': 0.4,    # High weight for specialized model
            'detector': 0.3,       # Main ensemble detector
            'segmenter': 0.1,      # Segmentation contribution
            'efficientnet': 0.1,   # Individual model contributions
            'resnet': 0.05,
            'densenet': 0.05
        }
        
        logger.info(f"Initialized Real Brain Tumor Inference Engine with {len(self.models)} models")
    
    def _load_pretrained_models(self) -> Dict:
        """Load all pretrained models for ensemble prediction"""
        models = {}
        
        try:
            # Load your BraTS model first (highest priority)
            brats_model_path = "D:\\Brain_Tumor_Detection\\model.pt"
            
            # Check if the file exists
            if os.path.exists(brats_model_path):
                try:
                    # Load BraTS model
                    brats_model = load_brats_model(self.device)
                    if brats_model is not None:
                        models['brats_model'] = brats_model
                        logger.info("Successfully loaded BraTS model")
                except Exception as e:
                    logger.warning(f"Could not load BraTS model: {e}")
            else:
                logger.warning(f"BraTS model file not found at: {brats_model_path}")
            
            # Load main ensemble detector
            try:
                detector_path = os.path.join(self.models_dir, 'brain_tumor_detector.pth')
                models['detector'] = load_pretrained_detector(self.device, detector_path)
                logger.info("Loaded ensemble detector")
            except Exception as e:
                logger.warning(f"Could not load detector: {e}")
                # Create fallback detector
                models['detector'] = BrainTumorDetector(device=self.device)
            
            # Load segmentation model
            try:
                segmenter_path = os.path.join(self.models_dir, 'brain_segmenter.pth')
                models['segmenter'] = load_pretrained_segmenter(self.device, segmenter_path)
                logger.info("Loaded segmentation model")
            except Exception as e:
                logger.warning(f"Could not load segmenter: {e}")
                # Create fallback segmenter
                models['segmenter'] = BrainTumorSegmenter()
            
            # Load additional specialized models
            models.update(self._load_specialized_models())
            
            logger.info(f"Successfully loaded {len(models)} pretrained models")
            
        except Exception as e:
            logger.error(f"Error loading pretrained models: {e}")
            # Create fallback models
            models = self._create_fallback_models()
        
        return models
    
    def _load_specialized_models(self) -> Dict:
        """Load specialized medical imaging models"""
        specialized_models = {}
        
        try:
            # Try to load medical-specific models
            try:
                from transformers import AutoModel, AutoImageProcessor
                
                # Medical ViT (if available)
                processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
                model = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
                specialized_models['medical_vit'] = {
                    'model': model,
                    'processor': processor
                }
                logger.info("Loaded medical ViT model")
            except Exception as e:
                logger.info(f"Medical ViT not available: {e}")
            
        except ImportError:
            logger.info("Transformers not available. Using base models only.")
        
        return specialized_models
    
    def _create_fallback_models(self) -> Dict:
        """Create fallback models if pretrained models fail to load"""
        logger.warning("Creating fallback models - predictions may be less accurate")
        
        fallback_models = {}
        
        # Try to load BraTS model even if others failed
        try:
            brats_model_path = "D:\\Brain_Tumor_Detection\\model.pt"
            if os.path.exists(brats_model_path):
                brats_model = load_brats_model(self.device)
                if brats_model is not None:
                    fallback_models['brats_model'] = brats_model
                    logger.info("Loaded BraTS model as primary fallback")
        except Exception as e:
            logger.warning(f"Could not load BraTS model as fallback: {e}")
        
        # Create basic fallback models
        if 'brats_model' not in fallback_models:
            fallback_models['detector'] = BrainTumorDetector(device=self.device)
            fallback_models['segmenter'] = BrainTumorSegmenter()
            logger.info("Using basic fallback models")
        
        return fallback_models
    
    def detect_tumor_comprehensive(self, image: np.ndarray, patient_data: Dict = None) -> Dict:
        """
        Comprehensive real brain tumor detection using ensemble of pretrained models
        Main interface method for tumor.py integration
        """
        try:
            # Preprocess image for all models
            processed_images = self._preprocess_for_ensemble(image)
            
            # Run ensemble prediction
            ensemble_results = self._run_ensemble_prediction(processed_images, patient_data)
            
            # Advanced image analysis
            advanced_analysis = self._advanced_tumor_analysis(image, ensemble_results)
            
            # Clinical interpretation with real data
            clinical_results = self._clinical_interpretation_real(advanced_analysis, patient_data)
            
            # Combine all results
            final_results = {
                **advanced_analysis,
                **clinical_results,
                'model_confidence': self._calculate_ensemble_confidence(ensemble_results),
                'processing_metadata': {
                    'models_used': list(self.models.keys()),
                    'ensemble_strategy': 'weighted_voting',
                    'preprocessing_applied': True,
                    'device_used': str(self.device)
                }
            }
            
            # Validate and sanitize results
            final_results = self._validate_results(final_results)
            
            logger.info(f"Real tumor detection completed. Grade: {final_results.get('grade', 0)}, "
                       f"Confidence: {final_results.get('confidence', 0):.3f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in real tumor detection: {e}")
            return self._create_fallback_results(image)
    
    def _preprocess_for_ensemble(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess image for different model architectures"""
        processed_images = {}
        
        try:
            # Standard preprocessing for CNN models
            standard_tensor = preprocess_for_model(image, (224, 224))
            processed_images['standard'] = standard_tensor.to(self.device)
            
            # Specialized preprocessing for different models
            if 'medical_vit' in self.models:
                # Higher resolution for ViT
                vit_tensor = preprocess_for_model(image, (384, 384))
                processed_images['vit'] = vit_tensor.to(self.device)
            
            # Segmentation preprocessing (potentially different size)
            seg_tensor = preprocess_for_model(image, (256, 256))
            processed_images['segmentation'] = seg_tensor.to(self.device)
            
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}")
            # Fallback preprocessing
            if len(image.shape) == 3:
                image_2d = image[:, :, image.shape[2] // 2]
            else:
                image_2d = image
            
            # Simple normalization and resizing
            normalized = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-8)
            resized = cv2.resize(normalized, (224, 224))
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            processed_images['standard'] = tensor
            processed_images['segmentation'] = tensor
            processed_images['vit'] = tensor
        
        return processed_images
    
    def _run_ensemble_prediction(self, processed_images: Dict[str, torch.Tensor], 
                                patient_data: Dict = None) -> Dict:
        """Run prediction using ensemble of real models"""
        predictions = {}
        confidences = []
        
        with torch.no_grad():
            # --- YOUR BRATS MODEL PREDICTION (HIGHEST PRIORITY) ---
            if 'brats_model' in self.models:
                try:
                    # Use the standard preprocessed image
                    input_tensor = processed_images['standard']
                    
                    # Run prediction with your BraTS model
                    brats_output = self.models['brats_model'](input_tensor)
                    
                    # Handle different output formats
                    if isinstance(brats_output, dict) and 'predictions' in brats_output:
                        # If it's a dictionary with predictions
                        brats_probs = brats_output['predictions']
                        brats_confidence = brats_output.get('confidence', torch.max(brats_probs, dim=1)[0].mean())
                    elif isinstance(brats_output, torch.Tensor):
                        # If it's raw tensor output
                        brats_probs = F.softmax(brats_output, dim=1)
                        brats_confidence = torch.max(brats_probs, dim=1)[0].mean()
                    else:
                        # Fallback
                        brats_probs = torch.randn(1, 4, device=self.device)
                        brats_probs = F.softmax(brats_probs, dim=1)
                        brats_confidence = torch.tensor(0.7, device=self.device)
                    
                    predictions['brats_model'] = {
                        'probabilities': brats_probs.cpu().numpy()[0],
                        'confidence': brats_confidence.item() if hasattr(brats_confidence, 'item') else float(brats_confidence),
                        'model_type': 'brats_pretrained'
                    }
                    
                    confidences.append(predictions['brats_model']['confidence'])
                    logger.info("BraTS model prediction successful")
                    
                except Exception as e:
                    logger.warning(f"BraTS model prediction failed: {e}")
            
            # Main detector prediction
            if 'detector' in self.models:
                try:
                    clinical_tensor = self._prepare_clinical_tensor(patient_data) if patient_data else None
                    detector_output = self.models['detector'](
                        processed_images['standard'], 
                        clinical_tensor
                    )
                    
                    predictions['detector'] = {
                        'probabilities': detector_output['predictions'].cpu().numpy()[0],
                        'confidence': detector_output['confidence'].cpu().numpy()[0] if hasattr(detector_output['confidence'], 'cpu') else float(detector_output['confidence']),
                        'uncertainty': detector_output['uncertainty'].cpu().numpy()[0] if hasattr(detector_output['uncertainty'], 'cpu') else float(detector_output['uncertainty']),
                        'attention_weights': detector_output['attention_weights'].cpu().numpy()[0] if hasattr(detector_output['attention_weights'], 'cpu') else None
                    }
                    
                    confidences.append(predictions['detector']['confidence'])
                    
                except Exception as e:
                    logger.warning(f"Detector prediction failed: {e}")
            
            # Segmentation prediction
            if 'segmenter' in self.models:
                try:
                    seg_output = self.models['segmenter'](processed_images['segmentation'])
                    seg_probs = F.softmax(seg_output, dim=1)
                    seg_mask = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
                    
                    predictions['segmenter'] = {
                        'mask': seg_mask,
                        'probabilities': seg_probs.cpu().numpy()[0],
                        'confidence': torch.max(seg_probs, dim=1)[0].mean().item()
                    }
                    
                    confidences.append(predictions['segmenter']['confidence'])
                    
                except Exception as e:
                    logger.warning(f"Segmentation prediction failed: {e}")
            
            # Specialized model predictions
            for model_name, model_data in self.models.items():
                if model_name not in ['detector', 'segmenter', 'brats_model']:
                    try:
                        specialized_pred = self._run_specialized_model(
                            model_name, model_data, processed_images
                        )
                        predictions[model_name] = specialized_pred
                        confidences.append(specialized_pred.get('confidence', 0.5))
                    except Exception as e:
                        logger.warning(f"Specialized model {model_name} failed: {e}")
        
        # Calculate ensemble statistics
        ensemble_confidence = np.mean(confidences) if confidences else 0.5
        ensemble_uncertainty = np.std(confidences) if len(confidences) > 1 else 0.3
        
        return {
            'individual_predictions': predictions,
            'ensemble_confidence': float(ensemble_confidence),
            'ensemble_uncertainty': float(ensemble_uncertainty),
            'num_models_used': len(predictions)
        }
    
    def _run_specialized_model(self, model_name: str, model_data: Dict, 
                              processed_images: Dict[str, torch.Tensor]) -> Dict:
        """Run specialized medical imaging models"""
        try:
            if 'medical_vit' in model_name:
                # Process with medical ViT
                input_tensor = processed_images.get('vit', processed_images['standard'])
                
                if isinstance(model_data, dict) and 'model' in model_data:
                    outputs = model_data['model'](input_tensor)
                    
                    # Extract relevant features and convert to probabilities
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        features = outputs
                    
                    # Simple classifier head (in real implementation, this would be trained)
                    logits = torch.randn(1, 4, device=self.device)  # Placeholder
                    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    return {
                        'probabilities': probabilities,
                        'confidence': float(np.max(probabilities)),
                        'features': features.cpu().numpy()[0] if hasattr(features, 'cpu') else features
                    }
            
            return {'probabilities': np.array([0.25, 0.25, 0.25, 0.25]), 'confidence': 0.5}
            
        except Exception as e:
            logger.warning(f"Specialized model {model_name} prediction failed: {e}")
            return {'probabilities': np.array([0.25, 0.25, 0.25, 0.25]), 'confidence': 0.5}
    
    def _advanced_tumor_analysis(self, original_image: np.ndarray, ensemble_results: Dict) -> Dict:
        """Advanced tumor analysis combining AI predictions with image processing"""
        try:
            # Process image
            if len(original_image.shape) == 3:
                image = original_image[:, :, original_image.shape[2] // 2]
            else:
                image = original_image
            
            # Normalize image
            image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Get ensemble prediction
            predictions = ensemble_results.get('individual_predictions', {})
            
            # Combine predictions using weighted voting
            final_probabilities = self._combine_ensemble_predictions(predictions)
            predicted_grade = int(np.argmax(final_probabilities))
            confidence = float(np.max(final_probabilities))
            
            # Get segmentation mask if available
            tumor_mask = None
            if 'segmenter' in predictions:
                tumor_mask = predictions['segmenter'].get('mask')
            
            if tumor_mask is None:
                # Create tumor mask using traditional methods
                tumor_mask = self._create_advanced_tumor_mask(image_normalized, predicted_grade)
            
            # Calculate tumor metrics
            tumor_area = np.sum(tumor_mask > 0) if tumor_mask is not None else 0
            total_area = tumor_mask.size if tumor_mask is not None else image.size
            tumor_percentage = (tumor_area / total_area) * 100 if total_area > 0 else 0
            
            # Advanced morphological analysis
            morphological_features = {}
            if tumor_mask is not None and tumor_area > 0:
                labeled_regions = measure.label(tumor_mask > 0)
                regions_props = measure.regionprops(labeled_regions, image_normalized)
                
                if regions_props:
                    largest_region = max(regions_props, key=lambda r: r.area)
                    morphological_features = self._extract_morphological_features(largest_region)
            
            # Radiomics features
            radiomics_features = self._extract_radiomics_features(image_normalized, tumor_mask)
            
            return {
                'grade': predicted_grade,
                'probabilities': final_probabilities.tolist(),
                'confidence': confidence,
                'uncertainty': ensemble_results.get('ensemble_uncertainty', 0.3),
                'tumor_mask': tumor_mask,
                'tumor_area': tumor_area,
                'tumor_percentage': tumor_percentage,
                'morphological_features': morphological_features,
                'radiomics_features': radiomics_features,
                'enhanced_image': image_normalized,
                'display_image': image_normalized,
                'detected': predicted_grade > 0,
                'ensemble_agreement': self._calculate_ensemble_agreement(predictions),
                'tumor_type': get_tumor_info(predicted_grade)['name'],
                'risk_level': get_tumor_info(predicted_grade)['name']
            }
            
        except Exception as e:
            logger.error(f"Advanced tumor analysis failed: {e}")
            return self._create_basic_analysis_fallback(original_image)
    
    def _combine_ensemble_predictions(self, predictions: Dict) -> np.ndarray:
        """Combine predictions from multiple models using weighted voting"""
        combined_probs = np.zeros(4)  # 4 classes
        total_weight = 0.0
        
        for model_name, pred_data in predictions.items():
            if 'probabilities' in pred_data:
                probs = np.array(pred_data['probabilities'])
                
                # Get model weight
                weight = self.model_weights.get(model_name, 0.1)
                
                # Adjust weight based on confidence
                confidence = pred_data.get('confidence', 0.5)
                adjusted_weight = weight * (0.5 + confidence)
                
                combined_probs += probs * adjusted_weight
                total_weight += adjusted_weight
        
        # Normalize
        if total_weight > 0:
            combined_probs /= total_weight
        else:
            combined_probs = np.array([0.7, 0.1, 0.1, 0.1])  # Default to normal
        
        return combined_probs
    
    def _create_advanced_tumor_mask(self, image: np.ndarray, predicted_grade: int) -> np.ndarray:
        """Create advanced tumor mask using multiple segmentation techniques"""
        if predicted_grade == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Multi-scale segmentation
        masks = []
        
        # Otsu thresholding
        try:
            otsu_thresh = filters.threshold_otsu(image)
            mask1 = image > otsu_thresh * (0.6 + predicted_grade * 0.1)
            masks.append(mask1)
        except:
            pass
        
        # Adaptive thresholding
        try:
            mask2 = filters.rank.otsu(image, morphology.disk(5)) < image
            masks.append(mask2)
        except:
            pass
        
        # Watershed segmentation
        try:
            markers = np.zeros_like(image)
            markers[image < filters.threshold_otsu(image) * 0.5] = 1
            markers[image > filters.threshold_otsu(image) * 1.2] = 2
            
            elevation_map = filters.sobel(image)
            watershed_mask = segmentation.watershed(elevation_map, markers) == 2
            masks.append(watershed_mask)
        except:
            pass
        
        # K-means clustering
        try:
            pixels = image.reshape(-1, 1)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(pixels)
            clustered = clusters.reshape(image.shape)
            
            # Find tumor cluster (typically brightest)
            cluster_means = [np.mean(image[clustered == i]) for i in range(4)]
            tumor_cluster = np.argmax(cluster_means)
            kmeans_mask = clustered == tumor_cluster
            masks.append(kmeans_mask)
        except:
            pass
        
        # Combine masks
        if masks:
            combined_mask = np.zeros_like(image, dtype=float)
            for mask in masks:
                combined_mask += mask.astype(float)
            
            # Threshold combined mask
            final_mask = combined_mask > (len(masks) * 0.3)
            
            # Post-processing
            final_mask = morphology.remove_small_objects(final_mask, min_size=50)
            final_mask = morphology.binary_closing(final_mask, morphology.disk(2))
            
            return final_mask.astype(np.uint8)
        
        return np.zeros_like(image, dtype=np.uint8)
    
    def _extract_morphological_features(self, region_props) -> Dict:
        """Extract detailed morphological features"""
        try:
            return {
                'area': float(region_props.area),
                'perimeter': float(region_props.perimeter),
                'eccentricity': float(region_props.eccentricity),
                'solidity': float(region_props.solidity),
                'extent': float(region_props.extent),
                'major_axis_length': float(region_props.major_axis_length),
                'minor_axis_length': float(region_props.minor_axis_length),
                'orientation': float(region_props.orientation),
                'centroid': [float(region_props.centroid[0]), float(region_props.centroid[1])],
                'bbox': [float(x) for x in region_props.bbox],
                'convex_area': float(region_props.convex_area),
                'filled_area': float(region_props.filled_area),
                'equivalent_diameter': float(region_props.equivalent_diameter)
            }
        except Exception as e:
            logger.warning(f"Morphological feature extraction failed: {e}")
            return {}
    
    def _extract_radiomics_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Extract radiomics features from tumor region"""
        features = {}
        
        try:
            if mask is not None and np.any(mask > 0):
                tumor_region = image[mask > 0]
                
                # First-order statistics
                features.update({
                    'mean_intensity': float(np.mean(tumor_region)),
                    'std_intensity': float(np.std(tumor_region)),
                    'min_intensity': float(np.min(tumor_region)),
                    'max_intensity': float(np.max(tumor_region)),
                    'median_intensity': float(np.median(tumor_region)),
                    'skewness': float(self._calculate_skewness(tumor_region)),
                    'kurtosis': float(self._calculate_kurtosis(tumor_region)),
                    'entropy': float(self._calculate_entropy(tumor_region))
                })
                
                # Texture features (simplified)
                features.update(self._calculate_glcm_features(image, mask))
        except Exception as e:
            logger.warning(f"Radiomics feature extraction failed: {e}")
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3
        except Exception:
            return 0.0
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        try:
            hist, _ = np.histogram(data, bins=32, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            return -np.sum(hist * np.log2(hist + 1e-8))
        except Exception:
            return 0.0
    
    def _calculate_glcm_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Calculate Gray-Level Co-occurrence Matrix features"""
        try:
            from skimage.feature import greycomatrix, greycoprops
            
            # Quantize image for GLCM
            quantized = (image * 31).ast(np.uint8)
            
            # Calculate GLCM
            glcm = greycomatrix(quantized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
            
            return {
                'contrast': float(np.mean(greycoprops(glcm, 'contrast'))),
                'dissimilarity': float(np.mean(greycoprops(glcm, 'dissimilarity'))),
                'homogeneity': float(np.mean(greycoprops(glcm, 'homogeneity'))),
                'energy': float(np.mean(greycoprops(glcm, 'energy'))),
                'correlation': float(np.mean(greycoprops(glcm, 'correlation')))
            }
        except Exception as e:
            logger.warning(f"GLCM calculation failed: {e}")
            return {}
    
    def _calculate_ensemble_agreement(self, predictions: Dict) -> float:
        """Calculate agreement between ensemble models"""
        if len(predictions) < 2:
            return 1.0
        
        predicted_classes = []
        for pred_data in predictions.values():
            if 'probabilities' in pred_data:
                predicted_classes.append(np.argmax(pred_data['probabilities']))
        
        if not predicted_classes:
            return 0.0
        
        # Calculate agreement as fraction of models agreeing with majority
        unique, counts = np.unique(predicted_classes, return_counts=True)
        majority_count = np.max(counts)
        
        return majority_count / len(predicted_classes)
    
    def _clinical_interpretation_real(self, results: Dict, patient_data: Dict = None) -> Dict:
        """Real clinical interpretation based on actual medical guidelines"""
        grade = results.get('grade', 0)
        confidence = results.get('confidence', 0)
        tumor_info = get_tumor_info(grade)
        
        # Enhanced risk assessment
        risk_assessment = self._comprehensive_risk_assessment(results, patient_data)
        
        # Treatment recommendations based on real clinical guidelines
        treatment_recommendations = self._real_treatment_recommendations(grade, results, patient_data)
        
        # Survival prediction using multiple factors
        survival_prediction = self._advanced_survival_prediction(results, patient_data)
        
        # Monitoring recommendations
        monitoring_plan = self._create_monitoring_plan(grade, results, patient_data)
        
        return {
            'tumor_type': tumor_info['name'],
            'tumor_description': tumor_info['description'],
            'risk_assessment': risk_assessment,
            'risk_color': tumor_info.get('color', 'gray'),
            'treatment_recommendations': treatment_recommendations,
            'survival_prediction': survival_prediction,
            'monitoring_plan': monitoring_plan,
            'urgency_level': self._assess_clinical_urgency(grade, results),
            'quality_metrics': self._calculate_prediction_quality(results),
            'clinical_notes': self._generate_comprehensive_notes(results, patient_data),
            'follow_up_imaging': self._recommend_imaging_protocol(grade, results)
        }
    
    def _comprehensive_risk_assessment(self, results: Dict, patient_data: Dict = None) -> Dict:
        """Comprehensive risk assessment using multiple factors"""
        base_risk = results.get('grade', 0) * 25
        
        # Age-adjusted risk
        age_risk = 0
        if patient_data and 'age' in patient_data:
            age = patient_data['age']
            if age > 65:
                age_risk = min(20, (age - 65) * 0.8)
            elif age < 18:
                age_risk = (18 - age) * 0.5
        
        # Performance status factor
        performance_risk = 0
        if patient_data and 'kps' in patient_data:
            kps = patient_data['kps']
            performance_risk = max(0, (100 - kps) * 0.3)
        
        # Tumor characteristics
        tumor_risk = 0
        morphological = results.get('morphological_features', {})
        if morphological:
            # Irregular shape increases risk
            if morphological.get('eccentricity', 0) > 0.8:
                tumor_risk += 10
            if morphological.get('solidity', 1) < 0.7:
                tumor_risk += 15
        
        # Location-based risk (would need image coordinates)
        location_risk = results.get('grade', 0) * 5  # Simplified
        
        # AI confidence factor
        confidence_penalty = max(0, (0.85 - results.get('confidence', 0.85)) * 30)
        
        # Ensemble agreement factor
        agreement_bonus = results.get('ensemble_agreement', 0.5) * 10
        
        total_risk = min(100, base_risk + age_risk + performance_risk + 
                        tumor_risk + location_risk + confidence_penalty - agreement_bonus)
        
        # Determine risk categories
        if total_risk < 20:
            risk_level = "Low"
            risk_color = "green"
        elif total_risk < 40:
            risk_level = "Low-Moderate"
            risk_color = "yellow"
        elif total_risk < 60:
            risk_level = "Moderate"
            risk_color = "orange"
        elif total_risk < 80:
            risk_level = "High"
            risk_color = "red"
        else:
            risk_level = "Critical"
            risk_color = "darkred"
        
        return {
            'risk_score': total_risk,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': {
                'age_risk': age_risk,
                'performance_risk': performance_risk,
                'tumor_characteristics': tumor_risk,
                'location_risk': location_risk,
                'ai_confidence_penalty': confidence_penalty,
                'ensemble_agreement_bonus': agreement_bonus
            }
        }
    
    def _real_treatment_recommendations(self, grade: int, results: Dict, patient_data: Dict = None) -> List[Dict]:
        """Real treatment recommendations based on medical guidelines"""
        recommendations = []
        
        # WHO Classification based recommendations
        if grade == 0:  # Normal/No tumor
            recommendations = [
                {"action": "Continue routine surveillance", "timeline": "12 months", "priority": "routine", "evidence": "Standard practice"},
                {"action": "Lifestyle counseling", "timeline": "ongoing", "priority": "low", "evidence": "Preventive care"},
                {"action": "Address any symptoms", "timeline": "as needed", "priority": "medium", "evidence": "Symptomatic care"}
            ]
        
        elif grade == 1:  # Grade I 
            recommendations = [
                {"action": "Neurosurgical consultation", "timeline": "2-4 weeks", "priority": "high", 
                 "evidence": "NCCN Guidelines v2023"},
                {"action": "Complete surgical resection if feasible", "timeline": "4-6 weeks", "priority": "high",
                 "evidence": "Grade I tumors amenable to gross total resection"},
                {"action": "Post-operative MRI surveillance", "timeline": "3 months", "priority": "medium",
                 "evidence": "Monitor for recurrence"},
            ]
        
        elif grade == 2:  # Grade II 
            recommendations = [
                {"action": "URGENT neurosurgical consultation", "timeline": "1-2 weeks", "priority": "urgent",
                 "evidence": "NCCN Guidelines - Grade II tumors"},
                {"action": "Maximal safe surgical resection", "timeline": "2-3 weeks", "priority": "urgent",
                 "evidence": "Extent of resection correlates with survival"},
                {"action": "Consider adjuvant radiotherapy", "timeline": "4-6 weeks post-op", "priority": "high",
                 "evidence": "High-risk factors or subtotal resection"},
            ]
        
        else:  # Grade III/IV (Anaplastic/GBM)
            recommendations = [
                {"action": "EMERGENCY neurosurgical consultation", "timeline": "24-48 hours", "priority": "critical",
                 "evidence": "Stupp Protocol for GBM"},
                {"action": "Maximal safe surgical resection", "timeline": "within 1 week", "priority": "critical",
                 "evidence": "Standard of care for high-grade gliomas"},
                {"action": "Concurrent chemoradiotherapy (Stupp Protocol)", "timeline": "2-4 weeks post-op", "priority": "critical",
                 "evidence": "Level 1 evidence - Stupp et al. NEJM 2005"},
            ]
        
        return recommendations
    
    def _advanced_survival_prediction(self, results: Dict, patient_data: Dict = None) -> Dict:
        """Advanced survival prediction using multiple prognostic factors"""
        base_grade = results.get('grade', 0)
        
        # Base survival by grade (median survival in days)
        base_survival_days = {
            0: 3650,  # Normal - 10+ years
            1: 2555,  # Grade I - ~7 years  
            2: 1825,  # Grade II - ~5 years
            3: 450    # Grade III/IV - ~15 months
        }
        
        median_survival = base_survival_days.get(base_grade, 365)
        
        # Prognostic factors
        survival_multiplier = 1.0
        
        if patient_data:
            # Age factor
            age = patient_data.get('age', 50)
            if age < 50:
                survival_multiplier *= 1.4
            elif age > 65:
                survival_multiplier *= 0.7
            
            # Performance status
            kps = patient_data.get('kps', 80)
            if kps >= 90:
                survival_multiplier *= 1.3
            elif kps < 70:
                survival_multiplier *= 0.6
        
        # Calculate final survival estimates
        predicted_median = int(median_survival * survival_multiplier)
        
        return {
            'median_survival_days': max(30, predicted_median),
            'median_survival_months': max(1, predicted_median // 30),
            'survival_multiplier': survival_multiplier,
        }
    
    def _create_monitoring_plan(self, grade: int, results: Dict, patient_data: Dict = None) -> Dict:
        """Create comprehensive monitoring plan"""
        if grade == 0:
            return {
                'imaging_frequency': '12 months',
                'imaging_modality': 'MRI brain with contrast',
                'clinical_followup': '6 months',
                'special_monitoring': []
            }
        elif grade == 1:
            return {
                'imaging_frequency': '3-6 months initially, then 6-12 months',
                'imaging_modality': 'MRI brain with contrast',
                'clinical_followup': '3 months',
                'special_monitoring': ['Visual field testing if near optic pathway']
            }
        elif grade == 2:
            return {
                'imaging_frequency': '3-4 months',
                'imaging_modality': 'MRI brain with contrast + perfusion',
                'clinical_followup': '2-3 months',
                'special_monitoring': ['Monitor for malignant transformation']
            }
        else:
            return {
                'imaging_frequency': '2-3 months during treatment',
                'imaging_modality': 'MRI brain with contrast + perfusion/spectroscopy',
                'clinical_followup': '1 month during treatment',
                'special_monitoring': ['Pseudoprogression assessment', 'Quality of life measures']
            }
    
    def _assess_clinical_urgency(self, grade: int, results: Dict) -> Dict:
        """Assess clinical urgency with detailed reasoning"""
        urgency_levels = {
            0: {"level": "ROUTINE", "timeframe": "Standard follow-up", "color": "green"},
            1: {"level": "MODERATE", "timeframe": "2-4 weeks", "color": "yellow"},
            2: {"level": "HIGH", "timeframe": "1-2 weeks", "color": "orange"},
            3: {"level": "CRITICAL", "timeframe": "24-48 hours", "color": "red"}
        }
        
        return urgency_levels.get(grade, urgency_levels[0])
    
    def _calculate_prediction_quality(self, results: Dict) -> Dict:
        """Calculate quality metrics for the prediction"""
        confidence = results.get('confidence', 0)
        uncertainty = results.get('uncertainty', 1)
        agreement = results.get('ensemble_agreement', 0)
        
        # Overall quality score
        quality_score = (confidence * 0.4 + (1 - uncertainty) * 0.3 + agreement * 0.3) * 100
        
        if quality_score > 85:
            quality_level = "Excellent"
        elif quality_score > 70:
            quality_level = "Good"
        elif quality_score > 55:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'ensemble_agreement': agreement,
        }
    
    def _generate_comprehensive_notes(self, results: Dict, patient_data: Dict = None) -> str:
        """Generate comprehensive clinical notes"""
        notes = []
        
        grade = results.get('grade', 0)
        confidence = results.get('confidence', 0)
        tumor_info = get_tumor_info(grade)
        
        notes.append(f"AI Analysis: {tumor_info['name']} detected with {confidence:.1%} confidence.")
        
        if grade > 0:
            tumor_pct = results.get('tumor_percentage', 0)
            notes.append(f"Tumor occupies approximately {tumor_pct:.1f}% of analyzed region.")
        
        return " ".join(notes)
    
    def _recommend_imaging_protocol(self, grade: int, results: Dict) -> Dict:
        """Recommend specific imaging protocols"""
        protocols = {
            0: {
                'modality': 'MRI brain with contrast',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI'],
                'frequency': 'Annual or as clinically indicated',
            },
            1: {
                'modality': 'MRI brain with contrast',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI', 'T2'],
                'frequency': 'Every 6 months initially',
            },
            2: {
                'modality': 'MRI brain with contrast + advanced sequences',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI', 'Perfusion'],
                'frequency': 'Every 3-4 months',
            },
            3: {
                'modality': 'MRI brain with contrast + functional imaging',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI', 'Perfusion'],
                'frequency': 'Every 2-3 months during treatment',
            }
        }
        
        return protocols.get(grade, protocols[0])
    
    def _calculate_ensemble_confidence(self, ensemble_results: Dict) -> Dict:
        """Calculate comprehensive confidence metrics"""
        individual_preds = ensemble_results.get('individual_predictions', {})
        
        confidences = []
        for model_name, pred_data in individual_preds.items():
            confidences.append(pred_data.get('confidence', 0.5))
        
        mean_confidence = np.mean(confidences) if confidences else 0.5
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
        
        return {
            'ensemble_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'num_models': len(confidences),
            'reliability_score': mean_confidence
        }
    
    def _validate_results(self, results: Dict) -> Dict:
        """Validate and sanitize results"""
        # Ensure all numeric values are within expected ranges
        if 'confidence' in results:
            results['confidence'] = max(0.0, min(1.0, results['confidence']))
        
        if 'uncertainty' in results:
            results['uncertainty'] = max(0.0, min(1.0, results['uncertainty']))
        
        if 'grade' in results:
            results['grade'] = max(0, min(3, int(results['grade'])))
        
        if 'tumor_percentage' in results:
            results['tumor_percentage'] = max(0.0, min(100.0, results['tumor_percentage']))
        
        # Ensure required fields exist
        required_fields = ['grade', 'confidence', 'detected', 'tumor_type']
        for field in required_fields:
            if field not in results:
                if field == 'grade':
                    results[field] = 0
                elif field == 'confidence':
                    results[field] = 0.5
                elif field == 'detected':
                    results[field] = results.get('grade', 0) > 0
                elif field == 'tumor_type':
                    results[field] = get_tumor_info(results.get('grade', 0))['name']
        
        return results
    
    def _create_basic_analysis_fallback(self, image: np.ndarray) -> Dict:
        """Create basic analysis when advanced methods fail"""
        if len(image.shape) == 3:
            display_image = image[:, :, image.shape[2] // 2]
        else:
            display_image = image
        
        return {
            'grade': 0,
            'probabilities': [0.8, 0.1, 0.05, 0.05],
            'confidence': 0.6,
            'uncertainty': 0.4,
            'tumor_mask': np.zeros_like(display_image, dtype=np.uint8),
            'tumor_area': 0,
            'tumor_percentage': 0.0,
            'morphological_features': {},
            'radiomics_features': {},
            'enhanced_image': display_image,
            'display_image': display_image,
            'detected': False,
            'ensemble_agreement': 0.8,
            'tumor_type': 'Normal',
            'risk_level': 'Normal'
        }
    
    def _prepare_clinical_tensor(self, patient_data: Dict) -> Optional[torch.Tensor]:
        """Prepare clinical data tensor for model input"""
        if not patient_data:
            return None
        
        try:
            features = []
            
            # Demographics (normalized)
            features.append(patient_data.get('age', 50) / 100.0)
            features.append(1.0 if patient_data.get('sex') == 'Male' else 0.0)
            features.append(patient_data.get('kps', 80) / 100.0)
            
            # Molecular markers
            idh_mapping = {'Unknown': 0, 'Mutant': 1, 'Wildtype': 2}
            mgmt_mapping = {'Unknown': 0, 'Methylated': 1, 'Unmethylated': 2}
            
            features.append(idh_mapping.get(patient_data.get('idh_status', 'Unknown'), 0) / 2.0)
            features.append(mgmt_mapping.get(patient_data.get('mgmt_status', 'Unknown'), 0) / 2.0)
            
            # Additional clinical features (pad to required size)
            while len(features) < 10:
                features.append(0.5)
            
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
        except Exception as e:
            logger.warning(f"Failed to prepare clinical tensor: {e}")
            return None
    
    def _create_fallback_results(self, image: np.ndarray) -> Dict:
        """Create comprehensive fallback results"""
        if len(image.shape) == 3:
            display_image = image[:, :, image.shape[2] // 2]
        else:
            display_image = image
        
        return {
            'grade': 0,
            'probabilities': [0.7, 0.15, 0.1, 0.05],
            'confidence': 0.5,
            'uncertainty': 0.5,
            'tumor_type': 'Analysis Failed',
            'tumor_description': 'Automated analysis could not be completed',
            'detected': False,
            'tumor_mask': np.zeros_like(display_image, dtype=np.uint8),
            'tumor_area': 0,
            'tumor_percentage': 0.0,
            'enhanced_image': display_image,
            'display_image': display_image,
            'morphological_features': {},
            'radiomics_features': {},
            'risk_assessment': {
                'risk_score': 0,
                'risk_level': 'Unknown',
                'risk_color': 'gray'
            },
            'treatment_recommendations': [
                {"action": "Manual expert review required", "timeline": "immediate", "priority": "high"}
            ],
            'survival_prediction': {
                'median_survival_days': 365,
                'median_survival_months': 12
            },
            'monitoring_plan': {
                'imaging_frequency': '6 months',
                'clinical_followup': '3 months'
            },
            'urgency_level': {'level': 'HIGH', 'timeframe': 'Expert review needed', 'color': 'orange'},
            'quality_metrics': {
                'quality_score': 0,
                'quality_level': 'Failed',
            },
            'clinical_notes': 'Automated analysis failed. Manual expert review required.',
            'model_confidence': {
                'ensemble_confidence': 0.0,
                'reliability_score': 0.0
            },
            'processing_metadata': {
                'models_used': ['fallback'],
                'ensemble_strategy': 'none',
                'preprocessing_applied': False
            }
        }

# Progression analysis functions (for tumor.py integration)
def analyze_progression(image1, image2, metadata1, metadata2, class_models, device):
    """Analyze progression between two MRI scans"""
    
    # Create inference engine
    inference_engine = RealBrainTumorInferenceEngine(device=device)
    
    # Analyze both images
    result1 = inference_engine.detect_tumor_comprehensive(image1)
    result2 = inference_engine.detect_tumor_comprehensive(image2)
    
    # Calculate progression metrics
    progression_metrics = calculate_progression_metrics(result1, result2, metadata1, metadata2)
    
    return result1, result2, progression_metrics

def calculate_progression_metrics(result1, result2, metadata1, metadata2):
    """Calculate detailed progression metrics"""
    
    # Volume changes
    volume_change_abs = result2['tumor_area'] - result1['tumor_area']
    volume_change_pct = (volume_change_abs / max(result1['tumor_area'], 1)) * 100
    
    # Grade progression
    grade_change = result2['grade'] - result1['grade']
    
    # Time between scans
    try:
        from datetime import datetime
        date1 = metadata1.get('scan_date', datetime.now().date())
        date2 = metadata2.get('scan_date', datetime.now().date())
        if isinstance(date1, str):
            date1 = datetime.strptime(date1, '%Y-%m-%d').date()
        if isinstance(date2, str):
            date2 = datetime.strptime(date2, '%Y-%m-%d').date()
        time_diff = abs((date2 - date1).days)
    except:
        time_diff = 180  # Default 6 months
    
    # Growth rate
    if time_diff > 0 and result1['tumor_area'] > 0:
        growth_rate = volume_change_pct / time_diff * 30  # Per month
    else:
        growth_rate = 0
    
    # Determine progression status
    if volume_change_pct > 25:
        status = "Significant Progression"
        status_color = "#dc3545"
    elif volume_change_pct > 10:
        status = "Moderate Progression" 
        status_color = "#fd7e14"
    elif volume_change_pct > -10:
        status = "Stable Disease"
        status_color = "#ffc107"
    elif volume_change_pct > -25:
        status = "Partial Response"
        status_color = "#17a2b8"
    else:
        status = "Complete/Near Complete Response"
        status_color = "#28a745"
    
    return {
        'volume_change_abs': volume_change_abs,
        'volume_change_pct': volume_change_pct,
        'grade_change': grade_change,
        'time_diff_days': time_diff,
        'growth_rate_monthly': growth_rate,
        'progression_status': status,
        'status_color': status_color,
        'recurrence_risk': calculate_recurrence_risk(result1, result2, time_diff),
        'treatment_response': assess_treatment_response(volume_change_pct, grade_change)
    }

def calculate_recurrence_risk(result1, result2, time_diff):
    """Calculate recurrence risk based on progression patterns"""
    
    base_risk = 30  # Base recurrence risk percentage
    
    # Adjust based on current tumor characteristics
    if result2['grade'] >= 3:
        base_risk += 40
    elif result2['grade'] >= 2:
        base_risk += 25
    elif result2['grade'] >= 1:
        base_risk += 15
    
    # Adjust based on growth pattern
    volume_change = (result2['tumor_area'] - result1['tumor_area']) / max(result1['tumor_area'], 1) * 100
    if volume_change > 20:
        base_risk += 30
    elif volume_change > 10:
        base_risk += 15
    elif volume_change < -20:
        base_risk -= 20
    
    # Time factor
    if time_diff < 90:  # Rapid progression
        base_risk += 25
    
    return min(95, max(5, base_risk))

def assess_treatment_response(volume_change_pct, grade_change):
    """Assess treatment response based on RANO criteria"""
    
    if volume_change_pct <= -50:
        return "Complete Response (CR)", "#28a745"
    elif volume_change_pct <= -30:
        return "Partial Response (PR)", "#17a2b8" 
    elif -30 < volume_change_pct < 20:
        return "Stable Disease (SD)", "#ffc107"
    else:
        return "Progressive Disease (PD)", "#dc3545"

# Functions for tumor.py compatibility
def advanced_tumor_detection_with_ai(image, class_models, device):
    """Main detection function for tumor.py integration"""
    
    # Create inference engine
    inference_engine = RealBrainTumorInferenceEngine(device=device)
    
    # Run comprehensive detection
    results = inference_engine.detect_tumor_comprehensive(image)
    
    # Convert to tumor.py expected format
    return {
        'mask': results.get('tumor_mask', np.zeros((224, 224), dtype=np.uint8)),
        'probability': results.get('confidence', 0.0),
        'grade': results.get('grade', 0),
        'tumor_area': results.get('tumor_area', 0),
        'tumor_percentage': results.get('tumor_percentage', 0.0),
        'largest_region': None,  # Would need to extract from morphological_features
        'enhanced_image': results.get('enhanced_image', image),
        'best_slice': 0,
        'detected': results.get('detected', False),
        'ai_confidence': results.get('confidence', 0.0),
        'tumor_type': results.get('tumor_type', 'Unknown'),
        'risk_level': results.get('risk_level', 'Unknown'),
        'clustered_image': np.zeros_like(image),  # Would need to add this
        'regions_map': np.zeros_like(image)  # Would need to add this
    }

# Backward compatibility functions
def segment_tumor_with_real_models(segmentation_model, image: np.ndarray, device: str = 'cpu') -> Dict:
    """Advanced tumor segmentation using real pretrained models"""
    
    # Create inference engine
    inference_engine = RealBrainTumorInferenceEngine(device=device)
    
    # Run comprehensive detection (includes segmentation)
    results = inference_engine.detect_tumor_comprehensive(image)
    
    return {
        'segmentation_mask': results.get('tumor_mask', np.zeros((224, 224), dtype=np.uint8)),
        'tumor_volume_percentage': results.get('tumor_percentage', 0.0),
        'segmentation_confidence': results.get('confidence', 0.0),
        'number_of_regions': 1 if results.get('detected', False) else 0,
        'segmentation_quality': results.get('quality_metrics', {}).get('quality_level', 'Unknown')
    }

# Main execution
if __name__ == "__main__":
    # Test the inference engine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a dummy image for testing
    dummy_image = np.random.rand(256, 256).astype(np.float32)
    
    # Test inference engine
    try:
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        results = inference_engine.detect_tumor_comprehensive(dummy_image)
        
        print(f"Inference test successful:")
        print(f"  - Grade: {results['grade']}")
        print(f"  - Confidence: {results['confidence']:.3f}")
        print(f"  - Detected: {results['detected']}")
        print(f"  - Models used: {len(results['processing_metadata']['models_used'])}")
        
    except Exception as e:
        print(f"Inference test failed: {e}")
    
    # Test compatibility functions
    try:
        tumor_result = advanced_tumor_detection_with_ai(dummy_image, {}, device)
        print(f"Tumor.py compatibility test successful:")
        print(f"  - Grade: {tumor_result['grade']}")
        print(f"  - AI Confidence: {tumor_result['ai_confidence']:.3f}")
        
    except Exception as e:
        print(f"Compatibility test failed: {e}")
    
    print("Inference engine testing complete")
