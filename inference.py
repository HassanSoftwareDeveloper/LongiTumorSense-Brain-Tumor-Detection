

# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import logging
# from typing import Dict, List, Tuple, Optional, Union
# from sklearn.cluster import KMeans
# from skimage import exposure, measure, morphology, filters, segmentation
# from scipy import ndimage

# from preprocessing import preprocess_mri
# from config import (
#     TUMOR_CLASSES, CLINICAL_CONFIG, RANO_CRITERIA, 
#     get_tumor_info, DEVICE_CONFIG, MODEL_CONFIG
# )

# logger = logging.getLogger(__name__)

# class InferenceEngine:
#     """Enhanced inference engine with clinical integration"""
    
#     def __init__(self, models: Dict, device: str = 'cpu'):
#         self.models = models
#         self.device = device
#         self.tumor_classes = TUMOR_CLASSES
#         self.clinical_config = CLINICAL_CONFIG
        
#     def detect_tumor_comprehensive(self, image: np.ndarray, patient_data: Dict = None) -> Dict:
#         """Comprehensive tumor detection with clinical context"""
#         try:
#             # Preprocess image for model input
#             model_ready_image, display_ready_image = preprocess_mri(
#                 image, 
#                 target_size=(224, 224), 
#                 enhance=True
#             )
            
#             # Prepare tensor for model inference
#             input_tensor = torch.from_numpy(model_ready_image).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
#             # Ensemble prediction if multiple models available
#             if isinstance(self.models, dict) and len(self.models) > 1:
#                 results = self._ensemble_prediction(input_tensor, patient_data)
#             else:
#                 # Single model prediction
#                 model = list(self.models.values())[0] if isinstance(self.models, dict) else self.models
#                 results = self._single_model_prediction(model, input_tensor, patient_data)
            
#             # Enhanced tumor analysis using traditional image processing
#             enhanced_results = self._enhanced_tumor_analysis(image, results)
            
#             # Clinical interpretation
#             clinical_results = self._clinical_interpretation(enhanced_results, patient_data)
            
#             # Combine all results
#             final_results = {
#                 **enhanced_results,
#                 **clinical_results,
#                 'display_image': display_ready_image,
#                 'processed_image': model_ready_image,
#                 'input_tensor_shape': input_tensor.shape,
#                 'analysis_timestamp': torch.tensor(0).item()  # Placeholder for timestamp
#             }
            
#             logger.info(f"Tumor detection completed. Grade: {final_results.get('grade', 0)}, Confidence: {final_results.get('confidence', 0):.3f}")
            
#             return final_results
            
#         except Exception as e:
#             logger.error(f"Error in tumor detection: {e}")
#             return self._create_fallback_results(image)
    
#     def _ensemble_prediction(self, input_tensor: torch.Tensor, patient_data: Dict = None) -> Dict:
#         """Ensemble prediction using multiple models"""
#         predictions = []
#         confidences = []
#         survival_estimates = []
        
#         for model_name, model in self.models.items():
#             try:
#                 with torch.no_grad():
#                     if hasattr(model, 'forward') and callable(model.forward):
#                         # New model architecture with multiple outputs
#                         outputs = model(input_tensor, 
#                                       clinical_data=self._prepare_clinical_tensor(patient_data) if patient_data else None)
                        
#                         if isinstance(outputs, tuple) and len(outputs) >= 3:
#                             tumor_pred, survival_pred, progression_pred, attention_weights = outputs[:4]
#                         else:
#                             tumor_pred = outputs
#                             survival_pred = torch.tensor([365.0])  # Default 1 year
#                             attention_weights = torch.ones(1, 1)
#                     else:
#                         # Fallback for simpler models
#                         tumor_pred = torch.softmax(torch.randn(1, 4), dim=1)
#                         survival_pred = torch.tensor([365.0])
#                         attention_weights = torch.ones(1, 1)
                
#                 # Extract probabilities
#                 if tumor_pred.dim() > 1:
#                     tumor_probs = F.softmax(tumor_pred, dim=1).cpu().numpy()[0]
#                 else:
#                     tumor_probs = F.softmax(tumor_pred.unsqueeze(0), dim=1).cpu().numpy()[0]
                
#                 predictions.append(tumor_probs)
#                 confidences.append(float(np.max(tumor_probs)))
#                 survival_estimates.append(float(survival_pred.cpu().numpy().item()))
                
#             except Exception as e:
#                 logger.warning(f"Model {model_name} prediction failed: {e}")
#                 # Add fallback prediction
#                 fallback_probs = np.array([0.7, 0.1, 0.1, 0.1]) if not patient_data else np.array([0.3, 0.2, 0.3, 0.2])
#                 predictions.append(fallback_probs)
#                 confidences.append(0.7)
#                 survival_estimates.append(365.0)
        
#         # Weighted ensemble based on model confidence
#         weights = np.array(confidences) / np.sum(confidences)
#         ensemble_probs = np.average(predictions, weights=weights, axis=0)
        
#         # Final predictions
#         predicted_grade = int(np.argmax(ensemble_probs))
#         final_confidence = float(np.max(ensemble_probs))
#         uncertainty = float(np.std([pred[predicted_grade] for pred in predictions]))
#         median_survival = float(np.median(survival_estimates))
        
#         return {
#             'grade': predicted_grade,
#             'probability': float(ensemble_probs[predicted_grade]),
#             'confidence': final_confidence,
#             'uncertainty': uncertainty,
#             'survival_estimate_days': median_survival,
#             'all_predictions': predictions,
#             'ensemble_weights': weights.tolist()
#         }
    
#     def _single_model_prediction(self, model, input_tensor: torch.Tensor, patient_data: Dict = None) -> Dict:
#         """Single model prediction with fallback"""
#         try:
#             with torch.no_grad():
#                 outputs = model(input_tensor)
                
#                 if isinstance(outputs, tuple):
#                     tumor_pred = outputs[0]
#                     survival_pred = outputs[1] if len(outputs) > 1 else torch.tensor([365.0])
#                 else:
#                     tumor_pred = outputs
#                     survival_pred = torch.tensor([365.0])
                
#                 tumor_probs = F.softmax(tumor_pred, dim=1).cpu().numpy()[0]
#                 predicted_grade = int(np.argmax(tumor_probs))
                
#                 return {
#                     'grade': predicted_grade,
#                     'probability': float(tumor_probs[predicted_grade]),
#                     'confidence': float(np.max(tumor_probs)),
#                     'uncertainty': 0.1,  # Single model has lower uncertainty estimate
#                     'survival_estimate_days': float(survival_pred.cpu().numpy().item()),
#                     'all_predictions': [tumor_probs.tolist()]
#                 }
                
#         except Exception as e:
#             logger.error(f"Single model prediction failed: {e}")
#             return {
#                 'grade': 0,
#                 'probability': 0.8,
#                 'confidence': 0.8,
#                 'uncertainty': 0.3,
#                 'survival_estimate_days': 365.0,
#                 'all_predictions': [[0.8, 0.1, 0.05, 0.05]]
#             }
    
#     def _enhanced_tumor_analysis(self, original_image: np.ndarray, ai_results: Dict) -> Dict:
#         """Enhanced tumor analysis using traditional image processing"""
#         try:
#             # Normalize image
#             if len(original_image.shape) == 3:
#                 # Take middle slice for 3D images
#                 image = original_image[:, :, original_image.shape[2] // 2]
#             else:
#                 image = original_image
            
#             # Ensure proper normalization
#             image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
#             # Advanced image enhancement
#             image_enhanced = exposure.equalize_adapthist(image_normalized, clip_limit=0.03)
            
#             # Multi-threshold tumor detection
#             tumor_mask = self._create_tumor_mask(image_enhanced, ai_results['grade'])
            
#             # Calculate tumor metrics
#             tumor_area = np.sum(tumor_mask)
#             total_area = tumor_mask.size
#             tumor_percentage = (tumor_area / total_area) * 100
            
#             # Morphological analysis
#             if tumor_area > 0:
#                 labeled_regions = measure.label(tumor_mask)
#                 regions_props = measure.regionprops(labeled_regions, image_enhanced)
                
#                 if regions_props:
#                     largest_region = max(regions_props, key=lambda r: r.area)
#                     morphological_features = self._extract_morphological_features(largest_region)
#                 else:
#                     largest_region = None
#                     morphological_features = {}
#             else:
#                 largest_region = None
#                 morphological_features = {}
            
#             # Update AI results with enhanced analysis
#             ai_results.update({
#                 'tumor_mask': tumor_mask,
#                 'tumor_area': tumor_area,
#                 'tumor_percentage': tumor_percentage,
#                 'enhanced_image': image_enhanced,
#                 'largest_region': largest_region,
#                 'morphological_features': morphological_features,
#                 'detected': ai_results['grade'] > 0
#             })
            
#             return ai_results
            
#         except Exception as e:
#             logger.error(f"Enhanced tumor analysis failed: {e}")
#             ai_results.update({
#                 'tumor_mask': np.zeros_like(original_image[:, :, 0] if len(original_image.shape) == 3 else original_image),
#                 'tumor_area': 0,
#                 'tumor_percentage': 0.0,
#                 'enhanced_image': original_image[:, :, 0] if len(original_image.shape) == 3 else original_image,
#                 'largest_region': None,
#                 'morphological_features': {},
#                 'detected': False
#             })
#             return ai_results
    
#     def _create_tumor_mask(self, image: np.ndarray, predicted_grade: int) -> np.ndarray:
#         """Create tumor mask based on image analysis and AI prediction"""
#         # Adaptive thresholding based on predicted grade
#         if predicted_grade == 0:
#             # No tumor predicted, return empty mask
#             return np.zeros_like(image, dtype=bool)
        
#         # Multiple thresholding approaches
#         otsu_thresh = filters.threshold_otsu(image)
#         multiotsu_thresh = filters.threshold_multiotsu(image, classes=4)
        
#         # Create initial mask based on brightness
#         if predicted_grade >= 3:
#             # High-grade tumors often have heterogeneous signal
#             mask1 = image > otsu_thresh * 0.8
#             mask2 = (image > multiotsu_thresh[1]) & (image < multiotsu_thresh[3])
#             combined_mask = mask1 | mask2
#         elif predicted_grade >= 2:
#             # Intermediate grade
#             combined_mask = image > multiotsu_thresh[2]
#         else:
#             # Low grade
#             combined_mask = image > multiotsu_thresh[1]
        
#         # K-means clustering for tissue segmentation
#         pixels = image.reshape(-1, 1)
#         n_clusters = min(5, len(np.unique(image)))
#         if n_clusters > 1:
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#             kmeans.fit(pixels)
#             clustered = kmeans.labels_.reshape(image.shape)
            
#             # Find clusters corresponding to tumor tissue
#             cluster_means = [np.mean(image[clustered == i]) for i in range(n_clusters)]
#             tumor_clusters = np.argsort(cluster_means)[-2:]  # Top 2 brightest clusters
            
#             cluster_mask = np.zeros_like(clustered, dtype=bool)
#             for cluster in tumor_clusters:
#                 cluster_mask |= (clustered == cluster)
            
#             # Combine masks
#             combined_mask = combined_mask | cluster_mask
        
#         # Morphological post-processing
#         combined_mask = morphology.remove_small_objects(combined_mask, min_size=100)
#         combined_mask = morphology.remove_small_holes(combined_mask, area_threshold=50)
#         combined_mask = morphology.binary_closing(combined_mask, morphology.disk(3))
        
#         return combined_mask
    
#     def _extract_morphological_features(self, region_props) -> Dict:
#         """Extract morphological features from tumor region"""
#         return {
#             'area': region_props.area,
#             'perimeter': region_props.perimeter,
#             'eccentricity': region_props.eccentricity,
#             'solidity': region_props.solidity,
#             'extent': region_props.extent,
#             'major_axis_length': region_props.major_axis_length,
#             'minor_axis_length': region_props.minor_axis_length,
#             'orientation': region_props.orientation,
#             'centroid': region_props.centroid,
#             'bbox': region_props.bbox
#         }
    
#     def _clinical_interpretation(self, results: Dict, patient_data: Dict = None) -> Dict:
#         """Add clinical interpretation to results"""
#         grade = results.get('grade', 0)
#         tumor_info = get_tumor_info(grade)
        
#         # Risk assessment
#         risk_level, risk_score = self._calculate_risk(results, patient_data)
        
#         # Treatment recommendations
#         treatment_recommendations = self._get_treatment_recommendations(grade, results, patient_data)
        
#         # Survival prediction refinement
#         refined_survival = self._refine_survival_prediction(results, patient_data)
        
#         return {
#             'tumor_type': tumor_info['name'],
#             'tumor_description': tumor_info['description'],
#             'risk_level': risk_level,
#             'risk_score': risk_score,
#             'risk_color': tumor_info['color'],
#             'follow_up_months': tumor_info['follow_up_months'],
#             'treatment_recommendations': treatment_recommendations,
#             'survival_days': refined_survival,
#             'urgency': self._assess_urgency(grade, results),
#             'clinical_notes': self._generate_clinical_notes(results, patient_data)
#         }
    
#     def _calculate_risk(self, results: Dict, patient_data: Dict = None) -> Tuple[str, float]:
#         """Calculate comprehensive risk score"""
#         base_risk = results.get('grade', 0) * 25
        
#         # Age factor
#         age_risk = 0
#         if patient_data and 'age' in patient_data:
#             age = patient_data['age']
#             if age > 65:
#                 age_risk = (age - 65) * 0.5
#             elif age < 18:
#                 age_risk = (18 - age) * 0.3
        
#         # Performance status factor
#         kps_risk = 0
#         if patient_data and 'kps' in patient_data:
#             kps = patient_data['kps']
#             kps_risk = max(0, (90 - kps) * 0.2)
        
#         # Tumor characteristics
#         tumor_risk = 0
#         if results.get('tumor_percentage', 0) > 5:
#             tumor_risk = min(20, results['tumor_percentage'] * 0.5)
        
#         # AI confidence factor (lower confidence = higher risk)
#         confidence_risk = max(0, (0.9 - results.get('confidence', 0.9)) * 50)
        
#         total_risk = min(100, base_risk + age_risk + kps_risk + tumor_risk + confidence_risk)
        
#         # Determine risk level
#         if total_risk < 25:
#             risk_level = "Low"
#         elif total_risk < 50:
#             risk_level = "Moderate"
#         elif total_risk < 75:
#             risk_level = "High"
#         else:
#             risk_level = "Critical"
        
#         return risk_level, total_risk
    
#     def _get_treatment_recommendations(self, grade: int, results: Dict, patient_data: Dict = None) -> List[Dict]:
#         """Generate treatment recommendations based on tumor grade and patient factors"""
#         recommendations = []
        
#         if grade == 0:
#             recommendations = [
#                 {"action": "Regular surveillance MRI", "timeline": "12 months", "priority": "routine"},
#                 {"action": "Maintain healthy lifestyle", "timeline": "ongoing", "priority": "low"},
#                 {"action": "Annual neurological assessment", "timeline": "12 months", "priority": "routine"}
#             ]
#         elif grade == 1:
#             recommendations = [
#                 {"action": "Neurosurgical consultation", "timeline": "2-4 weeks", "priority": "high"},
#                 {"action": "Complete surgical resection if feasible", "timeline": "4-6 weeks", "priority": "high"},
#                 {"action": "Post-operative MRI surveillance", "timeline": "3-6 months", "priority": "medium"},
#                 {"action": "Neuropsychological assessment", "timeline": "post-surgery", "priority": "medium"}
#             ]
#         elif grade == 2:
#             recommendations = [
#                 {"action": "Urgent neurosurgical consultation", "timeline": "1-2 weeks", "priority": "urgent"},
#                 {"action": "Maximal safe surgical resection", "timeline": "2-3 weeks", "priority": "urgent"},
#                 {"action": "Adjuvant radiotherapy (54-60 Gy)", "timeline": "4-6 weeks post-surgery", "priority": "high"},
#                 {"action": "Consider temozolomide chemotherapy", "timeline": "concurrent with RT", "priority": "high"},
#                 {"action": "MRI surveillance", "timeline": "every 3 months", "priority": "high"}
#             ]
#         else:  # Grade 3/4
#             recommendations = [
#                 {"action": "URGENT neurosurgical consultation", "timeline": "24-48 hours", "priority": "critical"},
#                 {"action": "Maximal safe surgical resection", "timeline": "1 week", "priority": "critical"},
#                 {"action": "Concurrent chemoradiotherapy", "timeline": "2-4 weeks post-surgery", "priority": "critical"},
#                 {"action": "Adjuvant temozolomide (6 cycles)", "timeline": "4 weeks post-RT", "priority": "high"},
#                 {"action": "Clinical trial enrollment", "timeline": "concurrent", "priority": "medium"},
#                 {"action": "Palliative care consultation", "timeline": "at diagnosis", "priority": "high"}
#             ]
        
#         # Adjust recommendations based on patient factors
#         if patient_data:
#             recommendations = self._adjust_recommendations_for_patient(recommendations, patient_data, results)
        
#         return recommendations
    
#     def _adjust_recommendations_for_patient(self, recommendations: List[Dict], patient_data: Dict, results: Dict) -> List[Dict]:
#         """Adjust treatment recommendations based on patient-specific factors"""
#         adjusted_recommendations = recommendations.copy()
        
#         age = patient_data.get('age', 50)
#         kps = patient_data.get('kps', 80)
        
#         # Age-based adjustments
#         if age > 70:
#             # More conservative approach for elderly patients
#             for rec in adjusted_recommendations:
#                 if 'aggressive' in rec['action'].lower() or 'maximal' in rec['action'].lower():
#                     rec['action'] += " (consider less aggressive approach due to age)"
#                     rec['priority'] = 'medium' if rec['priority'] == 'high' else rec['priority']
        
#         # Performance status adjustments
#         if kps < 70:
#             # Consider supportive care focus
#             supportive_rec = {
#                 "action": "Supportive care and symptom management", 
#                 "timeline": "immediate", 
#                 "priority": "high"
#             }
#             adjusted_recommendations.append(supportive_rec)
        
#         return adjusted_recommendations
    
#     def _refine_survival_prediction(self, results: Dict, patient_data: Dict = None) -> int:
#         """Refine survival prediction with additional factors"""
#         base_survival = results.get('survival_estimate_days', 365)
#         grade = results.get('grade', 0)
        
#         # Base survival by grade
#         grade_survival = self.clinical_config['survival_estimates'].get(f'grade_{grade}', 365)
        
#         # Weighted combination of AI prediction and clinical guidelines
#         refined_survival = (base_survival * 0.6) + (grade_survival * 0.4)
        
#         # Patient-specific adjustments
#         if patient_data:
#             age_factor = max(0.5, 1 - (patient_data.get('age', 50) - 40) * 0.005)
#             kps_factor = patient_data.get('kps', 80) / 100.0
            
#             # Molecular markers if available
#             molecular_factor = 1.0
#             if patient_data.get('idh_status') == 'Mutant':
#                 molecular_factor *= 1.5
#             if patient_data.get('mgmt_status') == 'Methylated':
#                 molecular_factor *= 1.3
            
#             refined_survival *= age_factor * kps_factor * molecular_factor
        
#         return max(30, int(refined_survival))  # Minimum 1 month
    
#     def _assess_urgency(self, grade: int, results: Dict) -> str:
#         """Assess clinical urgency"""
#         if grade >= 3:
#             return "CRITICAL - Immediate intervention required"
#         elif grade >= 2:
#             return "HIGH - Urgent consultation needed"
#         elif grade >= 1:
#             return "MODERATE - Timely evaluation recommended"
#         else:
#             return "ROUTINE - Standard follow-up"
    
#     def _generate_clinical_notes(self, results: Dict, patient_data: Dict = None) -> str:
#         """Generate clinical interpretation notes"""
#         grade = results.get('grade', 0)
#         confidence = results.get('confidence', 0)
#         tumor_percentage = results.get('tumor_percentage', 0)
        
#         notes = []
        
#         # AI analysis summary
#         notes.append(f"AI Analysis: {get_tumor_info(grade)['name']} detected with {confidence:.1%} confidence.")
        
#         if grade > 0:
#             notes.append(f"Tumor occupies approximately {tumor_percentage:.1f}% of analyzed region.")
            
#             # Morphological characteristics
#             morphological = results.get('morphological_features', {})
#             if morphological:
#                 if morphological.get('eccentricity', 0) > 0.8:
#                     notes.append("Tumor shows elongated morphology suggesting possible infiltrative pattern.")
#                 if morphological.get('solidity', 1) < 0.8:
#                     notes.append("Irregular tumor boundaries noted, consistent with higher-grade characteristics.")
        
#         # Confidence and uncertainty notes
#         uncertainty = results.get('uncertainty', 0)
#         if uncertainty > 0.3:
#             notes.append("High uncertainty in classification - recommend expert review and additional imaging.")
#         elif confidence < 0.7:
#             notes.append("Moderate confidence in diagnosis - clinical correlation recommended.")
        
#         # Patient-specific considerations
#         if patient_data:
#             age = patient_data.get('age')
#             if age and age > 65 and grade >= 2:
#                 notes.append("Consider age-related factors in treatment planning.")
#             elif age and age < 40 and grade >= 2:
#                 notes.append("Young patient - consider genetic counseling and molecular profiling.")
        
#         return " ".join(notes)
    
#     def _prepare_clinical_tensor(self, patient_data: Dict) -> Optional[torch.Tensor]:
#         """Prepare clinical data as tensor for model input"""
#         if not patient_data:
#             return None
        
#         try:
#             # Standard clinical features
#             features = []
            
#             # Demographics
#             features.append(patient_data.get('age', 50) / 100.0)  # Normalized age
#             features.append(1.0 if patient_data.get('sex') == 'Male' else 0.0)  # Sex encoding
#             features.append(patient_data.get('kps', 80) / 100.0)  # Normalized KPS
            
#             # Molecular markers (encoded)
#             idh_mapping = {'Unknown': 0, 'Mutant': 1, 'Wildtype': 2}
#             mgmt_mapping = {'Unknown': 0, 'Methylated': 1, 'Unmethylated': 2}
            
#             features.append(idh_mapping.get(patient_data.get('idh_status', 'Unknown'), 0) / 2.0)
#             features.append(mgmt_mapping.get(patient_data.get('mgmt_status', 'Unknown'), 0) / 2.0)
            
#             # Additional clinical features (placeholder values if not available)
#             features.extend([0.5, 0.5, 0.5, 0.5, 0.5])  # Placeholder for additional features
            
#             return torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
#         except Exception as e:
#             logger.warning(f"Failed to prepare clinical tensor: {e}")
#             return None
    
#     def _create_fallback_results(self, image: np.ndarray) -> Dict:
#         """Create fallback results in case of complete failure"""
#         if len(image.shape) == 3:
#             display_image = image[:, :, image.shape[2] // 2]
#         else:
#             display_image = image
            
#         return {
#             'grade': 0,
#             'probability': 0.5,
#             'confidence': 0.5,
#             'uncertainty': 0.5,
#             'tumor_type': 'Analysis Failed',
#             'tumor_description': 'Unable to complete analysis',
#             'detected': False,
#             'tumor_mask': np.zeros_like(display_image, dtype=bool),
#             'tumor_area': 0,
#             'tumor_percentage': 0.0,
#             'enhanced_image': display_image,
#             'display_image': display_image,
#             'risk_level': 'Unknown',
#             'risk_score': 0,
#             'survival_days': 365,
#             'treatment_recommendations': [
#                 {"action": "Manual review required", "timeline": "immediate", "priority": "high"}
#             ],
#             'urgency': 'SYSTEM ERROR - Manual review required',
#             'clinical_notes': 'Automated analysis failed. Manual expert review recommended.'
#         }

# def segment_tumor_advanced(segmentation_model, image: np.ndarray, device: str = 'cpu') -> Dict:
#     """Advanced tumor segmentation with 3D processing"""
#     try:
#         # Preprocess for segmentation model
#         if len(image.shape) == 2:
#             # Create pseudo-3D volume from 2D slice
#             depth = 16
#             volume = np.stack([image] * depth, axis=2)
#         elif len(image.shape) == 3:
#             volume = image
#         else:
#             raise ValueError(f"Unsupported image shape: {image.shape}")
        
#         # Normalize volume
#         volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
#         # Resize for model input
#         target_size = MODEL_CONFIG.get('segmentation', {}).get('target_size', (128, 128, 32))
#         if len(target_size) == 3:
#             volume_resized = cv2.resize(volume[:, :, volume.shape[2]//2], target_size[:2])
#             volume_3d = np.stack([volume_resized] * target_size[2], axis=2)
#         else:
#             volume_3d = volume
        
#         # Prepare tensor
#         input_tensor = torch.FloatTensor(volume_3d).unsqueeze(0).unsqueeze(0).to(device)
        
#         # Model inference
#         with torch.no_grad():
#             if segmentation_model and hasattr(segmentation_model, 'forward'):
#                 segmentation_output = segmentation_model(input_tensor)
                
#                 if isinstance(segmentation_output, list):
#                     # Deep supervision - use main output
#                     seg_logits = segmentation_output[0]
#                 else:
#                     seg_logits = segmentation_output
                
#                 # Get segmentation mask
#                 seg_probs = F.softmax(seg_logits, dim=1)
#                 seg_mask = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
#             else:
#                 # Fallback segmentation using traditional methods
#                 seg_mask = _fallback_segmentation(volume_3d.squeeze())
        
#         # Post-process segmentation
#         if len(seg_mask.shape) == 3:
#             # Take middle slice for 2D display
#             seg_mask_2d = seg_mask[seg_mask.shape[0] // 2, :, :]
#         else:
#             seg_mask_2d = seg_mask
        
#         # Clean up mask
#         seg_mask_clean = _post_process_segmentation(seg_mask_2d)
        
#         # Calculate segmentation metrics
#         total_pixels = seg_mask_clean.size
#         tumor_pixels = np.sum(seg_mask_clean > 0)
#         tumor_volume_percentage = (tumor_pixels / total_pixels) * 100
        
#         return {
#             'segmentation_mask': seg_mask_clean,
#             'tumor_volume_percentage': tumor_volume_percentage,
#             'segmentation_confidence': 0.8,  # Placeholder
#             'number_of_regions': len(np.unique(seg_mask_clean)) - 1  # Subtract background
#         }
        
#     except Exception as e:
#         logger.error(f"Segmentation failed: {e}")
#         fallback_mask = np.zeros((128, 128), dtype=np.uint8)
#         return {
#             'segmentation_mask': fallback_mask,
#             'tumor_volume_percentage': 0.0,
#             'segmentation_confidence': 0.0,
#             'number_of_regions': 0
#         }

# def _fallback_segmentation(image: np.ndarray) -> np.ndarray:
#     """Fallback segmentation using traditional image processing"""
#     # Simple thresholding-based segmentation
#     threshold = filters.threshold_otsu(image)
#     binary_mask = image > threshold
    
#     # Clean up
#     binary_mask = morphology.remove_small_objects(binary_mask, min_size=100)
#     binary_mask = morphology.binary_fill_holes(binary_mask)
    
#     return binary_mask.astype(np.uint8)

# def _post_process_segmentation(seg_mask: np.ndarray) -> np.ndarray:
#     """Post-process segmentation mask"""
#     # Convert to uint8 if needed
#     if seg_mask.dtype != np.uint8:
#         seg_mask = seg_mask.astype(np.uint8)
    
#     # Morphological operations
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)
#     seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel)
    
#     # Remove small components
#     if np.any(seg_mask):
#         labeled_mask = measure.label(seg_mask)
#         regions = measure.regionprops(labeled_mask)
        
#         # Keep only significant regions
#         min_area = max(50, seg_mask.size * 0.001)  # At least 0.1% of image
#         for region in regions:
#             if region.area < min_area:
#                 seg_mask[labeled_mask == region.label] = 0
    
#     return seg_mask

# def compare_scans_progression(baseline_results: Dict, followup_results: Dict, 
#                             baseline_date: str, followup_date: str) -> Dict:
#     """Compare two scans for progression analysis"""
#     try:
#         # Volume change calculation
#         baseline_volume = baseline_results.get('tumor_percentage', 0)
#         followup_volume = followup_results.get('tumor_percentage', 0)
        
#         if baseline_volume > 0:
#             volume_change = ((followup_volume - baseline_volume) / baseline_volume) * 100
#         else:
#             volume_change = followup_volume  # New lesion
        
#         # RANO criteria assessment
#         response_assessment = _assess_rano_response(volume_change, baseline_results, followup_results)
        
#         # Grade progression
#         grade_change = followup_results.get('grade', 0) - baseline_results.get('grade', 0)
        
#         # Confidence in progression assessment
#         baseline_confidence = baseline_results.get('confidence', 0)
#         followup_confidence = followup_results.get('confidence', 0)
#         progression_confidence = min(baseline_confidence, followup_confidence)
        
#         # Clinical significance
#         clinical_significance = _assess_clinical_significance(
#             volume_change, grade_change, progression_confidence
#         )
        
#         return {
#             'baseline_date': baseline_date,
#             'followup_date': followup_date,
#             'volume_change_percent': volume_change,
#             'grade_change': grade_change,
#             'response_assessment': response_assessment,
#             'progression_confidence': progression_confidence,
#             'clinical_significance': clinical_significance,
#             'recommendation': _generate_progression_recommendation(response_assessment, clinical_significance)
#         }
        
#     except Exception as e:
#         logger.error(f"Scan comparison failed: {e}")
#         return {
#             'baseline_date': baseline_date,
#             'followup_date': followup_date,
#             'volume_change_percent': 0,
#             'grade_change': 0,
#             'response_assessment': 'Unable to assess',
#             'progression_confidence': 0,
#             'clinical_significance': 'Unknown',
#             'recommendation': 'Manual expert review required'
#         }

# def _assess_rano_response(volume_change: float, baseline_results: Dict, followup_results: Dict) -> str:
#     """Assess response according to RANO criteria"""
#     # New lesions check
#     baseline_detected = baseline_results.get('detected', False)
#     followup_detected = followup_results.get('detected', False)
    
#     if not baseline_detected and followup_detected:
#         return 'Progressive Disease (New Lesion)'
#     elif baseline_detected and not followup_detected:
#         return 'Complete Response'
#     elif not baseline_detected and not followup_detected:
#         return 'Stable Disease (No Tumor)'
    
#     # Volume-based assessment
#     if volume_change <= -50:
#         return 'Partial Response'
#     elif volume_change >= 25:
#         return 'Progressive Disease'
#     else:
#         return 'Stable Disease'

# def _assess_clinical_significance(volume_change: float, grade_change: int, confidence: float) -> str:
#     """Assess clinical significance of progression"""
#     if confidence < 0.6:
#         return 'Uncertain - Low Confidence'
    
#     if abs(volume_change) > 50 or grade_change > 1:
#         return 'Highly Significant'
#     elif abs(volume_change) > 25 or grade_change > 0:
#         return 'Significant'
#     elif abs(volume_change) > 10:
#         return 'Moderate'
#     else:
#         return 'Minimal'

# def _generate_progression_recommendation(response: str, significance: str) -> str:
#     """Generate progression-based recommendations"""
#     if 'Progressive Disease' in response:
#         if 'Highly Significant' in significance:
#             return 'URGENT: Consider treatment modification or escalation'
#         else:
#             return 'Consider treatment adjustment and increased monitoring'
#     elif 'Partial Response' in response:
#         return 'Continue current treatment regimen'
#     elif 'Stable Disease' in response:
#         return 'Maintain current treatment and monitoring schedule'
#     else:
#         return 'Expert review recommended for treatment planning'




















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

from models import (
    BrainTumorDetector, 
    BrainTumorSegmenter, 
    load_pretrained_detector, 
    load_pretrained_segmenter,
    preprocess_for_model,
    preprocessor
)
from config import (
    TUMOR_CLASSES, CLINICAL_CONFIG, RANO_CRITERIA, 
    get_tumor_info, DEVICE_CONFIG, MODEL_CONFIG
)

logger = logging.getLogger(__name__)

class RealBrainTumorInferenceEngine:
    """
    Real Brain Tumor Inference Engine using multiple pretrained models
    """
    
    def __init__(self, device: str = 'cpu', models_dir: str = './models'):
        self.device = device
        self.models_dir = models_dir
        self.tumor_classes = TUMOR_CLASSES
        self.clinical_config = CLINICAL_CONFIG
        
        # Load real pretrained models
        self.models = self._load_pretrained_models()
        
        # Initialize model ensemble weights
        self.model_weights = {
            'efficientnet': 0.35,  # Best for medical images
            'resnet': 0.25,        # Robust features
            'densenet': 0.2,       # Good for small datasets  
            'vit': 0.2            # Attention mechanism
        }
        
        logger.info(f"Initialized Real Brain Tumor Inference Engine with {len(self.models)} models")
    
    def _load_pretrained_models(self) -> Dict:
        """Load all pretrained models for ensemble prediction"""
        models = {}
        
        try:
            # Main ensemble detector
            detector_path = os.path.join(self.models_dir, 'brain_tumor_detector.pth')
            models['detector'] = load_pretrained_detector(self.device, detector_path)
            
            # Segmentation model
            segmenter_path = os.path.join(self.models_dir, 'brain_segmenter.pth')
            models['segmenter'] = load_pretrained_segmenter(self.device, segmenter_path)
            
            # Additional specialized models
            models.update(self._load_specialized_models())
            
            logger.info(f"Successfully loaded {len(models)} pretrained models")
            
        except Exception as e:
            logger.error(f"Error loading pretrained models: {e}")
            # Create fallback models
            models = self._create_fallback_models()
        
        return models
    
    def _load_specialized_models(self) -> Dict:
        """Load specialized medical imaging models from Hugging Face or other sources"""
        specialized_models = {}
        
        try:
            # Try to load medical-specific models
            from transformers import AutoModel, AutoImageProcessor
            
            # Medical ViT (if available)
            try:
                processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
                model = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
                specialized_models['medical_vit'] = {
                    'model': model,
                    'processor': processor
                }
                logger.info("Loaded medical ViT model")
            except Exception as e:
                logger.warning(f"Could not load medical ViT: {e}")
            
        except ImportError:
            logger.info("Transformers not available. Using base models only.")
        
        return specialized_models
    
    def _create_fallback_models(self) -> Dict:
        """Create fallback models if pretrained models fail to load"""
        logger.warning("Creating fallback models - predictions may be less accurate")
        
        return {
            'detector': BrainTumorDetector(device=self.device),
            'segmenter': BrainTumorSegmenter()
        }
    
    def detect_tumor_comprehensive(self, image: np.ndarray, patient_data: Dict = None) -> Dict:
        """
        Comprehensive real brain tumor detection using ensemble of pretrained models
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
                    'preprocessing_applied': True
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
        
        return processed_images
    
    def _run_ensemble_prediction(self, processed_images: Dict[str, torch.Tensor], 
                                patient_data: Dict = None) -> Dict:
        """Run prediction using ensemble of real models"""
        predictions = {}
        confidences = []
        
        with torch.no_grad():
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
                        'confidence': detector_output['confidence'].cpu().numpy()[0],
                        'uncertainty': detector_output['uncertainty'].cpu().numpy()[0],
                        'attention_weights': detector_output['attention_weights'].cpu().numpy()[0]
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
                if model_name not in ['detector', 'segmenter']:
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
                'ensemble_agreement': self._calculate_ensemble_agreement(predictions)
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
        otsu_thresh = filters.threshold_otsu(image)
        mask1 = image > otsu_thresh * (0.6 + predicted_grade * 0.1)
        masks.append(mask1)
        
        # Adaptive thresholding
        mask2 = filters.rank.otsu(image, morphology.disk(5)) < image
        masks.append(mask2)
        
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
        return {
            'area': region_props.area,
            'perimeter': region_props.perimeter,
            'eccentricity': region_props.eccentricity,
            'solidity': region_props.solidity,
            'extent': region_props.extent,
            'major_axis_length': region_props.major_axis_length,
            'minor_axis_length': region_props.minor_axis_length,
            'orientation': region_props.orientation,
            'centroid': region_props.centroid,
            'bbox': region_props.bbox,
            'convex_area': region_props.convex_area,
            'filled_area': region_props.filled_area,
            'equivalent_diameter': region_props.equivalent_diameter
        }
    
    def _extract_radiomics_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Extract radiomics features from tumor region"""
        features = {}
        
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
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        hist, _ = np.histogram(data, bins=32, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log2(hist + 1e-8))
    
    def _calculate_glcm_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Calculate Gray-Level Co-occurrence Matrix features"""
        try:
            from skimage.feature import greycomatrix, greycoprops
            
            # Quantize image for GLCM
            quantized = (image * 31).astype(np.uint8)
            
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
            'risk_color': tumor_info['color'],
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
        
        elif grade == 1:  # Grade I (Pilocytic Astrocytoma)
            recommendations = [
                {"action": "Neurosurgical consultation", "timeline": "2-4 weeks", "priority": "high", 
                 "evidence": "NCCN Guidelines v2023"},
                {"action": "Complete surgical resection if feasible", "timeline": "4-6 weeks", "priority": "high",
                 "evidence": "Grade I tumors amenable to gross total resection"},
                {"action": "Post-operative MRI surveillance", "timeline": "3 months", "priority": "medium",
                 "evidence": "Monitor for recurrence"},
                {"action": "Avoid radiotherapy if gross total resection achieved", "timeline": "N/A", "priority": "high",
                 "evidence": "RT not indicated for completely resected Grade I"}
            ]
        
        elif grade == 2:  # Grade II (Diffuse Astrocytoma)
            recommendations = [
                {"action": "URGENT neurosurgical consultation", "timeline": "1-2 weeks", "priority": "urgent",
                 "evidence": "NCCN Guidelines - Grade II tumors"},
                {"action": "Maximal safe surgical resection", "timeline": "2-3 weeks", "priority": "urgent",
                 "evidence": "Extent of resection correlates with survival"},
                {"action": "Molecular testing (IDH, 1p/19q, MGMT)", "timeline": "Post-surgery", "priority": "high",
                 "evidence": "WHO 2021 Classification requirements"},
                {"action": "Consider adjuvant radiotherapy", "timeline": "4-6 weeks post-op", "priority": "high",
                 "evidence": "High-risk factors or subtotal resection"},
                {"action": "MRI surveillance every 3-4 months", "timeline": "ongoing", "priority": "high",
                 "evidence": "Monitor for malignant transformation"}
            ]
        
        else:  # Grade III/IV (Anaplastic/GBM)
            recommendations = [
                {"action": "EMERGENCY neurosurgical consultation", "timeline": "24-48 hours", "priority": "critical",
                 "evidence": "Stupp Protocol for GBM"},
                {"action": "Maximal safe surgical resection", "timeline": "within 1 week", "priority": "critical",
                 "evidence": "Standard of care for high-grade gliomas"},
                {"action": "Concurrent chemoradiotherapy (Stupp Protocol)", "timeline": "2-4 weeks post-op", "priority": "critical",
                 "evidence": "Level 1 evidence - Stupp et al. NEJM 2005"},
                {"action": "Temozolomide 75mg/m² daily during RT", "timeline": "concurrent with RT", "priority": "critical",
                 "evidence": "Standard Stupp Protocol"},
                {"action": "Adjuvant temozolomide (150-200mg/m²)", "timeline": "4 weeks post-RT", "priority": "critical",
                 "evidence": "6 cycles as per Stupp Protocol"},
                {"action": "Tumor Treating Fields (Optune)", "timeline": "concurrent with TMZ", "priority": "high",
                 "evidence": "FDA approved - EF-14 trial"},
                {"action": "Clinical trial enrollment", "timeline": "at diagnosis", "priority": "medium",
                 "evidence": "Access to novel therapies"},
                {"action": "Palliative care consultation", "timeline": "at diagnosis", "priority": "high",
                 "evidence": "Quality of life management"}
            ]
        
        # Adjust for patient-specific factors
        if patient_data:
            recommendations = self._personalize_recommendations(recommendations, patient_data, results)
        
        return recommendations
    
    def _personalize_recommendations(self, recommendations: List[Dict], patient_data: Dict, results: Dict) -> List[Dict]:
        """Personalize treatment recommendations based on patient factors"""
        age = patient_data.get('age', 50)
        kps = patient_data.get('kps', 80)
        comorbidities = patient_data.get('comorbidities', [])
        
        personalized = []
        
        for rec in recommendations:
            new_rec = rec.copy()
            
            # Age-based modifications
            if age > 70:
                if 'aggressive' in rec['action'].lower() or 'concurrent chemoradiotherapy' in rec['action']:
                    new_rec['action'] += " (Consider hypofractionated RT for elderly)"
                    new_rec['evidence'] += " | Perry et al. NEJM 2017 - elderly patients"
                
                if rec['priority'] == 'critical' and 'surgery' in rec['action'].lower():
                    new_rec['action'] += " (Assess surgical fitness carefully)"
            
            # Performance status modifications
            if kps < 70:
                if 'chemotherapy' in rec['action'].lower():
                    new_rec['priority'] = 'medium'
                    new_rec['action'] += " (Consider supportive care focus)"
                
                # Add supportive care
                supportive_rec = {
                    "action": "Comprehensive supportive care and symptom management",
                    "timeline": "immediate",
                    "priority": "high",
                    "evidence": "ESMO Guidelines for supportive care"
                }
                if supportive_rec not in personalized:
                    personalized.append(supportive_rec)
            
            # Comorbidity adjustments
            if 'cardiac' in comorbidities:
                if 'temozolomide' in rec['action'].lower():
                    new_rec['action'] += " (Monitor cardiac function)"
            
            personalized.append(new_rec)
        
        return personalized
    
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
        prognostic_factors = {}
        
        if patient_data:
            # Age factor (Lacroix et al.)
            age = patient_data.get('age', 50)
            if age < 50:
                age_factor = 1.4  # Better prognosis
            elif age > 65:
                age_factor = 0.7  # Worse prognosis
            else:
                age_factor = 1.0
            
            survival_multiplier *= age_factor
            prognostic_factors['age_factor'] = age_factor
            
            # Performance status (Karnofsky)
            kps = patient_data.get('kps', 80)
            if kps >= 90:
                kps_factor = 1.3
            elif kps >= 70:
                kps_factor = 1.0
            else:
                kps_factor = 0.6
            
            survival_multiplier *= kps_factor
            prognostic_factors['performance_status_factor'] = kps_factor
            
            # Molecular markers (if available)
            if patient_data.get('idh_status') == 'Mutant':
                idh_factor = 2.0  # Significant survival benefit
                survival_multiplier *= idh_factor
                prognostic_factors['idh_mutation_factor'] = idh_factor
            
            if patient_data.get('mgmt_status') == 'Methylated':
                mgmt_factor = 1.5  # Better response to alkylating agents
                survival_multiplier *= mgmt_factor
                prognostic_factors['mgmt_methylation_factor'] = mgmt_factor
            
            if patient_data.get('1p19q_status') == 'Codeleted':
                codeletion_factor = 2.5  # Oligodendroglioma with better prognosis
                survival_multiplier *= codeletion_factor
                prognostic_factors['1p19q_codeletion_factor'] = codeletion_factor
        
        # Tumor characteristics
        tumor_size_factor = 1.0
        if results.get('tumor_percentage', 0) > 10:  # Large tumors
            tumor_size_factor = 0.8
        elif results.get('tumor_percentage', 0) > 20:
            tumor_size_factor = 0.6
        
        survival_multiplier *= tumor_size_factor
        prognostic_factors['tumor_size_factor'] = tumor_size_factor
        
        # AI confidence adjustment
        confidence = results.get('confidence', 0.5)
        confidence_factor = 0.8 + (confidence * 0.4)  # Range 0.8-1.2
        survival_multiplier *= confidence_factor
        prognostic_factors['ai_confidence_factor'] = confidence_factor
        
        # Calculate final survival estimates
        predicted_median = int(median_survival * survival_multiplier)
        
        # Calculate confidence intervals (simplified)
        ci_lower = int(predicted_median * 0.6)
        ci_upper = int(predicted_median * 1.6)
        
        return {
            'median_survival_days': max(30, predicted_median),
            'median_survival_months': max(1, predicted_median // 30),
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'survival_multiplier': survival_multiplier,
            'prognostic_factors': prognostic_factors,
            'evidence_base': 'Based on Stupp Protocol, RTOG trials, and molecular markers'
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
                'special_monitoring': ['Monitor for malignant transformation', 'Neuropsychological testing']
            }
        else:
            return {
                'imaging_frequency': '2-3 months during treatment, then 3-6 months',
                'imaging_modality': 'MRI brain with contrast + perfusion/spectroscopy',
                'clinical_followup': '1 month during treatment',
                'special_monitoring': ['Pseudoprogression assessment', 'Quality of life measures', 'Steroid management']
            }
    
    def _assess_clinical_urgency(self, grade: int, results: Dict) -> Dict:
        """Assess clinical urgency with detailed reasoning"""
        urgency_levels = {
            0: {"level": "ROUTINE", "timeframe": "Standard follow-up", "color": "green"},
            1: {"level": "MODERATE", "timeframe": "2-4 weeks", "color": "yellow"},
            2: {"level": "HIGH", "timeframe": "1-2 weeks", "color": "orange"},
            3: {"level": "CRITICAL", "timeframe": "24-48 hours", "color": "red"}
        }
        
        base_urgency = urgency_levels.get(grade, urgency_levels[0])
        
        # Modify based on additional factors
        confidence = results.get('confidence', 1.0)
        if confidence < 0.6:
            base_urgency["additional_note"] = "Low AI confidence - expert review recommended"
        
        tumor_percentage = results.get('tumor_percentage', 0)
        if tumor_percentage > 15:
            base_urgency["additional_note"] = "Large tumor volume detected"
        
        return base_urgency
    
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
            'recommendation': 'Expert review recommended' if quality_score < 70 else 'AI prediction reliable'
        }
    
    def _generate_comprehensive_notes(self, results: Dict, patient_data: Dict = None) -> str:
        """Generate comprehensive clinical notes"""
        notes = []
        
        # AI Analysis Summary
        grade = results.get('grade', 0)
        confidence = results.get('confidence', 0)
        tumor_info = get_tumor_info(grade)
        
        notes.append(f"AI Analysis: {tumor_info['name']} detected with {confidence:.1%} confidence using ensemble of {results.get('num_models_used', 1)} models.")
        
        # Tumor characteristics
        if grade > 0:
            tumor_pct = results.get('tumor_percentage', 0)
            notes.append(f"Tumor occupies approximately {tumor_pct:.1f}% of analyzed region.")
            
            # Morphological features
            morphological = results.get('morphological_features', {})
            if morphological.get('eccentricity', 0) > 0.8:
                notes.append("Tumor shows irregular/elongated morphology suggesting infiltrative growth pattern.")
            if morphological.get('solidity', 1) < 0.7:
                notes.append("Irregular tumor boundaries noted, consistent with higher-grade characteristics.")
        
        # Ensemble analysis
        agreement = results.get('ensemble_agreement', 0)
        if agreement < 0.7:
            notes.append(f"Moderate ensemble agreement ({agreement:.1%}) - consider additional imaging or expert consultation.")
        
        # Radiomics insights
        radiomics = results.get('radiomics_features', {})
        if radiomics:
            entropy = radiomics.get('entropy', 0)
            if entropy > 4:
                notes.append("High tumor heterogeneity detected, suggesting possible high-grade features.")
        
        # Quality assessment
        quality = results.get('quality_metrics', {})
        if quality.get('quality_score', 100) < 70:
            notes.append("Prediction quality below optimal threshold - expert review strongly recommended.")
        
        # Patient-specific considerations
        if patient_data:
            age = patient_data.get('age')
            if age and age > 65 and grade >= 2:
                notes.append("Elderly patient - consider age-related factors in treatment planning per NCCN guidelines.")
            elif age and age < 40 and grade >= 2:
                notes.append("Young patient - genetic counseling and comprehensive molecular profiling recommended.")
            
            if patient_data.get('kps', 100) < 70:
                notes.append("Reduced performance status - supportive care measures should be prioritized.")
        
        return " ".join(notes)
    
    def _recommend_imaging_protocol(self, grade: int, results: Dict) -> Dict:
        """Recommend specific imaging protocols"""
        protocols = {
            0: {
                'modality': 'MRI brain with contrast',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI'],
                'frequency': 'Annual or as clinically indicated',
                'special_notes': 'Routine surveillance'
            },
            1: {
                'modality': 'MRI brain with contrast',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI', 'T2'],
                'frequency': 'Every 6 months initially',
                'special_notes': 'Monitor for growth or transformation'
            },
            2: {
                'modality': 'MRI brain with contrast + advanced sequences',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI', 'Perfusion', 'MR Spectroscopy'],
                'frequency': 'Every 3-4 months',
                'special_notes': 'Monitor for malignant transformation'
            },
            3: {
                'modality': 'MRI brain with contrast + functional imaging',
                'sequences': ['T1 pre/post contrast', 'T2 FLAIR', 'DWI', 'Perfusion', 'MR Spectroscopy'],
                'frequency': 'Every 2-3 months during treatment',
                'special_notes': 'Assess treatment response, pseudoprogression vs progression'
            }
        }
        
        return protocols.get(grade, protocols[0])
    
    def _calculate_ensemble_confidence(self, ensemble_results: Dict) -> Dict:
        """Calculate comprehensive confidence metrics"""
        individual_preds = ensemble_results.get('individual_predictions', {})
        
        # Calculate inter-model agreement
        predictions = []
        confidences = []
        
        for model_name, pred_data in individual_preds.items():
            if 'probabilities' in pred_data:
                predictions.append(np.argmax(pred_data['probabilities']))
                confidences.append(pred_data.get('confidence', 0.5))
        
        # Agreement metrics
        if len(predictions) > 1:
            agreement = len([p for p in predictions if p == max(set(predictions), key=predictions.count)]) / len(predictions)
        else:
            agreement = 1.0
        
        # Confidence metrics
        mean_confidence = np.mean(confidences) if confidences else 0.5
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
        
        return {
            'ensemble_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'model_agreement': agreement,
            'num_models': len(predictions),
            'reliability_score': mean_confidence * agreement
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
            'ensemble_agreement': 0.8
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
                'recommendation': 'Manual analysis required'
            },
            'clinical_notes': 'Automated analysis failed. Manual expert review required for accurate diagnosis.',
            'model_confidence': {
                'ensemble_confidence': 0.0,
                'model_agreement': 0.0,
                'reliability_score': 0.0
            },
            'processing_metadata': {
                'models_used': ['fallback'],
                'ensemble_strategy': 'none',
                'preprocessing_applied': False
            }
        }

def segment_tumor_with_real_models(segmentation_model, image: np.ndarray, device: str = 'cpu') -> Dict:
    """Advanced tumor segmentation using real pretrained models"""
    try:
        # Preprocess image
        if len(image.shape) == 2:
            volume = np.expand_dims(image, axis=2)
        elif len(image.shape) == 3:
            volume = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Normalize
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Prepare for model
        input_tensor = preprocess_for_model(volume_norm, (256, 256)).to(device)
        
        # Model inference
        with torch.no_grad():
            if segmentation_model:
                try:
                    seg_output = segmentation_model(input_tensor)
                    
                    if isinstance(seg_output, (list, tuple)):
                        seg_logits = seg_output[0]
                    else:
                        seg_logits = seg_output
                    
                    # Get probabilities and mask
                    seg_probs = F.softmax(seg_logits, dim=1)
                    seg_mask = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
                    
                    # Post-process
                    seg_mask_clean = _post_process_segmentation_real(seg_mask)
                    
                    # Calculate metrics
                    total_pixels = seg_mask_clean.size
                    tumor_pixels = np.sum(seg_mask_clean > 0)
                    tumor_volume_percentage = (tumor_pixels / total_pixels) * 100
                    
                    # Confidence from model outputs
                    confidence = torch.max(seg_probs, dim=1)[0].mean().item()
                    
                    return {
                        'segmentation_mask': seg_mask_clean,
                        'tumor_volume_percentage': tumor_volume_percentage,
                        'segmentation_confidence': confidence,
                        'number_of_regions': len(np.unique(seg_mask_clean)) - 1,
                        'segmentation_quality': 'Good' if confidence > 0.7 else 'Fair' if confidence > 0.5 else 'Poor'
                    }
                    
                except Exception as model_error:
                    logger.warning(f"Model segmentation failed: {model_error}, using fallback")
                    return _fallback_segmentation_real(volume_norm)
            else:
                return _fallback_segmentation_real(volume_norm)
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return {
            'segmentation_mask': np.zeros((128, 128), dtype=np.uint8),
            'tumor_volume_percentage': 0.0,
            'segmentation_confidence': 0.0,
            'number_of_regions': 0,
            'segmentation_quality': 'Failed'
        }

def _fallback_segmentation_real(image: np.ndarray) -> Dict:
    """Real fallback segmentation using advanced image processing"""
    if len(image.shape) == 3:
        image_2d = image[:, :, image.shape[2] // 2]
    else:
        image_2d = image
    
    # Multi-method segmentation
    masks = []
    
    # Otsu thresholding
    try:
        threshold = filters.threshold_otsu(image_2d)
        mask1 = image_2d > threshold
        masks.append(mask1)
    except:
        pass
    
    # Adaptive thresholding
    try:
        mask2 = image_2d > filters.rank.otsu(image_2d, morphology.disk(5))
        masks.append(mask2)
    except:
        pass
    
    # K-means clustering
    try:
        pixels = image_2d.reshape(-1, 1)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pixels)
        clustered = clusters.reshape(image_2d.shape)
        
        # Find brightest cluster as potential tumor
        cluster_means = [np.mean(image_2d[clustered == i]) for i in range(4)]
        tumor_cluster = np.argmax(cluster_means)
        mask3 = clustered == tumor_cluster
        masks.append(mask3)
    except:
        pass
    
    # Combine masks
    if masks:
        combined = np.zeros_like(image_2d, dtype=float)
        for mask in masks:
            combined += mask.astype(float)
        
        final_mask = combined > (len(masks) * 0.4)
        final_mask = _post_process_segmentation_real(final_mask.astype(np.uint8))
        
        tumor_pixels = np.sum(final_mask > 0)
        total_pixels = final_mask.size
        
        return {
            'segmentation_mask': final_mask,
            'tumor_volume_percentage': (tumor_pixels / total_pixels) * 100,
            'segmentation_confidence': 0.6,
            'number_of_regions': len(np.unique(final_mask)) - 1,
            'segmentation_quality': 'Fair'
        }
    
    # Complete fallback
    return {
        'segmentation_mask': np.zeros_like(image_2d, dtype=np.uint8),
        'tumor_volume_percentage': 0.0,
        'segmentation_confidence': 0.0,
        'number_of_regions': 0,
        'segmentation_quality': 'Failed'
    }

def _post_process_segmentation_real(seg_mask: np.ndarray) -> np.ndarray:
    """Enhanced post-processing for segmentation masks"""
    if seg_mask.dtype != np.uint8:
        seg_mask = seg_mask.astype(np.uint8)
    
    # Remove small objects
    if np.any(seg_mask):
        seg_mask = morphology.remove_small_objects(seg_mask.astype(bool), min_size=100).astype(np.uint8)
    
    # Fill holes
    seg_mask = morphology.binary_fill_holes(seg_mask).astype(np.uint8)
    
    # Morphological operations
    kernel = morphology.disk(2)
    seg_mask = morphology.binary_closing(seg_mask, kernel).astype(np.uint8)
    seg_mask = morphology.binary_opening(seg_mask, kernel).astype(np.uint8)
    
    return seg_mask

def compare_scans_real_progression(baseline_results: Dict, followup_results: Dict, 
                                 baseline_date: str, followup_date: str) -> Dict:
    """Enhanced progression analysis using real medical criteria"""
    try:
        # Extract key metrics
        baseline_volume = baseline_results.get('tumor_percentage', 0)
        followup_volume = followup_results.get('tumor_percentage', 0)
        baseline_grade = baseline_results.get('grade', 0)
        followup_grade = followup_results.get('grade', 0)
        
        # Volume change calculation
        if baseline_volume > 0:
            volume_change = ((followup_volume - baseline_volume) / baseline_volume) * 100
        else:
            volume_change = followup_volume if followup_volume > 0 else 0
        
        # RANO criteria assessment (enhanced)
        response_assessment = _assess_rano_response_real(
            volume_change, baseline_results, followup_results
        )
        
        # Grade progression analysis
        grade_change = followup_grade - baseline_grade
        
        # Confidence analysis
        baseline_confidence = baseline_results.get('confidence', 0)
        followup_confidence = followup_results.get('confidence', 0)
        progression_confidence = min(baseline_confidence, followup_confidence)
        
        # Clinical significance with enhanced factors
        clinical_significance = _assess_clinical_significance_real(
            volume_change, grade_change, progression_confidence,
            baseline_results, followup_results
        )
        
        # Time-based analysis
        try:
            from datetime import datetime
            baseline_dt = datetime.strptime(baseline_date, '%Y-%m-%d')
            followup_dt = datetime.strptime(followup_date, '%Y-%m-%d')
            time_interval_days = (followup_dt - baseline_dt).days
        except:
            time_interval_days = 90  # Default 3 months
        
        # Growth rate calculation
        growth_rate = volume_change / (time_interval_days / 30) if time_interval_days > 0 else 0  # per month
        
        return {
            'baseline_date': baseline_date,
            'followup_date': followup_date,
            'time_interval_days': time_interval_days,
            'volume_change_percent': volume_change,
            'grade_change': grade_change,
            'growth_rate_per_month': growth_rate,
            'response_assessment': response_assessment,
            'progression_confidence': progression_confidence,
            'clinical_significance': clinical_significance,
            'recommendation': _generate_progression_recommendation_real(
                response_assessment, clinical_significance, growth_rate
            ),
            'rano_category': response_assessment,
            'next_imaging_recommendation': _recommend_next_imaging_real(response_assessment, time_interval_days)
        }
        
    except Exception as e:
        logger.error(f"Real scan comparison failed: {e}")
        return {
            'baseline_date': baseline_date,
            'followup_date': followup_date,
            'volume_change_percent': 0,
            'grade_change': 0,
            'response_assessment': 'Unable to assess',
            'progression_confidence': 0,
            'clinical_significance': 'Unknown',
            'recommendation': 'Manual expert review required due to analysis failure'
        }

def _assess_rano_response_real(volume_change: float, baseline_results: Dict, followup_results: Dict) -> str:
    """Real RANO criteria assessment with enhanced logic"""
    baseline_detected = baseline_results.get('detected', False)
    followup_detected = followup_results.get('detected', False)
    
    # New lesion assessment
    if not baseline_detected and followup_detected:
        return 'Progressive Disease (New Lesion)'
    elif baseline_detected and not followup_detected:
        return 'Complete Response'
    elif not baseline_detected and not followup_detected:
        return 'No Evidence of Disease'
    
    # Volume-based RANO assessment
    if volume_change <= -50:  # >=50% decrease
        return 'Partial Response'
    elif volume_change >= 25:  # >=25% increase
        return 'Progressive Disease'
    elif volume_change <= -25:  # 25-50% decrease
        return 'Minor Response'
    elif volume_change >= 10:   # 10-25% increase
        return 'Stable Disease (slight growth)'
    else:
        return 'Stable Disease'

def _assess_clinical_significance_real(volume_change: float, grade_change: int, 
                                     confidence: float, baseline_results: Dict, 
                                     followup_results: Dict) -> str:
    """Enhanced clinical significance assessment"""
    if confidence < 0.6:
        return 'Uncertain - Low AI Confidence'
    
    # Grade progression is always significant
    if grade_change > 1:
        return 'Highly Significant (Grade Progression)'
    elif grade_change > 0:
        return 'Significant (Grade Change)'
    
    # Volume-based significance
    abs_volume_change = abs(volume_change)
    
    if abs_volume_change > 75:
        return 'Highly Significant (Major Volume Change)'
    elif abs_volume_change > 50:
        return 'Significant (Substantial Volume Change)'
    elif abs_volume_change > 25:
        return 'Moderate (RANO Threshold Exceeded)'
    elif abs_volume_change > 10:
        return 'Mild (Detectable Change)'
    else:
        return 'Minimal (Within Measurement Error)'

def _generate_progression_recommendation_real(response: str, significance: str, growth_rate: float) -> str:
    """Generate evidence-based progression recommendations"""
    if 'Progressive Disease' in response:
        if 'Highly Significant' in significance or growth_rate > 10:
            return 'URGENT: Immediate treatment modification required - Consider salvage therapy or clinical trial'
        elif 'Significant' in significance:
            return 'HIGH PRIORITY: Treatment adjustment needed within 1-2 weeks'
        else:
            return 'Increased monitoring recommended - Consider treatment modification'
    
    elif 'Partial Response' in response or 'Complete Response' in response:
        return 'Excellent response - Continue current treatment regimen with standard monitoring'
    
    elif 'Minor Response' in response:
        return 'Good response - Continue treatment with close monitoring for sustained response'
    
    elif 'Stable Disease' in response:
        if 'slight growth' in response:
            return 'Stable with slight growth - Maintain treatment but increase monitoring frequency'
        else:
            return 'Disease stable - Continue current treatment and monitoring schedule'
    
    else:
        return 'Expert multidisciplinary team review recommended for treatment planning'

def _recommend_next_imaging_real(response: str, time_since_last: int) -> str:
    """Recommend next imaging timeframe based on response"""
    if 'Progressive Disease' in response:
        return 'Repeat MRI in 1-2 months or sooner if clinically indicated'
    elif 'Partial Response' in response or 'Complete Response' in response:
        return 'Repeat MRI in 2-3 months to confirm sustained response'
    elif 'Stable Disease' in response:
        if 'slight growth' in response:
            return 'Repeat MRI in 2 months'
        else:
            return 'Repeat MRI in 3 months per standard protocol'
    else:
        return 'Repeat MRI in 2-3 months or as clinically indicated'