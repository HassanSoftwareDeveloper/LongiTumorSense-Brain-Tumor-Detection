# clinical.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict, fields
import logging
import io
import tempfile
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from pathlib import Path

logger = logging.getLogger(__name__)

# Import from unified modules
try:
    from config import (
        TUMOR_CLASSES, CLINICAL_CONFIG, RANO_CRITERIA,
        get_tumor_info, SURVIVAL_ESTIMATES, CLINICAL_PROTOCOLS,
        DATABASE_CONFIG, BASE_DIR
    )
except ImportError:
    # Fallback configurations
    TUMOR_CLASSES = {
        0: {'name': 'Normal', 'description': 'No tumor detected', 'color': 'green'},
        1: {'name': 'Grade I Glioma', 'description': 'Low-grade glioma', 'color': 'yellow'},
        2: {'name': 'Grade II Glioma', 'description': 'Intermediate-grade glioma', 'color': 'orange'},
        3: {'name': 'Grade III-IV Glioma', 'description': 'High-grade glioma (GBM)', 'color': 'red'}
    }
    
    def get_tumor_info(grade):
        return TUMOR_CLASSES.get(grade, TUMOR_CLASSES[0])
    
    CLINICAL_CONFIG = TUMOR_CLASSES
    RANO_CRITERIA = {}
    SURVIVAL_ESTIMATES = {'grade_0': 3650, 'grade_1': 2555, 'grade_2': 1825, 'grade_3': 450}
    CLINICAL_PROTOCOLS = {}
    DATABASE_CONFIG = {'clinical_db_path': 'data/clinical_database.json'}
    BASE_DIR = Path(__file__).parent

try:
    from inference import RealBrainTumorInferenceEngine
except ImportError:
    logger.warning("Could not import inference module")

try:
    from helper import get_device, predict_with_pretrained_models, get_model_predictions_unified
except ImportError:
    logger.warning("Could not import helper module")

@dataclass
class PatientRecord:
    """Patient record data structure - Compatible with unified system"""
    patient_id: str
    name: str = ""
    age: int = 0
    sex: str = ""
    kps: int = 80
    scan_date: str = ""
    tumor_detected: bool = False
    tumor_type: str = ""
    tumor_grade: int = 0
    confidence: float = 0.0
    tumor_area: float = 0.0
    tumor_percentage: float = 0.0  # Added for compatibility
    survival_estimate: int = 0
    idh_status: str = "Unknown"
    mgmt_status: str = "Unknown"
    treatment_status: str = "New"
    last_updated: str = ""
    scan_history: List[Dict] = None
    symptoms: str = ""
    risk_level: str = "Normal"  # Added for compatibility
    risk_score: float = 0.0  # Added for compatibility
    
    def __post_init__(self):
        if self.scan_history is None:
            self.scan_history = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()
        # Ensure compatibility with tumor_classes
        if self.tumor_detected and not self.tumor_type:
            self.tumor_type = get_tumor_info(self.tumor_grade)['name']

class ClinicalDatabase:
    """Enhanced clinical database with real-time updates - Compatible with unified system"""
    
    def __init__(self, db_path: str = None):
        # Use config path or default
        if db_path is None:
            db_path = DATABASE_CONFIG.get('clinical_db_path', 'data/clinical_database.json')
        self.db_path = db_path
        self.ensure_data_directory()
        self.patients = self.load_database()
        
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def load_database(self) -> Dict[str, PatientRecord]:
        """Load patient database from file"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    patients = {}
                    for pid, patient_data in data.items():
                        # Convert date strings to proper format
                        if 'scan_date' in patient_data and isinstance(patient_data['scan_date'], str):
                            try:
                                # Ensure date is in proper format
                                datetime.fromisoformat(patient_data['scan_date'].replace('Z', '+00:00'))
                            except ValueError:
                                # If date format is invalid, use current date
                                patient_data['scan_date'] = datetime.now().isoformat()
                        
                        # Ensure compatibility with new fields
                        if 'tumor_percentage' not in patient_data and 'tumor_area' in patient_data:
                            patient_data['tumor_percentage'] = patient_data['tumor_area']
                        
                        patients[pid] = PatientRecord(**patient_data)
                    return patients
            return {}
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return {}
    
    def save_database(self):
        """Save patient database to file"""
        try:
            data = {
                pid: asdict(patient) 
                for pid, patient in self.patients.items()
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_or_update_patient(self, patient_data: Dict, analysis_results: Dict = None):
        """Add new patient or update existing with analysis results"""
        patient_id = patient_data['patient_id']
        
        # FIXED: Convert date objects to strings
        if 'scan_date' in patient_data:
            if hasattr(patient_data['scan_date'], 'isoformat'):
                patient_data['scan_date'] = patient_data['scan_date'].isoformat()
        
        allowed_fields = {f.name for f in fields(PatientRecord)}
        filtered_data = {k: v for k, v in patient_data.items() if k in allowed_fields}
        
        if patient_id in self.patients:
            # Update existing patient
            patient = self.patients[patient_id]
            for key, value in filtered_data.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)
        else:
            # Create new patient record
            patient = PatientRecord(**filtered_data)
            self.patients[patient_id] = patient
        
        # Add analysis results if provided
        if analysis_results:
            self.add_scan_results(patient_id, analysis_results)
        
        patient.last_updated = datetime.now().isoformat()
        self.save_database()
        
        # Update session state for real-time dashboard updates
        if 'clinical_stats' not in st.session_state:
            st.session_state.clinical_stats = {}
        
        self.update_session_stats()
    
    def add_scan_results(self, patient_id: str, results: Dict):
        """Add scan analysis results to patient record - Compatible with unified inference results"""
        if patient_id not in self.patients:
            logger.warning(f"Patient {patient_id} not found")
            return
        
        patient = self.patients[patient_id]
        
        # Update main patient record with latest results
        # Handle both old and new result formats
        if 'tumor_result' in results:
            # Old format
            tumor_result = results.get('tumor_result', {})
            patient.tumor_detected = tumor_result.get('detected', False)
            patient.tumor_type = tumor_result.get('tumor_type', '')
            patient.tumor_grade = tumor_result.get('grade', 0)
            patient.confidence = tumor_result.get('ai_confidence', 0.0)
            patient.tumor_area = tumor_result.get('tumor_percentage', 0.0)
            patient.tumor_percentage = tumor_result.get('tumor_percentage', 0.0)
            patient.survival_estimate = results.get('survival_days', 0)
            patient.risk_level = results.get('risk_level', 'Normal')
            patient.risk_score = results.get('risk_score', 0)
        else:
            # New unified format from inference.py
            patient.tumor_detected = results.get('detected', False)
            patient.tumor_type = results.get('tumor_type', '')
            patient.tumor_grade = results.get('grade', 0)
            patient.confidence = results.get('confidence', 0.0)
            patient.tumor_area = results.get('tumor_area', 0.0)
            patient.tumor_percentage = results.get('tumor_percentage', 0.0)
            patient.survival_estimate = results.get('survival_prediction', {}).get('median_survival_days', 0)
            patient.risk_level = results.get('risk_assessment', {}).get('risk_level', 'Normal')
            patient.risk_score = results.get('risk_assessment', {}).get('risk_score', 0)
        
        # Add to scan history
        scan_record = {
            'scan_date': datetime.now().isoformat(),
            'tumor_detected': patient.tumor_detected,
            'tumor_type': patient.tumor_type,
            'tumor_grade': patient.tumor_grade,
            'confidence': patient.confidence,
            'tumor_area': patient.tumor_area,
            'tumor_percentage': patient.tumor_percentage,
            'survival_estimate': patient.survival_estimate,
            'risk_level': patient.risk_level,
            'risk_score': patient.risk_score
        }
        
        patient.scan_history.append(scan_record)
        self.save_database()
    
    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        """Get patient record by ID"""
        return self.patients.get(patient_id)
    
    def get_all_patients(self) -> List[PatientRecord]:
        """Get all patient records"""
        return list(self.patients.values())
    
    def get_dashboard_stats(self) -> Dict:
        """Generate dashboard statistics - Compatible with unified system"""
        patients = self.get_all_patients()
        total_patients = len(patients)
        
        if total_patients == 0:
            return {
                'total_scans': 0,
                'total_patients': 0,
                'tumors_detected': 0,
                'detection_rate': 0.0,
                'avg_confidence': 0.0,
                'grade_distribution': {},
                'recent_scans': []
            }
        
        total_scans = sum(len(p.scan_history) for p in patients)
        tumors_detected = sum(1 for p in patients if p.tumor_detected)
        detection_rate = (tumors_detected / total_patients) * 100 if total_patients > 0 else 0
        
        # Calculate average confidence for detected tumors
        detected_patients = [p for p in patients if p.tumor_detected]
        avg_confidence = np.mean([p.confidence for p in detected_patients]) if detected_patients else 0.0
        
        # Grade distribution using unified tumor classes
        grade_counts = {}
        for patient in patients:
            grade_info = get_tumor_info(patient.tumor_grade)
            grade_name = grade_info['name']
            grade_counts[grade_name] = grade_counts.get(grade_name, 0) + 1
        
        # Recent scans (last 10)
        recent_scans = []
        for patient in patients:
            for scan in patient.scan_history[-5:]:  # Last 5 scans per patient
                recent_scans.append({
                    'patient_id': patient.patient_id,
                    'scan_date': scan['scan_date'],
                    'tumor_detected': scan['tumor_detected'],
                    'tumor_grade': scan['tumor_grade'],
                    'confidence': scan['confidence'],
                    'tumor_type': scan.get('tumor_type', '')
                })
        
        recent_scans.sort(key=lambda x: x['scan_date'], reverse=True)
        recent_scans = recent_scans[:10]
        
        return {
            'total_scans': total_scans,
            'total_patients': total_patients,
            'tumors_detected': tumors_detected,
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'grade_distribution': grade_counts,
            'recent_scans': recent_scans
        }
    
    def update_session_stats(self):
        """Update session state with latest stats"""
        st.session_state.clinical_stats = self.get_dashboard_stats()
    
    def get_progression_data(self, patient_id: str) -> Tuple[List[str], List[float], List[str]]:
        """Get progression tracking data for a patient - Compatible with unified system"""
        patient = self.get_patient(patient_id)
        if not patient or not patient.scan_history:
            return [], [], []
        
        dates = []
        volumes = []
        responses = []
        
        for scan in patient.scan_history:
            dates.append(scan['scan_date'])
            # Use tumor_percentage if available, otherwise tumor_area
            volume = scan.get('tumor_percentage', scan.get('tumor_area', 0))
            volumes.append(volume)
            
            # Determine response based on tumor area changes using RANO-like criteria
            if len(volumes) == 1:
                responses.append('Baseline')
            else:
                prev_volume = volumes[-2]
                current_volume = volumes[-1]
                change = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
                
                if change < -50:
                    responses.append('Complete Response')
                elif change < -30:
                    responses.append('Partial Response')
                elif change > 25:
                    responses.append('Progressive Disease')
                else:
                    responses.append('Stable Disease')
        
        return dates, volumes, responses

# Clinical Report Generation - Compatible with unified system
class ClinicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        # BUG 1 FIX: getSampleStyleSheet() returns a module-level singleton.
        # On Streamlit reruns the styles already exist → KeyError crash.
        # Wrap every add() call in try/except KeyError so subsequent reruns
        # reuse the existing style instead of crashing.
        try:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            ))
        except KeyError:
            pass

        try:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.darkred,
                borderWidth=1,
                borderColor=colors.grey,
                borderPadding=5
            ))
        except KeyError:
            pass
    
    def generate_clinical_report(self, patient_data, analysis_results, report_type="comprehensive"):
        """
        Generate ACR-structured clinical report following ACR Practice Parameter
        for Communication of Diagnostic Imaging Findings (revised 2020).

        Standard sections: Indication / Technique / Findings / Impression /
        Differential Diagnosis / Recommendations / Radiologist Review / Disclaimer.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        story = []

        # ── Header ────────────────────────────────────────────────────────────
        story.append(Paragraph("LONGITUMOR SENSE", self.styles['CustomTitle']))
        story.append(Paragraph(
            "Brain MRI — AI-Assisted Analysis Report (RESEARCH / DECISION-SUPPORT ONLY)",
            self.styles['Title']))
        story.append(Spacer(1, 20))

        # ── Patient Information ────────────────────────────────────────────────
        story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))

        scan_date = patient_data.get('scan_date', 'N/A')
        if hasattr(scan_date, 'strftime'):
            scan_date = scan_date.strftime('%Y-%m-%d')

        patient_info = [
            ['Patient ID:', patient_data['patient_id']],
            ['Age:', f"{patient_data['age']} years"],
            ['Sex:', patient_data['sex']],
            ['KPS Score:', f"{patient_data['kps']}/100"],
            ['Scan Date:', str(scan_date)],
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['Referring Clinician:', patient_data.get('referring_clinician', 'Not specified')],
            ['Clinical Indication:', patient_data.get('clinical_indication', 'Not specified')],
        ]

        if patient_data.get('idh_status') and patient_data.get('idh_status') != 'Unknown':
            patient_info.append(['IDH Status:', patient_data['idh_status']])
        if patient_data.get('mgmt_status') and patient_data.get('mgmt_status') != 'Unknown':
            patient_info.append(['MGMT Status:', patient_data['mgmt_status']])
        if patient_data.get('symptoms'):
            patient_info.append(['Reported Symptoms:', patient_data.get('symptoms')])

        patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(patient_table)
        story.append(Spacer(1, 20))

        # ── Extract results (supports old and new format) ─────────────────────
        if 'tumor_result' in analysis_results:
            tumor_result    = analysis_results['tumor_result']
            tumor_detected  = tumor_result.get('detected', False)
            tumor_type      = tumor_result.get('tumor_type', '')
            tumor_grade     = tumor_result.get('grade', 0)
            confidence      = tumor_result.get('ai_confidence', 0.0)
            tumor_percentage= tumor_result.get('tumor_percentage', 0.0)
            risk_level      = analysis_results.get('risk_level', 'Normal')
            risk_score      = analysis_results.get('risk_score', 0)
        else:
            tumor_detected  = analysis_results.get('detected', False)
            tumor_type      = analysis_results.get('tumor_type', '')
            tumor_grade     = analysis_results.get('grade', 0)
            confidence      = analysis_results.get('confidence', 0.0)
            tumor_percentage= analysis_results.get('tumor_percentage', 0.0)
            risk_level      = analysis_results.get('risk_assessment', {}).get('risk_level', 'Normal')
            risk_score      = analysis_results.get('risk_assessment', {}).get('risk_score', 0)

        uncertainty = analysis_results.get('uncertainty', analysis_results.get('combined_uncertainty', None))
        n_models    = analysis_results.get('n_cls_models', analysis_results.get('num_models_used', 'N/A'))

        # ── Section 1: Indication ─────────────────────────────────────────────
        story.append(Paragraph("1. INDICATION (ACR)", self.styles['SectionHeader']))
        indication_text = patient_data.get('clinical_indication') or (
            f"Brain MRI for assessment of intracranial pathology. "
            f"Patient age {patient_data.get('age', 'unknown')}, "
            f"KPS {patient_data.get('kps', 'unknown')}. "
            f"Symptoms: {patient_data.get('symptoms', 'not documented')}."
        )
        story.append(Paragraph(indication_text, self.styles['Normal']))
        story.append(Spacer(1, 10))

        # ── Section 2: Technique ─────────────────────────────────────────────
        story.append(Paragraph("2. TECHNIQUE (ACR)", self.styles['SectionHeader']))
        technique_text = (
            "AI-assisted volumetric analysis of uploaded MRI image(s). "
            f"Multi-model ensemble: {n_models} classifier(s). "
            "Segmentation via U-Net / EfficientNet-B4 / SwinUNETR ensemble. "
            "Classification via RadImageNet ResNet-50 / Medical ViT / BiomedCLIP. "
            "GLCM radiomics (14 Haralick features). "
            "Modality as provided by operator — contrast status not verified by AI."
        )
        story.append(Paragraph(technique_text, self.styles['Normal']))
        story.append(Spacer(1, 10))

        # ── Section 3: AI Findings ────────────────────────────────────────────
        story.append(Paragraph("3. AI FINDINGS (ACR — Findings Section)", self.styles['SectionHeader']))

        if tumor_detected:
            status_text = f"<b>AI FINDING: ABNORMALITY DETECTED</b> — {tumor_type}"
        else:
            status_text = "<b>AI FINDING: NO DEFINITE ABNORMALITY DETECTED</b>"

        story.append(Paragraph(status_text, self.styles['Normal']))
        story.append(Spacer(1, 8))

        findings_data = [
            ['Parameter', 'AI Result', 'Notes'],
            ['Tumor Detection', 'Positive' if tumor_detected else 'Negative',
             'Requires radiologist confirmation'],
            ['WHO Grade (AI)', f"Grade {tumor_grade}" if tumor_grade > 0 else 'N/A',
             'Heuristic; not validated against WHO 2021 molecular criteria'],
            ['AI Confidence', f"{confidence:.1%}",
             'Soft-max probability — NOT a clinical certainty estimate'],
            ['Tumor Area (2D)', f"{tumor_percentage:.2f}% of analysed slice",
             '2D proxy; volumetric measurement requires 3D segmentation'],
            ['Risk Level', risk_level, ''],
            ['Risk Score', f"{risk_score:.0f}/100", 'Composite heuristic score'],
        ]
        if uncertainty is not None:
            findings_data.append([
                'AI Uncertainty (MC)', f"{float(uncertainty):.3f}",
                '0=certain, 1=highly uncertain'
            ])

        findings_table = Table(findings_data, colWidths=[1.8*inch, 1.6*inch, 2.6*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))

        story.append(findings_table)
        story.append(Spacer(1, 15))

        # ── Section 4: Impression ─────────────────────────────────────────────
        story.append(Paragraph("4. IMPRESSION (ACR)", self.styles['SectionHeader']))
        interpretation = self.generate_clinical_interpretation(
            {'detected': tumor_detected, 'tumor_type': tumor_type, 'grade': tumor_grade},
            patient_data)
        story.append(Paragraph(interpretation, self.styles['Normal']))
        story.append(Spacer(1, 15))

        # ── Section 5: Differential Diagnosis ────────────────────────────────
        story.append(Paragraph("5. DIFFERENTIAL DIAGNOSIS (ACR)", self.styles['SectionHeader']))
        differential = self._generate_differential_diagnosis(tumor_grade, tumor_detected, patient_data)
        story.append(Paragraph(differential, self.styles['Normal']))
        story.append(Spacer(1, 15))

        # ── Section 6: Survival Prediction ────────────────────────────────────
        if tumor_detected:
            story.append(Paragraph("6. SURVIVAL ESTIMATE (Statistical Reference Only)", self.styles['SectionHeader']))

            survival_days = analysis_results.get('survival_days') or (
                analysis_results.get('survival_prediction', {}).get('median_survival_days')
            ) or SURVIVAL_ESTIMATES.get(f'grade_{tumor_grade}', 480)

            if survival_days >= 730:
                survival_display = f"~{survival_days // 365} years (median, population-level)"
            elif survival_days >= 60:
                survival_display = f"~{survival_days // 30} months (median, population-level)"
            else:
                survival_display = f"~{survival_days} days (median, population-level)"

            survival_text = (
                f"Population-level median survival for WHO Grade {tumor_grade} glioma: "
                f"<b>{survival_display}</b>. "
                "Source: Stupp NEJM 2005, Buckner NEJM 2016, CBTRUS Statistical Report 2023. "
                "This estimate is highly individual and depends on molecular markers "
                "(IDH mutation, MGMT promoter methylation, 1p/19q codeletion), "
                "extent of resection, age, and performance status. "
                "<b>Interpretation must be performed by a qualified neuro-oncologist.</b>"
            )
            story.append(Paragraph(survival_text, self.styles['Normal']))
            story.append(Spacer(1, 15))

        # ── Section 7: Recommendations ────────────────────────────────────────
        story.append(Paragraph("7. RECOMMENDATIONS (ACR)", self.styles['SectionHeader']))
        recommendations = self.get_treatment_recommendations(tumor_grade)
        recommendations_text = "<br/>".join([f"• {rec}" for rec in recommendations])
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 15))

        # ── Section 8: Follow-up Plan ─────────────────────────────────────────
        story.append(Paragraph("8. FOLLOW-UP PLAN (ACR)", self.styles['SectionHeader']))
        followup_plan = self.generate_followup_plan(
            {'detected': tumor_detected, 'grade': tumor_grade})
        story.append(Paragraph(followup_plan, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # ── Section 9: Radiologist Review ─────────────────────────────────────
        story.append(Paragraph("9. RADIOLOGIST REVIEW (REQUIRED)", self.styles['SectionHeader']))
        review_block = [
            ['', ''],
            ['Reviewing Radiologist (print name):', '_' * 45],
            ['', ''],
            ['Signature:', '_' * 45],
            ['', ''],
            ['Date / Time of Review:', '_' * 30],
            ['', ''],
            ['Radiologist Impression / Override:', ''],
            ['', '_' * 60],
            ['', '_' * 60],
            ['', '_' * 60],
            ['', ''],
            ['AI Finding Confirmed:', '☐ Confirmed as reported'],
            ['', '☐ Partially confirmed — see notes above'],
            ['', '☐ Not confirmed / Overridden — see notes above'],
        ]
        review_table = Table(review_block, colWidths=[2.5*inch, 3.5*inch])
        review_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(review_table)
        story.append(Spacer(1, 20))

        # ── Section 10: Disclaimer ────────────────────────────────────────────
        story.append(Paragraph("DISCLAIMER & REGULATORY STATUS", self.styles['SectionHeader']))
        disclaimer = (
            "<b>IMPORTANT — PLEASE READ BEFORE CLINICAL USE</b><br/><br/>"
            "This report is generated by the LongiTumor Sense AI system "
            "(version 3.0, research build). "
            "<b>This system has NOT received FDA 510(k) clearance, CE marking, or any "
            "other regulatory approval for clinical diagnostic use.</b> "
            "It is provided solely as a research decision-support tool and must "
            "<b>NOT</b> be used as the sole basis for diagnosis, treatment planning, "
            "or any clinical decision.<br/><br/>"
            "All AI findings <b>MUST</b> be independently reviewed and confirmed by a "
            "qualified, licensed radiologist before any clinical action is taken. "
            "Findings should be correlated with clinical presentation, laboratory data, "
            "and additional imaging as clinically indicated.<br/><br/>"
            "The system has not been prospectively validated against radiologist ground truth "
            "on an independent patient cohort. Reported confidence values are model "
            "probabilities and do not represent calibrated clinical certainty.<br/><br/>"
            f"<b>Report generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
            f"<b>System version:</b> LongiTumor Sense v3.0 (research)<br/>"
            "<b>Regulatory status:</b> NOT FDA cleared / NOT CE marked — RESEARCH USE ONLY"
        )
        story.append(Paragraph(disclaimer, self.styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _generate_differential_diagnosis(self, tumor_grade: int, detected: bool,
                                          patient_data: dict) -> str:
        """
        Generate a grade-appropriate differential diagnosis list per ACR guidelines.
        NOTE: This is a static educational reference list, not a patient-specific
        AI diagnosis. The reviewing radiologist must confirm or expand upon this.
        """
        age  = patient_data.get('age', 0)
        if not detected:
            return (
                "No definite mass lesion detected by AI. Differential considerations include:<br/>"
                "1. Normal variant / scan artefact<br/>"
                "2. Reactive gliosis (non-neoplastic)<br/>"
                "3. Small or isointense lesion below AI detection threshold<br/>"
                "<i>Radiologist review required to exclude subtle pathology.</i>"
            )
        grade_differentials = {
            1: (
                "For a Grade I (low-grade) lesion, principal differentials include:<br/>"
                "1. Pilocytic astrocytoma (most common Grade I; typically cystic with mural nodule)<br/>"
                "2. Dysembryoplastic neuroepithelial tumour (DNET)<br/>"
                "3. Ganglioglioma<br/>"
                "4. Pleomorphic xanthoastrocytoma (PXA)<br/>"
                "5. Meningioma (extra-axial; if lesion is peripheral)<br/>"
                "6. Demyelinating pseudotumour (MS plaque, tumefactive MS)<br/>"
                "<i>IDH status and 1p/19q codeletion testing recommended per WHO CNS 2021.</i>"
            ),
            2: (
                "For a Grade II (intermediate-grade) lesion, principal differentials include:<br/>"
                "1. Diffuse astrocytoma, IDH-mutant (WHO 2021)<br/>"
                "2. Oligodendroglioma, IDH-mutant, 1p/19q codeleted (WHO 2021)<br/>"
                "3. Diffuse astrocytoma, IDH-wildtype (worse prognosis — exclude EGFR/TERT/+7/-10)<br/>"
                "4. Low-grade glioma, NOS (pending molecular testing)<br/>"
                "5. Tumefactive demyelination<br/>"
                "6. Cortical dysplasia / focal cortical malformation<br/>"
                "<i>Molecular profiling (IDH1/2, 1p/19q, MGMT, TERT, EGFR) mandatory per WHO CNS 2021.</i>"
            ),
            3: (
                "For a Grade III (high-grade) lesion, principal differentials include:<br/>"
                "1. Anaplastic astrocytoma, IDH-mutant (WHO 2021)<br/>"
                "2. Anaplastic oligodendroglioma, IDH-mutant, 1p/19q codeleted<br/>"
                "3. Anaplastic astrocytoma, IDH-wildtype<br/>"
                "4. Primary CNS lymphoma (especially if periventricular; consider in immunocompromised)<br/>"
                "5. Metastatic disease (if multiple lesions or known primary cancer)<br/>"
                "6. High-grade glioma NOS<br/>"
                "<i>MRS, perfusion MRI, and biopsy with molecular analysis strongly recommended.</i>"
            ),
            4: (
                "For a Grade IV (highest-grade) lesion, principal differentials include:<br/>"
                "1. Glioblastoma, IDH-wildtype (WHO Grade 4; most common primary brain tumour in adults)<br/>"
                "2. Glioblastoma, IDH-mutant (less common; better prognosis)<br/>"
                "3. Diffuse midline glioma, H3K27-altered (Grade 4 by definition per WHO 2021)<br/>"
                "4. Solitary brain metastasis (ring-enhancing; if known primary elsewhere)<br/>"
                "5. Primary CNS lymphoma (especially if steroid-responsive, periventricular)<br/>"
                "6. Cerebral abscess (ring-enhancing with restricted diffusion centrally on DWI)<br/>"
                "7. Tumefactive MS (rare; may mimic GBM — check DWI / spectroscopy)<br/>"
                "<i>Gadolinium-enhanced MRI, DWI, MRS, and stereotactic biopsy required. "
                "MGMT promoter methylation status guides chemotherapy decisions.</i>"
            ),
        }
        return grade_differentials.get(
            min(max(tumor_grade, 1), 4),
            grade_differentials[4]
        )

    def get_treatment_recommendations(self, tumor_grade: int) -> List[str]:
        """Get treatment recommendations from unified clinical protocols"""
        grade_key_map = {
            0: "Normal",
            1: "Grade I Glioma",
            2: "Grade II Glioma",
            3: "Grade III-IV Glioma",
        }
        grade_key = grade_key_map.get(tumor_grade, "Normal")
        return CLINICAL_PROTOCOLS.get('treatment_guidelines', {}).get(
            grade_key,
            ['Consult with neuro-oncology specialist', 'Individualized treatment planning']
        )
    
    def generate_progression_report(self, combined_data):
        """Generate progression analysis report - Compatible with unified system"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Header
        story.append(Paragraph("LONGITUMOR SENSE", self.styles['CustomTitle']))
        story.append(Paragraph("Progression Analysis Report", self.styles['Title']))
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        
        patient_info = [
            ['Patient ID:', combined_data['patient_id']],
            ['Age:', f"{combined_data['age']} years"],
            ['Sex:', combined_data['sex']],
            ['KPS Score:', f"{combined_data['kps']}/100"],
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Progression Analysis
        metrics = combined_data['progression_metrics']
        
        story.append(Paragraph("PROGRESSION ANALYSIS", self.styles['SectionHeader']))
        
        progression_status = f"<b>STATUS: {metrics['progression_status']}</b>"
        story.append(Paragraph(progression_status, self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Progression metrics table
        progression_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Volume Change', f"{metrics['volume_change_pct']:+.1f}%", 
             'Significant' if abs(metrics['volume_change_pct']) > 25 else 'Moderate' if abs(metrics['volume_change_pct']) > 10 else 'Stable'],
            ['Grade Change', f"{metrics['grade_change']:+d}", 
             'Progression' if metrics['grade_change'] > 0 else 'Regression' if metrics['grade_change'] < 0 else 'Stable'],
            ['Time Interval', f"{metrics['time_diff_days']} days", 
             'Rapid' if metrics['time_diff_days'] < 90 else 'Standard'],
            ['Growth Rate', f"{metrics['growth_rate_monthly']:+.2f}%/month", 
             'Fast' if abs(metrics['growth_rate_monthly']) > 5 else 'Slow'],
            ['Recurrence Risk', f"{metrics['recurrence_risk']:.0f}%", 
             'High' if metrics['recurrence_risk'] > 70 else 'Moderate' if metrics['recurrence_risk'] > 40 else 'Low']
        ]
        
        progression_table = Table(progression_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        progression_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(progression_table)
        story.append(Spacer(1, 20))
        
        # Treatment Response
        story.append(Paragraph("TREATMENT RESPONSE", self.styles['SectionHeader']))
        
        response_text = f"""
        <b>Assessment:</b> {metrics['treatment_response'][0]}<br/>
        <b>Volume Change:</b> {metrics['volume_change_pct']:+.1f}%<br/>
        <b>Clinical Significance:</b> {'Significant progression requiring immediate attention' if metrics['volume_change_pct'] > 25 else 'Stable disease with continued monitoring recommended'}
        """
        story.append(Paragraph(response_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Recommendations
        story.append(Paragraph("RECOMMENDATIONS", self.styles['SectionHeader']))
        
        if metrics['volume_change_pct'] > 25:
            recommendations = [
                "Urgent multidisciplinary team consultation",
                "Consider treatment modification or escalation",
                "Repeat imaging in 4-6 weeks",
                "Monitor for clinical symptoms closely"
            ]
        elif metrics['volume_change_pct'] > 10:
            recommendations = [
                "Continue current treatment with close monitoring",
                "Consider additional therapeutic options",
                "Repeat imaging in 8-12 weeks",
                "Regular clinical assessment"
            ]
        else:
            recommendations = [
                "Continue current treatment regimen",
                "Standard follow-up protocol",
                "Routine imaging surveillance",
                "Maintain quality of life measures"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("DISCLAIMER & REGULATORY STATUS", self.styles['SectionHeader']))
        disclaimer = (
            "This progression analysis report is generated by LongiTumor Sense AI (v3.0, research build). "
            "<b>This system has NOT received FDA 510(k) clearance or CE marking. "
            "It is provided for research decision-support only.</b> "
            "Results must be independently reviewed by a qualified radiologist before any clinical action. "
            "RANO-informed metrics are provided as a reference framework; full RANO compliance "
            "requires gadolinium-enhanced imaging, steroid documentation, and clinical assessment "
            "not available from pixel data alone.<br/><br/>"
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
            "<b>System:</b> LongiTumor Sense AI v3.0 (research) — NOT FDA cleared / NOT CE marked"
        )
        
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_clinical_interpretation(self, tumor_result, patient_data):
        """Generate clinical interpretation with appropriate uncertainty language."""

        if not tumor_result['detected']:
            return (
                "AI analysis did not detect a definite mass lesion in the provided image. "
                "This finding does not exclude pathology below the AI detection threshold, "
                "isointense lesions, or early/subtle changes. "
                "The reviewing radiologist should assess for any abnormal signal, "
                "enhancement, or structural changes not captured by the AI. "
                "Routine follow-up as clinically indicated."
            )

        tumor_info = get_tumor_info(tumor_result['grade'])
        interpretation_parts = [f"AI-assisted analysis suggests {tumor_info['description']}. "]

        if patient_data.get('age', 0) > 65:
            interpretation_parts.append(
                "Patient age above 65 is an adverse prognostic factor for high-grade gliomas; "
                "treatment tolerance and response rates are typically reduced in this cohort. "
            )

        if patient_data.get('kps', 100) < 70:
            interpretation_parts.append(
                "Reduced Karnofsky Performance Status (KPS < 70) may limit treatment options "
                "and is an independent adverse prognostic indicator. "
            )

        if patient_data.get('idh_status', '').lower() == 'mutant':
            interpretation_parts.append(
                "IDH-mutant status (if confirmed by molecular testing) is associated with "
                "improved prognosis and better response to alkylating chemotherapy. "
            )
        elif patient_data.get('idh_status', '').lower() == 'wildtype':
            interpretation_parts.append(
                "IDH-wildtype status (if confirmed) is associated with a more aggressive "
                "clinical course even for morphologically lower-grade lesions. "
            )

        interpretation_parts.append(
            "<br/><i>This interpretation is generated by AI and must be reviewed and "
            "confirmed by a qualified radiologist before any clinical use.</i>"
        )

        return " ".join(interpretation_parts)
    
    def generate_followup_plan(self, tumor_result):
        """Generate personalized follow-up plan using unified protocols"""
        
        if not tumor_result['detected']:
            return """
            • Routine clinical follow-up in 12 months or as clinically indicated
            • Patient education regarding warning signs and symptoms
            • No immediate imaging follow-up required unless symptoms develop
            """
        
        tumor_info = get_tumor_info(tumor_result['grade'])
        follow_up_months = tumor_info.get('follow_up_months', 6)
        
        if tumor_result['grade'] == 1:
            return f"""
            • MRI follow-up in {follow_up_months} months to assess for any changes
            • Neurosurgical consultation for surgical planning if indicated
            • Monitor for neurological symptoms
            • Consider neurocognitive assessment if appropriate
            """
        elif tumor_result['grade'] == 2:
            return f"""
            • MRI follow-up in {follow_up_months} months to monitor progression
            • Urgent neurosurgical consultation recommended
            • Consider radiation oncology consultation
            • Monitor closely for neurological deterioration
            • Discuss treatment options and prognosis with patient/family
            """
        else:
            return f"""
            • URGENT neurosurgical consultation within 24-48 hours
            • Consider immediate corticosteroids if not contraindicated
            • MRI follow-up in {follow_up_months} weeks post-intervention
            • Multidisciplinary team approach (neurosurgery, radiation oncology, medical oncology)
            • Palliative care consultation for symptom management and goals of care discussion
            • Close monitoring for complications and neurological changes
            """

    def generate_full_combined_report(self, single_results, progression_results):
        """BUG 6 FIX: Generate a combined PDF when the user ran both single
        analysis AND progression analysis in the same session."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        story = []

        # Cover header
        story.append(Paragraph("LONGITUMOR SENSE", self.styles['CustomTitle']))
        story.append(Paragraph("Full Combined Clinical Report", self.styles['Title']))
        story.append(Spacer(1, 20))

        # ── PART A: Detection findings ────────────────────────────────────────
        story.append(Paragraph("PART A – TUMOR DETECTION FINDINGS",
                               self.styles['SectionHeader']))
        story.append(Spacer(1, 10))

        patient_data = single_results.get('patient_data', {})
        scan_date = patient_data.get('scan_date', 'N/A')
        if hasattr(scan_date, 'strftime'):
            scan_date = scan_date.strftime('%Y-%m-%d')

        part_a_info = [
            ['Patient ID:', patient_data.get('patient_id', 'N/A')],
            ['Age:', f"{patient_data.get('age', 'N/A')} years"],
            ['Sex:', patient_data.get('sex', 'N/A')],
            ['KPS Score:', f"{patient_data.get('kps', 'N/A')}/100"],
            ['Scan Date:', str(scan_date)],
        ]
        part_a_table = Table(part_a_info, colWidths=[2 * inch, 3 * inch])
        part_a_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(part_a_table)
        story.append(Spacer(1, 12))

        tumor_result = single_results.get('tumor_result', {})
        detected = tumor_result.get('detected', False)
        story.append(Paragraph(
            f"<b>{'TUMOR DETECTED' if detected else 'NO TUMOR DETECTED'}</b> – "
            f"{tumor_result.get('tumor_type', 'Normal')}",
            self.styles['Normal']))
        story.append(Spacer(1, 10))

        findings = [
            ['Tumor Detection:', 'Positive' if detected else 'Negative'],
            ['WHO Grade:', f"Grade {tumor_result.get('grade', 0)}"
             if tumor_result.get('grade', 0) > 0 else 'N/A'],
            ['AI Confidence:', f"{tumor_result.get('ai_confidence', 0):.1%}"],
            ['Risk Level:', single_results.get('risk_level', 'N/A')],
        ]
        findings_table = Table(findings, colWidths=[2 * inch, 3 * inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(findings_table)
        story.append(Spacer(1, 20))

        # ── PART B: Progression comparison ────────────────────────────────────
        story.append(Paragraph("PART B – PROGRESSION COMPARISON",
                               self.styles['SectionHeader']))
        story.append(Spacer(1, 10))

        metrics = progression_results.get('progression_metrics', {})
        prog_data = [
            ['Progression Status:', metrics.get('progression_status', 'N/A')],
            ['Volume Change:', f"{metrics.get('volume_change_pct', 0):+.1f}%"],
            ['Time Interval:', f"{metrics.get('time_diff_days', 0)} days"],
            ['Growth Rate:', f"{metrics.get('growth_rate_monthly', 0):+.2f}%/month"],
            ['Recurrence Risk:', f"{metrics.get('recurrence_risk', 0):.0f}%"],
            ['Treatment Response:', metrics.get('treatment_response', ('N/A',))[0]],
        ]
        prog_table = Table(prog_data, colWidths=[2 * inch, 3 * inch])
        prog_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(prog_table)
        story.append(Spacer(1, 20))

        # ── Consolidated recommendations ──────────────────────────────────────
        story.append(Paragraph("CONSOLIDATED RECOMMENDATIONS",
                               self.styles['SectionHeader']))
        recs = self.get_treatment_recommendations(tumor_result.get('grade', 0))
        for rec in recs:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Footer / disclaimer
        story.append(Paragraph("DISCLAIMER", self.styles['SectionHeader']))
        story.append(Paragraph(
            f"This combined report was generated by LongiTumor Sense AI v3.0 (research) on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}. "
            "All findings must be reviewed and confirmed by a qualified radiologist.",
            self.styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer

    def generate_summary_report(self, patient_data, analysis_results):
        """BUG 6 FIX: Compact single-page referral / summary report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        story = []

        story.append(Paragraph("LONGITUMOR SENSE", self.styles['CustomTitle']))
        story.append(Paragraph("Clinical Summary Report", self.styles['Title']))
        story.append(Spacer(1, 20))

        scan_date = patient_data.get('scan_date', 'N/A')
        if hasattr(scan_date, 'strftime'):
            scan_date = scan_date.strftime('%Y-%m-%d')

        summary_info = [
            ['Patient ID:', patient_data.get('patient_id', 'N/A')],
            ['Age / Sex:', f"{patient_data.get('age', 'N/A')} / {patient_data.get('sex', 'N/A')}"],
            ['KPS:', f"{patient_data.get('kps', 'N/A')}/100"],
            ['Scan Date:', str(scan_date)],
            ['Report Date:', datetime.now().strftime('%Y-%m-%d')],
        ]
        t = Table(summary_info, colWidths=[2 * inch, 3 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        story.append(Paragraph("SUMMARY FINDINGS", self.styles['SectionHeader']))
        if 'tumor_result' in analysis_results:
            tr = analysis_results['tumor_result']
        else:
            tr = analysis_results
        detected = tr.get('detected', False)
        story.append(Paragraph(
            f"<b>{'TUMOR DETECTED' if detected else 'NO TUMOR DETECTED'}</b>. "
            f"AI Confidence: {tr.get('ai_confidence', tr.get('confidence', 0)):.1%}. "
            f"Type: {tr.get('tumor_type', 'Normal')}.",
            self.styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("REFERRAL RECOMMENDATION", self.styles['SectionHeader']))
        recs = self.get_treatment_recommendations(tr.get('grade', 0))
        story.append(Paragraph(recs[0] if recs else "Consult neuro-oncology specialist.",
                               self.styles['Normal']))
        story.append(Spacer(1, 20))

        story.append(Paragraph(
            f"Generated by LongiTumor Sense AI v3.0 (research) | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            self.styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer


# Dashboard functions - Compatible with unified system
def create_clinical_dashboard():
    """Create the clinical dashboard interface"""
    # Initialize database
    if 'clinical_db' not in st.session_state:
        st.session_state.clinical_db = ClinicalDatabase()
    
    db = st.session_state.clinical_db
    stats = db.get_dashboard_stats()
    
    # Dashboard header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2>[REPORT] Clinical Dashboard & Analytics</h2>
        <p>Real-time patient data, system performance, and clinical outcomes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=stats['total_patients'],
            help="Number of patients in the system"
        )
    
    with col2:
        st.metric(
            label="Total Scans",
            value=stats['total_scans'],
            help="Total number of MRI scans analyzed"
        )
    
    with col3:
        st.metric(
            label="Tumors Detected",
            value=stats['tumors_detected'],
            delta=f"{stats['detection_rate']:.1f}% detection rate",
            help="Number of patients with detected tumors"
        )
    
    with col4:
        st.metric(
            label="Avg Confidence",
            value=f"{stats['avg_confidence']:.1%}",
            help="Average AI confidence for tumor detections"
        )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Grade distribution pie chart using unified tumor classes
        if stats['grade_distribution']:
            fig_grades = go.Figure(data=[
                go.Pie(
                    labels=list(stats['grade_distribution'].keys()),
                    values=list(stats['grade_distribution'].values()),
                    hole=0.3,
                    marker_colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
                )
            ])
            fig_grades.update_layout(
                title="Tumor Grade Distribution",
                height=400
            )
            st.plotly_chart(fig_grades, use_container_width=True)
    
    with col2:
        # Recent activity timeline
        if stats['recent_scans']:
            recent_df = pd.DataFrame(stats['recent_scans'])
            recent_df['scan_date'] = pd.to_datetime(recent_df['scan_date'])
            
            fig_timeline = go.Figure()
            
            colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
            
            for grade in recent_df['tumor_grade'].unique():
                grade_data = recent_df[recent_df['tumor_grade'] == grade]
                grade_name = get_tumor_info(grade)['name']
                fig_timeline.add_trace(go.Scatter(
                    x=grade_data['scan_date'],
                    y=[grade_name] * len(grade_data),
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors.get(grade, 'gray'),
                        opacity=0.7
                    ),
                    name=grade_name,
                    text=grade_data['patient_id'],
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Grade: %{y}<extra></extra>"
                ))
            
            fig_timeline.update_layout(
                title="Recent Scan Activity",
                height=400,
                xaxis_title="Scan Date",
                yaxis_title="Tumor Grade"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Recent scans table
    st.subheader("Recent Scans")
    if stats['recent_scans']:
        recent_df = pd.DataFrame(stats['recent_scans'])
        recent_df['scan_date'] = pd.to_datetime(recent_df['scan_date']).dt.strftime('%Y-%m-%d %H:%M')
        recent_df['tumor_status'] = recent_df.apply(
            lambda row: get_tumor_info(row['tumor_grade'])['name'], 
            axis=1
        )
        recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        display_df = recent_df[['patient_id', 'scan_date', 'tumor_status', 'confidence']].copy()
        display_df.columns = ['Patient ID', 'Scan Date', 'Status', 'Confidence']
        
        st.dataframe(display_df, width='stretch')
    else:
        st.info("No recent scans to display")
    
    # Patient management
    st.subheader("Patient Management")
    
    tab1, tab2 = st.tabs(["All Patients", "Add Patient"])
    
    with tab1:
        patients = db.get_all_patients()
        if patients:
            patient_data = []
            for patient in patients:
                # FIXED: Convert date to string for display
                scan_date = patient.scan_date
                if hasattr(scan_date, 'strftime'):
                    scan_date = scan_date.strftime('%Y-%m-%d')
                
                patient_data.append({
                    'Patient ID': patient.patient_id,
                    'Name': patient.name,
                    'Age': patient.age,
                    'Sex': patient.sex,
                    'Tumor Status': patient.tumor_type,
                    'Last Scan': scan_date,
                    'Scans Count': len(patient.scan_history)
                })
            
            patients_df = pd.DataFrame(patient_data)
            st.dataframe(patients_df, width='stretch')
        else:
            st.info("No patients in the database")
    
    with tab2:
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("Patient ID", placeholder="PT-XXXX")
                name = st.text_input("Patient Name", placeholder="Full Name")
                age = st.number_input("Age", min_value=1, max_value=120, value=50)
            
            with col2:
                sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                kps = st.slider("KPS Score", 0, 100, 80)
                scan_date = st.date_input("Initial Scan Date")
            
            submitted = st.form_submit_button("Add Patient")
            
            if submitted and patient_id and name:
                patient_data = {
                    'patient_id': patient_id,
                    'name': name,
                    'age': age,
                    'sex': sex,
                    'kps': kps,
                    'scan_date': scan_date.isoformat()
                }
                
                db.add_or_update_patient(patient_data)
                st.success(f"Patient {patient_id} added successfully!")
                st.rerun()

def create_progression_tracker():
    """Create the progression tracking interface with dual scan comparison"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2>[METRICS] Tumor Progression Tracking</h2>
        <p>Compare serial MRI scans and track treatment response over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient selection
    if 'clinical_db' not in st.session_state:
        st.session_state.clinical_db = ClinicalDatabase()
    
    db = st.session_state.clinical_db
    patients = db.get_all_patients()
    
    if not patients:
        st.warning("No patients found in the database. Please add patients first.")
        return
    
    # Patient selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_patient = st.selectbox(
            "Select Patient",
            patients,
            format_func=lambda x: f"{x.patient_id} - {x.name}" if x.name else x.patient_id
        )
    
    with col2:
        if selected_patient:
            st.info(f"**Patient:** {selected_patient.name} | **Age:** {selected_patient.age} | **Current Status:** {selected_patient.tumor_type}")
    
    # Dual scan comparison interface
    st.subheader("[CAMERA] Dual Scan Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### [CALENDAR] Baseline/Previous Scan")
        baseline_file = st.file_uploader(
            "Upload Baseline MRI",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii'],
            key="baseline_scan"
        )
        baseline_date = st.date_input("Baseline Scan Date", key="baseline_date")
        
        if baseline_file:
            st.success(" Baseline scan uploaded")
    
    with col2:
        st.markdown("#####  Follow-up Scan")
        followup_file = st.file_uploader(
            "Upload Follow-up MRI",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii'],
            key="followup_scan"
        )
        followup_date = st.date_input("Follow-up Scan Date", key="followup_date")
        
        if followup_file:
            st.success(" Follow-up scan uploaded")
    
    # Analysis button
    if baseline_file and followup_file:
        if st.button("[ANALYSIS] Perform Comparative Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing progression..."):
                # Use unified inference for analysis
                try:
                    device = get_device()
                    inference_engine = RealBrainTumorInferenceEngine(device=device)
                    
                    # Analyze both scans
                    baseline_results = inference_engine.detect_tumor_comprehensive(
                        np.array(baseline_file)  # Simplified - would need proper image loading
                    )
                    followup_results = inference_engine.detect_tumor_comprehensive(
                        np.array(followup_file)  # Simplified - would need proper image loading
                    )
                    
                    # Calculate progression metrics
                    progression_metrics = calculate_progression_metrics(
                        baseline_results, followup_results, 
                        {'scan_date': baseline_date}, {'scan_date': followup_date}
                    )
                    
                    # Display results
                    display_progression_results(progression_metrics)
                    
                except Exception as e:
                    logger.error(f"Progression analysis failed: {e}")
                    st.error("Progression analysis failed. Please try again.")
    
    # Historical progression chart
    if selected_patient and selected_patient.scan_history:
        st.subheader("[METRICS] Historical Progression")
        
        dates, volumes, responses = db.get_progression_data(selected_patient.patient_id)
        
        if dates and volumes:
            fig = go.Figure()
            
            # Color mapping for responses
            color_map = {
                'Baseline': 'blue',
                'Stable Disease': 'green',
                'Partial Response': 'orange',
                'Progressive Disease': 'red',
                'Complete Response': 'purple'
            }
            
            for response in set(responses):
                response_indices = [i for i, r in enumerate(responses) if r == response]
                fig.add_trace(go.Scatter(
                    x=[dates[i] for i in response_indices],
                    y=[volumes[i] for i in response_indices],
                    mode='markers+lines',
                    name=response,
                    marker=dict(
                        color=color_map.get(response, 'gray'),
                        size=10
                    ),
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                title=f"Tumor Volume Progression - {selected_patient.patient_id}",
                xaxis_title="Scan Date",
                yaxis_title="Tumor Area (%)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No historical scan data available for the selected patient.")

def calculate_progression_metrics(result1, result2, metadata1, metadata2):
    """Calculate progression metrics - Compatible with unified results"""
    volume_change_abs = result2.get('tumor_area', 0) - result1.get('tumor_area', 0)
    volume_change_pct = (volume_change_abs / max(result1.get('tumor_area', 1), 1)) * 100
    
    grade_change = result2.get('grade', 0) - result1.get('grade', 0)
    
    try:
        date1 = metadata1.get('scan_date', datetime.now().date())
        date2 = metadata2.get('scan_date', datetime.now().date())
        if isinstance(date1, str):
            date1 = datetime.strptime(date1, '%Y-%m-%d').date()
        if isinstance(date2, str):
            date2 = datetime.strptime(date2, '%Y-%m-%d').date()
        time_diff = abs((date2 - date1).days)
    except:
        time_diff = 180
    
    if time_diff > 0 and result1.get('tumor_area', 0) > 0:
        growth_rate = volume_change_pct / time_diff * 30
    else:
        growth_rate = 0
    
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
        'recurrence_risk': min(95, max(5, 30 + volume_change_pct)),
        'treatment_response': (status, status_color)
    }

def display_progression_results(metrics):
    """Display progression analysis results"""
    st.markdown("###  Progression Analysis Results")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        volume_change = metrics['volume_change_pct']
        st.metric(
            "Volume Change",
            f"{volume_change:+.1f}%",
            delta_color="inverse" if volume_change > 0 else "normal"
        )
    
    with metric_col2:
        st.metric(
            "Response Assessment",
            metrics['progression_status']
        )
    
    with metric_col3:
        st.metric(
            "Recurrence Risk",
            f"{metrics['recurrence_risk']:.0f}%"
        )
    
    # Response assessment details
    st.markdown("####  RANO Criteria Assessment")
    
    response_details = {
        'Tumor Size': f"{metrics['volume_change_pct']:+.1f}%",
        'Grade Change': f"{metrics['grade_change']:+d}",
        'Time Interval': f"{metrics['time_diff_days']} days",
        'Growth Rate': f"{metrics['growth_rate_monthly']:+.2f}%/month",
        'Overall Response': metrics['progression_status']
    }
    
    response_df = pd.DataFrame(list(response_details.items()), columns=['Criteria', 'Assessment'])
    st.dataframe(response_df, width='stretch')
    
    # Treatment recommendation
    if metrics['progression_status'] in ['Significant Progression', 'Progressive Disease']:
        st.error(" **Recommendation:** Urgent multidisciplinary consultation required")
    else:
        st.success(" **Status:** Continue current treatment plan with monitoring")

# Initialize clinical database in session state
def init_clinical_system():
    """Initialize the clinical system"""
    if 'clinical_db' not in st.session_state:
        st.session_state.clinical_db = ClinicalDatabase()
        
        # Add some sample data if database is empty
        if not st.session_state.clinical_db.patients:
            sample_patients = [
                {
                    'patient_id': 'PT-001',
                    'name': 'John Doe',
                    'age': 45,
                    'sex': 'Male',
                    'kps': 90,
                    'scan_date': datetime.now().isoformat(),
                    'tumor_detected': True,
                    'tumor_type': get_tumor_info(3)['name'],
                    'tumor_grade': 3,
                    'confidence': 0.92,
                    'tumor_percentage': 15.5
                },
                {
                    'patient_id': 'PT-002',
                    'name': 'Jane Smith',
                    'age': 38,
                    'sex': 'Female',
                    'kps': 85,
                    'scan_date': datetime.now().isoformat(),
                    'tumor_detected': False,
                    'tumor_type': get_tumor_info(0)['name'],
                    'tumor_grade': 0,
                    'confidence': 0.98,
                    'tumor_percentage': 0.0
                }
            ]
            
            for patient_data in sample_patients:
                st.session_state.clinical_db.add_or_update_patient(patient_data)

# Export functions for use in other modules
def get_clinical_database():
    """Get the clinical database instance"""
    if 'clinical_db' not in st.session_state:
        init_clinical_system()
    return st.session_state.clinical_db

def add_patient_with_analysis(patient_data, image_data):
    """Add patient with AI analysis results"""
    db = get_clinical_database()
    
    # Perform AI analysis using unified inference
    try:
        device = get_device()
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        analysis_results = inference_engine.detect_tumor_comprehensive(image_data)
        
        # Add patient with analysis results
        db.add_or_update_patient(patient_data, analysis_results)
        return analysis_results
        
    except Exception as e:
        logger.error(f"Patient analysis failed: {e}")
        # Add patient without analysis results
        db.add_or_update_patient(patient_data)
        return None

def generate_clinical_report_pdf(patient_data, analysis_results):
    """Generate clinical report PDF"""
    report_generator = ClinicalReportGenerator()
    return report_generator.generate_clinical_report(patient_data, analysis_results)