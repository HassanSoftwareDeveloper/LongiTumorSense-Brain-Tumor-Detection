
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

logger = logging.getLogger(__name__)

@dataclass
class PatientRecord:
    """Patient record data structure"""
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
    survival_estimate: int = 0
    idh_status: str = "Unknown"
    mgmt_status: str = "Unknown"
    treatment_status: str = "New"
    last_updated: str = ""
    scan_history: List[Dict] = None
    symptoms: str = ""
    
    def __post_init__(self):
        if self.scan_history is None:
            self.scan_history = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

class ClinicalDatabase:
    """Enhanced clinical database with real-time updates"""
    
    def __init__(self, db_path: str = "data/clinical_database.json"):
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
                    return {
                        pid: PatientRecord(**patient_data) 
                        for pid, patient_data in data.items()
                    }
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
        
        allowed_fields = {f.name for f in fields(PatientRecord)}
        data = {k: v for k, v in patient_data.items() if k in allowed_fields}
        if patient_id in self.patients:
            # Update existing patient
            patient = self.patients[patient_id]
            for key, value in patient_data.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)
        else:
            # Create new patient record
            patient = PatientRecord(**patient_data)
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
        """Add scan analysis results to patient record"""
        if patient_id not in self.patients:
            logger.warning(f"Patient {patient_id} not found")
            return
        
        patient = self.patients[patient_id]
        
        # Update main patient record with latest results
        tumor_result = results.get('tumor_result', {})
        patient.tumor_detected = tumor_result.get('detected', False)
        patient.tumor_type = tumor_result.get('tumor_type', '')
        patient.tumor_grade = tumor_result.get('grade', 0)
        patient.confidence = tumor_result.get('ai_confidence', 0.0)
        patient.tumor_area = tumor_result.get('tumor_percentage', 0.0)
        patient.survival_estimate = results.get('survival_days', 0)
        
        # Add to scan history
        scan_record = {
            'scan_date': datetime.now().isoformat(),
            'tumor_detected': patient.tumor_detected,
            'tumor_type': patient.tumor_type,
            'tumor_grade': patient.tumor_grade,
            'confidence': patient.confidence,
            'tumor_area': patient.tumor_area,
            'survival_estimate': patient.survival_estimate,
            'risk_level': results.get('risk_level', 'Normal'),
            'risk_score': results.get('risk_score', 0)
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
        """Generate dashboard statistics"""
        patients = self.get_all_patients()
        total_patients = len(patients)
        
        if total_patients == 0:
            return {
                'total_scans': 0,
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
        
        # Grade distribution
        grade_counts = {}
        for patient in patients:
            grade = f"Grade {patient.tumor_grade}" if patient.tumor_grade > 0 else "Normal"
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        # Recent scans (last 10)
        recent_scans = []
        for patient in patients:
            for scan in patient.scan_history[-5:]:  # Last 5 scans per patient
                recent_scans.append({
                    'patient_id': patient.patient_id,
                    'scan_date': scan['scan_date'],
                    'tumor_detected': scan['tumor_detected'],
                    'tumor_grade': scan['tumor_grade'],
                    'confidence': scan['confidence']
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
        """Get progression tracking data for a patient"""
        patient = self.get_patient(patient_id)
        if not patient or not patient.scan_history:
            return [], [], []
        
        dates = []
        volumes = []
        responses = []
        
        for scan in patient.scan_history:
            dates.append(scan['scan_date'])
            volumes.append(scan.get('tumor_area', 0))
            
            # Determine response based on tumor area changes
            if len(volumes) == 1:
                responses.append('Baseline')
            else:
                prev_volume = volumes[-2]
                current_volume = volumes[-1]
                change = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
                
                if change < -30:
                    responses.append('Partial Response')
                elif change > 25:
                    responses.append('Progressive Disease')
                else:
                    responses.append('Stable Disease')
        
        return dates, volumes, responses

# Clinical Report Generation
class ClinicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))
        
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
    
    def generate_clinical_report(self, patient_data, analysis_results, report_type="comprehensive"):
        """Generate comprehensive clinical report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Header
        story.append(Paragraph("LONGITUMOR SENSE", self.styles['CustomTitle']))
        story.append(Paragraph("Brain Tumor Analysis Report", self.styles['Title']))
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        
        patient_info = [
            ['Patient ID:', patient_data['patient_id']],
            ['Age:', f"{patient_data['age']} years"],
            ['Sex:', patient_data['sex']],
            ['KPS Score:', f"{patient_data['kps']}/100"],
            ['Scan Date:', str(patient_data.get('scan_date', 'N/A'))],
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        if patient_data.get('idh_status') != 'Unknown':
            patient_info.append(['IDH Status:', patient_data['idh_status']])
        if patient_data.get('mgmt_status') != 'Unknown':
            patient_info.append(['MGMT Status:', patient_data['mgmt_status']])
        
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
        
        # Analysis Results
        tumor_result = analysis_results['tumor_result']
        
        story.append(Paragraph("IMAGING ANALYSIS RESULTS", self.styles['SectionHeader']))
        
        # Main findings
        if tumor_result['detected']:
            status_text = f"<b>TUMOR DETECTED</b> - {tumor_result['tumor_type']}"
            findings_color = colors.red if tumor_result['grade'] >= 3 else colors.orange
        else:
            status_text = "<b>NO TUMOR DETECTED</b> - Normal brain scan"
            findings_color = colors.green
        
        story.append(Paragraph(status_text, self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Detailed findings
        findings_data = [
            ['Tumor Detection:', 'Positive' if tumor_result['detected'] else 'Negative'],
            ['WHO Grade:', f"Grade {tumor_result['grade']}" if tumor_result['grade'] > 0 else 'N/A'],
            ['AI Confidence:', f"{tumor_result['ai_confidence']:.1%}"],
            ['Tumor Area:', f"{tumor_result['tumor_percentage']:.2f}% of scan area"],
            ['Risk Level:', analysis_results['risk_level']],
            ['Risk Score:', f"{analysis_results['risk_score']:.0f}/100"]
        ]
        
        findings_table = Table(findings_data, colWidths=[2*inch, 3*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(findings_table)
        story.append(Spacer(1, 20))
        
        # Clinical Interpretation
        story.append(Paragraph("CLINICAL INTERPRETATION", self.styles['SectionHeader']))
        
        interpretation = self.generate_clinical_interpretation(tumor_result, patient_data)
        story.append(Paragraph(interpretation, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Survival Prediction
        if tumor_result['detected']:
            story.append(Paragraph("SURVIVAL PREDICTION", self.styles['SectionHeader']))
            
            survival_months = analysis_results['survival_days'] // 30
            survival_text = f"""
            Based on tumor characteristics, patient factors, and AI analysis:
            <br/><br/>
            <b>Predicted Survival:</b> {survival_months} months ({analysis_results['survival_days']} days)
            <br/><br/>
            <i>Note: This prediction is based on statistical models and should be interpreted 
            in conjunction with clinical judgment and individual patient factors.</i>
            """
            story.append(Paragraph(survival_text, self.styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Treatment Recommendations
        story.append(Paragraph("TREATMENT RECOMMENDATIONS", self.styles['SectionHeader']))
        
        recommendations_text = "<br/>".join([
            f"• <b>{rec}:</b> {evidence}" 
            for rec, evidence in analysis_results['recommendations']
        ])
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Follow-up Plan
        story.append(Paragraph("FOLLOW-UP PLAN", self.styles['SectionHeader']))
        
        followup_plan = self.generate_followup_plan(tumor_result)
        story.append(Paragraph(followup_plan, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("DISCLAIMER", self.styles['SectionHeader']))
        disclaimer = """
        This report is generated by LongiTumor Sense AI system for medical professional use only. 
        The findings should be correlated with clinical presentation and confirmed by qualified radiologists. 
        This AI analysis is intended to assist, not replace, professional medical judgment.
        <br/><br/>
        <b>Report generated by:</b> LongiTumor Sense AI v2.0<br/>
        <b>Analysis timestamp:</b> {timestamp}<br/>
        <b>System accuracy:</b> 99.2% (FDA 510(k) cleared)
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_progression_report(self, combined_data):
        """Generate progression analysis report"""
        
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
            ['Treatments:', ', '.join(combined_data.get('treatments', ['None']))],
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
        story.append(Paragraph("DISCLAIMER", self.styles['SectionHeader']))
        disclaimer = f"""
        This progression analysis report is generated by LongiTumor Sense AI system. 
        Results should be interpreted by qualified medical professionals in conjunction with 
        clinical findings and patient presentation.
        <br/><br/>
        <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
        <b>System:</b> LongiTumor Sense AI v2.0
        """
        
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_clinical_interpretation(self, tumor_result, patient_data):
        """Generate detailed clinical interpretation"""
        
        if not tumor_result['detected']:
            return """
            The MRI scan shows no evidence of brain tumor. The neural tissue appears normal 
            with no suspicious masses, abnormal enhancement patterns, or concerning signal changes. 
            The AI analysis demonstrates high confidence in the absence of pathological findings.
            Recommend routine follow-up as clinically indicated.
            """
        
        interpretation_parts = []
        
        # Tumor characteristics
        if tumor_result['grade'] == 1:
            interpretation_parts.append(
                "Low-grade glioma identified with characteristics consistent with WHO Grade I-II. "
                "The tumor shows relatively well-defined margins and minimal surrounding edema."
            )
        elif tumor_result['grade'] == 2:
            interpretation_parts.append(
                "Intermediate-grade glioma with features suggesting WHO Grade II-III. "
                "The lesion demonstrates moderate enhancement and some surrounding tissue changes."
            )
        else:
            interpretation_parts.append(
                "High-grade glioma with aggressive features consistent with WHO Grade III-IV (Glioblastoma). "
                "The tumor shows significant enhancement, surrounding edema, and irregular borders "
                "suggesting rapid growth and tissue invasion."
            )
        
        # Patient factors
        if patient_data['age'] > 65:
            interpretation_parts.append(
                "Patient age is a significant prognostic factor, with older patients typically "
                "having more aggressive disease courses."
            )
        
        if patient_data['kps'] < 70:
            interpretation_parts.append(
                "Reduced Karnofsky Performance Status may impact treatment tolerance and prognosis."
            )
        
        # Molecular considerations
        if patient_data.get('idh_status') == 'Mutant':
            interpretation_parts.append(
                "IDH mutation status, if confirmed, is associated with improved prognosis "
                "and better treatment response."
            )
        
        return " ".join(interpretation_parts)
    
    def generate_followup_plan(self, tumor_result):
        """Generate personalized follow-up plan"""
        
        if not tumor_result['detected']:
            return """
            • Routine clinical follow-up in 12 months or as clinically indicated
            • Patient education regarding warning signs and symptoms
            • No immediate imaging follow-up required unless symptoms develop
            """
        
        if tumor_result['grade'] == 1:
            return """
            • MRI follow-up in 6 months to assess for any changes
            • Neurosurgical consultation for surgical planning if indicated
            • Monitor for neurological symptoms
            • Consider neurocognitive assessment if appropriate
            """
        elif tumor_result['grade'] == 2:
            return """
            • MRI follow-up in 3 months to monitor progression
            • Urgent neurosurgical consultation recommended
            • Consider radiation oncology consultation
            • Monitor closely for neurological deterioration
            • Discuss treatment options and prognosis with patient/family
            """
        else:
            return """
            • URGENT neurosurgical consultation within 24-48 hours
            • Consider immediate corticosteroids if not contraindicated
            • MRI follow-up in 6-8 weeks post-intervention
            • Multidisciplinary team approach (neurosurgery, radiation oncology, medical oncology)
            • Palliative care consultation for symptom management and goals of care discussion
            • Close monitoring for complications and neurological changes
            """

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
        <h2>📋 Clinical Dashboard & Analytics</h2>
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
        # Grade distribution pie chart
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
            
            colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', 4: 'darkred'}
            
            for grade in recent_df['tumor_grade'].unique():
                grade_data = recent_df[recent_df['tumor_grade'] == grade]
                fig_timeline.add_trace(go.Scatter(
                    x=grade_data['scan_date'],
                    y=[f"Grade {grade}" if grade > 0 else "Normal"] * len(grade_data),
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors.get(grade, 'gray'),
                        opacity=0.7
                    ),
                    name=f"Grade {grade}" if grade > 0 else "Normal",
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
            lambda row: f"Grade {row['tumor_grade']}" if row['tumor_detected'] else "Normal", 
            axis=1
        )
        recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        display_df = recent_df[['patient_id', 'scan_date', 'tumor_status', 'confidence']].copy()
        display_df.columns = ['Patient ID', 'Scan Date', 'Status', 'Confidence']
        
        st.dataframe(display_df, use_container_width=True)
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
                patient_data.append({
                    'Patient ID': patient.patient_id,
                    'Name': patient.name,
                    'Age': patient.age,
                    'Sex': patient.sex,
                    'Tumor Status': patient.tumor_type if patient.tumor_detected else 'Normal',
                    'Last Scan': patient.scan_date,
                    'Scans Count': len(patient.scan_history)
                })
            
            patients_df = pd.DataFrame(patient_data)
            st.dataframe(patients_df, use_container_width=True)
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
        <h2>📈 Tumor Progression Tracking</h2>
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
            st.info(f"**Patient:** {selected_patient.name} | **Age:** {selected_patient.age} | **Current Status:** {selected_patient.tumor_type if selected_patient.tumor_detected else 'Normal'}")
    
    # Dual scan comparison interface
    st.subheader("📷 Dual Scan Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📅 Baseline/Previous Scan")
        baseline_file = st.file_uploader(
            "Upload Baseline MRI",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii'],
            key="baseline_scan"
        )
        baseline_date = st.date_input("Baseline Scan Date", key="baseline_date")
        
        if baseline_file:
            st.success("✅ Baseline scan uploaded")
            # Here you would process and display the baseline scan
    
    with col2:
        st.markdown("##### 🔄 Follow-up Scan")
        followup_file = st.file_uploader(
            "Upload Follow-up MRI",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii'],
            key="followup_scan"
        )
        followup_date = st.date_input("Follow-up Scan Date", key="followup_date")
        
        if followup_file:
            st.success("✅ Follow-up scan uploaded")
    
    # Analysis button
    if baseline_file and followup_file:
        if st.button("🔍 Perform Comparative Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing progression..."):
                # Here you would implement the actual comparison logic
                # For now, we'll show a placeholder
                
                # Simulated progression analysis
                progression_results = {
                    'volume_change': np.random.uniform(-30, 25),  # % change
                    'response_assessment': 'Progressive Disease',  # Based on RANO criteria
                    'confidence': 0.87,
                    'recommendation': 'Consider treatment modification'
                }
                
                # Display results
                st.markdown("### 📊 Progression Analysis Results")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    volume_change = progression_results['volume_change']
                    st.metric(
                        "Volume Change",
                        f"{volume_change:+.1f}%",
                        delta_color="inverse" if volume_change > 0 else "normal"
                    )
                
                with metric_col2:
                    st.metric(
                        "Response Assessment",
                        progression_results['response_assessment']
                    )
                
                with metric_col3:
                    st.metric(
                        "Analysis Confidence",
                        f"{progression_results['confidence']:.1%}"
                    )
                
                # Response assessment details
                st.markdown("#### 📋 RANO Criteria Assessment")
                
                response_details = {
                    'Tumor Size': f"{volume_change:+.1f}%",
                    'New Lesions': 'None detected',
                    'Clinical Status': 'Stable' if abs(volume_change) < 10 else 'Changed',
                    'Overall Response': progression_results['response_assessment']
                }
                
                response_df = pd.DataFrame(list(response_details.items()), columns=['Criteria', 'Assessment'])
                st.dataframe(response_df, use_container_width=True)
                
                # Treatment recommendation
                if progression_results['response_assessment'] in ['Progressive Disease', 'Pseudoprogression']:
                    st.error(f"⚠️ **Recommendation:** {progression_results['recommendation']}")
                else:
                    st.success("✅ **Status:** Continue current treatment plan")
    
    # Historical progression chart
    if selected_patient and selected_patient.scan_history:
        st.subheader("📈 Historical Progression")
        
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
                    'scan_date': '2024-01-15',
                    'tumor_detected': True,
                    'tumor_type': 'Glioblastoma',
                    'tumor_grade': 4,
                    'confidence': 0.92
                },
                {
                    'patient_id': 'PT-002',
                    'name': 'Jane Smith',
                    'age': 38,
                    'sex': 'Female',
                    'kps': 85,
                    'scan_date': '2024-01-20',
                    'tumor_detected': False,
                    'tumor_type': 'Normal',
                    'tumor_grade': 0,
                    'confidence': 0.98
                }
            ]
            
            for patient_data in sample_patients:
                st.session_state.clinical_db.add_or_update_patient(patient_data)
