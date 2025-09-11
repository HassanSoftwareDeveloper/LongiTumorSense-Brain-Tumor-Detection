# import numpy as np
# import pandas as pd

# def generate_treatment_recommendations(tumor_info, survival_days, patient_data):
#     """Generate evidence-based treatment recommendations with grading"""
#     recommendations = []
#     evidence_levels = []
    
#     # Base recommendations on tumor grade with evidence levels
#     if tumor_info['grade'] == 0:
#         recommendations = [
#             "🔍 Continue regular monitoring with MRI scans every 6-12 months",
#             "📊 Consider additional imaging modalities if symptoms develop",
#             "💊 No active treatment required at this time"
#         ]
#         evidence_levels = ["Level II", "Level III", "Level IV"]
        
#     elif tumor_info['grade'] == 1:
#         recommendations = [
#             "🏥 Surgical consultation for complete resection (GTR)",
#             "📅 Regular follow-up every 3-6 months with contrast-enhanced MRI",
#             "🧬 Genetic testing for BRAF V600E mutation recommended",
#             "💊 Consider adjuvant therapy if incomplete resection or recurrence"
#         ]
#         evidence_levels = ["Level I", "Level II", "Level II", "Level III"]
        
#     elif tumor_info['grade'] == 2:
#         recommendations = [
#             "⚕️ Maximal safe surgical resection recommended (GTR if possible)",
#             "☢️ Post-operative radiation therapy (54-60 Gy in 30-33 fractions)",
#             "💉 Consider adjuvant chemotherapy (Temozolomide or PCV)",
#             "🔬 Molecular profiling for IDH mutation, 1p/19q codeletion status",
#             "📅 Close monitoring every 2-3 months with advanced MRI sequences"
#         ]
#         evidence_levels = ["Level I", "Level I", "Level I", "Level I", "Level II"]
        
#     elif tumor_info['grade'] >= 3:
#         recommendations = [
#             "🚨 Urgent multidisciplinary team consultation",
#             "🏥 Maximal safe surgical resection (neuronavigation, intraoperative monitoring)",
#             "☢️ Concurrent chemoradiotherapy (60 Gy + Temozolomide 75 mg/m²)",
#             "💊 Maintenance Temozolomide for 6-12 cycles (150-200 mg/m² 5/28 days)",
#             "🧬 Comprehensive molecular testing (IDH, 1p/19q, MGMT, TERT, EGFR)",
#             "📊 Consider clinical trial enrollment or targeted therapies",
#             "🔄 Advanced imaging every 2 months during treatment (DSC/DCE perfusion, DWI)"
#         ]
#         evidence_levels = ["Level I", "Level I", "Level I", "Level I", "Level I", "Level II", "Level II"]
    
#     # Add personalized recommendations based on survival prediction
#     if survival_days < 180:
#         recommendations.insert(0, "🕐 Focus on quality of life and palliative symptom management")
#         recommendations.append("❤️ Early palliative care and family support services integration")
#         evidence_levels.insert(0, "Level I")
#         evidence_levels.append("Level I")
    
#     # Add age-specific considerations
#     if patient_data['age'] > 70:
#         recommendations.append("👴 Consider hypofractionated radiotherapy (40 Gy in 15 fractions) and modified chemotherapy dosing")
#         evidence_levels.append("Level II")
    
#     # Add molecular marker-based recommendations
#     if patient_data.get('idh_status') == 'mutant':
#         recommendations.append("🧪 IDH-mutant: Consider IDH inhibitors in clinical trial setting")
#         evidence_levels.append("Level II")
    
#     if patient_data.get('mgmt_status') == 'methylated':
#         recommendations.append("🔬 MGMT methylated: Improved response to Temozolomide expected")
#         evidence_levels.append("Level I")
    
#     return list(zip(recommendations, evidence_levels))

# def create_patient_database():
#     """Create a more realistic patient database"""
#     patients = []
    
#     # Sample patient data with more clinical details
#     patient_templates = [
#         {
#             'Patient_ID': 'GLI-2024-001', 'MRN': '78912346', 'Age': 45, 'Sex': 'M', 
#             'Grade': 4, 'Diagnosis': 'Glioblastoma, IDH-wildtype',
#             'Survival_Days': 287, 'Status': 'Active Treatment', 
#             'Last_Scan': '2024-01-15', 'Next_Appointment': '2024-02-15',
#             'KPS': 70, 'IDH_Status': 'Wildtype', 'MGMT_Status': 'Unmethylated',
#             'Location': 'Right Temporal', 'Treatment_Phase': 'Concurrent Chemoradiation',
#             'Oncologist': 'Dr. Smith', 'Surgeon': 'Dr. Johnson'
#         },
#         {
#             'Patient_ID': 'GLI-2024-002', 'MRN': '78912347', 'Age': 62, 'Sex': 'F', 
#             'Grade': 2, 'Diagnosis': 'Astrocytoma, IDH-mutant',
#             'Survival_Days': 892, 'Status': 'Follow-up', 
#             'Last_Scan': '2024-01-20', 'Next_Appointment': '2024-04-20',
#             'KPS': 90, 'IDH_Status': 'Mutant', 'MGMT_Status': 'Methylated',
#             'Location': 'Left Frontal', 'Treatment_Phase': 'Monitoring',
#             'Oncologist': 'Dr. Williams', 'Surgeon': 'Dr. Brown'
#         },
#         {
#             'Patient_ID': 'GLI-2024-003', 'MRN': '78912348', 'Age': 38, 'Sex': 'M', 
#             'Grade': 3, 'Diagnosis': 'Anaplastic Astrocytoma, IDH-wildtype',
#             'Survival_Days': 456, 'Status': 'Active Treatment', 
#             'Last_Scan': '2024-01-25', 'Next_Appointment': '2024-03-10',
#             'KPS': 80, 'IDH_Status': 'Wildtype', 'MGMT_Status': 'Methylated',
#             'Location': 'Right Parietal', 'Treatment_Phase': 'Adjuvant Chemotherapy',
#             'Oncologist': 'Dr. Davis', 'Surgeon': 'Dr. Miller'
#         },
#         {
#             'Patient_ID': 'GLI-2024-004', 'MRN': '78912349', 'Age': 55, 'Sex': 'F', 
#             'Grade': 1, 'Diagnosis': 'Pilocytic Astrocytoma',
#             'Survival_Days': 1245, 'Status': 'Monitoring', 
#             'Last_Scan': '2024-01-28', 'Next_Appointment': '2024-07-28',
#             'KPS': 100, 'IDH_Status': 'Not Applicable', 'MGMT_Status': 'Not Tested',
#             'Location': 'Cerebellum', 'Treatment_Phase': 'Post-operative Surveillance',
#             'Oncologist': 'Dr. Wilson', 'Surgeon': 'Dr. Anderson'
#         },
#         {
#             'Patient_ID': 'GLI-2024-005', 'MRN': '78912350', 'Age': 72, 'Sex': 'M', 
#             'Grade': 4, 'Diagnosis': 'Glioblastoma, IDH-wildtype',
#             'Survival_Days': 154, 'Status': 'Palliative Care', 
#             'Last_Scan': '2024-02-01', 'Next_Appointment': '2024-02-15',
#             'KPS': 50, 'IDH_Status': 'Wildtype', 'MGMT_Status': 'Unmethylated',
#             'Location': 'Multifocal', 'Treatment_Phase': 'Best Supportive Care',
#             'Oncologist': 'Dr. Taylor', 'Surgeon': 'Dr. Thomas'
#         }
#     ]
    
#     return patient_templates

# def calculate_risk_level(tumor_info, patient_data):
#     """Calculate comprehensive risk assessment"""
#     risk_score = 0
    
#     # Tumor grade factor (0-40 points)
#     grade_weights = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40}
#     risk_score += grade_weights.get(tumor_info['grade'], 0)
    
#     # Probability factor (0-20 points)
#     risk_score += tumor_info['probability'] * 20
    
#     # Age factor (0-15 points)
#     if patient_data['age'] > 65:
#         risk_score += 15
#     elif patient_data['age'] > 50:
#         risk_score += 10
#     elif patient_data['age'] > 40:
#         risk_score += 5
    
#     # KPS factor (0-15 points)
#     risk_score += (100 - patient_data['kps']) * 0.15
    
#     # Progression factor (0-10 points)
#     risk_score += tumor_info['progression_score'] * 10
    
#     # Determine risk level
#     if risk_score >= 70:
#         return "Critical", risk_score, "#dc3545"
#     elif risk_score >= 50:
#         return "High", risk_score, "#fd7e14"
#     elif risk_score >= 30:
#         return "Moderate", risk_score, "#ffc107"
#     elif risk_score >= 15:
#         return "Low", risk_score, "#20c997"
#     else:
#         return "Very Low", risk_score, "#198754"







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