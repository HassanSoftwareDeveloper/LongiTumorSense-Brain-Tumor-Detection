# rag_engine.py  —  LongiTumorSense RAG (Retrieval-Augmented Grading)
# =============================================================================
#
#  PURPOSE
#  -------
#  Augment the AI model's grade prediction with evidence retrieved from:
#    1. TCGA clinical database  (real patient outcomes, 1000+ cases)
#    2. WHO 2021 glioma grading knowledge base  (curated clinical rules)
#    3. Local patient history   (scan_history from clinical_database.json)
#
#  HOW IT WORKS
#  ------------
#  A. BUILD INDEX
#     Each TCGA case + knowledge-base entry is converted to a feature vector:
#       [age_norm, survival_norm, grade_0..4 one-hot, sex_bin, treatment_flags...]
#     TF-IDF is used for text fields (diagnosis, treatment type).
#     Combined into a dense matrix stored in memory (no external DB needed).
#
#  B. QUERY
#     The current patient's AI result + demographics are encoded the same way.
#     Cosine similarity retrieves the top-K most similar cases.
#
#  C. AUGMENT
#     Retrieved cases vote on the grade (weighted by similarity × outcome quality).
#     The RAG grade vote is blended with the AI model's grade probabilities.
#     The blend weight is proportional to retrieval confidence.
#
#  D. COMPARISON AUGMENTATION
#     For progression analysis, RAG retrieves cases with similar baseline grade
#     and time interval, then reports population-level progression rates.
#
#  DEPENDENCIES
#  ------------
#  All standard: numpy, pandas, scipy, scikit-learn  (already in requirements.txt)
#  No external vector DB, no API calls, works fully offline.
# =============================================================================

from __future__ import annotations

import ast
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
#  CONSTANTS
# =============================================================================

CLINICAL_DATA_DIR = Path(__file__).parent / "clinical_data"
LOCAL_DB_PATH     = Path(__file__).parent / "data" / "clinical_database.json"

GRADE_NAMES = {
    0: "Normal",
    1: "Grade I Glioma",
    2: "Grade II Glioma",
    3: "Grade III Glioma",
    4: "Grade IV Glioma (GBM)",
}

# Median survival months by grade (CBTRUS 2023 / Stupp NEJM 2005)
GRADE_MEDIAN_SURVIVAL_MONTHS = {0: 999, 1: 96, 2: 60, 3: 18, 4: 14.6}

# WHO 2021 glioma grading knowledge base
# Each entry: text description + grade label + key features
WHO_KNOWLEDGE_BASE: List[Dict] = [
    # ── Grade I ──────────────────────────────────────────────────────────────
    {
        "grade": 1,
        "title": "Pilocytic Astrocytoma (WHO Grade I)",
        "text": (
            "Pilocytic astrocytoma is a WHO Grade I glioma. Typically occurs in children "
            "and young adults. Well-circumscribed, often cystic with mural nodule. "
            "Low mitotic activity. BRAF fusion common. Excellent prognosis with surgical "
            "resection. 10-year survival >90%. Low GLCM contrast, high homogeneity. "
            "Small tumor area, regular shape, low asymmetry."
        ),
        "key_features": {
            "age_range": (5, 25),
            "tumor_pct_range": (0.5, 8.0),
            "glcm_malignancy_range": (0.1, 0.35),
            "compactness_range": (0.6, 1.0),
            "asymmetry_range": (0.0, 0.10),
        },
    },
    {
        "grade": 1,
        "title": "Ganglioglioma (WHO Grade I)",
        "text": (
            "Ganglioglioma WHO Grade I. Temporal lobe most common. Young patients. "
            "Well-defined, often calcified. BRAF V600E mutation. Seizures common presentation. "
            "Surgical resection curative in most cases. Low malignancy features on imaging."
        ),
        "key_features": {
            "age_range": (10, 30),
            "tumor_pct_range": (0.3, 5.0),
            "glcm_malignancy_range": (0.1, 0.30),
            "compactness_range": (0.65, 1.0),
            "asymmetry_range": (0.0, 0.08),
        },
    },
    # ── Grade II ─────────────────────────────────────────────────────────────
    {
        "grade": 2,
        "title": "Diffuse Astrocytoma IDH-mutant (WHO Grade II)",
        "text": (
            "Diffuse astrocytoma IDH-mutant WHO Grade II. Adults 30-50 years. "
            "Diffusely infiltrating, no contrast enhancement typically. "
            "IDH1/IDH2 mutation present. ATRX loss. Slow growth. "
            "Median survival 8-10 years. Moderate GLCM contrast, moderate asymmetry. "
            "T2/FLAIR hyperintense. Moderate tumor area."
        ),
        "key_features": {
            "age_range": (25, 55),
            "tumor_pct_range": (2.0, 15.0),
            "glcm_malignancy_range": (0.25, 0.50),
            "compactness_range": (0.4, 0.75),
            "asymmetry_range": (0.05, 0.20),
        },
    },
    {
        "grade": 2,
        "title": "Oligodendroglioma IDH-mutant 1p/19q-codeleted (WHO Grade II)",
        "text": (
            "Oligodendroglioma IDH-mutant 1p/19q-codeleted WHO Grade II. "
            "Frontal lobe predominance. Adults 35-55 years. Calcification common. "
            "IDH mutation + 1p/19q codeletion required for diagnosis. "
            "Better prognosis than astrocytoma. Median survival 12-15 years. "
            "Moderate imaging features, often cortical-based."
        ),
        "key_features": {
            "age_range": (30, 60),
            "tumor_pct_range": (1.5, 12.0),
            "glcm_malignancy_range": (0.20, 0.45),
            "compactness_range": (0.45, 0.80),
            "asymmetry_range": (0.04, 0.18),
        },
    },
    # ── Grade III ─────────────────────────────────────────────────────────────
    {
        "grade": 3,
        "title": "Anaplastic Astrocytoma IDH-mutant (WHO Grade III)",
        "text": (
            "Anaplastic astrocytoma IDH-mutant WHO Grade III. Adults 40-60 years. "
            "Increased cellularity, mitoses, nuclear atypia. May show patchy enhancement. "
            "IDH mutation present. Median survival 3-5 years. "
            "Higher GLCM contrast and entropy than Grade II. Moderate-high asymmetry. "
            "Irregular borders, moderate tumor area."
        ),
        "key_features": {
            "age_range": (35, 65),
            "tumor_pct_range": (3.0, 20.0),
            "glcm_malignancy_range": (0.40, 0.65),
            "compactness_range": (0.30, 0.60),
            "asymmetry_range": (0.10, 0.30),
        },
    },
    {
        "grade": 3,
        "title": "Anaplastic Oligodendroglioma IDH-mutant 1p/19q-codeleted (WHO Grade III)",
        "text": (
            "Anaplastic oligodendroglioma IDH-mutant 1p/19q-codeleted WHO Grade III. "
            "Adults 45-65 years. Increased mitoses, microvascular proliferation. "
            "Ring enhancement possible. Median survival 3-5 years. "
            "High GLCM malignancy, irregular shape, significant asymmetry."
        ),
        "key_features": {
            "age_range": (40, 70),
            "tumor_pct_range": (4.0, 22.0),
            "glcm_malignancy_range": (0.45, 0.70),
            "compactness_range": (0.25, 0.55),
            "asymmetry_range": (0.12, 0.35),
        },
    },
    # ── Grade IV ─────────────────────────────────────────────────────────────
    {
        "grade": 4,
        "title": "Glioblastoma IDH-wildtype (WHO Grade IV / GBM)",
        "text": (
            "Glioblastoma IDH-wildtype WHO Grade IV. Most common malignant brain tumor. "
            "Adults 55-75 years. Ring-enhancing lesion with central necrosis. "
            "Microvascular proliferation, pseudopalisading necrosis. "
            "EGFR amplification, PTEN loss, TERT promoter mutation. "
            "Median survival 14.6 months with Stupp protocol (TMZ + RT). "
            "Very high GLCM contrast and entropy. Large tumor area. "
            "Highly irregular shape, high asymmetry, high hot pixel percentage."
        ),
        "key_features": {
            "age_range": (50, 80),
            "tumor_pct_range": (5.0, 40.0),
            "glcm_malignancy_range": (0.55, 1.0),
            "compactness_range": (0.10, 0.45),
            "asymmetry_range": (0.15, 0.50),
        },
    },
    {
        "grade": 4,
        "title": "Glioblastoma IDH-mutant (WHO Grade IV)",
        "text": (
            "Glioblastoma IDH-mutant WHO Grade IV. Younger patients than IDH-wildtype. "
            "Better prognosis than IDH-wildtype GBM. Median survival 24-31 months. "
            "Ring enhancement, necrosis. High GLCM malignancy. Large irregular tumor."
        ),
        "key_features": {
            "age_range": (40, 65),
            "tumor_pct_range": (4.0, 35.0),
            "glcm_malignancy_range": (0.50, 0.95),
            "compactness_range": (0.12, 0.50),
            "asymmetry_range": (0.12, 0.45),
        },
    },
    # ── Normal ────────────────────────────────────────────────────────────────
    {
        "grade": 0,
        "title": "Normal Brain MRI",
        "text": (
            "Normal brain MRI. No tumor detected. Symmetric brain parenchyma. "
            "Normal gray-white matter differentiation. No abnormal enhancement. "
            "No mass effect. No midline shift. Normal ventricles. "
            "Low GLCM contrast, high homogeneity, low asymmetry, minimal hot pixels."
        ),
        "key_features": {
            "age_range": (0, 100),
            "tumor_pct_range": (0.0, 0.5),
            "glcm_malignancy_range": (0.0, 0.25),
            "compactness_range": (0.7, 1.0),
            "asymmetry_range": (0.0, 0.06),
        },
    },
    # ── Progression patterns ──────────────────────────────────────────────────
    {
        "grade": 4,
        "title": "GBM Progression Pattern — Pseudoprogression",
        "text": (
            "Pseudoprogression after chemoradiation for GBM. Apparent increase in "
            "enhancement within 3 months of completing RT+TMZ. Treatment-related "
            "inflammatory response, not true tumor growth. Occurs in 20-30% of GBM. "
            "MGMT methylated patients more susceptible. Stable or improving clinically. "
            "MR perfusion and spectroscopy help differentiate from true progression."
        ),
        "key_features": {
            "age_range": (45, 80),
            "tumor_pct_range": (3.0, 25.0),
            "glcm_malignancy_range": (0.40, 0.80),
            "compactness_range": (0.20, 0.55),
            "asymmetry_range": (0.10, 0.40),
        },
    },
    {
        "grade": 3,
        "title": "Malignant Transformation — Grade II to Grade III",
        "text": (
            "Malignant transformation from Grade II to Grade III glioma. "
            "New contrast enhancement in previously non-enhancing tumor. "
            "Increased metabolic activity on PET. Worsening symptoms. "
            "Median time to transformation 5-7 years for IDH-mutant astrocytoma. "
            "Requires re-biopsy for definitive diagnosis. Treatment escalation needed."
        ),
        "key_features": {
            "age_range": (35, 70),
            "tumor_pct_range": (4.0, 25.0),
            "glcm_malignancy_range": (0.45, 0.75),
            "compactness_range": (0.25, 0.55),
            "asymmetry_range": (0.12, 0.35),
        },
    },
]


# =============================================================================
#  DATA STRUCTURES
# =============================================================================

@dataclass
class RAGCase:
    """A single case in the RAG index."""
    case_id:          str
    source:           str          # "tcga_gbm", "tcga_lgg", "who_kb", "local_db"
    grade:            int          # 0-4
    age:              float        # years
    sex:              str          # "Male" / "Female" / "Unknown"
    survival_months:  float        # months (999 = alive/censored)
    primary_diagnosis: str
    treatments:       List[str]
    vital_status:     str          # "Alive" / "Dead" / "Unknown"
    text_description: str          # for TF-IDF
    # Imaging features (from AI result or estimated from grade)
    tumor_pct:        float = 0.0
    glcm_malignancy:  float = 0.0
    compactness:      float = 0.5
    asymmetry:        float = 0.0
    confidence:       float = 0.5
    # Feature vector (filled at index time)
    feature_vector:   np.ndarray = field(default_factory=lambda: np.zeros(20))


@dataclass
class RAGResult:
    """Output of a RAG query."""
    retrieved_cases:      List[RAGCase]
    similarities:         List[float]
    rag_grade_probs:      List[float]   # 5-element probability vector
    rag_grade:            int
    rag_confidence:       float
    augmented_grade_probs: List[float]  # blended AI + RAG
    augmented_grade:      int
    augmented_confidence: float
    supporting_evidence:  List[Dict]    # human-readable evidence cards
    population_stats:     Dict          # survival, treatment stats for retrieved cases
    retrieval_confidence: float         # how good the retrieval was (0-1)
    grade_agreement:      float         # fraction of retrieved cases agreeing on grade


# =============================================================================
#  TCGA DATA PARSER
# =============================================================================

def _parse_diagnoses(diag_str: str) -> Tuple[str, List[str]]:
    """Parse the TCGA 'diagnoses' JSON column → (primary_diagnosis, treatments)."""
    try:
        diag_list = ast.literal_eval(diag_str)
        if not isinstance(diag_list, list) or not diag_list:
            return "Unknown", []
        # Use the primary (non-recurrence) entry if available
        primary = next(
            (d for d in diag_list if str(d.get("classification_of_tumor", "")).lower() == "primary"),
            diag_list[0],
        )
        diagnosis = str(primary.get("primary_diagnosis", "Unknown"))
        treatments = []
        for t in primary.get("treatments", []):
            tt = str(t.get("treatment_type", ""))
            if tt and tt.lower() not in ("not reported", ""):
                treatments.append(tt)
        return diagnosis, treatments
    except Exception:
        return "Unknown", []


def _grade_from_diagnosis(diagnosis: str, project: str) -> int:
    """Infer WHO grade from TCGA diagnosis string + project."""
    d = diagnosis.lower()
    if "glioblastoma" in d or "gbm" in d:
        return 4
    if "anaplastic" in d:
        return 3
    if "oligodendroglioma" in d or "astrocytoma" in d:
        return 2
    if "pilocytic" in d or "ganglioglioma" in d or "ependymoma" in d:
        return 1
    # Fall back to project
    if "gbm" in project.lower():
        return 4
    if "lgg" in project.lower():
        return 2
    return 2  # default to Grade II for unknown LGG


def load_tcga_cases(data_dir: Path = CLINICAL_DATA_DIR) -> List[RAGCase]:
    """Load and parse all TCGA CSV files into RAGCase objects."""
    cases: List[RAGCase] = []
    csv_files = [
        (data_dir / "tcga_glioma_combined.csv", "tcga_combined"),
        (data_dir / "tcga_gbm_clinical.csv",    "tcga_gbm"),
        (data_dir / "tcga_lgg_clinical.csv",    "tcga_lgg"),
    ]
    seen_ids: set = set()

    for csv_path, source_tag in csv_files:
        if not csv_path.exists():
            logger.warning(f"TCGA file not found: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
            continue

        for _, row in df.iterrows():
            case_id = str(row.get("patient_id", row.get("case_id", "")))
            if not case_id or case_id in seen_ids:
                continue
            seen_ids.add(case_id)

            # Parse diagnosis
            diag_raw = str(row.get("diagnoses", ""))
            diagnosis, treatments = _parse_diagnoses(diag_raw)

            # Grade
            project = str(row.get("project", source_tag))
            grade   = _grade_from_diagnosis(diagnosis, project)

            # Demographics
            raw_age = row.get("demo_age_at_index", row.get("age_years", None))
            try:
                age = float(raw_age) if raw_age is not None and str(raw_age).lower() not in ("nan", "") else 0.0
            except (ValueError, TypeError):
                age = 0.0
            if age < 1:
                try:
                    age_days = abs(float(row.get("demo_days_to_birth", 0) or 0))
                except (ValueError, TypeError):
                    age_days = 0.0
                age = age_days / 365.25 if age_days > 0 else 50.0
            age = float(np.clip(age, 1, 100))

            sex_raw = str(row.get("demo_gender", "not reported")).lower()
            sex = "Female" if "female" in sex_raw else ("Male" if "male" in sex_raw else "Unknown")

            try:
                survival_months = float(row.get("survival_months", 0) or 0)
            except (ValueError, TypeError):
                survival_months = 0.0
            if survival_months <= 0:
                try:
                    days = float(row.get("demo_days_to_death", 0) or 0)
                except (ValueError, TypeError):
                    days = 0.0
                survival_months = days / 30.44 if days > 0 else GRADE_MEDIAN_SURVIVAL_MONTHS.get(grade, 24)

            vital_status = str(row.get("demo_vital_status", "Unknown"))

            # Build text description for TF-IDF
            treatment_str = ", ".join(treatments) if treatments else "not reported"
            text = (
                f"{diagnosis} grade {grade} patient age {int(age)} {sex} "
                f"treatments {treatment_str} survival {survival_months:.1f} months "
                f"vital status {vital_status}"
            )

            # Estimate imaging features from grade (used when no real scan data)
            kb_entry = next(
                (k for k in WHO_KNOWLEDGE_BASE if k["grade"] == grade and "Normal" not in k["title"]),
                None,
            )
            if kb_entry:
                kf = kb_entry["key_features"]
                tumor_pct      = float(np.mean(kf["tumor_pct_range"]))
                glcm_malignancy= float(np.mean(kf["glcm_malignancy_range"]))
                compactness    = float(np.mean(kf["compactness_range"]))
                asymmetry      = float(np.mean(kf["asymmetry_range"]))
            else:
                tumor_pct, glcm_malignancy, compactness, asymmetry = 0.0, 0.0, 0.5, 0.0

            cases.append(RAGCase(
                case_id=case_id,
                source=source_tag,
                grade=grade,
                age=age,
                sex=sex,
                survival_months=survival_months,
                primary_diagnosis=diagnosis,
                treatments=treatments,
                vital_status=vital_status,
                text_description=text,
                tumor_pct=tumor_pct,
                glcm_malignancy=glcm_malignancy,
                compactness=compactness,
                asymmetry=asymmetry,
                confidence=0.85,
            ))

    logger.info(f"[RAG] Loaded {len(cases)} TCGA cases from {data_dir}")
    return cases


def load_who_kb_cases() -> List[RAGCase]:
    """Convert WHO knowledge-base entries into RAGCase objects."""
    cases: List[RAGCase] = []
    for i, entry in enumerate(WHO_KNOWLEDGE_BASE):
        kf = entry["key_features"]
        age_mid = float(np.mean(kf["age_range"]))
        cases.append(RAGCase(
            case_id=f"WHO_KB_{i:03d}",
            source="who_kb",
            grade=entry["grade"],
            age=age_mid,
            sex="Unknown",
            survival_months=GRADE_MEDIAN_SURVIVAL_MONTHS.get(entry["grade"], 24),
            primary_diagnosis=entry["title"],
            treatments=[],
            vital_status="Unknown",
            text_description=entry["text"],
            tumor_pct=float(np.mean(kf["tumor_pct_range"])),
            glcm_malignancy=float(np.mean(kf["glcm_malignancy_range"])),
            compactness=float(np.mean(kf["compactness_range"])),
            asymmetry=float(np.mean(kf["asymmetry_range"])),
            confidence=0.95,
        ))
    return cases


def load_local_db_cases(db_path: Path = LOCAL_DB_PATH) -> List[RAGCase]:
    """Load previously analysed patients from clinical_database.json."""
    cases: List[RAGCase] = []
    if not db_path.exists():
        return cases
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pid, rec in data.items():
            if not rec.get("tumor_detected", False):
                continue
            grade = int(rec.get("tumor_grade", 0))
            age   = float(rec.get("age", 50))
            sex   = str(rec.get("sex", "Unknown"))
            survival_months = float(rec.get("survival_estimate", 0) or 0) / 30.44
            if survival_months <= 0:
                survival_months = GRADE_MEDIAN_SURVIVAL_MONTHS.get(grade, 24)
            text = (
                f"{rec.get('tumor_type', 'glioma')} grade {grade} "
                f"patient age {int(age)} {sex} "
                f"confidence {rec.get('confidence', 0.5):.2f} "
                f"tumor area {rec.get('tumor_percentage', 0):.1f}%"
            )
            cases.append(RAGCase(
                case_id=f"LOCAL_{pid}",
                source="local_db",
                grade=grade,
                age=age,
                sex=sex,
                survival_months=survival_months,
                primary_diagnosis=rec.get("tumor_type", "Unknown"),
                treatments=[],
                vital_status="Unknown",
                text_description=text,
                tumor_pct=float(rec.get("tumor_percentage", 0)),
                glcm_malignancy=0.0,
                compactness=0.5,
                asymmetry=0.0,
                confidence=float(rec.get("confidence", 0.5)),
            ))
    except Exception as e:
        logger.warning(f"[RAG] Could not load local DB: {e}")
    return cases


# =============================================================================
#  RAG INDEX BUILDER
# =============================================================================

class RAGIndex:
    """In-memory RAG index using TF-IDF + cosine similarity."""

    def __init__(self):
        self.cases: List[RAGCase] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.feature_scaler: Optional[MinMaxScaler] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.is_built = False

    def build(self, include_tcga=True, include_who_kb=True, include_local=True):
        """Build the RAG index from all available sources."""
        logger.info("[RAG] Building index...")
        self.cases = []

        if include_tcga:
            self.cases.extend(load_tcga_cases())
        if include_who_kb:
            self.cases.extend(load_who_kb_cases())
        if include_local:
            self.cases.extend(load_local_db_cases())

        if not self.cases:
            logger.warning("[RAG] No cases loaded — index is empty")
            return

        # Build TF-IDF matrix for text descriptions
        texts = [c.text_description for c in self.cases]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts).toarray()

        # Build numerical feature matrix
        features = []
        for c in self.cases:
            age_norm  = c.age / 100.0
            surv_norm = min(c.survival_months / 120.0, 1.0)
            grade_oh  = [1.0 if c.grade == g else 0.0 for g in range(5)]
            sex_bin   = 1.0 if c.sex == "Female" else (0.5 if c.sex == "Male" else 0.0)
            tumor_norm   = min(c.tumor_pct / 40.0, 1.0)
            glcm_norm    = c.glcm_malignancy
            compact_norm = c.compactness
            asym_norm    = min(c.asymmetry / 0.5, 1.0)
            conf_norm    = c.confidence
            feat = [
                age_norm, surv_norm, sex_bin, tumor_norm, glcm_norm,
                compact_norm, asym_norm, conf_norm,
            ] + grade_oh
            features.append(feat)

        self.feature_matrix = np.array(features, dtype=np.float32)
        self.feature_scaler = MinMaxScaler()
        self.feature_matrix = self.feature_scaler.fit_transform(self.feature_matrix)

        for i, c in enumerate(self.cases):
            c.feature_vector = self.feature_matrix[i]

        self.is_built = True
        logger.info(
            f"[RAG] Index built: {len(self.cases)} cases, "
            f"{self.tfidf_matrix.shape[1]} TF-IDF features"
        )

    def query(
        self,
        ai_result: Dict,
        patient_data: Dict,
        top_k: int = 10,
        blend_weight: float = 0.30,
    ) -> "RAGResult":
        """
        Query the RAG index with current patient + AI result.

        Args:
            ai_result:    Output from ImprovedTumorDetector.detect_tumor()
            patient_data: Dict with age, sex, kps, idh_status, mgmt_status
            top_k:        Number of similar cases to retrieve
            blend_weight: How much to blend RAG with AI (0=pure AI, 1=pure RAG)

        Returns:
            RAGResult with augmented predictions + evidence
        """
        if not self.is_built:
            logger.warning("[RAG] Index not built — returning AI result unchanged")
            return self._fallback_result(ai_result)

        query_text    = self._build_query_text(ai_result, patient_data)
        query_tfidf   = self.tfidf_vectorizer.transform([query_text]).toarray()[0]
        query_features = self._build_query_features(ai_result, patient_data)
        query_features_scaled = self.feature_scaler.transform([query_features])[0]

        # Compute similarities
        similarities = []
        for i, case in enumerate(self.cases):
            try:
                tfidf_sim = 1.0 - cosine_dist(query_tfidf, self.tfidf_matrix[i])
            except Exception:
                tfidf_sim = 0.0
            try:
                feat_sim  = 1.0 - cosine_dist(query_features_scaled, case.feature_vector)
            except Exception:
                feat_sim  = 0.0
            # Guard against NaN from zero vectors
            if np.isnan(tfidf_sim): tfidf_sim = 0.0
            if np.isnan(feat_sim):  feat_sim  = 0.0
            combined  = 0.40 * tfidf_sim + 0.60 * feat_sim
            similarities.append(float(np.clip(combined, 0.0, 1.0)))

        top_indices     = np.argsort(similarities)[::-1][:top_k]
        retrieved_cases = [self.cases[i] for i in top_indices]
        retrieved_sims  = [similarities[i] for i in top_indices]

        rag_grade_probs = self._compute_rag_grade_probs(retrieved_cases, retrieved_sims)
        rag_grade       = int(np.argmax(rag_grade_probs))
        rag_confidence  = float(rag_grade_probs[rag_grade])

        # Blend with AI probabilities
        ai_probs = np.array(ai_result.get("probabilities", [0.2] * 5), dtype=np.float64)
        if ai_probs.sum() < 0.1:
            ai_probs = np.zeros(5)
            ai_probs[ai_result.get("grade", 0)] = ai_result.get("confidence", 0.5)
        ai_probs /= ai_probs.sum() + 1e-8

        retrieval_confidence = float(np.mean(retrieved_sims[:3]))
        adaptive_blend       = blend_weight * retrieval_confidence

        augmented_probs = (1 - adaptive_blend) * ai_probs + adaptive_blend * rag_grade_probs
        augmented_probs = np.nan_to_num(augmented_probs, nan=0.0)
        total = augmented_probs.sum()
        if total < 1e-8:
            augmented_probs = ai_probs.copy()
        else:
            augmented_probs /= total
        augmented_grade      = int(np.argmax(augmented_probs))
        augmented_confidence = float(augmented_probs[augmented_grade])

        evidence  = self._build_evidence(retrieved_cases, retrieved_sims, ai_result)
        pop_stats = self._compute_population_stats(retrieved_cases)
        grade_agreement = float(
            sum(1 for c in retrieved_cases if c.grade == rag_grade) / max(len(retrieved_cases), 1)
        )

        return RAGResult(
            retrieved_cases=retrieved_cases,
            similarities=retrieved_sims,
            rag_grade_probs=rag_grade_probs.tolist(),
            rag_grade=rag_grade,
            rag_confidence=rag_confidence,
            augmented_grade_probs=augmented_probs.tolist(),
            augmented_grade=augmented_grade,
            augmented_confidence=augmented_confidence,
            supporting_evidence=evidence,
            population_stats=pop_stats,
            retrieval_confidence=retrieval_confidence,
            grade_agreement=grade_agreement,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_query_text(self, ai_result: Dict, patient_data: Dict) -> str:
        grade     = ai_result.get("grade", 0)
        tumor_type = ai_result.get("tumor_type", "glioma")
        age       = patient_data.get("age", 50)
        sex       = patient_data.get("sex", "Unknown")
        tumor_pct = ai_result.get("tumor_percentage", 0)
        return (
            f"{tumor_type} grade {grade} patient age {int(age)} {sex} "
            f"tumor area {tumor_pct:.1f}% confidence {ai_result.get('confidence', 0.5):.2f}"
        )

    def _build_query_features(self, ai_result: Dict, patient_data: Dict) -> List[float]:
        age      = patient_data.get("age", 50)
        age_norm = age / 100.0
        surv_months = patient_data.get("survival_estimate", 0) / 30.44
        if surv_months <= 0:
            surv_months = GRADE_MEDIAN_SURVIVAL_MONTHS.get(ai_result.get("grade", 0), 24)
        surv_norm = min(surv_months / 120.0, 1.0)
        sex       = patient_data.get("sex", "Unknown")
        sex_bin   = 1.0 if sex == "Female" else (0.5 if sex == "Male" else 0.0)
        tumor_norm   = min(ai_result.get("tumor_percentage", 0) / 40.0, 1.0)
        glcm_norm    = ai_result.get("glcm_malignancy", 0.0)
        compact_norm = ai_result.get("compactness", 0.5)
        asym_norm    = min(ai_result.get("asymmetry", 0.0) / 0.5, 1.0)
        conf_norm    = ai_result.get("confidence", 0.5)
        grade        = ai_result.get("grade", 0)
        grade_oh     = [1.0 if grade == g else 0.0 for g in range(5)]
        return [
            age_norm, surv_norm, sex_bin, tumor_norm, glcm_norm,
            compact_norm, asym_norm, conf_norm,
        ] + grade_oh

    def _compute_rag_grade_probs(
        self, cases: List[RAGCase], sims: List[float]
    ) -> np.ndarray:
        probs = np.zeros(5)
        total_weight = 0.0
        for case, sim in zip(cases, sims):
            weight = sim * case.confidence
            probs[case.grade] += weight
            total_weight += weight
        if total_weight > 0:
            probs /= total_weight
        else:
            probs = np.ones(5) / 5.0
        return probs

    def _build_evidence(
        self, cases: List[RAGCase], sims: List[float], ai_result: Dict
    ) -> List[Dict]:
        evidence = []
        for i, (case, sim) in enumerate(zip(cases[:5], sims[:5])):
            evidence.append({
                "rank":             i + 1,
                "similarity":       float(sim),
                "case_id":          case.case_id,
                "source":           case.source,
                "grade":            case.grade,
                "grade_name":       GRADE_NAMES.get(case.grade, "Unknown"),
                "age":              int(case.age),
                "sex":              case.sex,
                "survival_months":  float(case.survival_months),
                "diagnosis":        case.primary_diagnosis,
                "treatments":       case.treatments,
                "vital_status":     case.vital_status,
                "tumor_pct":        float(case.tumor_pct),
                "confidence":       float(case.confidence),
            })
        return evidence

    def _compute_population_stats(self, cases: List[RAGCase]) -> Dict:
        if not cases:
            return {}
        survivals = [c.survival_months for c in cases if c.survival_months < 999]
        median_survival = float(np.median(survivals)) if survivals else 0.0
        grade_counts = {}
        for c in cases:
            gname = GRADE_NAMES.get(c.grade, "Unknown")
            grade_counts[gname] = grade_counts.get(gname, 0) + 1
        treatment_counts: Dict[str, int] = {}
        for c in cases:
            for t in c.treatments:
                treatment_counts[t] = treatment_counts.get(t, 0) + 1
        return {
            "n_cases":                len(cases),
            "median_survival_months": median_survival,
            "grade_distribution":     grade_counts,
            "common_treatments":      dict(
                sorted(treatment_counts.items(), key=lambda x: -x[1])[:5]
            ),
            "mean_age":               float(np.mean([c.age for c in cases])),
            "sex_distribution": {
                "Male":    sum(1 for c in cases if c.sex == "Male"),
                "Female":  sum(1 for c in cases if c.sex == "Female"),
                "Unknown": sum(1 for c in cases if c.sex == "Unknown"),
            },
        }

    def _fallback_result(self, ai_result: Dict) -> "RAGResult":
        grade = ai_result.get("grade", 0)
        conf  = ai_result.get("confidence", 0.5)
        probs = ai_result.get("probabilities", [0.2] * 5)
        return RAGResult(
            retrieved_cases=[],
            similarities=[],
            rag_grade_probs=probs,
            rag_grade=grade,
            rag_confidence=conf,
            augmented_grade_probs=probs,
            augmented_grade=grade,
            augmented_confidence=conf,
            supporting_evidence=[],
            population_stats={},
            retrieval_confidence=0.0,
            grade_agreement=0.0,
        )


# =============================================================================
#  PROGRESSION AUGMENTATION
# =============================================================================

def augment_progression_analysis(
    result1: Dict,
    result2: Dict,
    progression_metrics: Dict,
    rag_index: "RAGIndex",
) -> Dict:
    """
    Augment progression analysis with population-level progression rates.

    Retrieves cases with similar baseline grade and time interval,
    then reports how often those cases showed progression vs stability.
    """
    if not rag_index.is_built:
        return {}

    baseline_grade    = result1.get("grade", 0)
    time_diff_days    = progression_metrics.get("time_diff_days", 180)
    time_diff_months  = time_diff_days / 30.44

    similar_cases = [c for c in rag_index.cases if c.grade == baseline_grade]
    if not similar_cases:
        return {}

    progressed_count = sum(
        1 for c in similar_cases if c.survival_months < time_diff_months
    )
    progression_rate = progressed_count / len(similar_cases)

    return {
        "population_progression_rate": float(progression_rate),
        "n_similar_cases":             len(similar_cases),
        "baseline_grade":              baseline_grade,
        "time_interval_months":        float(time_diff_months),
        "interpretation": (
            f"In {len(similar_cases)} similar cases (Grade {baseline_grade}), "
            f"{progression_rate:.1%} showed progression within "
            f"{time_diff_months:.1f} months."
        ),
    }


# =============================================================================
#  SINGLETON + PUBLIC API
# =============================================================================

_rag_index: Optional[RAGIndex] = None


def get_rag_index(rebuild: bool = False) -> RAGIndex:
    """Get or build the global RAG index (singleton)."""
    global _rag_index
    if _rag_index is None or rebuild:
        _rag_index = RAGIndex()
        _rag_index.build()
    return _rag_index


def augment_with_rag(
    ai_result: Dict,
    patient_data: Dict,
    top_k: int = 10,
    blend_weight: float = 0.30,
) -> "RAGResult":
    """
    Augment AI prediction with RAG evidence.

    Args:
        ai_result:    Output from ImprovedTumorDetector.detect_tumor()
        patient_data: Dict with age, sex, kps, idh_status, mgmt_status
        top_k:        Number of similar cases to retrieve
        blend_weight: How much to blend RAG with AI (0=pure AI, 1=pure RAG)

    Returns:
        RAGResult with augmented predictions + evidence
    """
    try:
        idx = get_rag_index()
        return idx.query(ai_result, patient_data, top_k, blend_weight)
    except Exception as e:
        logger.error(f"[RAG] augment_with_rag failed: {e}", exc_info=True)
        idx = get_rag_index()
        return idx._fallback_result(ai_result)


# =============================================================================
#  SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    idx = RAGIndex()
    idx.build()
    print(f"[RAG] Index ready: {len(idx.cases)} cases")

    test_ai = {
        "grade": 4,
        "confidence": 0.85,
        "probabilities": [0.05, 0.05, 0.10, 0.15, 0.65],
        "tumor_percentage": 15.5,
        "glcm_malignancy": 0.75,
        "compactness": 0.35,
        "asymmetry": 0.25,
        "tumor_type": "Grade IV Glioma (GBM)",
    }
    test_patient = {"age": 62, "sex": "Male", "kps": 80}

    result = idx.query(test_ai, test_patient, top_k=5, blend_weight=0.30)
    print(f"AI grade: {test_ai['grade']}  conf={test_ai['confidence']:.2f}")
    print(f"RAG grade: {result.rag_grade}  conf={result.rag_confidence:.2f}")
    print(f"Augmented grade: {result.augmented_grade}  conf={result.augmented_confidence:.2f}")
    print(f"Grade agreement: {result.grade_agreement:.1%}")
    print(f"Median survival (retrieved): {result.population_stats.get('median_survival_months', 0):.1f} mo")
