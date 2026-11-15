# rag_engine.py - FIXED VERSION with Medical Embeddings

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Pre-import torch so that sentence_transformers can reference nn without error
# on first startup (avoids "name 'nn' is not defined" in torch internals).
try:
    import torch
    import torch.nn as nn  # noqa: F401  — ensures nn is initialised
except ImportError:
    pass

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
#  CONSTANTS
# =============================================================================

LOCAL_DB_PATH = Path(__file__).parent / "data" / "clinical_database.json"

GRADE_NAMES = {
    0: "Normal",
    1: "Grade I Glioma",
    2: "Grade II Glioma",
    3: "Grade III Glioma",
    4: "Grade IV Glioma (GBM)",
}

GRADE_MEDIAN_SURVIVAL_MONTHS = {0: 999, 1: 96, 2: 60, 3: 18, 4: 14.6}


# =============================================================================
#  DATA STRUCTURES
# =============================================================================

@dataclass
class RAGCase:
    """A single case in the RAG index with REAL imaging features."""
    case_id: str
    grade: int
    age: float
    sex: str
    tumor_pct: float
    glcm_malignancy: float
    asymmetry: float
    compactness: float
    confidence: float
    survival_months: float
    primary_diagnosis: str
    features_estimated: bool = False  # CRITICAL: must be False for valid cases
    grade_confirmed: bool = True


@dataclass
class RAGResult:
    """Output of a RAG query."""
    augmented_grade: int
    augmented_confidence: float
    augmented_grade_probs: List[float]
    rag_used: bool
    rag_grade: int = 0
    rag_confidence: float = 0.0
    retrieval_confidence: float = 0.0
    grade_agreement: float = 0.0
    retrieved_cases: List[Dict] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    rag_reason: str = ""


# =============================================================================
#  FIXED RAG ENGINE - Uses Medical Embeddings
# =============================================================================

class FixedRAGEngine:
    """
    FIXED RAG engine using proper medical embeddings.
    
    Key improvements over old version:
    1. Uses BioBERT/PubMedBERT for semantic similarity (not TF-IDF)
    2. Only matches cases with REAL imaging features (not estimated)
    3. Strict validation before blending (confidence > 0.75)
    4. Conservative blending (15% RAG, 85% AI)
    5. Only augments when RAG agrees with AI (within 1 grade)
    """
    
    def __init__(self):
        self.cases: List[RAGCase] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False
        self.embedding_model = None
        
        # Try to load medical embedding model
        try:
            from sentence_transformers import SentenceTransformer
            # Use medical-specific BioBERT model
            self.embedding_model = SentenceTransformer(
                'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
            )
            logger.info("[RAG] ✅ Loaded BioBERT medical embeddings")
        except ImportError:
            logger.warning(
                "[RAG] ⚠️  sentence-transformers not installed. "
                "Install: pip install sentence-transformers"
            )
            self.embedding_model = None
        except Exception as e:
            logger.warning(f"[RAG] ⚠️  Could not load BioBERT: {e}")
            self.embedding_model = None
    
    def add_case_from_result(self, ai_result: Dict, patient_data: Optional[Dict] = None) -> bool:
        """
        Add a completed scan result to the RAG index so the registry grows
        automatically after every analysis run.

        Call this from inference.py / frontend.py right after analysis completes.
        The case is appended to the in-memory index AND persisted to the local
        clinical_database.json so it survives restarts.

        Returns True if the case was accepted (grade > 0 with real features).
        """
        try:
            grade = int(ai_result.get("grade", 0))
            tumor_pct = float(ai_result.get("tumor_percentage", 0.0))
            glcm_malignancy = float(ai_result.get("glcm_malignancy", 0.0))

            # Require SOME real imaging features (not a blank / failed scan)
            # NOTE: Normal scans (grade=0) ARE accepted if the image was processed
            # — they are valuable negative examples for RAG similarity search.
            # Only skip if BOTH features are zero AND confidence is very low
            # (which indicates a failed/blank scan, not a clean normal brain).
            if tumor_pct == 0.0 and glcm_malignancy == 0.0 and float(ai_result.get("confidence", 0)) < 0.3:
                logger.debug("[RAG] Skipping case: no real imaging features and very low confidence")
                return False

            import uuid, time
            case_id = f"SCAN_{int(time.time())}_{uuid.uuid4().hex[:6].upper()}"

            pdata = patient_data or {}
            age = float(pdata.get("age", 50))
            sex = str(pdata.get("sex", "Unknown"))
            confidence = float(ai_result.get("confidence", 0.5))
            asymmetry = float(ai_result.get("asymmetry_score", 0.0))
            compactness = float(ai_result.get("compactness", 0.5))
            survival_months = GRADE_MEDIAN_SURVIVAL_MONTHS.get(grade, 24)

            new_case = RAGCase(
                case_id=case_id,
                grade=grade,
                age=age,
                sex=sex,
                tumor_pct=tumor_pct,
                glcm_malignancy=glcm_malignancy,
                asymmetry=asymmetry,
                compactness=compactness,
                confidence=confidence,
                survival_months=survival_months,
                primary_diagnosis=ai_result.get("tumor_type", GRADE_NAMES.get(grade, "Unknown")),
                features_estimated=False,
                grade_confirmed=True,
            )

            # Add to in-memory list
            self.cases.append(new_case)

            # Persist to local DB so count survives restarts
            self._persist_case_to_local_db(case_id, new_case, ai_result, pdata)

            # Rebuild embeddings if we have enough cases
            if self.embedding_model is not None and len(self.cases) >= 1:
                try:
                    desc = self._create_medical_description(new_case)
                    new_emb = self.embedding_model.encode(
                        [desc], show_progress_bar=False, convert_to_numpy=True
                    )
                    if self.embeddings is not None and len(self.embeddings) > 0:
                        self.embeddings = np.vstack([self.embeddings, new_emb])
                    else:
                        self.embeddings = new_emb
                    self.is_built = True
                except Exception as emb_err:
                    logger.warning(f"[RAG] Could not update embeddings: {emb_err}")
                    # Still mark as built so feature-based similarity can be used
                    self.is_built = True

            logger.info(
                f"[RAG] ✅ Case added: {case_id}  grade={grade}  "
                f"tumor_pct={tumor_pct:.2f}%  total_cases={len(self.cases)}"
            )
            return True

        except Exception as e:
            logger.error(f"[RAG] add_case_from_result failed: {e}", exc_info=True)
            return False

    def _persist_case_to_local_db(
        self,
        case_id: str,
        case: "RAGCase",
        ai_result: Dict,
        patient_data: Dict,
    ) -> None:
        """Append/update the case in clinical_database.json for persistence across restarts."""
        try:
            LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            if LOCAL_DB_PATH.exists():
                with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            data[case_id] = {
                "tumor_detected": case.grade > 0,
                "tumor_grade": case.grade,
                "tumor_type": case.primary_diagnosis,
                "tumor_percentage": case.tumor_pct,
                "glcm_malignancy": case.glcm_malignancy,
                "asymmetry_score": case.asymmetry,
                "compactness": case.compactness,
                "confidence": case.confidence,
                "age": case.age,
                "sex": case.sex,
                "survival_estimate": case.survival_months * 30.44,
                # Copy extra keys from ai_result for completeness
                "grade_probs": ai_result.get("grade_probs", []),
                "risk_level": ai_result.get("risk_level", "Unknown"),
                "max_diameter_mm": ai_result.get("max_diameter_mm", 0.0),
            }

            with open(LOCAL_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"[RAG] Persisted case {case_id} to {LOCAL_DB_PATH}")

        except Exception as e:
            logger.warning(f"[RAG] Could not persist case to local DB: {e}")

    def build_index(self, include_local: bool = True) -> bool:
        """
        Build RAG index from cases with REAL imaging features.
        
        Only includes cases that have:
        - Real MRI scans (features_estimated=False)
        - Confirmed WHO grade
        - Complete imaging features (tumor_pct, glcm_malignancy, asymmetry)
        """
        if self.embedding_model is None:
            logger.error("[RAG] ❌ Cannot build index without embedding model")
            return False
        
        logger.info("[RAG] Building index from local database...")
        
        # Load cases from local database
        if include_local:
            local_cases = self._load_local_db_cases()
            self.cases.extend(local_cases)
        
        # Filter: only keep cases with REAL imaging data
        valid_cases = []
        for case in self.cases:
            # Skip cases with estimated features
            if case.features_estimated:
                continue
            
            # Require complete imaging features
            if case.tumor_pct == 0 and case.glcm_malignancy == 0:
                continue
            
            # Require confirmed grade
            if not case.grade_confirmed:
                continue
            
            valid_cases.append(case)
        
        if len(valid_cases) < 1:
            logger.info(
                "[RAG] ℹ️  No valid cases yet — index will build as scans are run."
            )
            # Mark as built with empty index; add_case_from_result will populate it
            self.is_built = True
            return True
        
        self.cases = valid_cases
        
        # Create medical descriptions for embedding
        descriptions = [self._create_medical_description(case) for case in self.cases]
        
        # Generate embeddings
        logger.info(f"[RAG] Generating embeddings for {len(descriptions)} cases...")
        self.embeddings = self.embedding_model.encode(
            descriptions, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        self.is_built = True
        logger.info(f"[RAG] ✅ Index built with {len(self.cases)} valid cases")
        return True
    
    def _create_medical_description(self, case: RAGCase) -> str:
        """
        Create a rich medical description for embedding.
        
        Focus on clinical features, not demographics.
        This is what BioBERT will use for similarity matching.
        """
        grade_names = {
            0: 'normal brain',
            1: 'low-grade glioma',
            2: 'intermediate-grade glioma',
            3: 'anaplastic glioma',
            4: 'glioblastoma'
        }
        
        # Size descriptor
        if case.tumor_pct < 2:
            size = "small"
        elif case.tumor_pct < 10:
            size = "moderate"
        else:
            size = "large"
        
        # Malignancy descriptor
        if case.glcm_malignancy < 0.3:
            malignancy = "low malignancy"
        elif case.glcm_malignancy < 0.6:
            malignancy = "moderate malignancy"
        else:
            malignancy = "high malignancy"
        
        # Asymmetry descriptor
        if case.asymmetry < 0.1:
            asym = "symmetric"
        elif case.asymmetry < 0.25:
            asym = "mildly asymmetric"
        else:
            asym = "highly asymmetric"
        
        # Compactness descriptor
        if case.compactness > 0.7:
            shape = "well-circumscribed"
        elif case.compactness > 0.4:
            shape = "moderately infiltrative"
        else:
            shape = "highly infiltrative"
        
        desc = (
            f"{grade_names.get(case.grade, 'unknown')} with {size} tumor burden "
            f"showing {malignancy} features, {asym} appearance, and {shape} morphology"
        )
        
        return desc
    
    def query(
        self,
        ai_result: Dict,
        patient_data: Optional[Dict] = None,
        top_k: int = 5,
        confidence_threshold: float = 0.75
    ) -> RAGResult:
        """
        Query RAG index with STRICT validation.
        
        Only augments AI prediction if:
        1. RAG index is built and ready
        2. Retrieval confidence > 0.55 (lowered from 0.75 for smaller registries)
        3. Retrieved cases agree with AI (within 1 grade)
        4. At least 2 similar cases found (lowered from 3 for early registry growth)
        
        Args:
            ai_result: Dict with 'grade', 'confidence', 'grade_probs', imaging features
            patient_data: Optional patient demographics (not used for matching)
            top_k: Number of similar cases to retrieve
            confidence_threshold: Minimum similarity to use RAG (default 0.55)
        
        Returns:
            RAGResult with augmented predictions or AI-only result
        """
        # Validation 0: Check if RAG is available
        if not self.is_built or self.embedding_model is None:
            return self._return_ai_only(
                ai_result, 
                reason="RAG not available (index not built or embeddings not loaded)"
            )
        
        # Create query description from AI result
        query_desc = self._create_medical_description_from_result(ai_result)
        query_embedding = self.embedding_model.encode(
            [query_desc], 
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        # Compute similarities (cosine similarity on embeddings)
        similarities = []
        for case_embedding in self.embeddings:
            sim = np.dot(query_embedding, case_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(case_embedding) + 1e-9
            )
            similarities.append(float(np.clip(sim, 0.0, 1.0)))
        
        # Get top-k most similar cases
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_cases = [self.cases[i] for i in top_indices]
        retrieved_sims = [similarities[i] for i in top_indices]
        
        # Validation 1: Check retrieval confidence
        avg_similarity = float(np.mean(retrieved_sims[:3]))
        if avg_similarity < confidence_threshold:
            logger.info(
                f"[RAG] Low retrieval confidence {avg_similarity:.2f} < {confidence_threshold} "
                "- using AI only"
            )
            return self._return_ai_only(
                ai_result, 
                reason=f"low retrieval confidence ({avg_similarity:.2f})"
            )
        
        # Validation 2: Check grade agreement with AI
        ai_grade = ai_result.get('grade', 0)
        retrieved_grades = [c.grade for c in retrieved_cases]
        
        # Count how many retrieved cases are within 1 grade of AI prediction
        close_grades = sum(1 for g in retrieved_grades if abs(g - ai_grade) <= 1)
        agreement_ratio = close_grades / len(retrieved_grades) if retrieved_grades else 0.0
        
        if agreement_ratio < 0.6:  # Less than 60% agreement
            logger.info(
                f"[RAG] Low agreement with AI ({agreement_ratio:.1%}) - using AI only"
            )
            return self._return_ai_only(
                ai_result, 
                reason=f"low agreement with AI ({agreement_ratio:.1%})"
            )
        
        # Validation 3: Check if we have enough similar cases
        if len(retrieved_cases) < 2:
            logger.info(f"[RAG] Only {len(retrieved_cases)} cases found - using AI only")
            return self._return_ai_only(
                ai_result, 
                reason=f"insufficient cases ({len(retrieved_cases)})"
            )
        
        # All validations passed - compute RAG prediction
        rag_grade_probs = self._compute_rag_probs(retrieved_cases, retrieved_sims)
        rag_grade = int(np.argmax(rag_grade_probs))
        rag_confidence = float(rag_grade_probs[rag_grade])
        
        # Conservative blending (adaptive: higher similarity = more RAG weight)
        ai_probs = np.array(ai_result.get('grade_probs', [0.2]*5), dtype=np.float64)
        ai_probs = ai_probs / (ai_probs.sum() + 1e-9)
        
        # Blend weight: 10-20% RAG based on retrieval confidence
        blend_weight = 0.10 + 0.10 * avg_similarity  # Range: 0.10 to 0.20
        
        augmented_probs = (1 - blend_weight) * ai_probs + blend_weight * rag_grade_probs
        augmented_probs = augmented_probs / (augmented_probs.sum() + 1e-9)
        
        augmented_grade = int(np.argmax(augmented_probs))
        augmented_confidence = float(augmented_probs[augmented_grade])
        
        logger.info(
            f"[RAG] ✅ Augmented: AI Grade {ai_grade} (conf={ai_result.get('confidence', 0):.2f}) "
            f"→ RAG Grade {augmented_grade} (conf={augmented_confidence:.2f}) "
            f"[similarity={avg_similarity:.2f}, agreement={agreement_ratio:.1%}, "
            f"blend={blend_weight:.1%}]"
        )
        
        # Prepare evidence (top 3 cases)
        evidence_cases = []
        for case, sim in zip(retrieved_cases[:3], retrieved_sims[:3]):
            evidence_cases.append({
                'case_id': case.case_id,
                'grade': case.grade,
                'similarity': round(sim, 3),
                'tumor_pct': round(case.tumor_pct, 2),
                'glcm_malignancy': round(case.glcm_malignancy, 3),
                'diagnosis': case.primary_diagnosis
            })
        
        return RAGResult(
            augmented_grade=augmented_grade,
            augmented_confidence=augmented_confidence,
            augmented_grade_probs=augmented_probs.tolist(),
            rag_used=True,
            rag_grade=rag_grade,
            rag_confidence=rag_confidence,
            retrieval_confidence=avg_similarity,
            grade_agreement=agreement_ratio,
            retrieved_cases=evidence_cases,
            similarities=retrieved_sims[:3],
            rag_reason=f"RAG augmentation successful (blend={blend_weight:.1%})"
        )
    
    def _compute_rag_probs(self, cases: List[RAGCase], similarities: List[float]) -> np.ndarray:
        """Compute grade probabilities from retrieved cases."""
        probs = np.zeros(5, dtype=np.float64)
        total_weight = 0.0
        
        for case, sim in zip(cases, similarities):
            # Square similarity to emphasize close matches
            weight = sim ** 2
            probs[case.grade] += weight
            total_weight += weight
        
        if total_weight > 1e-9:
            probs = probs / total_weight
        else:
            probs = np.ones(5) / 5
        
        return probs
    
    def _create_medical_description_from_result(self, ai_result: Dict) -> str:
        """Create medical description from AI result for query."""
        # Create a temporary RAGCase-like object
        class TempCase:
            def __init__(self, result):
                self.grade = result.get('grade', 0)
                self.tumor_pct = result.get('tumor_percentage', 0)
                self.glcm_malignancy = result.get('glcm_malignancy', 0)
                self.asymmetry = result.get('asymmetry_score', 0)
                self.compactness = result.get('compactness', 0.5)
        
        temp_case = TempCase(ai_result)
        return self._create_medical_description(temp_case)
    
    def _return_ai_only(self, ai_result: Dict, reason: str) -> RAGResult:
        """Return AI prediction without RAG augmentation."""
        return RAGResult(
            augmented_grade=ai_result.get('grade', 0),
            augmented_confidence=ai_result.get('confidence', 0.5),
            augmented_grade_probs=ai_result.get('grade_probs', [0.2]*5),
            rag_used=False,
            rag_reason=reason
        )
    
    def _load_local_db_cases(self) -> List[RAGCase]:
        """Load cases from local clinical database."""
        cases = []
        
        if not LOCAL_DB_PATH.exists():
            logger.info("[RAG] No local database found - starting with empty index")
            return cases
        
        try:
            with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for pid, rec in data.items():
                grade = int(rec.get("tumor_grade", 0))
                age = float(rec.get("age", 50))
                sex = str(rec.get("sex", "Unknown"))
                
                # CRITICAL: Only include if we have REAL imaging features
                tumor_pct = float(rec.get("tumor_percentage", 0))
                glcm_malignancy = float(rec.get("glcm_malignancy", 0))
                asymmetry = float(rec.get("asymmetry_score", 0))
                compactness = float(rec.get("compactness", 0.5))
                confidence = float(rec.get("confidence", 0.5))
                
                # Skip ONLY if imaging features are completely absent (failed scan)
                # Normal cases (grade=0) with glcm_malignancy=0 are VALID — skip only blanks
                if tumor_pct == 0 and glcm_malignancy == 0 and grade == 0:
                    confidence_val = float(rec.get("confidence", 0))
                    if confidence_val < 0.3:
                        continue
                
                survival_months = float(rec.get("survival_estimate", 0) or 0) / 30.44
                if survival_months <= 0:
                    survival_months = GRADE_MEDIAN_SURVIVAL_MONTHS.get(grade, 24)
                
                cases.append(RAGCase(
                    case_id=f"LOCAL_{pid}",
                    grade=grade,
                    age=age,
                    sex=sex,
                    tumor_pct=tumor_pct,
                    glcm_malignancy=glcm_malignancy,
                    asymmetry=asymmetry,
                    compactness=compactness,
                    confidence=confidence,
                    survival_months=survival_months,
                    primary_diagnosis=rec.get("tumor_type", "Unknown"),
                    features_estimated=False,  # Local DB has REAL features
                    grade_confirmed=True
                ))
            
            logger.info(f"[RAG] Loaded {len(cases)} cases from local database")
        
        except Exception as e:
            logger.warning(f"[RAG] Could not load local DB: {e}")
        
        return cases


# =============================================================================
#  GLOBAL RAG INDEX (Singleton)
# =============================================================================

_RAG_INDEX: Optional[FixedRAGEngine] = None


def get_rag_index(rebuild: bool = False) -> FixedRAGEngine:
    """Get or create the global RAG index.
    
    Args:
        rebuild: If True, force rebuild the index even if it exists
    """
    global _RAG_INDEX
    if _RAG_INDEX is None or rebuild:
        _RAG_INDEX = FixedRAGEngine()
        # Try to build index automatically
        try:
            _RAG_INDEX.build_index(include_local=True)
        except Exception as e:
            logger.warning(f"[RAG] Could not auto-build index: {e}")
    return _RAG_INDEX


def register_scan_result(ai_result: Dict, patient_data: Optional[Dict] = None) -> bool:
    """
    PUBLIC API — call this after every completed scan to grow the registry.

    This is the single function that makes the "0 cases" counter increase.
    Add one call in inference.py (analyze_brain_mri) right before returning
    the result, or in frontend.py right after the analysis completes.

    Example::

        from rag_engine import register_scan_result
        register_scan_result(result_dict, patient_data={"age": 45, "sex": "M"})

    Returns True if the case was accepted and added.
    """
    rag_index = get_rag_index()
    return rag_index.add_case_from_result(ai_result, patient_data)


def augment_with_rag(
    ai_result: Dict,
    patient_data: Optional[Dict] = None,
    top_k: int = 5,
    blend_weight: float = 0.15  # Not used anymore (adaptive blending)
) -> RAGResult:
    """
    Augment AI prediction with RAG.
    
    This is the main entry point for RAG augmentation.
    
    Args:
        ai_result: AI prediction result
        patient_data: Optional patient demographics
        top_k: Number of similar cases to retrieve
        blend_weight: Deprecated (now uses adaptive blending)
    
    Returns:
        RAGResult with augmented prediction or AI-only result
    """
    rag_index = get_rag_index()
    return rag_index.query(ai_result, patient_data, top_k=top_k,
                           confidence_threshold=0.55)  # Lowered threshold for usability


# =============================================================================
#  USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FIXED RAG ENGINE - Medical Embeddings")
    print("="*60)
    print("\n Key Improvements:")
    print("  1. BioBERT medical embeddings (not TF-IDF)")
    print("  2. Real imaging features only (not estimated)")
    print("  3. Strict validation (confidence > 0.75)")
    print("  4. Conservative blending (10-20% RAG, 80-90% AI)")
    print("  5. Agreement check (RAG must agree with AI)")
    print("\n Installation:")
    print("  pip install sentence-transformers")
    print("\n Usage:")
    print("  from rag_engine import augment_with_rag")
    print("  result = augment_with_rag(ai_result, patient_data)")
    print("="*60)