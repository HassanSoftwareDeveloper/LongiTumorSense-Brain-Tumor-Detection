"""
audit_logger.py — HIPAA-Compliant Audit Trail for LongiTumorSense

Every access to patient data and every AI result must be logged.
Log entries are append-only (JSONL format) and include:
 - WHO accessed the data (user ID, role, institution, license)
 - WHAT was done (action type)
 - WHEN it happened (UTC timestamp)
 - WHAT the AI concluded (grade, confidence)
 - Anonymised patient reference (SHA-256 of patient_id)

DO NOT store full patient names or identifiable data in this log.
Log files must be stored in a secure, access-controlled location.
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
AUDIT_LOG_DIR = Path(__file__).parent / "logs"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit_trail.jsonl"
SYSTEM_VERSION = "3.0.0-research"  # NOT FDA cleared / NOT CE marked — research use only

AUDIT_LOG_DIR.mkdir(exist_ok=True)


# ── Action Type Constants ─────────────────────────────────────────────────────
class AuditAction:
    LOGIN            = "USER_LOGIN"
    LOGOUT           = "USER_LOGOUT"
    LOGIN_FAILED     = "USER_LOGIN_FAILED"
    IMAGE_UPLOAD     = "IMAGE_UPLOAD"
    AI_ANALYSIS      = "AI_ANALYSIS_RUN"
    RESULT_VIEWED    = "RESULT_VIEWED"
    REPORT_GENERATED = "REPORT_GENERATED"
    REPORT_EXPORTED  = "REPORT_EXPORTED"
    PROGRESSION_RUN  = "PROGRESSION_ANALYSIS_RUN"
    RESULT_CONFIRMED = "RESULT_RADIOLOGIST_CONFIRMED"  # When radiologist reviews


def _anonymise_patient_id(patient_id: str) -> str:
    """One-way hash of patient ID — identifies record without exposing PHI."""
    if not patient_id or patient_id.strip() == "":
        return "UNKNOWN"
    return hashlib.sha256(patient_id.strip().encode("utf-8")).hexdigest()[:16]


def write_audit_entry(
    action: str,
    user_info: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
    ai_result: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Write a single audit log entry.

    Parameters
    ----------
    action      : AuditAction constant (str)
    user_info   : dict with keys: username, role, institution, license_number
    patient_id  : raw patient ID string (will be anonymised before logging)
    ai_result   : dict with keys: grade, confidence, detected, num_models_used,
                  detection_source, tumor_type (all non-PHI)
    extra       : any additional non-PHI metadata
    """
    entry: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "system_version": SYSTEM_VERSION,
    }

    # User info (no passwords, no tokens)
    if user_info:
        entry["user"] = {
            "username": user_info.get("username", "unknown"),
            "role": user_info.get("role", "unknown"),
            "institution": user_info.get("institution", "unknown"),
            "license_number": user_info.get("license_number", "unknown"),
        }

    # Anonymised patient reference
    if patient_id:
        entry["patient_ref"] = _anonymise_patient_id(patient_id)

    # AI result — non-PHI summary only
    if ai_result:
        entry["ai_output"] = {
            "grade": ai_result.get("grade"),
            "confidence": round(float(ai_result.get("confidence", 0)), 4),
            "detected": ai_result.get("detected"),
            "tumor_type": ai_result.get("tumor_type"),
            "num_models_used": ai_result.get("num_models_used"),
            "detection_source": ai_result.get("detection_source"),
            "tumor_percentage": round(float(ai_result.get("tumor_percentage", 0)), 4),
        }

    if extra:
        entry["extra"] = extra

    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception as e:
        logger.error(f"AUDIT LOG WRITE FAILURE: {e} — Entry was: {entry}")
        return False


def log_login(user_info: Dict, success: bool, reason: str = "") -> None:
    """Log a login attempt (success or failure)."""
    action = AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED
    write_audit_entry(
        action=action,
        user_info=user_info,
        extra={"success": success, "reason": reason if not success else ""},
    )


def log_ai_analysis(
    user_info: Dict,
    patient_id: str,
    ai_result: Dict,
    image_format: str = "unknown",
) -> None:
    """Log every AI analysis run — the most critical audit event."""
    write_audit_entry(
        action=AuditAction.AI_ANALYSIS,
        user_info=user_info,
        patient_id=patient_id,
        ai_result=ai_result,
        extra={
            "image_format": image_format,
            "disclaimer": (
                "Result is AI decision-support only. "
                "Requires independent radiologist review before clinical use."
            ),
        },
    )


def log_report_export(user_info: Dict, patient_id: str, report_format: str) -> None:
    """Log report generation/export."""
    write_audit_entry(
        action=AuditAction.REPORT_EXPORTED,
        user_info=user_info,
        patient_id=patient_id,
        extra={"report_format": report_format},
    )


def log_logout(user_info: Dict) -> None:
    """Log user logout."""
    write_audit_entry(action=AuditAction.LOGOUT, user_info=user_info)


def get_audit_log_path() -> Path:
    """Return the path to the current audit log file."""
    return AUDIT_LOG_FILE


def read_recent_entries(n: int = 100) -> list:
    """
    Read the last N entries from the audit log.
    Returns list of parsed dicts. Use only for admin review interfaces.
    """
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-n:]
        return [json.loads(line) for line in recent]
    except Exception as e:
        logger.error(f"Failed to read audit log: {e}")
        return []