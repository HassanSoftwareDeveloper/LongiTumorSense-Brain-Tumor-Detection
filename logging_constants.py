"""
Logging Constants Module
=========================
Centralized logging constants and utilities following software engineering best practices.

This module provides:
- Standardized log message prefixes
- Consistent logging levels
- Reusable logging utilities

Author: LongiTumorSense Team
Date: 2026-04-20
"""

from enum import Enum
from typing import Optional
import logging


class LogPrefix(str, Enum):
    """Standardized log message prefixes for consistent logging across the application."""
    
    # Status indicators
    SUCCESS = "[SUCCESS]"
    COMPLETE = "[COMPLETE]"
    WARNING = "[WARNING]"
    ERROR = "[ERROR]"
    INFO = "[INFO]"
    DEBUG = "[DEBUG]"
    
    # Process stages
    INIT = "[INIT]"
    PREP = "[PREP]"
    ANALYSIS = "[ANALYSIS]"
    CALIBRATION = "[CALIBRATION]"
    
    # Model-specific
    UNET = "[UNET]"
    EFFICIENTNET = "[ENET]"
    RESNET = "[RESNET]"
    DENSENET = "[DENSENET]"
    ENSEMBLE = "[ENSEMBLE]"
    
    # Medical/Clinical
    EMERGENCY = "[EMERGENCY]"
    MEDICATION = "[MEDICATION]"
    LAB = "[LAB]"
    REPORT = "[REPORT]"
    RECURRENCE = "[RECURRENCE]"
    DIAGNOSIS = "[DIAGNOSIS]"
    TREATMENT = "[TREATMENT]"
    
    # System operations
    LAUNCH = "[LAUNCH]"
    METRICS = "[METRICS]"
    ALERT = "[ALERT]"
    FILE = "[FILE]"
    SAVE = "[SAVE]"
    DELETE = "[DELETE]"
    SECURE = "[SECURE]"
    USER = "[USER]"


class LogLevel(str, Enum):
    """Standard logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def format_log_message(prefix: LogPrefix, message: str, details: Optional[str] = None) -> str:
    """
    Format a log message with consistent structure.
    
    Args:
        prefix: Log prefix from LogPrefix enum
        message: Main log message
        details: Optional additional details
        
    Returns:
        Formatted log message string
        
    Example:
        >>> format_log_message(LogPrefix.SUCCESS, "Model loaded", "EfficientNet-B4")
        '[SUCCESS] Model loaded - EfficientNet-B4'
    """
    if details:
        return f"{prefix.value} {message} - {details}"
    return f"{prefix.value} {message}"


def log_success(logger: logging.Logger, message: str, details: Optional[str] = None):
    """Log a success message."""
    logger.info(format_log_message(LogPrefix.SUCCESS, message, details))


def log_warning(logger: logging.Logger, message: str, details: Optional[str] = None):
    """Log a warning message."""
    logger.warning(format_log_message(LogPrefix.WARNING, message, details))


def log_error(logger: logging.Logger, message: str, details: Optional[str] = None):
    """Log an error message."""
    logger.error(format_log_message(LogPrefix.ERROR, message, details))


def log_info(logger: logging.Logger, message: str, details: Optional[str] = None):
    """Log an info message."""
    logger.info(format_log_message(LogPrefix.INFO, message, details))


def log_model_load(logger: logging.Logger, model_name: str, status: str = "loaded"):
    """
    Log model loading status.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        status: Status message (default: "loaded")
    """
    log_success(logger, f"{model_name} {status}")


def log_process_stage(logger: logging.Logger, stage: LogPrefix, message: str):
    """
    Log a processing stage.
    
    Args:
        logger: Logger instance
        stage: Processing stage prefix
        message: Stage description
    """
    logger.info(format_log_message(stage, message))


# Processing stage messages for inference pipeline
INFERENCE_STAGES = [
    (0.00, LogPrefix.INIT, "Initialising AI pipeline"),
    (0.08, LogPrefix.PREP, "Preprocessing MRI image"),
    (0.15, LogPrefix.UNET, "Running U-Net segmentation model (Dice ~0.85)"),
    (0.30, LogPrefix.ANALYSIS, "Analysing tumour mask & radiological features"),
    (0.45, LogPrefix.EFFICIENTNET, "Running EfficientNet-B4 (texture analysis)"),
    (0.58, LogPrefix.RESNET, "Running ResNet-50 (structural analysis)"),
    (0.70, LogPrefix.DENSENET, "Running DenseNet-121 (edge & border analysis)"),
    (0.82, LogPrefix.ENSEMBLE, "Computing ensemble weighted vote"),
    (0.90, LogPrefix.CALIBRATION, "Calibrating confidence scores"),
    (0.96, LogPrefix.COMPLETE, "Finalising diagnosis"),
]


def get_inference_stage_message(progress: float) -> str:
    """
    Get the appropriate inference stage message for a given progress value.
    
    Args:
        progress: Progress value between 0.0 and 1.0
        
    Returns:
        Formatted stage message
    """
    for i, (threshold, prefix, message) in enumerate(INFERENCE_STAGES):
        if progress <= threshold or i == len(INFERENCE_STAGES) - 1:
            return format_log_message(prefix, message)
    
    return format_log_message(LogPrefix.INFO, "Processing")


# Export all public symbols
__all__ = [
    'LogPrefix',
    'LogLevel',
    'format_log_message',
    'log_success',
    'log_warning',
    'log_error',
    'log_info',
    'log_model_load',
    'log_process_stage',
    'INFERENCE_STAGES',
    'get_inference_stage_message',
]