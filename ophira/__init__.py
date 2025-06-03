"""
Ophira AI - Advanced Medical AI System

A multi-agent AI system specialized in medical analysis with real-time sensor integration,
focusing on ophthalmology, cardiovascular health, and metabolic monitoring.
"""

__version__ = "0.1.0"
__author__ = "Ophira AI Team"
__description__ = "Advanced Medical AI System with Multi-Agent Architecture"

# Import core components
from .core.config import get_settings, validate_required_settings
from .core.logging import setup_logging

# Validate configuration on import
try:
    validate_required_settings()
except ValueError as e:
    import warnings
    warnings.warn(f"Configuration validation failed: {e}", UserWarning)

# Set up logging
setup_logging()

__all__ = [
    "get_settings",
    "validate_required_settings", 
    "setup_logging"
] 