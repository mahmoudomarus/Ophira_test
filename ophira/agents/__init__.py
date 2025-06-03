"""
Agent system for Ophira AI.

Contains the hierarchical agent architecture including the primary voice agent,
specialized medical agents, and coordination mechanisms.
"""

from .base import BaseAgent, AgentMessage, AgentRegistry
from .primary import OphiraVoiceAgent
from .medical_specialist import MedicalSpecialistAgent
from .sensor_control import SensorControlAgent
from .supervisory import SupervisoryAgent
from .report_organization import ReportOrganizationAgent

__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentRegistry",
    "OphiraVoiceAgent",
    "MedicalSpecialistAgent",
    "SensorControlAgent",
    "SupervisoryAgent",
    "ReportOrganizationAgent",
] 