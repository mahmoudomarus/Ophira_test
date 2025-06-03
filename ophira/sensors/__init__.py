"""
Sensor Integration Package for Ophira AI Medical System
Provides real-time health monitoring capabilities through various sensors
"""

from .base_sensor import BaseSensor, SensorData, SensorStatus
from .heart_rate_sensor import HeartRateSensor
from .ppg_sensor import PPGSensor
from .nir_camera import NIRCamera
from .temperature_sensor import TemperatureSensor
from .blood_pressure_sensor import BloodPressureSensor
from .sensor_manager import SensorManager

__all__ = [
    "BaseSensor", 
    "SensorData", 
    "SensorStatus",
    "HeartRateSensor",
    "PPGSensor", 
    "NIRCamera",
    "TemperatureSensor",
    "BloodPressureSensor",
    "SensorManager"
] 