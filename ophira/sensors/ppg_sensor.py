"""
PPG Sensor Implementation for Ophira AI Medical System
Photoplethysmography sensor for blood volume and circulation analysis
"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from .base_sensor import BaseSensor, SensorData, SensorStatus


class PPGSensor(BaseSensor):
    """PPG Sensor for photoplethysmography measurements"""
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "ppg", config)
        self.sampling_rate = self.config.get("sampling_rate", 100)  # Hz
        self.led_wavelengths = self.config.get("led_wavelengths", [660, 940])  # nm
        
    async def connect(self) -> bool:
        try:
            self.status = SensorStatus.CONNECTING
            await asyncio.sleep(0.3)
            self.status = SensorStatus.CONNECTED if random.random() > 0.03 else SensorStatus.ERROR
            return self.status == SensorStatus.CONNECTED
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        self.status = SensorStatus.DISCONNECTED
        return True
    
    async def read_raw_data(self) -> Any:
        if self.status != SensorStatus.CONNECTED:
            return None
        
        try:
            await asyncio.sleep(0.05)
            
            # Generate PPG waveform
            t = np.linspace(0, 2, self.sampling_rate * 2)
            hr = random.uniform(60, 100)
            signal = np.sin(2 * np.pi * hr/60 * t) + np.random.normal(0, 0.1, len(t))
            
            return {
                "waveform": signal.tolist(),
                "red_signal": (signal + np.random.normal(0, 0.05, len(t))).tolist(),
                "ir_signal": (signal * 0.8 + np.random.normal(0, 0.05, len(t))).tolist(),
                "spo2": random.uniform(95, 100),
                "perfusion_index": random.uniform(1, 15),
                "timestamp": datetime.now()
            }
        except Exception:
            return None
    
    async def calibrate(self) -> bool:
        try:
            self.status = SensorStatus.CALIBRATING
            await asyncio.sleep(1.5)
            self.calibration_data = {"dark_current": random.uniform(0, 0.1)}
            self.status = SensorStatus.CONNECTED
            return True
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        if not raw_data:
            return None
        
        spo2 = raw_data["spo2"]
        perfusion = raw_data["perfusion_index"]
        
        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=raw_data["timestamp"],
            value={
                "spo2": round(spo2, 1),
                "perfusion_index": round(perfusion, 1),
                "waveform_quality": random.uniform(0.7, 1.0)
            },
            unit="mixed",
            confidence=0.9 if spo2 > 95 else 0.7,
            metadata={
                "sampling_rate": self.sampling_rate,
                "waveform_length": len(raw_data["waveform"])
            },
            quality_score=random.uniform(0.8, 1.0),
            is_critical=spo2 < 95
        ) 