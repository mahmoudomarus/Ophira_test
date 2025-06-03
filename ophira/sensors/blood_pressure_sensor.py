"""Blood Pressure Sensor Implementation"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from .base_sensor import BaseSensor, SensorData, SensorStatus


class BloodPressureSensor(BaseSensor):
    """Blood pressure sensor implementation"""
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "blood_pressure", config)
        self.base_systolic = self.config.get("base_systolic", 120)
        self.base_diastolic = self.config.get("base_diastolic", 80)
        
    async def connect(self) -> bool:
        try:
            self.status = SensorStatus.CONNECTING
            await asyncio.sleep(0.8)  # BP sensors take longer to connect
            self.status = SensorStatus.CONNECTED if random.random() > 0.04 else SensorStatus.ERROR
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
            await asyncio.sleep(0.5)  # BP measurement takes time
            
            systolic = self.base_systolic + np.random.normal(0, 15)
            diastolic = self.base_diastolic + np.random.normal(0, 10)
            
            # Ensure realistic relationship
            if diastolic >= systolic - 20:
                diastolic = systolic - 20 - random.uniform(5, 15)
            
            return {
                "systolic": max(80, systolic),
                "diastolic": max(50, diastolic),
                "pulse_pressure": systolic - diastolic,
                "timestamp": datetime.now()
            }
        except Exception:
            return None
    
    async def calibrate(self) -> bool:
        try:
            self.status = SensorStatus.CALIBRATING
            await asyncio.sleep(3.0)  # BP calibration takes time
            self.calibration_data = {
                "pressure_offset": random.uniform(-2, 2),
                "pump_efficiency": random.uniform(0.95, 1.05)
            }
            self.status = SensorStatus.CONNECTED
            return True
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        if not raw_data:
            return None
        
        systolic = round(raw_data["systolic"])
        diastolic = round(raw_data["diastolic"])
        pulse_pressure = round(raw_data["pulse_pressure"])
        
        is_critical = systolic > 140 or diastolic > 90 or systolic < 90 or diastolic < 60
        
        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=raw_data["timestamp"],
            value={
                "systolic": systolic,
                "diastolic": diastolic,
                "pulse_pressure": pulse_pressure
            },
            unit="mmHg",
            confidence=0.9,
            metadata={
                "measurement_duration": 0.5,
                "cuff_pressure_max": systolic + 40
            },
            quality_score=0.9,
            is_critical=is_critical
        ) 