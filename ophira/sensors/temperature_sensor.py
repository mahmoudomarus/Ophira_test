"""Temperature Sensor Implementation"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from .base_sensor import BaseSensor, SensorData, SensorStatus


class TemperatureSensor(BaseSensor):
    """Temperature sensor implementation"""
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "temperature", config)
        self.base_temp = self.config.get("base_temperature", 36.8)
        
    async def connect(self) -> bool:
        try:
            self.status = SensorStatus.CONNECTING
            await asyncio.sleep(0.2)
            self.status = SensorStatus.CONNECTED if random.random() > 0.02 else SensorStatus.ERROR
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
            await asyncio.sleep(0.02)
            temp = self.base_temp + np.random.normal(0, 0.3)
            return {
                "temperature": temp,
                "ambient": temp - random.uniform(5, 15),
                "timestamp": datetime.now()
            }
        except Exception:
            return None
    
    async def calibrate(self) -> bool:
        try:
            self.status = SensorStatus.CALIBRATING
            await asyncio.sleep(1.0)
            self.calibration_data = {"offset": random.uniform(-0.1, 0.1)}
            self.status = SensorStatus.CONNECTED
            return True
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        if not raw_data:
            return None
        
        temp = raw_data["temperature"]
        is_critical = temp > 38.5 or temp < 36.0
        
        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=raw_data["timestamp"],
            value=round(temp, 1),
            unit="°C",
            confidence=0.95,
            metadata={"ambient_temp": round(raw_data["ambient"], 1)},
            quality_score=0.95,
            is_critical=is_critical
        ) 