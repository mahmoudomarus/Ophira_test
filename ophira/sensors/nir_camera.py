"""NIR Camera Sensor Implementation"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from .base_sensor import BaseSensor, SensorData, SensorStatus


class NIRCamera(BaseSensor):
    """Near-infrared camera sensor for retinal imaging"""
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "nir_camera", config)
        self.resolution = self.config.get("resolution", (1024, 1024))
        self.frame_rate = self.config.get("frame_rate", 30)
        
    async def connect(self) -> bool:
        try:
            self.status = SensorStatus.CONNECTING
            await asyncio.sleep(1.2)  # Camera initialization takes time
            self.status = SensorStatus.CONNECTED if random.random() > 0.05 else SensorStatus.ERROR
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
            await asyncio.sleep(0.1)  # Frame capture time
            
            # Simulate NIR image data
            width, height = self.resolution
            image_data = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            
            # Simulate retinal features
            vessel_density = random.uniform(0.15, 0.25)
            optic_disc_ratio = random.uniform(0.3, 0.5)
            
            return {
                "image_data": image_data,
                "vessel_density": vessel_density,
                "optic_disc_ratio": optic_disc_ratio,
                "image_quality": random.uniform(0.7, 1.0),
                "focus_score": random.uniform(0.8, 1.0),
                "timestamp": datetime.now()
            }
        except Exception:
            return None
    
    async def calibrate(self) -> bool:
        try:
            self.status = SensorStatus.CALIBRATING
            await asyncio.sleep(2.5)  # Camera calibration
            self.calibration_data = {
                "dark_frame": np.random.randint(0, 10, self.resolution).tolist(),
                "lens_distortion": {"k1": random.uniform(-0.1, 0.1)}
            }
            self.status = SensorStatus.CONNECTED
            return True
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        if not raw_data:
            return None
        
        vessel_density = raw_data["vessel_density"]
        image_quality = raw_data["image_quality"]
        
        # Detect potential issues
        is_critical = vessel_density < 0.15 or image_quality < 0.7
        
        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=raw_data["timestamp"],
            value={
                "vessel_density": round(vessel_density, 3),
                "optic_disc_ratio": round(raw_data["optic_disc_ratio"], 3),
                "image_quality": round(image_quality, 2),
                "focus_score": round(raw_data["focus_score"], 2)
            },
            unit="analysis",
            confidence=image_quality,
            metadata={
                "resolution": self.resolution,
                "image_size_kb": (self.resolution[0] * self.resolution[1]) // 1024
            },
            quality_score=image_quality,
            is_critical=is_critical
        ) 