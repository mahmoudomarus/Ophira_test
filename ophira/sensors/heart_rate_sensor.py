"""
Heart Rate Sensor Implementation for Ophira AI Medical System
Simulates a realistic heart rate sensor with variable data
"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from .base_sensor import BaseSensor, SensorData, SensorStatus


class HeartRateSensor(BaseSensor):
    """
    Heart Rate Sensor implementation
    Simulates realistic heart rate monitoring with variability
    """
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "heart_rate", config)
        self.base_hr = self.config.get("base_heart_rate", 70)
        self.variance = self.config.get("variance", 10)
        self.activity_multiplier = 1.0
        self.stress_level = 0.0
        self.measurement_noise = self.config.get("measurement_noise", 2.0)
        
    async def connect(self) -> bool:
        """Simulate connection to heart rate sensor"""
        try:
            self.status = SensorStatus.CONNECTING
            await asyncio.sleep(0.5)  # Simulate connection time
            
            # Simulate successful connection
            if random.random() > 0.05:  # 95% success rate
                self.status = SensorStatus.CONNECTED
                return True
            else:
                self.status = SensorStatus.ERROR
                return False
                
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from heart rate sensor"""
        self.status = SensorStatus.DISCONNECTED
        return True
    
    async def read_raw_data(self) -> Any:
        """Read raw heart rate data"""
        if self.status != SensorStatus.CONNECTED:
            return None
        
        try:
            # Simulate sensor read delay
            await asyncio.sleep(0.1)
            
            # Generate realistic heart rate with variability
            current_time = datetime.now()
            time_factor = np.sin(current_time.second / 60 * 2 * np.pi) * 0.1  # Natural variation
            
            # Add activity and stress influences
            activity_effect = self.activity_multiplier - 1.0
            stress_effect = self.stress_level * 20
            
            # Calculate heart rate
            hr = (self.base_hr + 
                  np.random.normal(0, self.variance) +
                  time_factor * 5 +
                  activity_effect * 30 +
                  stress_effect +
                  np.random.normal(0, self.measurement_noise))
            
            # Ensure realistic bounds
            hr = max(40, min(200, hr))
            
            return {
                "bpm": round(hr, 1),
                "raw_signal": np.random.normal(hr, 2, 10).tolist(),  # Simulate raw PPG signal
                "quality": random.uniform(0.7, 1.0),
                "timestamp": current_time
            }
            
        except Exception as e:
            print(f"Heart rate sensor read error: {e}")
            return None
    
    async def calibrate(self) -> bool:
        """Calibrate heart rate sensor"""
        try:
            self.status = SensorStatus.CALIBRATING
            await asyncio.sleep(2.0)  # Simulate calibration time
            
            # Store calibration data
            self.calibration_data = {
                "baseline_offset": random.uniform(-2, 2),
                "sensitivity": random.uniform(0.95, 1.05),
                "calibrated_at": datetime.now().isoformat()
            }
            
            self.status = SensorStatus.CONNECTED
            return True
            
        except Exception:
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        """Process raw heart rate data into standardized format"""
        if not raw_data:
            return None
        
        bpm = raw_data["bpm"]
        quality = raw_data["quality"]
        timestamp = raw_data["timestamp"]
        
        # Apply calibration if available
        if self.calibration_data:
            offset = self.calibration_data.get("baseline_offset", 0)
            sensitivity = self.calibration_data.get("sensitivity", 1.0)
            bpm = (bpm + offset) * sensitivity
        
        # Determine if reading is critical
        is_critical = bpm > 100 or bpm < 60
        
        # Calculate confidence based on quality and bounds
        confidence = quality
        if bpm < 40 or bpm > 200:
            confidence *= 0.5
        
        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=timestamp,
            value=round(bpm, 1),
            unit="bpm",
            confidence=confidence,
            metadata={
                "raw_signal_length": len(raw_data["raw_signal"]),
                "signal_quality": quality,
                "calibration_applied": bool(self.calibration_data)
            },
            quality_score=quality,
            is_critical=is_critical
        )
    
    def set_activity_level(self, level: float):
        """Set activity level (1.0 = rest, 2.0 = moderate, 3.0 = intense)"""
        self.activity_multiplier = max(1.0, min(3.0, level))
    
    def set_stress_level(self, level: float):
        """Set stress level (0.0 = calm, 1.0 = high stress)"""
        self.stress_level = max(0.0, min(1.0, level)) 