"""
Base Sensor Interface for Ophira AI Medical System
Provides foundation for all sensor implementations
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

# Get logger
logger = logging.getLogger(__name__)

class SensorStatus(Enum):
    """Sensor operational status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READING = "reading"
    ERROR = "error"
    CALIBRATING = "calibrating"


@dataclass
class SensorData:
    """Standardized sensor data structure"""
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    value: Union[float, int, np.ndarray, Dict[str, Any]]
    unit: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    quality_score: float = 1.0
    is_critical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value if not isinstance(self.value, np.ndarray) else self.value.tolist(),
            "unit": self.unit,
            "confidence": self.confidence,
            "metadata": self.metadata or {},
            "quality_score": self.quality_score,
            "is_critical": self.is_critical
        }


class BaseSensor(ABC):
    """Abstract base class for all sensors"""
    
    def __init__(self, sensor_id: str, sensor_type: str, config: Optional[Dict[str, Any]] = None):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config or {}
        self.status = SensorStatus.DISCONNECTED
        self.last_reading = None
        self.error_count = 0
        self.calibration_data = {}
        self._reading_task = None
        self._data_buffer: List[SensorData] = []
        self.max_buffer_size = self.config.get("max_buffer_size", 1000)
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{sensor_type}.{sensor_id}")
        self.is_connected = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the sensor hardware"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the sensor hardware"""
        pass
    
    @abstractmethod
    async def read_raw_data(self) -> Any:
        """Read raw data from sensor"""
        pass
    
    @abstractmethod
    async def calibrate(self) -> bool:
        """Calibrate the sensor"""
        pass
    
    @abstractmethod
    def process_raw_data(self, raw_data: Any) -> SensorData:
        """Process raw sensor data into standardized format"""
        pass
    
    async def start_continuous_reading(self, interval: float = 1.0):
        """Start continuous sensor reading"""
        if self._reading_task and not self._reading_task.done():
            return
            
        self._reading_task = asyncio.create_task(
            self._continuous_reading_loop(interval)
        )
    
    async def stop_continuous_reading(self):
        """Stop continuous sensor reading"""
        if self._reading_task:
            self._reading_task.cancel()
            try:
                await self._reading_task
            except asyncio.CancelledError:
                pass
    
    async def _continuous_reading_loop(self, interval: float):
        """Internal continuous reading loop"""
        while True:
            try:
                if self.status == SensorStatus.CONNECTED:
                    await self.read_single()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                self.status = SensorStatus.ERROR
                print(f"Sensor {self.sensor_id} error: {e}")
                await asyncio.sleep(interval * 2)  # Back off on error
    
    async def read_single(self) -> Optional[SensorData]:
        """Read a single data point from sensor"""
        try:
            self.status = SensorStatus.READING
            raw_data = await self.read_raw_data()
            
            if raw_data is not None:
                sensor_data = self.process_raw_data(raw_data)
                self.last_reading = sensor_data
                self._add_to_buffer(sensor_data)
                self.status = SensorStatus.CONNECTED
                return sensor_data
            else:
                self.status = SensorStatus.ERROR
                return None
                
        except Exception as e:
            self.error_count += 1
            self.status = SensorStatus.ERROR
            print(f"Error reading from sensor {self.sensor_id}: {e}")
            return None
    
    def _add_to_buffer(self, data: SensorData):
        """Add data to internal buffer"""
        self._data_buffer.append(data)
        if len(self._data_buffer) > self.max_buffer_size:
            self._data_buffer.pop(0)
    
    def get_recent_data(self, count: int = 10) -> List[SensorData]:
        """Get recent sensor data"""
        return self._data_buffer[-count:]
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive sensor status"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "status": self.status.value,
            "error_count": self.error_count,
            "last_reading_time": self.last_reading.timestamp.isoformat() if self.last_reading else None,
            "buffer_size": len(self._data_buffer),
            "config": self.config
        }
    
    async def health_check(self) -> bool:
        """Perform health check on sensor"""
        try:
            if self.status == SensorStatus.DISCONNECTED:
                return await self.connect()
            elif self.status == SensorStatus.ERROR:
                # Try to reconnect
                await self.disconnect()
                return await self.connect()
            else:
                # Test reading
                data = await self.read_single()
                return data is not None
        except Exception:
            return False 