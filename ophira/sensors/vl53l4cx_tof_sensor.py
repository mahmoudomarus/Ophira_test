#!/usr/bin/env python3
"""
VL53L4CX Time of Flight Distance Sensor for Ophira AI Medical System

Interfaces with Adafruit VL53L4CX ToF sensor via Arduino serial communication
for precise distance measurements in medical applications.
"""

import asyncio
import serial
import serial.tools.list_ports
import time
import re
import statistics
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from ophira.sensors.base_sensor import BaseSensor, SensorData


@dataclass
class ToFData:
    """Time of Flight sensor specific data structure"""
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    distance_mm: float
    signal_rate_mcps: float
    ambient_rate_mcps: float
    range_status: int
    objects_found: int
    stream_count: int
    quality_score: float
    is_valid_reading: bool
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    is_critical: bool = False


class VL53L4CXToFSensor(BaseSensor):
    """
    VL53L4CX Time of Flight Distance sensor
    
    Communicates with Arduino running VL53L4CX sensor code via serial port
    Provides precise distance measurements for medical applications
    """
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "vl53l4cx_tof", config)
        
        # Serial communication settings
        self.port = config.get('port') if config else None
        self.baudrate = config.get('baudrate', 115200) if config else 115200
        self.timeout = config.get('timeout', 2.0) if config else 2.0
        
        # Serial connection
        self.serial_connection = None
        
        # Data parsing
        self.data_pattern = re.compile(
            r'VL53L4CX Satellite: Count=(\d+), #Objs=(\d+)\s+'
            r'status=(\d+), D=(\d+)mm, Signal=([\d.]+) Mcps, Ambient=([\d.]+) Mcps'
        )
        
        # Measurement tracking
        self.measurement_history = []
        self.max_history = 100
        self.last_stream_count = 0
        
        # Quality thresholds
        self.min_signal_rate = 0.1  # Minimum signal rate for valid reading
        self.max_ambient_rate = 5.0  # Maximum ambient rate for good conditions
        self.distance_variance_threshold = 50  # mm variance for stability check
        
        # Calibration data
        self.offset_calibration = 0.0
        self.crosstalk_calibration = 0.0
        
        self.logger.info(f"VL53L4CX ToF sensor initialized - Port: {self.port}")

    async def connect(self) -> bool:
        """Connect to Arduino via serial port"""
        try:
            # Auto-detect port if not specified
            if not self.port:
                self.port = self._auto_detect_arduino_port()
                if not self.port:
                    self.logger.error("No Arduino port detected")
                    return False
            
            self.logger.info(f"Connecting to VL53L4CX sensor on port {self.port}")
            
            # Establish serial connection
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Wait for Arduino to initialize
            await asyncio.sleep(3.0)
            
            # Clear any initial data
            self.serial_connection.flushInput()
            
            # Test communication by reading a few measurements
            test_readings = 0
            for _ in range(10):
                data = await self._read_serial_line()
                if data and self._parse_measurement(data):
                    test_readings += 1
                if test_readings >= 3:
                    break
                await asyncio.sleep(0.1)
            
            if test_readings < 3:
                self.logger.error("Failed to receive valid data from VL53L4CX sensor")
                return False
            
            self.is_connected = True
            self.logger.info("VL53L4CX ToF sensor connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to VL53L4CX sensor: {e}")
            return False

    def _auto_detect_arduino_port(self) -> Optional[str]:
        """Auto-detect Arduino port"""
        try:
            arduino_ports = []
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                # Look for common Arduino identifiers
                if any(keyword in port.description.lower() for keyword in 
                       ['arduino', 'ch340', 'ch341', 'ftdi', 'usb serial']):
                    arduino_ports.append(port.device)
                    self.logger.info(f"Found potential Arduino port: {port.device} - {port.description}")
                # Also check for ACM devices which are common for Arduino
                elif 'ACM' in port.device or 'USB' in port.description:
                    arduino_ports.append(port.device)
                    self.logger.info(f"Found potential Arduino port: {port.device} - {port.description}")
            
            # Manually add common Arduino ports if not found
            common_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
            for common_port in common_ports:
                if common_port not in arduino_ports:
                    try:
                        # Test if port exists and can be opened
                        test_serial = serial.Serial(common_port, 115200, timeout=0.1)
                        test_serial.close()
                        arduino_ports.append(common_port)
                        self.logger.info(f"Found working port: {common_port}")
                    except:
                        pass  # Port doesn't exist or can't be opened
            
            if arduino_ports:
                return arduino_ports[0]  # Return the first found
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting Arduino port: {e}")
            return None

    async def _read_serial_line(self) -> Optional[str]:
        """Read a line from serial connection"""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                return None
                
            # Read line with timeout
            line = self.serial_connection.readline()
            if line:
                return line.decode('utf-8', errors='ignore').strip()
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading serial data: {e}")
            return None

    def _parse_measurement(self, data_line: str) -> Optional[Dict[str, Any]]:
        """Parse measurement data from Arduino output"""
        try:
            match = self.data_pattern.search(data_line)
            if not match:
                return None
            
            stream_count = int(match.group(1))
            objects_found = int(match.group(2))
            range_status = int(match.group(3))
            distance_mm = int(match.group(4))
            signal_rate = float(match.group(5))
            ambient_rate = float(match.group(6))
            
            return {
                'stream_count': stream_count,
                'objects_found': objects_found,
                'range_status': range_status,
                'distance_mm': distance_mm,
                'signal_rate_mcps': signal_rate,
                'ambient_rate_mcps': ambient_rate,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing measurement: {e}")
            return None

    async def disconnect(self) -> bool:
        """Disconnect from serial port"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.is_connected = False
            self.logger.info("VL53L4CX ToF sensor disconnected")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from sensor: {e}")
            return False

    async def read_raw_data(self) -> Any:
        """Read raw measurement data from sensor"""
        if not self.is_connected or not self.serial_connection:
            self.logger.error("Sensor not connected")
            return None
            
        try:
            # Read multiple lines to find valid measurement
            for _ in range(10):  # Try up to 10 times
                line = await self._read_serial_line()
                if line:
                    parsed_data = self._parse_measurement(line)
                    if parsed_data:
                        return parsed_data
                await asyncio.sleep(0.01)
            
            self.logger.warning("No valid measurement received")
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading ToF data: {e}")
            return None

    def _assess_measurement_quality(self, measurement: Dict[str, Any]) -> float:
        """Assess the quality of a measurement (0.0 to 1.0)"""
        try:
            quality_factors = []
            
            # Range status (0 is good)
            if measurement['range_status'] == 0:
                quality_factors.append(1.0)
            elif measurement['range_status'] < 5:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.2)
            
            # Signal rate (higher is better)
            signal_rate = measurement['signal_rate_mcps']
            if signal_rate >= self.min_signal_rate:
                signal_quality = min(1.0, signal_rate / 2.0)  # Normalize to 2.0 as max
                quality_factors.append(signal_quality)
            else:
                quality_factors.append(0.1)
            
            # Ambient rate (lower is better)
            ambient_rate = measurement['ambient_rate_mcps']
            if ambient_rate <= self.max_ambient_rate:
                ambient_quality = 1.0 - (ambient_rate / (self.max_ambient_rate * 2))
                quality_factors.append(max(0.1, ambient_quality))
            else:
                quality_factors.append(0.1)
            
            # Distance stability (if we have history)
            if len(self.measurement_history) >= 5:
                recent_distances = [m['distance_mm'] for m in self.measurement_history[-5:]]
                variance = statistics.variance(recent_distances)
                if variance <= self.distance_variance_threshold:
                    stability_quality = 1.0 - min(1.0, variance / self.distance_variance_threshold)
                    quality_factors.append(stability_quality)
                else:
                    quality_factors.append(0.3)
            
            return statistics.mean(quality_factors)
            
        except Exception:
            return 0.5  # Default moderate quality

    def _is_valid_reading(self, measurement: Dict[str, Any]) -> bool:
        """Determine if a reading is valid for medical use"""
        try:
            # Check range status
            if measurement['range_status'] > 10:
                return False
            
            # Check signal strength
            if measurement['signal_rate_mcps'] < self.min_signal_rate:
                return False
            
            # Check distance range (1mm to 6000mm for VL53L4CX)
            distance = measurement['distance_mm']
            if distance < 1 or distance > 6000:
                return False
            
            # Check objects found
            if measurement['objects_found'] == 0:
                return False
            
            return True
            
        except Exception:
            return False

    async def calibrate(self) -> bool:
        """Calibrate the ToF sensor"""
        try:
            self.logger.info("Starting ToF sensor calibration...")
            
            # Collect baseline measurements for offset calibration
            # User should place target at known distance (e.g., 100mm)
            self.logger.info("Please place a target at exactly 100mm distance")
            await asyncio.sleep(5.0)  # Give time to position target
            
            calibration_measurements = []
            for i in range(20):
                raw_data = await self.read_raw_data()
                if raw_data and self._is_valid_reading(raw_data):
                    calibration_measurements.append(raw_data['distance_mm'])
                await asyncio.sleep(0.1)
            
            if len(calibration_measurements) < 10:
                self.logger.error("Insufficient calibration measurements")
                return False
            
            # Calculate offset calibration
            mean_distance = statistics.mean(calibration_measurements)
            self.offset_calibration = 100.0 - mean_distance  # Target is at 100mm
            
            self.logger.info(f"Calibration complete - Offset: {self.offset_calibration:.2f}mm")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False

    def process_raw_data(self, raw_data: Any) -> ToFData:
        """Process raw sensor data into structured format"""
        if not raw_data:
            return None
            
        try:
            # Apply calibrations
            calibrated_distance = raw_data['distance_mm'] + self.offset_calibration
            
            # Assess quality
            quality_score = self._assess_measurement_quality(raw_data)
            is_valid = self._is_valid_reading(raw_data)
            
            # Add to history
            self.measurement_history.append(raw_data)
            if len(self.measurement_history) > self.max_history:
                self.measurement_history.pop(0)
            
            # Create metadata
            metadata = {
                'raw_distance': raw_data['distance_mm'],
                'offset_calibration': self.offset_calibration,
                'crosstalk_calibration': self.crosstalk_calibration,
                'port': self.port,
                'baudrate': self.baudrate,
                'history_count': len(self.measurement_history)
            }
            
            return ToFData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=raw_data['timestamp'],
                distance_mm=calibrated_distance,
                signal_rate_mcps=raw_data['signal_rate_mcps'],
                ambient_rate_mcps=raw_data['ambient_rate_mcps'],
                range_status=raw_data['range_status'],
                objects_found=raw_data['objects_found'],
                stream_count=raw_data['stream_count'],
                quality_score=quality_score,
                is_valid_reading=is_valid,
                confidence=1.0,
                metadata=metadata,
                is_critical=False
            )
            
        except Exception as e:
            self.logger.error(f"Error processing ToF data: {e}")
            return None

    async def get_multiple_readings(self, count: int = 10) -> List[ToFData]:
        """Get multiple readings for statistical analysis"""
        readings = []
        for _ in range(count):
            raw_data = await self.read_raw_data()
            if raw_data:
                processed_data = self.process_raw_data(raw_data)
                if processed_data and processed_data.is_valid_reading:
                    readings.append(processed_data)
            await asyncio.sleep(0.1)
        
        return readings

    def get_distance_statistics(self, readings: List[ToFData]) -> Dict[str, float]:
        """Calculate statistics from multiple readings"""
        if not readings:
            return {}
        
        distances = [r.distance_mm for r in readings]
        
        return {
            'mean': statistics.mean(distances),
            'median': statistics.median(distances),
            'std_dev': statistics.stdev(distances) if len(distances) > 1 else 0.0,
            'min': min(distances),
            'max': max(distances),
            'range': max(distances) - min(distances),
            'count': len(distances)
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        """Get sensor information"""
        return {
            'sensor_type': 'VL53L4CX',
            'manufacturer': 'STMicroelectronics',
            'range': '1mm to 6000mm',
            'accuracy': '±3mm',
            'port': self.port,
            'baudrate': self.baudrate,
            'connected': self.is_connected,
            'offset_calibration': self.offset_calibration,
            'measurement_count': len(self.measurement_history)
        } 