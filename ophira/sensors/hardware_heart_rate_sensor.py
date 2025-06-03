"""
Hardware Heart Rate Sensor Implementation for Ophira AI Medical System
Real hardware interface for heart rate sensor connected via microcontroller over serial
"""

import asyncio
import serial
import serial.tools.list_ports
import json
import time
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, List
import re

from .base_sensor import BaseSensor, SensorData, SensorStatus


class HardwareHeartRateSensor(BaseSensor):
    """
    Hardware Heart Rate Sensor implementation
    Interfaces with heart rate sensor connected via microcontroller over serial
    """
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "hardware_heart_rate", config)
        
        # Serial communication settings
        self.port = self.config.get("port", None)  # Auto-detect if None
        self.baudrate = self.config.get("baudrate", 115200)
        self.timeout = self.config.get("timeout", 1.0)
        self.data_bits = self.config.get("data_bits", 8)
        self.stop_bits = self.config.get("stop_bits", 1)
        self.parity = self.config.get("parity", "N")
        
        # Heart rate processing parameters
        self.sampling_window = self.config.get("sampling_window", 10)  # seconds
        self.min_heart_rate = self.config.get("min_heart_rate", 40)
        self.max_heart_rate = self.config.get("max_heart_rate", 200)
        
        # Serial connection
        self.serial_connection = None
        self.data_buffer = []
        self.last_heartbeat_time = None
        
        # Data parsing settings
        self.data_format = self.config.get("data_format", "json")  # json, csv, or raw
        self.expected_fields = self.config.get("expected_fields", ["bpm", "timestamp"])
        
        # Signal processing
        self.signal_buffer = []
        self.signal_buffer_size = self.config.get("signal_buffer_size", 1000)
        
    async def connect(self) -> bool:
        """Connect to heart rate sensor via microcontroller serial port"""
        try:
            self.status = SensorStatus.CONNECTING
            
            # Auto-detect port if not specified
            if self.port is None:
                detected_port = self._auto_detect_port()
                if detected_port is None:
                    print("❌ No suitable serial port found for heart rate sensor")
                    self.status = SensorStatus.ERROR
                    return False
                self.port = detected_port
            
            # Open serial connection
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.data_bits,
                stopbits=self.stop_bits,
                parity=self.parity,
                timeout=self.timeout
            )
            
            # Wait for connection to stabilize
            await asyncio.sleep(2)
            
            # Test communication
            if self.serial_connection.is_open:
                # Clear any existing data
                self.serial_connection.flushInput()
                self.serial_connection.flushOutput()
                
                # Send test command (if your microcontroller supports it)
                self._send_command("STATUS")
                
                # Try to read some initial data
                test_data = await self._read_serial_data()
                if test_data:
                    self.status = SensorStatus.CONNECTED
                    print(f"✅ Hardware Heart Rate Sensor connected on {self.port}")
                    print(f"   Baudrate: {self.baudrate}, Test data received: {len(test_data)} bytes")
                    return True
                else:
                    print(f"⚠️ Connected to {self.port} but no data received")
                    # Still consider it connected - data might come later
                    self.status = SensorStatus.CONNECTED
                    return True
            else:
                self.status = SensorStatus.ERROR
                return False
                
        except serial.SerialException as e:
            print(f"❌ Serial connection failed: {e}")
            self.status = SensorStatus.ERROR
            return False
        except Exception as e:
            print(f"❌ Unexpected error connecting to heart rate sensor: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    def _auto_detect_port(self) -> Optional[str]:
        """Auto-detect the microcontroller serial port"""
        try:
            ports = serial.tools.list_ports.comports()
            
            # Common microcontroller identifiers
            microcontroller_ids = [
                "Arduino", "ESP32", "ESP8266", "Teensy", "STM32",
                "CH340", "CP210", "FTDI", "USB Serial"
            ]
            
            for port in ports:
                port_info = f"{port.description} {port.manufacturer or ''}"
                print(f"🔍 Found port: {port.device} - {port_info}")
                
                # Check if it matches known microcontroller patterns
                for mc_id in microcontroller_ids:
                    if mc_id.lower() in port_info.lower():
                        print(f"✅ Detected microcontroller: {port.device} ({mc_id})")
                        return port.device
            
            # If no specific microcontroller found, try the first available port
            if ports:
                print(f"⚠️ No known microcontroller found, trying first port: {ports[0].device}")
                return ports[0].device
            
            return None
            
        except Exception as e:
            print(f"❌ Error auto-detecting port: {e}")
            return None
    
    def _send_command(self, command: str):
        """Send command to microcontroller"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                command_bytes = f"{command}\n".encode()
                self.serial_connection.write(command_bytes)
                self.serial_connection.flush()
        except Exception as e:
            print(f"❌ Error sending command: {e}")
    
    async def _read_serial_data(self) -> Optional[str]:
        """Read data from serial port"""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                return None
            
            # Check if data is available
            if self.serial_connection.in_waiting > 0:
                # Read line
                line = self.serial_connection.readline()
                if line:
                    return line.decode('utf-8', errors='ignore').strip()
            
            return None
            
        except Exception as e:
            print(f"❌ Error reading serial data: {e}")
            return None
    
    async def disconnect(self) -> bool:
        """Disconnect from heart rate sensor"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.status = SensorStatus.DISCONNECTED
            print("🔌 Hardware Heart Rate Sensor disconnected")
            return True
        except Exception as e:
            print(f"❌ Error disconnecting: {e}")
            return False
    
    async def read_raw_data(self) -> Any:
        """Read heart rate data from microcontroller"""
        if self.status != SensorStatus.CONNECTED:
            return None
        
        try:
            # Read multiple lines to get a good sample
            readings = []
            for _ in range(5):  # Try to get 5 readings
                data_line = await self._read_serial_data()
                if data_line:
                    readings.append(data_line)
                await asyncio.sleep(0.1)  # Small delay between reads
            
            if not readings:
                return None
            
            # Parse the readings
            parsed_data = []
            for reading in readings:
                parsed = self._parse_data_line(reading)
                if parsed:
                    parsed_data.append(parsed)
            
            if not parsed_data:
                return None
            
            # Calculate heart rate from parsed data
            bpm_values = [d.get("bpm", 0) for d in parsed_data if d.get("bpm", 0) > 0]
            
            if bpm_values:
                # Use most recent valid BPM
                current_bpm = bpm_values[-1]
                avg_bpm = np.mean(bpm_values)
                bpm_variance = np.var(bpm_values) if len(bpm_values) > 1 else 0
            else:
                # Try to calculate from raw signal if available
                raw_signals = [d.get("raw_signal", []) for d in parsed_data]
                flat_signals = [val for sublist in raw_signals for val in sublist if isinstance(sublist, list)]
                
                if len(flat_signals) > 10:
                    current_bpm = self._calculate_bpm_from_signal(flat_signals)
                    avg_bpm = current_bpm
                    bpm_variance = 0
                else:
                    return None
            
            # Quality assessment
            signal_quality = self._assess_signal_quality(parsed_data)
            
            return {
                "bpm": current_bpm,
                "avg_bpm": avg_bpm,
                "bpm_variance": bpm_variance,
                "signal_quality": signal_quality,
                "raw_readings": readings,
                "parsed_data": parsed_data,
                "timestamp": datetime.now(),
                "sample_count": len(parsed_data)
            }
            
        except Exception as e:
            print(f"❌ Error reading heart rate data: {e}")
            return None
    
    def _parse_data_line(self, data_line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line of data from microcontroller"""
        try:
            if self.data_format == "json":
                # Try to parse as JSON
                try:
                    return json.loads(data_line)
                except json.JSONDecodeError:
                    pass
            
            elif self.data_format == "csv":
                # Parse as CSV (bpm,timestamp,raw_value,etc.)
                parts = data_line.split(',')
                if len(parts) >= 2:
                    return {
                        "bpm": float(parts[0]) if parts[0].replace('.','').isdigit() else 0,
                        "timestamp": parts[1],
                        "raw_signal": [float(p) for p in parts[2:] if p.replace('.','').replace('-','').isdigit()]
                    }
            
            # Try to extract numbers (raw format)
            numbers = re.findall(r'-?\d+\.?\d*', data_line)
            if numbers:
                bpm = float(numbers[0])
                if self.min_heart_rate <= bpm <= self.max_heart_rate:
                    return {
                        "bpm": bpm,
                        "raw_signal": [float(n) for n in numbers[1:]] if len(numbers) > 1 else [],
                        "raw_line": data_line
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing data line '{data_line}': {e}")
            return None
    
    def _calculate_bpm_from_signal(self, signal: List[float]) -> float:
        """Calculate BPM from raw signal data using peak detection"""
        try:
            if len(signal) < 20:
                return 0
            
            # Simple peak detection
            signal_array = np.array(signal)
            mean_val = np.mean(signal_array)
            std_val = np.std(signal_array)
            threshold = mean_val + 0.5 * std_val
            
            # Find peaks
            peaks = []
            for i in range(1, len(signal_array) - 1):
                if (signal_array[i] > signal_array[i-1] and 
                    signal_array[i] > signal_array[i+1] and 
                    signal_array[i] > threshold):
                    peaks.append(i)
            
            if len(peaks) < 2:
                return 0
            
            # Calculate average interval between peaks
            intervals = np.diff(peaks)
            if len(intervals) == 0:
                return 0
            
            avg_interval = np.mean(intervals)
            
            # Assuming data comes at ~100Hz (adjust based on your microcontroller)
            sampling_rate = self.config.get("sampling_rate", 100)
            bpm = (60 * sampling_rate) / avg_interval
            
            # Validate result
            if self.min_heart_rate <= bpm <= self.max_heart_rate:
                return round(bpm, 1)
            
            return 0
            
        except Exception as e:
            print(f"❌ Error calculating BPM from signal: {e}")
            return 0
    
    def _assess_signal_quality(self, parsed_data: List[Dict[str, Any]]) -> float:
        """Assess the quality of heart rate signal"""
        try:
            if not parsed_data:
                return 0.0
            
            bpm_values = [d.get("bpm", 0) for d in parsed_data if d.get("bpm", 0) > 0]
            
            if len(bpm_values) < 2:
                return 0.5  # Moderate quality if only one reading
            
            # Check consistency
            bpm_variance = np.var(bpm_values)
            consistency_score = max(0, 1 - (bpm_variance / 100))  # Lower variance = better
            
            # Check if values are in reasonable range
            valid_readings = sum(1 for bpm in bpm_values if self.min_heart_rate <= bpm <= self.max_heart_rate)
            validity_score = valid_readings / len(bpm_values)
            
            # Overall quality score
            quality = (consistency_score * 0.6 + validity_score * 0.4)
            return min(1.0, max(0.0, quality))
            
        except Exception:
            return 0.3  # Default moderate quality
    
    async def calibrate(self) -> bool:
        """Calibrate heart rate sensor"""
        try:
            self.status = SensorStatus.CALIBRATING
            print("🔧 Starting heart rate sensor calibration...")
            
            # Send calibration command to microcontroller
            self._send_command("CALIBRATE")
            
            # Collect baseline data for 10 seconds
            baseline_readings = []
            start_time = time.time()
            
            while (time.time() - start_time) < 10:
                data = await self._read_serial_data()
                if data:
                    parsed = self._parse_data_line(data)
                    if parsed and parsed.get("bpm", 0) > 0:
                        baseline_readings.append(parsed["bpm"])
                await asyncio.sleep(0.2)
            
            if len(baseline_readings) < 5:
                print("❌ Insufficient calibration data")
                self.status = SensorStatus.ERROR
                return False
            
            # Calculate calibration parameters
            baseline_bpm = np.mean(baseline_readings)
            baseline_variance = np.var(baseline_readings)
            
            self.calibration_data = {
                "baseline_bpm": float(baseline_bpm),
                "baseline_variance": float(baseline_variance),
                "calibration_readings": len(baseline_readings),
                "calibrated_at": datetime.now().isoformat(),
                "port": self.port,
                "baudrate": self.baudrate
            }
            
            self.status = SensorStatus.CONNECTED
            print(f"✅ Heart rate sensor calibration complete")
            print(f"   Baseline BPM: {baseline_bpm:.1f} ± {np.sqrt(baseline_variance):.1f}")
            return True
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        """Process raw heart rate data into standardized format"""
        if not raw_data:
            return None
        
        try:
            bpm = raw_data["bpm"]
            signal_quality = raw_data["signal_quality"]
            timestamp = raw_data["timestamp"]
            
            # Validate heart rate
            is_valid = self.min_heart_rate <= bpm <= self.max_heart_rate
            
            # Determine if critical
            is_critical = (bpm > 100 or bpm < 60) and is_valid
            
            # Calculate confidence based on signal quality and validity
            confidence = signal_quality if is_valid else signal_quality * 0.5
            
            return SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=timestamp,
                value=round(bpm, 1),
                unit="bpm",
                confidence=confidence,
                metadata={
                    "signal_quality": signal_quality,
                    "avg_bpm": round(raw_data["avg_bpm"], 1),
                    "bpm_variance": round(raw_data["bpm_variance"], 2),
                    "sample_count": raw_data["sample_count"],
                    "port": self.port,
                    "baudrate": self.baudrate,
                    "calibration_applied": bool(self.calibration_data),
                    "hardware_type": "microcontroller_serial"
                },
                quality_score=signal_quality,
                is_critical=is_critical
            )
            
        except Exception as e:
            print(f"❌ Error processing heart rate data: {e}")
            return None
    
    def get_port_info(self) -> Dict[str, Any]:
        """Get detailed port and connection information"""
        info = {
            "port": self.port,
            "baudrate": self.baudrate,
            "is_connected": self.serial_connection is not None and self.serial_connection.is_open,
            "data_format": self.data_format,
            "expected_fields": self.expected_fields
        }
        
        if self.serial_connection:
            try:
                info.update({
                    "bytes_waiting": self.serial_connection.in_waiting,
                    "timeout": self.serial_connection.timeout,
                    "data_bits": self.serial_connection.bytesize,
                    "stop_bits": self.serial_connection.stopbits,
                    "parity": self.serial_connection.parity
                })
            except Exception:
                pass
        
        return info 