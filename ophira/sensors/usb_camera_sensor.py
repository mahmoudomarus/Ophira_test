#!/usr/bin/env python3
"""
USB Camera Sensor for Ophira AI Medical System

Provides interface for standard USB cameras for general purpose image capture
and basic medical image analysis.
"""

import asyncio
import cv2
import numpy as np
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ophira.sensors.base_sensor import BaseSensor, SensorData


@dataclass
class USBCameraData:
    """USB Camera specific data structure"""
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    frame: np.ndarray
    frame_width: int
    frame_height: int
    fps: float
    brightness: float
    contrast: float
    focus_score: float
    motion_detected: bool
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    quality_score: float = 1.0
    is_critical: bool = False


class USBCameraSensor(BaseSensor):
    """
    USB Camera sensor for general purpose medical imaging
    
    Supports standard USB cameras with OpenCV backend
    """
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "usb_camera", config)
        
        # Camera settings
        self.camera_index = config.get('camera_index', 0) if config else 0
        self.width = config.get('width', 640) if config else 640
        self.height = config.get('height', 480) if config else 480
        self.fps_target = config.get('fps', 30) if config else 30
        
        # OpenCV camera object
        self.camera = None
        self.is_recording = False
        
        # Image analysis
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.previous_frame = None
        self.frame_count = 0
        self.fps_actual = 0.0
        self.last_fps_time = time.time()
        
        # Calibration data
        self.brightness_baseline = 128
        self.contrast_baseline = 1.0
        
        self.logger.info(f"USB Camera sensor initialized - Index: {self.camera_index}")

    async def connect(self) -> bool:
        """Connect to USB camera"""
        try:
            self.logger.info(f"Connecting to USB camera at index {self.camera_index}")
            
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                # Try different backends
                backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]
                for backend in backends:
                    self.camera = cv2.VideoCapture(self.camera_index, backend)
                    if self.camera.isOpened():
                        self.logger.info(f"Camera connected using backend: {backend}")
                        break
                else:
                    self.logger.error("Failed to connect to camera with any backend")
                    return False
            
            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            # Test frame capture
            ret, frame = self.camera.read()
            if not ret:
                self.logger.error("Failed to capture test frame")
                return False
            
            self.is_connected = True
            self.logger.info("USB camera connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to USB camera: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from camera"""
        try:
            if self.camera and self.camera.isOpened():
                self.camera.release()
            self.is_connected = False
            self.is_recording = False
            self.logger.info("USB camera disconnected")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting camera: {e}")
            return False

    async def read_raw_data(self) -> Any:
        """Capture frame from camera"""
        if not self.is_connected or not self.camera:
            self.logger.error("Camera not connected")
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                return None
                
            # Update FPS calculation
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps_actual = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            return {
                'frame': frame,
                'timestamp': datetime.now(),
                'frame_number': self.frame_count,
                'fps': self.fps_actual
            }
            
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None

    def _calculate_focus_score(self, image: np.ndarray) -> float:
        """Calculate focus score using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()
        except:
            return 0.0

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion using background subtraction"""
        try:
            fg_mask = self.background_subtractor.apply(frame)
            motion_pixels = cv2.countNonZero(fg_mask)
            motion_threshold = (frame.shape[0] * frame.shape[1]) * 0.01  # 1% of pixels
            return motion_pixels > motion_threshold
        except:
            return False

    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        except:
            return 0.0

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate RMS contrast"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return float(np.std(gray))
        except:
            return 0.0

    async def calibrate(self) -> bool:
        """Calibrate camera settings"""
        try:
            self.logger.info("Starting camera calibration...")
            
            # Capture multiple frames for baseline
            frames = []
            for i in range(10):
                raw_data = await self.read_raw_data()
                if raw_data and raw_data['frame'] is not None:
                    frames.append(raw_data['frame'])
                await asyncio.sleep(0.1)
            
            if not frames:
                self.logger.error("No frames captured for calibration")
                return False
            
            # Calculate baseline values
            brightness_values = [self._calculate_brightness(frame) for frame in frames]
            contrast_values = [self._calculate_contrast(frame) for frame in frames]
            
            self.brightness_baseline = np.mean(brightness_values)
            self.contrast_baseline = np.mean(contrast_values)
            
            self.logger.info(f"Calibration complete - Brightness: {self.brightness_baseline:.2f}, Contrast: {self.contrast_baseline:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False

    def process_raw_data(self, raw_data: Any) -> USBCameraData:
        """Process raw camera data into structured format"""
        if not raw_data or raw_data.get('frame') is None:
            return None
            
        try:
            frame = raw_data['frame']
            timestamp = raw_data['timestamp']
            
            # Calculate image metrics
            focus_score = self._calculate_focus_score(frame)
            motion_detected = self._detect_motion(frame)
            brightness = self._calculate_brightness(frame)
            contrast = self._calculate_contrast(frame)
            
            # Create metadata
            metadata = {
                'camera_index': self.camera_index,
                'frame_number': raw_data.get('frame_number', 0),
                'capture_fps': raw_data.get('fps', 0),
                'brightness_baseline': self.brightness_baseline,
                'contrast_baseline': self.contrast_baseline,
                'device_info': self.get_camera_info()
            }
            
            return USBCameraData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=timestamp,
                frame=frame,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                fps=raw_data.get('fps', 0),
                brightness=brightness,
                contrast=contrast,
                focus_score=focus_score,
                motion_detected=motion_detected,
                confidence=1.0,
                metadata=metadata,
                quality_score=1.0,
                is_critical=False
            )
            
        except Exception as e:
            self.logger.error(f"Error processing camera data: {e}")
            return None

    def save_frame(self, frame: np.ndarray, filename: str = None) -> str:
        """Save frame to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"usb_camera_frame_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            self.logger.info(f"Frame saved: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return ""

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        if not self.camera:
            return {}
            
        try:
            return {
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
                'backend': self.camera.getBackendName(),
                'is_opened': self.camera.isOpened()
            }
        except:
            return {}

    async def start_recording(self, duration: float = 10.0) -> List[np.ndarray]:
        """Record frames for specified duration"""
        frames = []
        self.is_recording = True
        start_time = time.time()
        
        try:
            while self.is_recording and (time.time() - start_time) < duration:
                raw_data = await self.read_raw_data()
                if raw_data and raw_data['frame'] is not None:
                    frames.append(raw_data['frame'])
                await asyncio.sleep(1.0 / self.fps_target)
                
            self.logger.info(f"Recording complete: {len(frames)} frames captured")
            return frames
            
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
            return frames
        finally:
            self.is_recording = False

    def stop_recording(self):
        """Stop current recording"""
        self.is_recording = False
        self.logger.info("Recording stopped") 