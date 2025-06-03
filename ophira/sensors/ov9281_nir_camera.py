"""
OV 9281 NIR Camera Hardware Implementation for Ophira AI Medical System
Real hardware interface for USB-connected OV 9281 NIR camera
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional
import logging

from .base_sensor import BaseSensor, SensorData, SensorStatus


class OV9281NIRCamera(BaseSensor):
    """
    OV 9281 NIR Camera hardware implementation
    Interfaces with real USB-connected OV 9281 camera for retinal imaging
    """
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id, "ov9281_nir_camera", config)
        
        # Camera configuration
        self.device_id = self.config.get("device_id", 0)  # USB camera device ID
        self.resolution = self.config.get("resolution", (1280, 720))  # OV9281 typical resolution
        self.fps = self.config.get("fps", 60)  # OV9281 can do up to 120fps
        self.exposure = self.config.get("exposure", -7)  # Auto exposure by default
        self.gain = self.config.get("gain", 0)  # Auto gain by default
        
        # OpenCV camera object
        self.camera = None
        self.frame_count = 0
        
        # Image processing parameters
        self.roi_enabled = self.config.get("roi_enabled", True)
        self.roi_coords = self.config.get("roi_coords", (160, 90, 960, 540))  # Center crop for retina
        
        # NIR-specific parameters
        self.nir_wavelength = self.config.get("nir_wavelength", 850)  # nm
        self.auto_white_balance = self.config.get("auto_white_balance", False)
        
    async def connect(self) -> bool:
        """Connect to OV 9281 NIR camera via USB"""
        try:
            self.status = SensorStatus.CONNECTING
            
            # Initialize camera with OpenCV
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                self.status = SensorStatus.ERROR
                return False
            
            # Configure camera parameters for OV9281
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set exposure and gain for NIR imaging
            self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            self.camera.set(cv2.CAP_PROP_GAIN, self.gain)
            
            # Disable auto white balance for consistent NIR imaging
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 0 if not self.auto_white_balance else 1)
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.camera.release()
                self.status = SensorStatus.ERROR
                return False
            
            self.status = SensorStatus.CONNECTED
            print(f"✅ OV9281 NIR Camera connected - Resolution: {frame.shape[:2]}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to OV9281 camera: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from OV 9281 camera"""
        try:
            if self.camera and self.camera.isOpened():
                self.camera.release()
            self.status = SensorStatus.DISCONNECTED
            print("🔌 OV9281 NIR Camera disconnected")
            return True
        except Exception as e:
            print(f"❌ Error disconnecting camera: {e}")
            return False
    
    async def read_raw_data(self) -> Any:
        """Capture frame from OV 9281 camera"""
        if self.status != SensorStatus.CONNECTED or not self.camera:
            return None
        
        try:
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                return None
            
            self.frame_count += 1
            
            # Convert to grayscale for NIR analysis (OV9281 is monochrome)
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            # Apply ROI if enabled (focus on retinal area)
            if self.roi_enabled and self.roi_coords:
                x, y, w, h = self.roi_coords
                roi_frame = gray_frame[y:y+h, x:x+w]
            else:
                roi_frame = gray_frame
            
            # Calculate image quality metrics
            frame_mean = np.mean(roi_frame)
            frame_std = np.std(roi_frame)
            contrast = frame_std / frame_mean if frame_mean > 0 else 0
            
            # Detect potential retinal features using basic image processing
            vessel_mask = self._detect_vessels(roi_frame)
            vessel_density = np.sum(vessel_mask > 0) / vessel_mask.size
            
            # Calculate focus score using gradient magnitude
            focus_score = self._calculate_focus_score(roi_frame)
            
            return {
                "frame": roi_frame,
                "full_frame": gray_frame,
                "timestamp": datetime.now(),
                "frame_count": self.frame_count,
                "image_quality": {
                    "mean_intensity": float(frame_mean),
                    "contrast": float(contrast),
                    "focus_score": float(focus_score)
                },
                "retinal_features": {
                    "vessel_density": float(vessel_density),
                    "vessel_mask": vessel_mask
                },
                "camera_params": {
                    "exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE),
                    "gain": self.camera.get(cv2.CAP_PROP_GAIN),
                    "fps": self.camera.get(cv2.CAP_PROP_FPS)
                }
            }
            
        except Exception as e:
            print(f"❌ Error reading from OV9281 camera: {e}")
            return None
    
    def _detect_vessels(self, image: np.ndarray) -> np.ndarray:
        """Basic vessel detection using morphological operations"""
        try:
            # Apply CLAHE to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
            
            # Create vessel detection kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Top-hat transform to highlight vessels
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold to create vessel mask
            _, vessel_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return vessel_mask
            
        except Exception:
            return np.zeros_like(image)
    
    def _calculate_focus_score(self, image: np.ndarray) -> float:
        """Calculate focus score using Laplacian variance"""
        try:
            # Calculate Laplacian and return variance
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            focus_score = laplacian.var()
            
            # Normalize to 0-1 range
            return min(focus_score / 1000.0, 1.0)
            
        except Exception:
            return 0.0
    
    async def calibrate(self) -> bool:
        """Calibrate OV 9281 camera for optimal NIR imaging"""
        try:
            self.status = SensorStatus.CALIBRATING
            
            print("🔧 Starting OV9281 camera calibration...")
            
            # Capture multiple frames for calibration
            calibration_frames = []
            for i in range(10):
                await asyncio.sleep(0.1)  # Allow camera to adjust
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame
                    calibration_frames.append(gray)
            
            if len(calibration_frames) < 5:
                self.status = SensorStatus.ERROR
                return False
            
            # Calculate optimal parameters
            avg_frame = np.mean(calibration_frames, axis=0).astype(np.uint8)
            avg_intensity = np.mean(avg_frame)
            
            # Store calibration data
            self.calibration_data = {
                "dark_frame": np.min(calibration_frames, axis=0).tolist(),
                "avg_intensity": float(avg_intensity),
                "optimal_exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE),
                "optimal_gain": self.camera.get(cv2.CAP_PROP_GAIN),
                "calibrated_at": datetime.now().isoformat(),
                "roi_coords": self.roi_coords
            }
            
            self.status = SensorStatus.CONNECTED
            print("✅ OV9281 camera calibration complete")
            return True
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    def process_raw_data(self, raw_data: Any) -> SensorData:
        """Process raw camera data into standardized format"""
        if not raw_data:
            return None
        
        try:
            frame = raw_data["frame"]
            quality_metrics = raw_data["image_quality"]
            retinal_features = raw_data["retinal_features"]
            timestamp = raw_data["timestamp"]
            
            # Determine image quality and critical conditions
            focus_score = quality_metrics["focus_score"]
            contrast = quality_metrics["contrast"]
            vessel_density = retinal_features["vessel_density"]
            
            # Quality score based on focus, contrast, and vessel visibility
            quality_score = (focus_score * 0.4 + min(contrast / 0.3, 1.0) * 0.3 + 
                           min(vessel_density / 0.2, 1.0) * 0.3)
            
            # Critical if image quality is too poor for analysis
            is_critical = (focus_score < 0.3 or contrast < 0.1 or 
                          quality_score < 0.5)
            
            # Calculate confidence based on quality metrics
            confidence = quality_score
            
            return SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=timestamp,
                value={
                    "vessel_density": round(vessel_density, 4),
                    "focus_score": round(focus_score, 3),
                    "contrast": round(contrast, 3),
                    "mean_intensity": round(quality_metrics["mean_intensity"], 1),
                    "frame_shape": frame.shape,
                    "frame_count": raw_data["frame_count"]
                },
                unit="nir_analysis",
                confidence=confidence,
                metadata={
                    "camera_model": "OV9281",
                    "nir_wavelength": self.nir_wavelength,
                    "resolution": self.resolution,
                    "roi_enabled": self.roi_enabled,
                    "roi_coords": self.roi_coords,
                    "camera_params": raw_data["camera_params"],
                    "calibration_applied": bool(self.calibration_data)
                },
                quality_score=quality_score,
                is_critical=is_critical
            )
            
        except Exception as e:
            print(f"❌ Error processing OV9281 data: {e}")
            return None
    
    def save_frame(self, frame: np.ndarray, filename: str = None) -> str:
        """Save captured frame to disk"""
        try:
            if filename is None:
                filename = f"ov9281_frame_{self.frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving frame: {e}")
            return ""
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get detailed camera information"""
        if not self.camera or not self.camera.isOpened():
            return {}
        
        return {
            "device_id": self.device_id,
            "resolution": (
                int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "fps": self.camera.get(cv2.CAP_PROP_FPS),
            "exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE),
            "gain": self.camera.get(cv2.CAP_PROP_GAIN),
            "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST),
            "backend": self.camera.getBackendName(),
            "frame_count": self.frame_count
        } 