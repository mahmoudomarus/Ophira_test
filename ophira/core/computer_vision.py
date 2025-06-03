"""
Advanced Computer Vision Module for Medical Monitoring
Provides facial expression analysis, pain assessment, and visual monitoring capabilities
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

from .logging import get_logger

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    PAIN = "pain"
    DISTRESS = "distress"
    FEAR = "fear"
    ANGER = "anger"
    SADNESS = "sadness"
    CONFUSION = "confusion"

class PainLevel(Enum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4

@dataclass
class FacialAnalysisResult:
    timestamp: datetime
    emotional_state: EmotionalState
    pain_level: PainLevel
    confidence_score: float
    facial_landmarks: Dict[str, Any]
    analysis_metrics: Dict[str, float]
    alerts: List[str]

@dataclass
class GaitAnalysisResult:
    timestamp: datetime
    stability_score: float
    fall_risk_level: str
    gait_speed: float
    stride_length: float
    asymmetry_index: float
    recommendations: List[str]

class FacialExpressionAnalyzer:
    """Advanced facial expression analysis for pain and emotional state detection"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.face_cascade = None
        self.emotion_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize OpenCV and facial analysis models"""
        try:
            # Load face cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # In a production system, you would load trained models here
            # For now, we'll use rule-based analysis
            self.logger.info("Facial analysis models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing facial analysis models: {str(e)}")
    
    def analyze_facial_expression(self, image: np.ndarray, patient_id: str = None) -> FacialAnalysisResult:
        """Analyze facial expressions for pain and emotional state"""
        try:
            timestamp = datetime.now()
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return self._create_no_face_result(timestamp)
            
            # Analyze the largest face (assumed to be the patient)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            face_roi = gray[y:y+h, x:x+w]
            color_face_roi = image[y:y+h, x:x+w]
            
            # Extract facial features and analyze
            analysis_metrics = self._extract_facial_metrics(face_roi, color_face_roi)
            
            # Determine emotional state and pain level
            emotional_state, confidence = self._classify_emotional_state(analysis_metrics)
            pain_level = self._assess_pain_level(analysis_metrics, emotional_state)
            
            # Generate alerts if necessary
            alerts = self._generate_facial_alerts(emotional_state, pain_level, analysis_metrics)
            
            # Extract facial landmarks (simplified)
            landmarks = self._extract_facial_landmarks(face_roi)
            
            result = FacialAnalysisResult(
                timestamp=timestamp,
                emotional_state=emotional_state,
                pain_level=pain_level,
                confidence_score=confidence,
                facial_landmarks=landmarks,
                analysis_metrics=analysis_metrics,
                alerts=alerts
            )
            
            self.logger.info(f"Facial analysis completed: {emotional_state.value}, pain level: {pain_level.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in facial expression analysis: {str(e)}")
            return self._create_error_result(timestamp, str(e))
    
    def _extract_facial_metrics(self, face_gray: np.ndarray, face_color: np.ndarray) -> Dict[str, float]:
        """Extract key facial metrics for analysis"""
        metrics = {}
        
        try:
            # Eye region analysis
            eye_region = face_gray[int(face_gray.shape[0]*0.2):int(face_gray.shape[0]*0.6), :]
            
            # Mouth region analysis
            mouth_region = face_gray[int(face_gray.shape[0]*0.6):int(face_gray.shape[0]*0.9), :]
            
            # Forehead region analysis (for tension)
            forehead_region = face_gray[0:int(face_gray.shape[0]*0.3), :]
            
            # Calculate texture and contrast metrics
            metrics['eye_openness'] = self._calculate_eye_openness(eye_region)
            metrics['mouth_curvature'] = self._calculate_mouth_curvature(mouth_region)
            metrics['forehead_tension'] = self._calculate_forehead_tension(forehead_region)
            metrics['facial_symmetry'] = self._calculate_facial_symmetry(face_gray)
            metrics['skin_color_variation'] = self._calculate_skin_color_variation(face_color)
            
            # Pain-specific indicators
            metrics['brow_furrow_intensity'] = self._calculate_brow_furrow(forehead_region)
            metrics['lip_compression'] = self._calculate_lip_compression(mouth_region)
            metrics['cheek_tension'] = self._calculate_cheek_tension(face_gray)
            
        except Exception as e:
            self.logger.warning(f"Error extracting facial metrics: {str(e)}")
            # Provide default values
            for key in ['eye_openness', 'mouth_curvature', 'forehead_tension', 'facial_symmetry', 
                       'skin_color_variation', 'brow_furrow_intensity', 'lip_compression', 'cheek_tension']:
                metrics[key] = 0.5
        
        return metrics
    
    def _calculate_eye_openness(self, eye_region: np.ndarray) -> float:
        """Calculate eye openness ratio"""
        # Simplified implementation - in practice would use eye landmark detection
        hist = cv2.calcHist([eye_region], [0], None, [256], [0, 256])
        dark_pixels = np.sum(hist[:50])  # Dark pixels likely represent closed/partially closed eyes
        total_pixels = eye_region.size
        return 1.0 - (dark_pixels / total_pixels)
    
    def _calculate_mouth_curvature(self, mouth_region: np.ndarray) -> float:
        """Calculate mouth curvature (positive for smile, negative for frown)"""
        # Simplified edge detection to find mouth corners
        edges = cv2.Canny(mouth_region, 50, 150)
        
        # Analyze upper and lower mouth regions
        upper_mouth = edges[:edges.shape[0]//2, :]
        lower_mouth = edges[edges.shape[0]//2:, :]
        
        upper_edge_density = np.sum(upper_mouth) / upper_mouth.size
        lower_edge_density = np.sum(lower_mouth) / lower_mouth.size
        
        # Positive value suggests smile, negative suggests frown
        return (lower_edge_density - upper_edge_density) * 10
    
    def _calculate_forehead_tension(self, forehead_region: np.ndarray) -> float:
        """Calculate forehead tension based on wrinkle detection"""
        # Use Sobel operator to detect horizontal lines (wrinkles)
        sobel_x = cv2.Sobel(forehead_region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(forehead_region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Focus on horizontal features (wrinkles)
        horizontal_features = np.abs(sobel_y)
        tension_score = np.mean(horizontal_features) / 255.0
        
        return min(1.0, tension_score * 5)  # Scale appropriately
    
    def _calculate_facial_symmetry(self, face: np.ndarray) -> float:
        """Calculate facial symmetry"""
        # Split face in half and compare
        height, width = face.shape
        left_half = face[:, :width//2]
        right_half = cv2.flip(face[:, width//2:], 1)  # Flip right half
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate normalized correlation
        correlation = cv2.matchTemplate(left_half.astype(np.float32), 
                                      right_half.astype(np.float32), 
                                      cv2.TM_CCOEFF_NORMED)
        
        return float(correlation[0, 0])
    
    def _calculate_skin_color_variation(self, face_color: np.ndarray) -> float:
        """Calculate skin color variation (can indicate stress/pain)"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
        
        # Calculate standard deviation of hue and saturation
        hue_std = np.std(hsv[:, :, 0])
        sat_std = np.std(hsv[:, :, 1])
        val_std = np.std(hsv[:, :, 2])
        
        # Combine variations
        color_variation = (hue_std + sat_std + val_std) / (3 * 255)
        return min(1.0, color_variation)
    
    def _calculate_brow_furrow(self, forehead_region: np.ndarray) -> float:
        """Calculate brow furrowing intensity (pain indicator)"""
        # Focus on vertical lines between eyebrows
        sobel_x = cv2.Sobel(forehead_region, cv2.CV_64F, 1, 0, ksize=3)
        
        # Look for vertical features in the center region
        center_region = sobel_x[:, forehead_region.shape[1]//3:2*forehead_region.shape[1]//3]
        furrow_intensity = np.mean(np.abs(center_region)) / 255.0
        
        return min(1.0, furrow_intensity * 3)
    
    def _calculate_lip_compression(self, mouth_region: np.ndarray) -> float:
        """Calculate lip compression (pain/stress indicator)"""
        # Analyze horizontal edges in mouth region
        edges = cv2.Canny(mouth_region, 50, 150)
        
        # Calculate edge density in horizontal strips
        height = edges.shape[0]
        compression_scores = []
        
        for i in range(3):  # Analyze 3 horizontal strips
            start_row = i * height // 3
            end_row = (i + 1) * height // 3
            strip = edges[start_row:end_row, :]
            edge_density = np.sum(strip) / strip.size
            compression_scores.append(edge_density)
        
        # Higher compression typically shows as more defined edges
        return min(1.0, max(compression_scores) * 5)
    
    def _calculate_cheek_tension(self, face: np.ndarray) -> float:
        """Calculate cheek muscle tension"""
        # Analyze cheek regions (sides of face)
        height, width = face.shape
        
        left_cheek = face[height//3:2*height//3, :width//4]
        right_cheek = face[height//3:2*height//3, 3*width//4:]
        
        # Calculate texture variation (tension creates different texture patterns)
        left_tension = np.std(left_cheek) / 255.0
        right_tension = np.std(right_cheek) / 255.0
        
        return (left_tension + right_tension) / 2
    
    def _classify_emotional_state(self, metrics: Dict[str, float]) -> Tuple[EmotionalState, float]:
        """Classify emotional state based on facial metrics"""
        # Rule-based classification (in production, would use trained models)
        pain_indicators = (
            metrics.get('brow_furrow_intensity', 0) * 0.3 +
            metrics.get('lip_compression', 0) * 0.2 +
            metrics.get('forehead_tension', 0) * 0.3 +
            metrics.get('cheek_tension', 0) * 0.2
        )
        
        happiness_indicators = (
            max(0, metrics.get('mouth_curvature', 0)) * 0.5 +
            metrics.get('eye_openness', 0) * 0.3 +
            (1 - metrics.get('forehead_tension', 0)) * 0.2
        )
        
        distress_indicators = (
            metrics.get('facial_symmetry', 1) < 0.7,  # Asymmetry
            metrics.get('skin_color_variation', 0) > 0.3,  # Color changes
            pain_indicators > 0.4
        )
        
        # Determine state based on strongest indicators
        if pain_indicators > 0.6:
            return EmotionalState.PAIN, min(0.95, pain_indicators + 0.1)
        elif happiness_indicators > 0.6:
            return EmotionalState.HAPPY, min(0.9, happiness_indicators)
        elif sum(distress_indicators) >= 2:
            return EmotionalState.DISTRESS, 0.8
        else:
            return EmotionalState.NEUTRAL, 0.7
    
    def _assess_pain_level(self, metrics: Dict[str, float], emotional_state: EmotionalState) -> PainLevel:
        """Assess pain level based on facial analysis"""
        if emotional_state != EmotionalState.PAIN:
            return PainLevel.NONE
        
        pain_score = (
            metrics.get('brow_furrow_intensity', 0) * 0.4 +
            metrics.get('lip_compression', 0) * 0.3 +
            metrics.get('forehead_tension', 0) * 0.3
        )
        
        if pain_score >= 0.8:
            return PainLevel.EXTREME
        elif pain_score >= 0.6:
            return PainLevel.SEVERE
        elif pain_score >= 0.4:
            return PainLevel.MODERATE
        elif pain_score >= 0.2:
            return PainLevel.MILD
        else:
            return PainLevel.NONE
    
    def _generate_facial_alerts(self, emotional_state: EmotionalState, pain_level: PainLevel, 
                              metrics: Dict[str, float]) -> List[str]:
        """Generate alerts based on facial analysis"""
        alerts = []
        
        if pain_level in [PainLevel.SEVERE, PainLevel.EXTREME]:
            alerts.append(f"High pain level detected: {pain_level.name}")
            alerts.append("Consider immediate pain assessment and intervention")
        
        if emotional_state == EmotionalState.DISTRESS:
            alerts.append("Patient distress detected - may require attention")
        
        if metrics.get('facial_symmetry', 1) < 0.6:
            alerts.append("Facial asymmetry detected - potential neurological concern")
        
        return alerts
    
    def _extract_facial_landmarks(self, face: np.ndarray) -> Dict[str, Any]:
        """Extract simplified facial landmarks"""
        # Simplified landmark extraction (in production would use dlib or similar)
        height, width = face.shape
        
        return {
            'face_center': (width // 2, height // 2),
            'eye_region': (width // 4, height // 3, 3 * width // 4, height // 2),
            'mouth_region': (width // 4, 2 * height // 3, 3 * width // 4, 5 * height // 6),
            'forehead_region': (width // 4, 0, 3 * width // 4, height // 3)
        }
    
    def _create_no_face_result(self, timestamp: datetime) -> FacialAnalysisResult:
        """Create result when no face is detected"""
        return FacialAnalysisResult(
            timestamp=timestamp,
            emotional_state=EmotionalState.NEUTRAL,
            pain_level=PainLevel.NONE,
            confidence_score=0.0,
            facial_landmarks={},
            analysis_metrics={},
            alerts=["No face detected in image"]
        )
    
    def _create_error_result(self, timestamp: datetime, error_msg: str) -> FacialAnalysisResult:
        """Create result when analysis fails"""
        return FacialAnalysisResult(
            timestamp=timestamp,
            emotional_state=EmotionalState.NEUTRAL,
            pain_level=PainLevel.NONE,
            confidence_score=0.0,
            facial_landmarks={},
            analysis_metrics={},
            alerts=[f"Analysis error: {error_msg}"]
        )

class GaitAnalyzer:
    """Gait analysis for fall risk assessment"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        
    def analyze_gait(self, video_frames: List[np.ndarray], patient_id: str = None) -> GaitAnalysisResult:
        """Analyze gait pattern from video frames"""
        try:
            timestamp = datetime.now()
            
            if len(video_frames) < 10:
                return self._create_insufficient_data_result(timestamp)
            
            # Extract motion features
            motion_features = self._extract_motion_features(video_frames)
            
            # Analyze gait parameters
            stability_score = self._calculate_stability_score(motion_features)
            gait_speed = self._estimate_gait_speed(motion_features)
            stride_length = self._estimate_stride_length(motion_features)
            asymmetry_index = self._calculate_asymmetry(motion_features)
            
            # Assess fall risk
            fall_risk_level = self._assess_fall_risk(stability_score, gait_speed, asymmetry_index)
            
            # Generate recommendations
            recommendations = self._generate_gait_recommendations(fall_risk_level, stability_score)
            
            return GaitAnalysisResult(
                timestamp=timestamp,
                stability_score=stability_score,
                fall_risk_level=fall_risk_level,
                gait_speed=gait_speed,
                stride_length=stride_length,
                asymmetry_index=asymmetry_index,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in gait analysis: {str(e)}")
            return self._create_gait_error_result(timestamp, str(e))
    
    def _extract_motion_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Extract motion features from video frames"""
        motion_vectors = []
        centroids = []
        
        for i, frame in enumerate(frames):
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Find contours (representing the person)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assumed to be the person)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    
                    # Calculate motion vector if we have previous centroid
                    if len(centroids) > 1:
                        prev_cx, prev_cy = centroids[-2]
                        motion_vectors.append((cx - prev_cx, cy - prev_cy))
        
        return {
            'motion_vectors': motion_vectors,
            'centroids': centroids,
            'frame_count': len(frames)
        }
    
    def _calculate_stability_score(self, motion_features: Dict[str, Any]) -> float:
        """Calculate gait stability score"""
        motion_vectors = motion_features.get('motion_vectors', [])
        
        if not motion_vectors:
            return 0.5  # Neutral score when no motion detected
        
        # Calculate variability in motion
        x_movements = [mv[0] for mv in motion_vectors]
        y_movements = [mv[1] for mv in motion_vectors]
        
        x_std = np.std(x_movements) if x_movements else 0
        y_std = np.std(y_movements) if y_movements else 0
        
        # Lower variability indicates higher stability
        # Normalize to 0-1 scale (higher = more stable)
        variability = (x_std + y_std) / 2
        stability = max(0, 1 - (variability / 50))  # Assuming max reasonable variability of 50 pixels
        
        return min(1.0, stability)
    
    def _estimate_gait_speed(self, motion_features: Dict[str, Any]) -> float:
        """Estimate gait speed in relative units"""
        motion_vectors = motion_features.get('motion_vectors', [])
        
        if not motion_vectors:
            return 0.0
        
        # Calculate average speed as magnitude of motion vectors
        speeds = [np.sqrt(mv[0]**2 + mv[1]**2) for mv in motion_vectors]
        return np.mean(speeds) if speeds else 0.0
    
    def _estimate_stride_length(self, motion_features: Dict[str, Any]) -> float:
        """Estimate stride length in relative units"""
        centroids = motion_features.get('centroids', [])
        
        if len(centroids) < 4:
            return 0.0
        
        # Calculate distances between consecutive positions
        distances = []
        for i in range(1, len(centroids)):
            x1, y1 = centroids[i-1]
            x2, y2 = centroids[i]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distances.append(distance)
        
        # Return average distance as stride length estimate
        return np.mean(distances) if distances else 0.0
    
    def _calculate_asymmetry(self, motion_features: Dict[str, Any]) -> float:
        """Calculate gait asymmetry index"""
        motion_vectors = motion_features.get('motion_vectors', [])
        
        if len(motion_vectors) < 4:
            return 0.0
        
        # Analyze left vs right movements (simplified)
        left_movements = []
        right_movements = []
        
        for mv in motion_vectors:
            if mv[0] < 0:  # Moving left
                left_movements.append(abs(mv[0]))
            elif mv[0] > 0:  # Moving right
                right_movements.append(mv[0])
        
        if not left_movements or not right_movements:
            return 0.0
        
        left_avg = np.mean(left_movements)
        right_avg = np.mean(right_movements)
        
        # Calculate asymmetry index
        asymmetry = abs(left_avg - right_avg) / max(left_avg, right_avg)
        return min(1.0, asymmetry)
    
    def _assess_fall_risk(self, stability_score: float, gait_speed: float, asymmetry_index: float) -> str:
        """Assess fall risk level"""
        risk_factors = 0
        
        if stability_score < 0.6:
            risk_factors += 2
        elif stability_score < 0.8:
            risk_factors += 1
        
        if gait_speed < 20:  # Slow gait
            risk_factors += 1
        elif gait_speed > 100:  # Very fast, potentially unstable
            risk_factors += 1
        
        if asymmetry_index > 0.3:
            risk_factors += 2
        elif asymmetry_index > 0.2:
            risk_factors += 1
        
        if risk_factors >= 4:
            return "High"
        elif risk_factors >= 2:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_gait_recommendations(self, fall_risk_level: str, stability_score: float) -> List[str]:
        """Generate gait-based recommendations"""
        recommendations = []
        
        if fall_risk_level == "High":
            recommendations.extend([
                "High fall risk detected - consider immediate assessment",
                "Implement fall prevention measures",
                "Consider mobility assistance devices",
                "Increase monitoring frequency"
            ])
        elif fall_risk_level == "Moderate":
            recommendations.extend([
                "Moderate fall risk - monitor gait pattern",
                "Consider physical therapy evaluation",
                "Review medications affecting balance"
            ])
        
        if stability_score < 0.5:
            recommendations.append("Poor gait stability - balance training recommended")
        
        return recommendations
    
    def _create_insufficient_data_result(self, timestamp: datetime) -> GaitAnalysisResult:
        """Create result when insufficient data is available"""
        return GaitAnalysisResult(
            timestamp=timestamp,
            stability_score=0.0,
            fall_risk_level="Unknown",
            gait_speed=0.0,
            stride_length=0.0,
            asymmetry_index=0.0,
            recommendations=["Insufficient video data for gait analysis"]
        )
    
    def _create_gait_error_result(self, timestamp: datetime, error_msg: str) -> GaitAnalysisResult:
        """Create result when gait analysis fails"""
        return GaitAnalysisResult(
            timestamp=timestamp,
            stability_score=0.0,
            fall_risk_level="Error",
            gait_speed=0.0,
            stride_length=0.0,
            asymmetry_index=0.0,
            recommendations=[f"Gait analysis error: {error_msg}"]
        )

class MedicationComplianceMonitor:
    """Visual monitoring for medication compliance"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def detect_medication_taking(self, image: np.ndarray, expected_medication: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if patient is taking medication as expected"""
        try:
            timestamp = datetime.now()
            
            # Simplified medication detection (in production would use trained models)
            # This is a proof of concept implementation
            
            # Detect hand-to-mouth motion
            hand_mouth_motion = self._detect_hand_mouth_motion(image)
            
            # Detect pill/container in image
            medication_present = self._detect_medication_objects(image)
            
            # Analyze compliance
            compliance_score = self._calculate_compliance_score(hand_mouth_motion, medication_present)
            
            return {
                'timestamp': timestamp.isoformat(),
                'medication_detected': medication_present,
                'hand_mouth_motion': hand_mouth_motion,
                'compliance_score': compliance_score,
                'compliance_status': 'compliant' if compliance_score > 0.7 else 'non_compliant',
                'confidence': 0.6 if medication_present else 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Error in medication compliance monitoring: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'compliance_status': 'unknown'
            }
    
    def _detect_hand_mouth_motion(self, image: np.ndarray) -> bool:
        """Detect hand-to-mouth motion"""
        # Simplified implementation - would use pose estimation in production
        # For now, just check for skin-colored regions near face area
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Analyze upper portion of image (where face/hands would be)
        upper_region = skin_mask[:image.shape[0]//2, :]
        skin_pixels = np.sum(upper_region > 0)
        
        # Simple heuristic: if significant skin area in upper region, assume hand-mouth motion
        return skin_pixels > (upper_region.size * 0.1)
    
    def _detect_medication_objects(self, image: np.ndarray) -> bool:
        """Detect medication objects (pills, bottles, etc.)"""
        # Simplified object detection - would use trained models in production
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply circular Hough transform to detect pill-like objects
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=50)
        
        pill_detected = circles is not None and len(circles[0, :]) > 0
        
        # Also check for rectangular objects (pill bottles)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_objects = 0
        for contour in contours:
            # Approximate contour to polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Rectangular object
                rectangular_objects += 1
        
        bottle_detected = rectangular_objects > 0
        
        return pill_detected or bottle_detected
    
    def _calculate_compliance_score(self, hand_mouth_motion: bool, medication_present: bool) -> float:
        """Calculate medication compliance score"""
        score = 0.0
        
        if medication_present:
            score += 0.5
        
        if hand_mouth_motion:
            score += 0.4
        
        if medication_present and hand_mouth_motion:
            score += 0.1  # Bonus for both indicators
        
        return min(1.0, score)

class AdvancedComputerVision:
    """Main computer vision controller coordinating all visual analysis"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.facial_analyzer = FacialExpressionAnalyzer()
        self.gait_analyzer = GaitAnalyzer()
        self.medication_monitor = MedicationComplianceMonitor()
        
    def comprehensive_visual_analysis(self, image: np.ndarray, patient_id: str = None) -> Dict[str, Any]:
        """Perform comprehensive visual analysis"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_id,
                'analysis_results': {}
            }
            
            # Facial expression analysis
            facial_result = self.facial_analyzer.analyze_facial_expression(image, patient_id)
            results['analysis_results']['facial_analysis'] = {
                'emotional_state': facial_result.emotional_state.value,
                'pain_level': facial_result.pain_level.value,
                'confidence_score': facial_result.confidence_score,
                'alerts': facial_result.alerts,
                'metrics': facial_result.analysis_metrics
            }
            
            # Medication compliance check
            medication_result = self.medication_monitor.detect_medication_taking(image, {})
            results['analysis_results']['medication_compliance'] = medication_result
            
            # Generate combined alerts
            combined_alerts = self._generate_combined_alerts(facial_result, medication_result)
            results['combined_alerts'] = combined_alerts
            
            # Calculate overall risk assessment
            risk_assessment = self._calculate_visual_risk_assessment(facial_result, medication_result)
            results['risk_assessment'] = risk_assessment
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive visual analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'analysis_results': {}
            }
    
    def _generate_combined_alerts(self, facial_result: FacialAnalysisResult, 
                                 medication_result: Dict[str, Any]) -> List[str]:
        """Generate combined alerts from all visual analyses"""
        alerts = []
        
        # Add facial analysis alerts
        alerts.extend(facial_result.alerts)
        
        # Add medication compliance alerts
        if medication_result.get('compliance_status') == 'non_compliant':
            alerts.append("Medication non-compliance detected")
        
        return alerts
    
    def _calculate_visual_risk_assessment(self, facial_result: FacialAnalysisResult,
                                        medication_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk assessment from visual analysis"""
        risk_factors = []
        overall_risk_score = 0.0
        
        # Assess pain-related risk
        if facial_result.pain_level in [PainLevel.SEVERE, PainLevel.EXTREME]:
            risk_factors.append("High pain level detected")
            overall_risk_score += 0.4
        elif facial_result.pain_level in [PainLevel.MODERATE]:
            risk_factors.append("Moderate pain level detected")
            overall_risk_score += 0.2
        
        # Assess emotional distress risk
        if facial_result.emotional_state == EmotionalState.DISTRESS:
            risk_factors.append("Patient distress detected")
            overall_risk_score += 0.3
        
        # Assess medication compliance risk
        if medication_result.get('compliance_status') == 'non_compliant':
            risk_factors.append("Medication non-compliance")
            overall_risk_score += 0.3
        
        # Determine risk level
        if overall_risk_score >= 0.7:
            risk_level = "High"
        elif overall_risk_score >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            'overall_risk_score': min(1.0, overall_risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._generate_visual_recommendations(risk_level, risk_factors)
        }
    
    def _generate_visual_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on visual risk assessment"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.extend([
                "Immediate patient assessment recommended",
                "Consider increasing monitoring frequency",
                "Review pain management protocol"
            ])
        elif risk_level == "Moderate":
            recommendations.extend([
                "Monitor patient closely",
                "Consider nursing assessment"
            ])
        
        if "Medication non-compliance" in risk_factors:
            recommendations.append("Review medication administration with patient")
        
        if "High pain level detected" in risk_factors:
            recommendations.append("Pain assessment and intervention may be needed")
        
        return recommendations 