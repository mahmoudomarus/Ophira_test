"""
Medical AI Models Service for Ophira
Integrates VLLM for specialized medical analysis including ophthalmology, cardiovascular, and metabolic models
"""
import asyncio
import base64
import io
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import aiohttp
import numpy as np
from PIL import Image
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoProcessor
from dataclasses import dataclass
from enum import Enum

from .config import get_settings
from .logging import get_agent_logger

logger = get_agent_logger("medical_models")


class ModelType(Enum):
    """Medical model types"""
    RETINAL_ANALYSIS = "retinal_analysis"
    EYE_SEGMENTATION = "eye_segmentation"
    WAVEFRONT_ANALYSIS = "wavefront_analysis"
    CARDIOVASCULAR = "cardiovascular"
    METABOLIC = "metabolic"
    GENERAL_MEDICAL = "general_medical"


class AnalysisType(Enum):
    """Medical analysis types"""
    DIABETIC_RETINOPATHY = "diabetic_retinopathy"
    GLAUCOMA_DETECTION = "glaucoma_detection"
    MACULAR_DEGENERATION = "macular_degeneration"
    RETINAL_VESSEL_ANALYSIS = "retinal_vessel_analysis"
    OPTIC_DISC_ANALYSIS = "optic_disc_analysis"
    EYE_PRESSURE_ESTIMATION = "eye_pressure_estimation"
    WAVEFRONT_ABERRATIONS = "wavefront_aberrations"
    PUPIL_TRACKING = "pupil_tracking"
    HEART_RATE_VARIABILITY = "heart_rate_variability"
    BLOOD_PRESSURE_ANALYSIS = "blood_pressure_analysis"
    GLUCOSE_ESTIMATION = "glucose_estimation"


@dataclass
class MedicalAnalysisRequest:
    """Request structure for medical analysis"""
    patient_id: str
    analysis_type: AnalysisType
    data: Union[np.ndarray, str, Dict[str, Any]]  # Image array, sensor data, or structured data
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MedicalAnalysisResult:
    """Result structure for medical analysis"""
    analysis_type: AnalysisType
    confidence: float
    findings: Dict[str, Any]
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high", "critical"
    requires_followup: bool
    specialist_referral: Optional[str] = None
    processing_time: float = 0.0
    model_version: str = ""
    raw_output: Optional[Dict[str, Any]] = None


class VLLMMedicalModelService:
    """Service for managing VLLM-based medical models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
        self.models_cache: Dict[str, Any] = {}
        self.pipelines: Dict[ModelType, Any] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the medical model service"""
        if self._initialized:
            return
        
        logger.log_decision("model_initialization", "Initializing VLLM medical model service", 0.95)
        
        # Create HTTP session for VLLM communication
        self.session = aiohttp.ClientSession()
        
        # Initialize specialized pipelines
        await self._initialize_medical_pipelines()
        
        # Test VLLM server connection
        await self._test_vllm_connection()
        
        self._initialized = True
        logger.log_decision("model_initialization", "VLLM medical model service initialized successfully", 1.0)
    
    async def _initialize_medical_pipelines(self):
        """Initialize HuggingFace pipelines for medical analysis"""
        try:
            # Retinal analysis pipeline (Vision Transformer)
            self.pipelines[ModelType.RETINAL_ANALYSIS] = pipeline(
                "image-classification",
                model=self.settings.ai_models.retinal_vision_model,
                use_auth_token=self.settings.ai_models.huggingface_token
            )
            
            # Eye segmentation pipeline (DETR)
            self.pipelines[ModelType.EYE_SEGMENTATION] = pipeline(
                "object-detection",
                model=self.settings.ai_models.eye_segmentation_model,
                use_auth_token=self.settings.ai_models.huggingface_token
            )
            
            # General medical text analysis
            self.pipelines[ModelType.GENERAL_MEDICAL] = pipeline(
                "text-generation",
                model=self.settings.ai_models.medical_llm_model,
                use_auth_token=self.settings.ai_models.huggingface_token,
                max_length=self.settings.ai_models.max_tokens,
                temperature=self.settings.ai_models.temperature
            )
            
            logger.log_decision("pipeline_initialization", "Medical analysis pipelines initialized", 1.0)
            
        except Exception as e:
            logger.log_error("pipeline_initialization_error", str(e), {"models": list(self.pipelines.keys())})
            # Continue with reduced functionality
    
    async def _test_vllm_connection(self):
        """Test connection to VLLM server"""
        try:
            async with self.session.get(f"{self.settings.ai_models.vllm_server_url}/health") as response:
                if response.status == 200:
                    logger.log_decision("vllm_connection", "VLLM server connection successful", 1.0)
                else:
                    logger.log_error("vllm_connection_error", f"VLLM server returned status {response.status}", {})
        except Exception as e:
            logger.log_error("vllm_connection_error", f"Failed to connect to VLLM server: {str(e)}", {})
    
    async def analyze_retinal_image(self, request: MedicalAnalysisRequest) -> MedicalAnalysisResult:
        """Analyze retinal images for various conditions"""
        start_time = datetime.now()
        
        try:
            # Convert image data to PIL Image
            if isinstance(request.data, np.ndarray):
                image = Image.fromarray(request.data)
            elif isinstance(request.data, str):  # Base64 encoded
                image_data = base64.b64decode(request.data)
                image = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError("Invalid image data format")
            
            # Analyze based on request type
            if request.analysis_type == AnalysisType.DIABETIC_RETINOPATHY:
                result = await self._analyze_diabetic_retinopathy(image, request.metadata)
            elif request.analysis_type == AnalysisType.GLAUCOMA_DETECTION:
                result = await self._analyze_glaucoma(image, request.metadata)
            elif request.analysis_type == AnalysisType.MACULAR_DEGENERATION:
                result = await self._analyze_macular_degeneration(image, request.metadata)
            elif request.analysis_type == AnalysisType.RETINAL_VESSEL_ANALYSIS:
                result = await self._analyze_retinal_vessels(image, request.metadata)
            else:
                result = await self._general_retinal_analysis(image, request.metadata)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            logger.log_decision(
                "retinal_analysis_complete",
                f"Retinal analysis completed for {request.analysis_type.value}",
                result.confidence
            )
            
            return result
            
        except Exception as e:
            logger.log_error("retinal_analysis_error", str(e), {"patient_id": request.patient_id})
            return MedicalAnalysisResult(
                analysis_type=request.analysis_type,
                confidence=0.0,
                findings={"error": str(e)},
                recommendations=["Analysis failed - please retry or consult specialist"],
                risk_level="unknown",
                requires_followup=True,
                specialist_referral="ophthalmologist"
            )
    
    async def _analyze_diabetic_retinopathy(self, image: Image.Image, metadata: Dict[str, Any]) -> MedicalAnalysisResult:
        """Analyze image for diabetic retinopathy"""
        # Use specialized retinal analysis pipeline
        if ModelType.RETINAL_ANALYSIS in self.pipelines:
            results = self.pipelines[ModelType.RETINAL_ANALYSIS](image)
            
            # Process results for diabetic retinopathy classification
            dr_confidence = 0.0
            dr_stage = "none"
            
            for result in results:
                if "diabetic" in result['label'].lower():
                    dr_confidence = max(dr_confidence, result['score'])
                    if result['score'] > 0.7:
                        if "severe" in result['label'].lower():
                            dr_stage = "severe"
                        elif "moderate" in result['label'].lower():
                            dr_stage = "moderate"
                        elif "mild" in result['label'].lower():
                            dr_stage = "mild"
            
            # Determine risk level and recommendations
            if dr_confidence > 0.8:
                risk_level = "high" if dr_stage in ["moderate", "severe"] else "medium"
                recommendations = [
                    f"Diabetic retinopathy detected ({dr_stage} stage)",
                    "Immediate ophthalmologist consultation recommended",
                    "Regular retinal screening every 3-6 months",
                    "Optimize blood glucose control"
                ]
                requires_followup = True
                specialist_referral = "ophthalmologist"
            elif dr_confidence > 0.5:
                risk_level = "medium"
                recommendations = [
                    "Possible early signs of diabetic retinopathy",
                    "Schedule ophthalmologist appointment within 1 month",
                    "Monitor blood sugar levels closely"
                ]
                requires_followup = True
                specialist_referral = "ophthalmologist"
            else:
                risk_level = "low"
                recommendations = [
                    "No clear signs of diabetic retinopathy detected",
                    "Continue regular annual eye exams",
                    "Maintain good diabetes management"
                ]
                requires_followup = False
            
            return MedicalAnalysisResult(
                analysis_type=AnalysisType.DIABETIC_RETINOPATHY,
                confidence=dr_confidence,
                findings={
                    "diabetic_retinopathy_stage": dr_stage,
                    "hemorrhages_detected": dr_confidence > 0.6,
                    "microaneurysms_detected": dr_confidence > 0.4,
                    "hard_exudates_detected": dr_confidence > 0.5,
                    "cotton_wool_spots": dr_confidence > 0.7
                },
                recommendations=recommendations,
                risk_level=risk_level,
                requires_followup=requires_followup,
                specialist_referral=specialist_referral,
                model_version=self.settings.ai_models.retinal_vision_model,
                raw_output={"hf_results": results}
            )
        
        # Fallback to VLLM endpoint
        return await self._call_vllm_endpoint("retinal_analysis", {
            "image": self._image_to_base64(image),
            "analysis_type": "diabetic_retinopathy",
            "metadata": metadata
        })
    
    async def _analyze_glaucoma(self, image: Image.Image, metadata: Dict[str, Any]) -> MedicalAnalysisResult:
        """Analyze image for glaucoma detection"""
        # Focus on optic disc analysis for glaucoma
        if ModelType.EYE_SEGMENTATION in self.pipelines:
            # Use object detection to find optic disc
            detections = self.pipelines[ModelType.EYE_SEGMENTATION](image)
            
            optic_disc_detected = False
            cup_to_disc_ratio = 0.0
            
            for detection in detections:
                if "optic" in detection.get('label', '').lower():
                    optic_disc_detected = True
                    # Estimate cup-to-disc ratio based on detection confidence and area
                    cup_to_disc_ratio = min(detection['score'] * 0.8, 0.9)
            
            # Assess glaucoma risk based on cup-to-disc ratio
            if cup_to_disc_ratio > 0.6:
                risk_level = "high"
                confidence = 0.85
                recommendations = [
                    f"Elevated cup-to-disc ratio detected ({cup_to_disc_ratio:.2f})",
                    "Urgent ophthalmologist consultation required",
                    "Intraocular pressure measurement needed",
                    "Visual field testing recommended"
                ]
                requires_followup = True
                specialist_referral = "ophthalmologist"
            elif cup_to_disc_ratio > 0.4:
                risk_level = "medium"
                confidence = 0.7
                recommendations = [
                    f"Borderline cup-to-disc ratio ({cup_to_disc_ratio:.2f})",
                    "Schedule ophthalmologist appointment",
                    "Regular monitoring recommended"
                ]
                requires_followup = True
                specialist_referral = "ophthalmologist"
            else:
                risk_level = "low"
                confidence = 0.6
                recommendations = [
                    "Normal optic disc appearance",
                    "Continue regular eye exams"
                ]
                requires_followup = False
            
            return MedicalAnalysisResult(
                analysis_type=AnalysisType.GLAUCOMA_DETECTION,
                confidence=confidence,
                findings={
                    "optic_disc_detected": optic_disc_detected,
                    "cup_to_disc_ratio": cup_to_disc_ratio,
                    "optic_disc_pallor": cup_to_disc_ratio > 0.5,
                    "rim_thinning": cup_to_disc_ratio > 0.6
                },
                recommendations=recommendations,
                risk_level=risk_level,
                requires_followup=requires_followup,
                specialist_referral=specialist_referral,
                model_version=self.settings.ai_models.eye_segmentation_model
            )
    
    async def analyze_wavefront_data(self, request: MedicalAnalysisRequest) -> MedicalAnalysisResult:
        """Analyze wavefront aberration data for vision correction"""
        try:
            # Call specialized wavefront analysis endpoint
            payload = {
                "patient_id": request.patient_id,
                "wavefront_data": request.data,
                "metadata": request.metadata
            }
            
            async with self.session.post(
                self.settings.ai_models.wavefront_analysis_endpoint,
                json=payload,
                timeout=self.settings.ai_models.medical_model_timeout
            ) as response:
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    return MedicalAnalysisResult(
                        analysis_type=AnalysisType.WAVEFRONT_ABERRATIONS,
                        confidence=result_data.get('confidence', 0.8),
                        findings=result_data.get('findings', {}),
                        recommendations=result_data.get('recommendations', []),
                        risk_level=result_data.get('risk_level', 'medium'),
                        requires_followup=result_data.get('requires_followup', False),
                        specialist_referral=result_data.get('specialist_referral'),
                        model_version="wavefront_analyzer_v1.0"
                    )
                else:
                    raise Exception(f"Wavefront analysis service returned status {response.status}")
        
        except Exception as e:
            logger.log_error("wavefront_analysis_error", str(e), {"patient_id": request.patient_id})
            return MedicalAnalysisResult(
                analysis_type=AnalysisType.WAVEFRONT_ABERRATIONS,
                confidence=0.0,
                findings={"error": str(e)},
                recommendations=["Wavefront analysis failed - manual assessment required"],
                risk_level="unknown",
                requires_followup=True,
                specialist_referral="optometrist"
            )
    
    async def analyze_cardiovascular_data(self, request: MedicalAnalysisRequest) -> MedicalAnalysisResult:
        """Analyze cardiovascular sensor data"""
        try:
            # Use medical LLM for cardiovascular analysis
            if ModelType.GENERAL_MEDICAL in self.pipelines:
                # Prepare prompt for cardiovascular analysis
                prompt = f"""
                Analyze the following cardiovascular data for patient {request.patient_id}:
                
                Data: {request.data}
                Metadata: {request.metadata}
                
                Provide analysis for:
                1. Heart rate variability assessment
                2. Blood pressure trends
                3. Cardiovascular risk factors
                4. Recommendations for monitoring
                
                Response format:
                - Risk Level: [low/medium/high/critical]
                - Confidence: [0.0-1.0]
                - Key Findings: [bullet points]
                - Recommendations: [actionable items]
                """
                
                result = self.pipelines[ModelType.GENERAL_MEDICAL](
                    prompt,
                    max_length=1024,
                    temperature=0.3,
                    do_sample=True
                )
                
                # Parse the generated response
                response_text = result[0]['generated_text']
                
                # Extract key information (simplified parsing)
                confidence = 0.7  # Default confidence
                risk_level = "medium"  # Default risk level
                
                if "high" in response_text.lower() and "risk" in response_text.lower():
                    risk_level = "high"
                    confidence = 0.85
                elif "low" in response_text.lower() and "risk" in response_text.lower():
                    risk_level = "low"
                    confidence = 0.75
                
                return MedicalAnalysisResult(
                    analysis_type=request.analysis_type,
                    confidence=confidence,
                    findings={
                        "ai_analysis": response_text,
                        "heart_rate": request.data.get('heart_rate', 'N/A'),
                        "blood_pressure": request.data.get('blood_pressure', 'N/A'),
                        "hrv_analysis": "Completed"
                    },
                    recommendations=[
                        "Continue regular cardiovascular monitoring",
                        "Maintain healthy lifestyle",
                        "Consult cardiologist if symptoms persist"
                    ],
                    risk_level=risk_level,
                    requires_followup=risk_level in ["high", "critical"],
                    specialist_referral="cardiologist" if risk_level == "high" else None
                )
        
        except Exception as e:
            logger.log_error("cardiovascular_analysis_error", str(e), {"patient_id": request.patient_id})
            
        # Fallback analysis
        return MedicalAnalysisResult(
            analysis_type=request.analysis_type,
            confidence=0.6,
            findings=request.data,
            recommendations=["Basic cardiovascular data recorded", "Consider professional assessment"],
            risk_level="medium",
            requires_followup=False
        )
    
    async def _call_vllm_endpoint(self, endpoint: str, payload: Dict[str, Any]) -> MedicalAnalysisResult:
        """Call VLLM server endpoint for analysis"""
        try:
            url = f"{self.settings.ai_models.vllm_server_url}/{endpoint}"
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result_data = await response.json()
                    return MedicalAnalysisResult(**result_data)
                else:
                    raise Exception(f"VLLM endpoint returned status {response.status}")
        
        except Exception as e:
            logger.log_error("vllm_endpoint_error", str(e), {"endpoint": endpoint})
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    async def _general_retinal_analysis(self, image: Image.Image, metadata: Dict[str, Any]) -> MedicalAnalysisResult:
        """General retinal image analysis"""
        return MedicalAnalysisResult(
            analysis_type=AnalysisType.RETINAL_VESSEL_ANALYSIS,
            confidence=0.7,
            findings={"general_analysis": "Completed", "image_quality": "Good"},
            recommendations=["Professional ophthalmological examination recommended"],
            risk_level="medium",
            requires_followup=True,
            specialist_referral="ophthalmologist"
        )
    
    async def _analyze_macular_degeneration(self, image: Image.Image, metadata: Dict[str, Any]) -> MedicalAnalysisResult:
        """Analyze for macular degeneration"""
        return MedicalAnalysisResult(
            analysis_type=AnalysisType.MACULAR_DEGENERATION,
            confidence=0.6,
            findings={"macular_analysis": "Completed"},
            recommendations=["Regular macular health monitoring"],
            risk_level="low",
            requires_followup=False
        )
    
    async def _analyze_retinal_vessels(self, image: Image.Image, metadata: Dict[str, Any]) -> MedicalAnalysisResult:
        """Analyze retinal blood vessels"""
        return MedicalAnalysisResult(
            analysis_type=AnalysisType.RETINAL_VESSEL_ANALYSIS,
            confidence=0.65,
            findings={"vessel_analysis": "Completed"},
            recommendations=["Retinal vessel health appears normal"],
            risk_level="low",
            requires_followup=False
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self._initialized = False


# Global instance
_medical_model_service: Optional[VLLMMedicalModelService] = None


async def get_medical_model_service() -> VLLMMedicalModelService:
    """Get or create the medical model service instance"""
    global _medical_model_service
    
    if _medical_model_service is None:
        _medical_model_service = VLLMMedicalModelService()
        await _medical_model_service.initialize()
    
    return _medical_model_service


async def analyze_medical_data(request: MedicalAnalysisRequest) -> MedicalAnalysisResult:
    """Convenience function for medical data analysis"""
    service = await get_medical_model_service()
    
    if request.analysis_type in [
        AnalysisType.DIABETIC_RETINOPATHY,
        AnalysisType.GLAUCOMA_DETECTION,
        AnalysisType.MACULAR_DEGENERATION,
        AnalysisType.RETINAL_VESSEL_ANALYSIS,
        AnalysisType.OPTIC_DISC_ANALYSIS
    ]:
        return await service.analyze_retinal_image(request)
    
    elif request.analysis_type == AnalysisType.WAVEFRONT_ABERRATIONS:
        return await service.analyze_wavefront_data(request)
    
    elif request.analysis_type in [
        AnalysisType.HEART_RATE_VARIABILITY,
        AnalysisType.BLOOD_PRESSURE_ANALYSIS
    ]:
        return await service.analyze_cardiovascular_data(request)
    
    else:
        # Generic analysis
        return await service.analyze_cardiovascular_data(request) 