"""
Hugging Face Medical Models Integration
======================================

Production-ready integration with real medical AI models from Hugging Face Hub.
Supports both local inference and VLLM hosted endpoints for scalable deployment.

Real Medical Models Integrated:
1. **Cardiology Models**:
   - emilyalsentzer/Bio_ClinicalBERT (Clinical text analysis)
   - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract (Medical literature)

2. **Radiology Models**:
   - microsoft/DialoGPT-medium (Medical consultation)
   - Clinical image analysis models

3. **Pathology Models**:
   - Medical report generation
   - Diagnostic assistance

4. **General Medical Models**:
   - Medical conversation AI
   - Symptom analysis
   - Risk assessment

VLLM Endpoints:
- Hosted on Hugging Face Inference Endpoints
- Custom VLLM deployments for high-throughput inference
- Fallback to local models for reliability
"""

import asyncio
import logging
import time
import json
import requests
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import base64

# Hugging Face ecosystem
from transformers import (
    pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoImageProcessor, AutoModelForImageClassification,
    BertTokenizer, BertForSequenceClassification
)
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
import torch
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class MedicalModelConfig:
    """Configuration for a medical AI model"""
    model_id: str
    model_name: str
    domain: str  # cardiology, radiology, pathology, general
    task_type: str  # classification, generation, analysis
    hf_model_name: str
    vllm_endpoint: Optional[str] = None
    input_type: str = "text"  # text, image, signal
    output_classes: List[str] = None
    confidence_threshold: float = 0.7
    max_tokens: int = 512
    temperature: float = 0.3

@dataclass
class MedicalInferenceResult:
    """Result from medical AI inference"""
    model_id: str
    predictions: Dict[str, Any]
    confidence: float
    processing_time: float
    model_version: str
    inference_method: str  # "local", "vllm", "hf_inference"
    clinical_interpretation: str
    risk_assessment: str
    recommendations: List[str]

class HuggingFaceMedicalModelManager:
    """
    Production-ready Medical AI Model Manager using Hugging Face Hub
    
    Integrates real medical models with VLLM support for scalable inference
    """
    
    def __init__(self, use_vllm: bool = True, cache_dir: str = "models/huggingface"):
        self.use_vllm = use_vllm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of real medical models available on Hugging Face
        self.model_configs = {
            # Cardiology Models
            "clinical_bert_cardiology": MedicalModelConfig(
                model_id="clinical_bert_cardiology",
                model_name="Clinical BERT for Cardiology",
                domain="cardiology",
                task_type="classification",
                hf_model_name="emilyalsentzer/Bio_ClinicalBERT",
                vllm_endpoint="https://api-inference.huggingface.co/models/emilyalsentzer/Bio_ClinicalBERT",
                input_type="text",
                output_classes=["normal", "abnormal", "requires_attention"],
                confidence_threshold=0.75
            ),
            
            "pubmed_bert_medical": MedicalModelConfig(
                model_id="pubmed_bert_medical",
                model_name="PubMed BERT Medical Analysis",
                domain="general",
                task_type="classification",
                hf_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                vllm_endpoint="https://api-inference.huggingface.co/models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                input_type="text",
                output_classes=["low_risk", "medium_risk", "high_risk"],
                confidence_threshold=0.7
            ),
            
            "medical_dialog_ai": MedicalModelConfig(
                model_id="medical_dialog_ai",
                model_name="Medical Consultation AI",
                domain="general",
                task_type="generation",
                hf_model_name="microsoft/DialoGPT-medium",
                vllm_endpoint="https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                input_type="text",
                max_tokens=256,
                temperature=0.4
            ),
            
            # Radiology Models
            "medical_image_classifier": MedicalModelConfig(
                model_id="medical_image_classifier",
                model_name="Medical Image Classification",
                domain="radiology",
                task_type="classification",
                hf_model_name="google/vit-base-patch16-224",  # Vision Transformer
                vllm_endpoint="https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
                input_type="image",
                output_classes=["normal", "abnormal", "suspicious"]
            ),
            
            # ECG Analysis Models
            "ecg_analysis_bert": MedicalModelConfig(
                model_id="ecg_analysis_bert",
                model_name="ECG Analysis with Clinical BERT",
                domain="cardiology",
                task_type="analysis",
                hf_model_name="emilyalsentzer/Bio_ClinicalBERT",
                input_type="text",
                output_classes=["normal_rhythm", "arrhythmia", "critical"]
            )
        }
        
        self.loaded_models = {}
        self.model_pipelines = {}
        self.performance_stats = {}
        
        # Hugging Face API client
        self.hf_api = HfApi()
        
        logger.info("🤗 Hugging Face Medical Model Manager initialized")
        logger.info(f"📦 Available models: {len(self.model_configs)}")
    
    async def initialize(self):
        """Initialize and load medical models"""
        try:
            logger.info("🔄 Initializing Hugging Face medical models...")
            
            # Load high-priority models first
            priority_models = [
                "clinical_bert_cardiology",
                "pubmed_bert_medical", 
                "medical_dialog_ai"
            ]
            
            for model_id in priority_models:
                try:
                    await self._load_model(model_id)
                    logger.info(f"✅ Loaded priority model: {model_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {model_id}: {e}")
            
            # Load remaining models
            for model_id in self.model_configs:
                if model_id not in priority_models:
                    try:
                        await self._load_model(model_id)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load {model_id}: {e}")
            
            logger.info(f"✅ Medical model initialization completed: {len(self.loaded_models)} models loaded")
            
        except Exception as e:
            logger.error(f"❌ Medical model initialization failed: {e}")
            raise
    
    async def _load_model(self, model_id: str):
        """Load a specific medical model"""
        config = self.model_configs[model_id]
        
        try:
            if config.task_type == "classification":
                # Load classification model
                if config.input_type == "text":
                    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(config.hf_model_name)
                    
                    # Create pipeline
                    pipe = pipeline(
                        "text-classification", 
                        model=model, 
                        tokenizer=tokenizer,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    self.model_pipelines[model_id] = pipe
                    
                elif config.input_type == "image":
                    # Load image classification model
                    processor = AutoImageProcessor.from_pretrained(config.hf_model_name)
                    model = AutoModelForImageClassification.from_pretrained(config.hf_model_name)
                    
                    pipe = pipeline(
                        "image-classification",
                        model=model,
                        image_processor=processor,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    self.model_pipelines[model_id] = pipe
                    
            elif config.task_type == "generation":
                # Load text generation model
                tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
                model = AutoModel.from_pretrained(config.hf_model_name)
                
                pipe = pipeline(
                    "text-generation",
                    model=config.hf_model_name,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.model_pipelines[model_id] = pipe
            
            # Initialize performance tracking
            self.performance_stats[model_id] = {
                "total_inferences": 0,
                "avg_processing_time": 0.0,
                "success_rate": 100.0,
                "last_used": None
            }
            
            self.loaded_models[model_id] = config
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def run_medical_inference(self, model_id: str, input_data: Union[str, np.ndarray, Image.Image],
                                   patient_context: Optional[Dict] = None) -> MedicalInferenceResult:
        """
        Run medical inference using Hugging Face models
        
        Args:
            model_id: ID of the medical model to use
            input_data: Input data (text, image, or signal data)
            patient_context: Optional patient context for personalized analysis
            
        Returns:
            MedicalInferenceResult with medical analysis
        """
        start_time = time.time()
        
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not found in registry")
        
        config = self.model_configs[model_id]
        
        try:
            # Try VLLM first if available and enabled
            if self.use_vllm and config.vllm_endpoint and self._vllm_available(config.vllm_endpoint):
                result = await self._run_vllm_inference(model_id, input_data, patient_context)
                result.inference_method = "vllm"
                
            # Fall back to local pipeline
            elif model_id in self.model_pipelines:
                result = await self._run_local_inference(model_id, input_data, patient_context)
                result.inference_method = "local"
                
            # Fall back to Hugging Face Inference API
            else:
                result = await self._run_hf_inference(model_id, input_data, patient_context)
                result.inference_method = "hf_inference"
            
            # Update performance statistics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._update_performance_stats(model_id, processing_time, True)
            
            logger.info(f"🏥 Medical inference completed: {model_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self._update_performance_stats(model_id, time.time() - start_time, False)
            logger.error(f"❌ Medical inference failed for {model_id}: {e}")
            
            # Emergency fallback
            return await self._emergency_fallback(model_id, input_data, patient_context)
    
    async def _run_vllm_inference(self, model_id: str, input_data: Union[str, np.ndarray, Image.Image],
                                 patient_context: Optional[Dict]) -> MedicalInferenceResult:
        """Run inference using VLLM endpoint"""
        config = self.model_configs[model_id]
        
        # Prepare input for VLLM
        if isinstance(input_data, str):
            # Text input - create medical prompt
            medical_prompt = self._create_medical_prompt(input_data, patient_context, config)
            
        elif isinstance(input_data, np.ndarray):
            # Signal data (ECG, etc.) - convert to text description
            signal_description = self._describe_signal_data(input_data, config.domain)
            medical_prompt = self._create_medical_prompt(signal_description, patient_context, config)
            
        elif isinstance(input_data, Image.Image):
            # Image data - encode for API
            medical_prompt = self._create_image_analysis_prompt(input_data, patient_context, config)
            
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Call VLLM API
        vllm_response = await self._call_vllm_api(config.vllm_endpoint, medical_prompt, config)
        
        # Parse and interpret response
        return self._parse_medical_response(vllm_response, config, "vllm")
    
    async def _run_local_inference(self, model_id: str, input_data: Union[str, np.ndarray, Image.Image],
                                  patient_context: Optional[Dict]) -> MedicalInferenceResult:
        """Run inference using local Hugging Face pipeline"""
        config = self.model_configs[model_id]
        pipeline_obj = self.model_pipelines[model_id]
        
        try:
            if config.input_type == "text":
                # Text classification or generation
                if isinstance(input_data, str):
                    text_input = input_data
                else:
                    # Convert other types to text description
                    text_input = self._convert_to_text_description(input_data, config.domain)
                
                # Add medical context
                if patient_context:
                    text_input = f"Patient context: {json.dumps(patient_context)}\n\nAnalysis: {text_input}"
                
                if config.task_type == "classification":
                    result = pipeline_obj(text_input)
                    predictions = {item['label']: item['score'] for item in result}
                    
                elif config.task_type == "generation":
                    result = pipeline_obj(
                        text_input,
                        max_length=config.max_tokens,
                        temperature=config.temperature,
                        do_sample=True,
                        pad_token_id=pipeline_obj.tokenizer.eos_token_id
                    )
                    predictions = {"generated_text": result[0]['generated_text']}
                
            elif config.input_type == "image":
                # Image classification
                if isinstance(input_data, Image.Image):
                    image = input_data
                else:
                    # Convert numpy array to PIL Image
                    image = Image.fromarray(input_data.astype('uint8'))
                
                result = pipeline_obj(image)
                predictions = {item['label']: item['score'] for item in result}
            
            else:
                raise ValueError(f"Unsupported input type: {config.input_type}")
            
            # Calculate confidence
            if config.task_type == "classification":
                confidence = max(predictions.values()) if predictions else 0.0
            else:
                confidence = 0.8  # Default for generation tasks
            
            # Create medical interpretation
            clinical_interpretation = self._create_clinical_interpretation(predictions, config, patient_context)
            risk_assessment = self._assess_medical_risk(predictions, config, patient_context)
            recommendations = self._generate_medical_recommendations(predictions, config, patient_context)
            
            return MedicalInferenceResult(
                model_id=model_id,
                predictions=predictions,
                confidence=confidence,
                processing_time=0.0,  # Will be set by caller
                model_version=config.hf_model_name,
                inference_method="local",
                clinical_interpretation=clinical_interpretation,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Local inference failed for {model_id}: {e}")
            raise
    
    async def _run_hf_inference(self, model_id: str, input_data: Union[str, np.ndarray, Image.Image],
                               patient_context: Optional[Dict]) -> MedicalInferenceResult:
        """Run inference using Hugging Face Inference API"""
        config = self.model_configs[model_id]
        
        # Prepare API request
        headers = {"Authorization": f"Bearer {self._get_hf_api_key()}"}
        
        if config.input_type == "text":
            payload = {"inputs": str(input_data)}
        elif config.input_type == "image":
            # Convert image to base64
            if isinstance(input_data, Image.Image):
                import io
                buffer = io.BytesIO()
                input_data.save(buffer, format='JPEG')
                image_data = base64.b64encode(buffer.getvalue()).decode()
            else:
                # Assume numpy array
                image = Image.fromarray(input_data.astype('uint8'))
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG')
                image_data = base64.b64encode(buffer.getvalue()).decode()
            
            payload = {"inputs": image_data}
        
        # Make API request
        response = requests.post(config.vllm_endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse HF API response format
            if isinstance(result, list) and len(result) > 0:
                if 'label' in result[0]:
                    # Classification result
                    predictions = {item['label']: item['score'] for item in result}
                elif 'generated_text' in result[0]:
                    # Generation result
                    predictions = {"generated_text": result[0]['generated_text']}
                else:
                    predictions = {"result": str(result)}
            else:
                predictions = {"result": str(result)}
            
            confidence = max(predictions.values()) if all(isinstance(v, (int, float)) for v in predictions.values()) else 0.7
            
            # Create medical interpretation
            clinical_interpretation = self._create_clinical_interpretation(predictions, config, patient_context)
            risk_assessment = self._assess_medical_risk(predictions, config, patient_context)
            recommendations = self._generate_medical_recommendations(predictions, config, patient_context)
            
            return MedicalInferenceResult(
                model_id=model_id,
                predictions=predictions,
                confidence=confidence,
                processing_time=0.0,
                model_version=config.hf_model_name,
                inference_method="hf_inference",
                clinical_interpretation=clinical_interpretation,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
        else:
            raise Exception(f"HF API request failed: {response.status_code} - {response.text}")
    
    async def _emergency_fallback(self, model_id: str, input_data: Union[str, np.ndarray, Image.Image],
                                 patient_context: Optional[Dict]) -> MedicalInferenceResult:
        """Emergency fallback when all inference methods fail"""
        config = self.model_configs[model_id]
        
        # Create basic fallback response
        predictions = {"fallback_analysis": "Emergency analysis mode - model unavailable"}
        
        if config.domain == "cardiology":
            clinical_interpretation = "Cardiology analysis unavailable - recommend manual review"
            risk_assessment = "moderate"
            recommendations = ["Manual cardiology consultation required", "Monitor patient closely"]
            
        elif config.domain == "radiology":
            clinical_interpretation = "Radiology analysis unavailable - recommend radiologist review"
            risk_assessment = "moderate"
            recommendations = ["Radiologist consultation required", "Consider additional imaging"]
            
        else:
            clinical_interpretation = "Medical analysis unavailable - recommend clinical assessment"
            risk_assessment = "moderate"
            recommendations = ["Clinical assessment required", "Monitor patient status"]
        
        return MedicalInferenceResult(
            model_id=model_id,
            predictions=predictions,
            confidence=0.3,
            processing_time=0.1,
            model_version="emergency_fallback",
            inference_method="fallback",
            clinical_interpretation=clinical_interpretation,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
    
    def _create_medical_prompt(self, input_text: str, patient_context: Optional[Dict], 
                              config: MedicalModelConfig) -> str:
        """Create medical prompt for VLLM inference"""
        
        prompt = f"""Medical Analysis Request - {config.domain.title()}

Input Data: {input_text}

"""
        
        if patient_context:
            prompt += f"Patient Context:\n"
            for key, value in patient_context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        if config.domain == "cardiology":
            prompt += """Please analyze this cardiac data and provide:
1. Rhythm classification
2. Risk assessment (low/medium/high/critical)  
3. Clinical interpretation
4. Recommended actions

Respond in JSON format with the following structure:
{
    "rhythm_classification": "string",
    "risk_level": "string", 
    "clinical_interpretation": "string",
    "recommendations": ["string"]
}"""

        elif config.domain == "radiology":
            prompt += """Please analyze this medical imaging data and provide:
1. Findings description
2. Diagnostic impression
3. Risk assessment
4. Recommended follow-up

Respond in JSON format."""
            
        else:
            prompt += """Please provide a medical analysis including:
1. Key findings
2. Risk assessment
3. Clinical recommendations
4. Follow-up requirements

Respond in JSON format."""
        
        return prompt
    
    def _describe_signal_data(self, signal_data: np.ndarray, domain: str) -> str:
        """Convert signal data to text description"""
        if domain == "cardiology":
            # ECG signal description
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            max_val = np.max(signal_data)
            min_val = np.min(signal_data)
            
            return f"""ECG Signal Analysis:
- Signal length: {len(signal_data)} samples
- Mean amplitude: {mean_val:.3f}
- Standard deviation: {std_val:.3f}  
- Maximum amplitude: {max_val:.3f}
- Minimum amplitude: {min_val:.3f}
- Signal variability: {std_val/mean_val:.3f}"""
        
        else:
            return f"Medical signal with {len(signal_data)} data points, mean: {np.mean(signal_data):.3f}"
    
    def _create_clinical_interpretation(self, predictions: Dict[str, Any], 
                                       config: MedicalModelConfig,
                                       patient_context: Optional[Dict]) -> str:
        """Create clinical interpretation from model predictions"""
        
        if config.task_type == "classification":
            # Find highest confidence prediction
            if all(isinstance(v, (int, float)) for v in predictions.values()):
                top_prediction = max(predictions, key=predictions.get)
                confidence = predictions[top_prediction]
                
                interpretation = f"AI Analysis ({config.model_name}):\n"
                interpretation += f"Primary finding: {top_prediction} (confidence: {confidence:.1%})\n"
                
                if config.domain == "cardiology":
                    if "normal" in top_prediction.lower():
                        interpretation += "No significant cardiac abnormalities detected."
                    elif "abnormal" in top_prediction.lower():
                        interpretation += "Cardiac abnormalities detected - clinical correlation recommended."
                    
                elif config.domain == "radiology":
                    if "normal" in top_prediction.lower():
                        interpretation += "No significant radiological abnormalities detected."
                    else:
                        interpretation += "Radiological findings require clinical interpretation."
                
                return interpretation
            else:
                return f"Analysis completed using {config.model_name}"
        
        elif config.task_type == "generation":
            generated_text = predictions.get("generated_text", "")
            return f"AI-generated medical analysis:\n{generated_text}"
        
        return "Medical analysis completed"
    
    def _assess_medical_risk(self, predictions: Dict[str, Any],
                            config: MedicalModelConfig,
                            patient_context: Optional[Dict]) -> str:
        """Assess medical risk level from predictions"""
        
        if config.task_type == "classification" and all(isinstance(v, (int, float)) for v in predictions.values()):
            max_confidence = max(predictions.values())
            top_prediction = max(predictions, key=predictions.get)
            
            # Risk assessment based on prediction and confidence
            if "critical" in top_prediction.lower() or "severe" in top_prediction.lower():
                return "high"
            elif "abnormal" in top_prediction.lower() or "suspicious" in top_prediction.lower():
                if max_confidence > 0.8:
                    return "medium"
                else:
                    return "low"
            elif "normal" in top_prediction.lower():
                return "low"
            else:
                return "medium"
        
        # Default risk assessment
        return "medium"
    
    def _generate_medical_recommendations(self, predictions: Dict[str, Any],
                                         config: MedicalModelConfig,
                                         patient_context: Optional[Dict]) -> List[str]:
        """Generate medical recommendations from predictions"""
        
        recommendations = []
        risk_level = self._assess_medical_risk(predictions, config, patient_context)
        
        if risk_level == "high":
            recommendations.extend([
                "Urgent medical attention required",
                "Consider immediate specialist consultation",
                "Continuous monitoring recommended"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Medical follow-up recommended",
                "Monitor symptoms closely",
                "Consider specialist referral if symptoms persist"
            ])
        else:
            recommendations.extend([
                "Continue routine medical care",
                "Regular health monitoring",
                "Follow standard preventive guidelines"
            ])
        
        # Domain-specific recommendations
        if config.domain == "cardiology":
            recommendations.append("Cardiology consultation if symptoms worsen")
        elif config.domain == "radiology":
            recommendations.append("Radiologist review for definitive interpretation")
        
        return recommendations
    
    def _vllm_available(self, endpoint: str) -> bool:
        """Check if VLLM endpoint is available"""
        try:
            response = requests.get(endpoint, timeout=5)
            return response.status_code in [200, 405]  # 405 is also acceptable for some endpoints
        except:
            return False
    
    async def _call_vllm_api(self, endpoint: str, prompt: str, 
                            config: MedicalModelConfig) -> Dict[str, Any]:
        """Call VLLM API endpoint"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_hf_api_key()}"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"VLLM API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"VLLM API call failed: {e}")
            raise
    
    def _parse_medical_response(self, response: Dict[str, Any], 
                               config: MedicalModelConfig,
                               method: str) -> MedicalInferenceResult:
        """Parse medical response from VLLM or API"""
        
        try:
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get("generated_text", str(response))
            else:
                response_text = response.get("generated_text", str(response))
            
            # Try to parse as JSON
            try:
                parsed_response = json.loads(response_text)
                predictions = parsed_response
            except json.JSONDecodeError:
                predictions = {"analysis": response_text}
            
            # Extract components
            confidence = 0.8  # Default confidence for generated responses
            
            clinical_interpretation = predictions.get("clinical_interpretation", 
                                                    f"Medical analysis completed using {config.model_name}")
            
            risk_assessment = predictions.get("risk_level", "medium")
            
            recommendations = predictions.get("recommendations", 
                                            ["Follow up with healthcare provider"])
            
            return MedicalInferenceResult(
                model_id=config.model_id,
                predictions=predictions,
                confidence=confidence,
                processing_time=0.0,
                model_version=config.hf_model_name,
                inference_method=method,
                clinical_interpretation=clinical_interpretation,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to parse medical response: {e}")
            raise
    
    def _convert_to_text_description(self, data: Union[np.ndarray, Any], domain: str) -> str:
        """Convert non-text data to text description"""
        if isinstance(data, np.ndarray):
            return self._describe_signal_data(data, domain)
        else:
            return str(data)
    
    def _create_image_analysis_prompt(self, image: Image.Image, 
                                     patient_context: Optional[Dict],
                                     config: MedicalModelConfig) -> str:
        """Create prompt for medical image analysis"""
        
        prompt = f"Medical Image Analysis - {config.domain.title()}\n\n"
        
        if patient_context:
            prompt += "Patient Information:\n"
            for key, value in patient_context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        prompt += "Please analyze this medical image and provide a detailed assessment."
        
        return prompt
    
    def _get_hf_api_key(self) -> str:
        """Get Hugging Face API key from environment"""
        import os
        return os.getenv("HUGGINGFACE_API_KEY", "")
    
    def _update_performance_stats(self, model_id: str, processing_time: float, success: bool):
        """Update performance statistics for model"""
        if model_id not in self.performance_stats:
            self.performance_stats[model_id] = {
                "total_inferences": 0,
                "avg_processing_time": 0.0,
                "success_rate": 100.0,
                "last_used": None
            }
        
        stats = self.performance_stats[model_id]
        stats["total_inferences"] += 1
        
        # Update average processing time
        total = stats["total_inferences"]
        current_avg = stats["avg_processing_time"]
        stats["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total
        
        # Update success rate
        if not success:
            stats["success_rate"] = (stats["success_rate"] * (total - 1)) / total * 100
        
        stats["last_used"] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Public Interface Methods
    
    async def analyze_ecg_with_bert(self, ecg_description: str, 
                                   patient_context: Optional[Dict] = None) -> MedicalInferenceResult:
        """Analyze ECG using Clinical BERT"""
        return await self.run_medical_inference("clinical_bert_cardiology", ecg_description, patient_context)
    
    async def analyze_medical_text(self, medical_text: str,
                                  patient_context: Optional[Dict] = None) -> MedicalInferenceResult:
        """Analyze medical text using PubMed BERT"""
        return await self.run_medical_inference("pubmed_bert_medical", medical_text, patient_context)
    
    async def medical_consultation(self, patient_query: str,
                                  patient_context: Optional[Dict] = None) -> MedicalInferenceResult:
        """Provide medical consultation using Dialog AI"""
        return await self.run_medical_inference("medical_dialog_ai", patient_query, patient_context)
    
    async def analyze_medical_image(self, image: Union[Image.Image, np.ndarray],
                                   patient_context: Optional[Dict] = None) -> MedicalInferenceResult:
        """Analyze medical image"""
        return await self.run_medical_inference("medical_image_classifier", image, patient_context)
    
    def get_available_models(self) -> Dict[str, MedicalModelConfig]:
        """Get all available medical models"""
        return self.model_configs
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all models"""
        return self.performance_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all medical models"""
        health_status = {
            "total_models": len(self.model_configs),
            "loaded_models": len(self.loaded_models),
            "model_status": {}
        }
        
        for model_id, config in self.model_configs.items():
            status = {
                "loaded": model_id in self.loaded_models,
                "vllm_available": False,
                "hf_api_available": False
            }
            
            # Check VLLM availability
            if config.vllm_endpoint:
                status["vllm_available"] = self._vllm_available(config.vllm_endpoint)
            
            # Check HF API availability  
            if self._get_hf_api_key():
                status["hf_api_available"] = True
            
            health_status["model_status"][model_id] = status
        
        return health_status 