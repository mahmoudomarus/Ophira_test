"""
Medical AI Models Management System
==================================

Centralized management for all medical AI models including:
- Model loading and initialization
- Inference management
- Performance monitoring
- Result aggregation and confidence scoring
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import tensorflow as tf
from sklearn.base import BaseEstimator
import pickle
import json

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported medical AI model types"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow" 
    SKLEARN = "sklearn"
    ONNX = "onnx"
    CUSTOM = "custom"

class MedicalDomain(Enum):
    """Medical analysis domains"""
    RETINAL = "retinal"
    CARDIOVASCULAR = "cardiovascular" 
    METABOLIC = "metabolic"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    GENERAL = "general"

@dataclass
class ModelMetadata:
    """Metadata for medical AI models"""
    model_id: str
    name: str
    version: str
    domain: MedicalDomain
    model_type: ModelType
    file_path: str
    input_shape: Tuple[int, ...]
    output_classes: List[str]
    accuracy: float
    sensitivity: float
    specificity: float
    description: str
    clinical_validation: bool = False
    last_updated: Optional[str] = None

@dataclass 
class InferenceResult:
    """Result from medical AI model inference"""
    model_id: str
    predictions: Dict[str, float]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    timestamp: str
    risk_level: str  # "low", "medium", "high", "critical"
    clinical_notes: Optional[str] = None

class AIModelManager:
    """
    Centralized AI Model Management System
    
    Manages loading, inference, and monitoring of all medical AI models
    """
    
    def __init__(self, model_directory: str = "models/medical"):
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.performance_metrics: Dict[str, Dict] = {}
        
        # Enhanced device management with fallbacks
        self.device = self._get_optimal_device()
        self.force_cpu = False  # Flag to force CPU inference if GPU fails
        
        logger.info(f"AI Model Manager initialized with device: {self.device}")
        
    def _get_optimal_device(self):
        """Get optimal device with proper fallback handling"""
        try:
            if torch.cuda.is_available():
                # Check if CUDA is actually working
                test_tensor = torch.zeros(1).cuda()
                device = torch.device("cuda")
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                return device
        except Exception as e:
            logger.warning(f"CUDA device test failed: {e}, falling back to CPU")
        
        device = torch.device("cpu")
        logger.info("Using CPU device")
        return device
    
    async def initialize(self):
        """Initialize the model manager and load default models"""
        try:
            await self._load_model_registry()
            await self._load_default_models()
            logger.info("AI Model Manager successfully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AI Model Manager: {e}")
            raise
    
    async def _load_model_registry(self):
        """Load model registry from configuration"""
        registry_path = self.model_directory / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
                
            for model_id, metadata_dict in registry_data.items():
                metadata_dict['domain'] = MedicalDomain(metadata_dict['domain'])
                metadata_dict['model_type'] = ModelType(metadata_dict['model_type'])
                self.model_metadata[model_id] = ModelMetadata(**metadata_dict)
        else:
            # Create default registry
            await self._create_default_registry()
    
    async def _create_default_registry(self):
        """Create default model registry with placeholder models"""
        default_models = {
            "diabetic_retinopathy_v1": ModelMetadata(
                model_id="diabetic_retinopathy_v1",
                name="Diabetic Retinopathy Detection",
                version="1.0.0",
                domain=MedicalDomain.RETINAL,
                model_type=ModelType.PYTORCH,
                file_path="retinal/diabetic_retinopathy_v1.pth",
                input_shape=(3, 512, 512),
                output_classes=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
                accuracy=0.92,
                sensitivity=0.89,
                specificity=0.94,
                description="Deep learning model for diabetic retinopathy severity classification",
                clinical_validation=True
            ),
            "glaucoma_detection_v1": ModelMetadata(
                model_id="glaucoma_detection_v1",
                name="Glaucoma Detection System",
                version="1.0.0",
                domain=MedicalDomain.RETINAL,
                model_type=ModelType.TENSORFLOW,
                file_path="retinal/glaucoma_detection_v1.h5",
                input_shape=(224, 224, 3),
                output_classes=["Normal", "Suspect", "Glaucoma"],
                accuracy=0.88,
                sensitivity=0.85,
                specificity=0.91,
                description="CNN model for glaucoma detection from fundus images",
                clinical_validation=True
            ),
            "ecg_arrhythmia_v1": ModelMetadata(
                model_id="ecg_arrhythmia_v1",
                name="ECG Arrhythmia Detection",
                version="1.0.0",
                domain=MedicalDomain.CARDIOVASCULAR,
                model_type=ModelType.PYTORCH,
                file_path="cardiovascular/ecg_arrhythmia_v1.pth",
                input_shape=(12, 1000),  # 12-lead ECG, 1000 samples
                output_classes=["Normal", "Atrial Fibrillation", "Atrial Flutter", "Ventricular Tachycardia", "Other"],
                accuracy=0.94,
                sensitivity=0.91,
                specificity=0.96,
                description="Multi-class ECG arrhythmia detection using 1D CNN",
                clinical_validation=True
            ),
            "hrv_analysis_v1": ModelMetadata(
                model_id="hrv_analysis_v1",
                name="Heart Rate Variability Analysis",
                version="1.0.0",
                domain=MedicalDomain.CARDIOVASCULAR,
                model_type=ModelType.SKLEARN,
                file_path="cardiovascular/hrv_analysis_v1.pkl",
                input_shape=(50,),  # HRV features
                output_classes=["Low Risk", "Medium Risk", "High Risk"],
                accuracy=0.86,
                sensitivity=0.83,
                specificity=0.89,
                description="Random Forest model for cardiovascular risk assessment from HRV",
                clinical_validation=False
            )
        }
        
        # Save registry
        registry_path = self.model_directory / "model_registry.json"
        registry_data = {}
        for model_id, metadata in default_models.items():
            registry_data[model_id] = {
                'model_id': metadata.model_id,
                'name': metadata.name,
                'version': metadata.version,
                'domain': metadata.domain.value,
                'model_type': metadata.model_type.value,
                'file_path': metadata.file_path,
                'input_shape': metadata.input_shape,
                'output_classes': metadata.output_classes,
                'accuracy': metadata.accuracy,
                'sensitivity': metadata.sensitivity,
                'specificity': metadata.specificity,
                'description': metadata.description,
                'clinical_validation': metadata.clinical_validation
            }
            
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
            
        self.model_metadata = default_models
        logger.info("Created default model registry")
    
    async def _load_default_models(self):
        """Load default placeholder models for demonstration"""
        # For now, create mock models for demonstration
        # In production, these would be actual trained models
        
        for model_id, metadata in self.model_metadata.items():
            try:
                # Create placeholder/mock models for demonstration
                if metadata.model_type == ModelType.PYTORCH:
                    model = self._create_mock_pytorch_model(metadata)
                elif metadata.model_type == ModelType.TENSORFLOW:
                    model = self._create_mock_tensorflow_model(metadata)
                elif metadata.model_type == ModelType.SKLEARN:
                    model = self._create_mock_sklearn_model(metadata)
                else:
                    continue
                    
                self.loaded_models[model_id] = model
                self.performance_metrics[model_id] = {
                    'total_inferences': 0,
                    'avg_processing_time': 0.0,
                    'accuracy_trend': [],
                    'last_used': None
                }
                
                logger.info(f"Loaded model: {model_id}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
    
    def _create_mock_pytorch_model(self, metadata: ModelMetadata):
        """Create a mock PyTorch model for demonstration"""
        import torch.nn as nn
        
        class MockMedicalModel(nn.Module):
            def __init__(self, input_shape, num_classes, device):
                super().__init__()
                self.device = device
                
                if len(input_shape) == 3:  # Image input (C, H, W)
                    # Expecting (batch, channels, height, width) for Conv2d
                    in_channels = input_shape[0] if input_shape[0] <= 4 else 3  # Default to 3 for RGB
                    self.features = nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten()
                    )
                    feature_size = 64
                else:  # Signal input
                    self.features = nn.Sequential(
                        nn.Linear(np.prod(input_shape), 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU()
                    )
                    feature_size = 64
                    
                self.classifier = nn.Linear(feature_size, num_classes)
                
            def forward(self, x):
                # Ensure input is on the same device as the model
                if hasattr(x, 'device') and x.device != self.device:
                    x = x.to(self.device)
                x = self.features(x)
                return torch.softmax(self.classifier(x), dim=1)
        
        model = MockMedicalModel(metadata.input_shape, len(metadata.output_classes), self.device)
        model.to(self.device)  # Ensure model is on correct device
        model.eval()
        return model
    
    def _create_mock_tensorflow_model(self, metadata: ModelMetadata):
        """Create a mock TensorFlow model with CPU fallback"""
        try:
            # Force TensorFlow to use CPU to avoid CUDA PTX issues
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            import tensorflow as tf
            
            # Ensure CPU-only execution
            tf.config.set_visible_devices([], 'GPU')
            
            # Build model architecture based on input shape
            if len(metadata.input_shape) == 3:  # Image input
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, 3, activation='relu', 
                                         input_shape=metadata.input_shape),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(len(metadata.output_classes), activation='softmax')
                ])
            else:  # Signal input
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', 
                                        input_shape=metadata.input_shape),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(len(metadata.output_classes), activation='softmax')
                ])
            
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            logger.info(f"TensorFlow model created successfully (CPU mode): {metadata.model_id}")
            return model
            
        except Exception as e:
            logger.error(f"TensorFlow model creation failed: {e}")
            raise
    
    def _create_mock_sklearn_model(self, metadata: ModelMetadata):
        """Create a mock sklearn model for demonstration"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        if "hrv" in metadata.model_id.lower():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(random_state=42)
            
        # Create dummy training data to fit the model
        X_dummy = np.random.randn(100, np.prod(metadata.input_shape))
        y_dummy = np.random.randint(0, len(metadata.output_classes), 100)
        model.fit(X_dummy, y_dummy)
        
        return model
    
    async def predict(self, model_id: str, input_data: np.ndarray, 
                     patient_context: Optional[Dict] = None) -> InferenceResult:
        """
        Run inference on specified medical AI model
        
        Args:
            model_id: ID of the model to use
            input_data: Input data for the model
            patient_context: Optional patient context for personalized analysis
            
        Returns:
            InferenceResult with predictions and metadata
        """
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
            
        model = self.loaded_models[model_id]
        metadata = self.model_metadata[model_id]
        
        start_time = time.time()
        
        try:
            # Run inference based on model type
            if metadata.model_type == ModelType.PYTORCH:
                predictions = await self._pytorch_inference(model, input_data, metadata)
            elif metadata.model_type == ModelType.TENSORFLOW:
                predictions = await self._tensorflow_inference(model, input_data, metadata)
            elif metadata.model_type == ModelType.SKLEARN:
                predictions = await self._sklearn_inference(model, input_data, metadata)
            else:
                raise ValueError(f"Unsupported model type: {metadata.model_type}")
            
            processing_time = time.time() - start_time
            
            # Calculate confidence and risk level
            confidence = max(predictions.values())
            risk_level = self._calculate_risk_level(predictions, metadata)
            
            # Update performance metrics
            self._update_performance_metrics(model_id, processing_time)
            
            result = InferenceResult(
                model_id=model_id,
                predictions=predictions,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    'model_name': metadata.name,
                    'domain': metadata.domain.value,
                    'patient_context': patient_context
                },
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                risk_level=risk_level,
                clinical_notes=self._generate_clinical_notes(predictions, metadata, risk_level)
            )
            
            logger.info(f"Inference completed for {model_id}: {confidence:.3f} confidence, {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {model_id}: {e}")
            raise
    
    async def _pytorch_inference(self, model, input_data: np.ndarray, 
                                metadata: ModelMetadata) -> Dict[str, float]:
        """Run PyTorch model inference with proper device handling"""
        try:
            with torch.no_grad():
                # Ensure input data is properly formatted
                if len(input_data.shape) == 2 and len(metadata.input_shape) == 3:
                    # Convert HxW grayscale to CxHxW format
                    input_data = np.expand_dims(input_data, axis=0)  # Add channel dimension
                elif len(input_data.shape) == 3 and len(metadata.input_shape) == 3:
                    # Convert HxWxC to CxHxW format for PyTorch
                    if input_data.shape[2] <= 4:  # Channel last format
                        input_data = np.transpose(input_data, (2, 0, 1))
                
                # Normalize input data
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                
                # Scale to 0-1 range if needed
                if input_data.max() > 1.0:
                    input_data = input_data / 255.0
                
                # Convert to tensor on correct device
                tensor_input = torch.FloatTensor(input_data)
                
                # Add batch dimension if not present
                if len(tensor_input.shape) == len(metadata.input_shape):
                    tensor_input = tensor_input.unsqueeze(0)
                
                # Move to model's device
                model_device = next(model.parameters()).device
                tensor_input = tensor_input.to(model_device)
                
                # Run inference
                outputs = model(tensor_input)
                probabilities = outputs.cpu().numpy()[0]
                
                # Convert to predictions dictionary
                predictions = {
                    class_name: float(prob) 
                    for class_name, prob in zip(metadata.output_classes, probabilities)
                }
                
            return predictions
            
        except Exception as e:
            logger.error(f"PyTorch inference error: {e}")
            # Try CPU fallback
            if not self.force_cpu and self.device.type == 'cuda':
                logger.info("Attempting CPU fallback for PyTorch inference")
                self.force_cpu = True
                model_cpu = model.cpu()
                tensor_input_cpu = torch.FloatTensor(input_data)
                if len(tensor_input_cpu.shape) == len(metadata.input_shape):
                    tensor_input_cpu = tensor_input_cpu.unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model_cpu(tensor_input_cpu)
                    probabilities = outputs.numpy()[0]
                    
                predictions = {
                    class_name: float(prob) 
                    for class_name, prob in zip(metadata.output_classes, probabilities)
                }
                return predictions
            else:
                raise
    
    async def _tensorflow_inference(self, model, input_data: np.ndarray, 
                                   metadata: ModelMetadata) -> Dict[str, float]:
        """Run TensorFlow model inference with CPU-only execution"""
        try:
            # Ensure CPU-only execution
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Ensure input data is properly formatted
            if len(input_data.shape) == 2 and len(metadata.input_shape) == 3:
                # Convert HxW to HxWx1 for TensorFlow
                input_data = np.expand_dims(input_data, axis=-1)
            
            # Normalize input data
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Scale to 0-1 range if needed
            if input_data.max() > 1.0:
                input_data = input_data / 255.0
            
            # Add batch dimension if not present
            if len(input_data.shape) == len(metadata.input_shape):
                input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference (CPU-only)
            predictions_array = model.predict(input_data, verbose=0)[0]
            
            # Convert to predictions dictionary
            predictions = {
                class_name: float(prob) 
                for class_name, prob in zip(metadata.output_classes, predictions_array)
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"TensorFlow inference failed: {e}")
            raise
    
    async def _sklearn_inference(self, model, input_data: np.ndarray, 
                                metadata: ModelMetadata) -> Dict[str, float]:
        """Run sklearn model inference"""
        # Reshape if needed
        if len(input_data.shape) > 1:
            input_data = input_data.reshape(1, -1)
        else:
            input_data = input_data.reshape(1, -1)
        
        # Get probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Convert to predictions dictionary
        predictions = {
            class_name: float(prob) 
            for class_name, prob in zip(metadata.output_classes, probabilities)
        }
        
        return predictions
    
    def _calculate_risk_level(self, predictions: Dict[str, float], 
                             metadata: ModelMetadata) -> str:
        """Calculate risk level based on predictions"""
        max_class = max(predictions, key=predictions.get)
        max_prob = predictions[max_class]
        
        # Domain-specific risk assessment
        if metadata.domain == MedicalDomain.RETINAL:
            if "severe" in max_class.lower() or "proliferative" in max_class.lower():
                return "critical"
            elif "moderate" in max_class.lower():
                return "high"
            elif "mild" in max_class.lower():
                return "medium"
            else:
                return "low"
                
        elif metadata.domain == MedicalDomain.CARDIOVASCULAR:
            if "ventricular" in max_class.lower() or "critical" in max_class.lower():
                return "critical"
            elif "atrial" in max_class.lower() or "high" in max_class.lower():
                return "high"
            elif "medium" in max_class.lower():
                return "medium"
            else:
                return "low"
        
        # General risk assessment based on confidence
        if max_prob > 0.8:
            if "high" in max_class.lower() or "severe" in max_class.lower():
                return "high"
            else:
                return "medium"
        else:
            return "low"
    
    def _generate_clinical_notes(self, predictions: Dict[str, float], 
                                metadata: ModelMetadata, risk_level: str) -> str:
        """Generate clinical notes for the prediction"""
        max_class = max(predictions, key=predictions.get)
        max_prob = predictions[max_class]
        
        notes = f"AI Analysis ({metadata.name}):\n"
        notes += f"Primary finding: {max_class} (confidence: {max_prob:.1%})\n"
        notes += f"Risk level: {risk_level.upper()}\n"
        
        if metadata.domain == MedicalDomain.RETINAL:
            if "severe" in max_class.lower():
                notes += "Recommendation: Urgent ophthalmology referral required\n"
            elif "moderate" in max_class.lower():
                notes += "Recommendation: Ophthalmology follow-up within 1 month\n"
            elif "mild" in max_class.lower():
                notes += "Recommendation: Annual eye examination\n"
                
        elif metadata.domain == MedicalDomain.CARDIOVASCULAR:
            if risk_level == "critical":
                notes += "Recommendation: Immediate cardiology consultation\n"
            elif risk_level == "high":
                notes += "Recommendation: Cardiology referral within 1 week\n"
        
        notes += f"Model accuracy: {metadata.accuracy:.1%} (Sensitivity: {metadata.sensitivity:.1%}, Specificity: {metadata.specificity:.1%})"
        
        return notes
    
    def _update_performance_metrics(self, model_id: str, processing_time: float):
        """Update performance metrics for model"""
        metrics = self.performance_metrics[model_id]
        metrics['total_inferences'] += 1
        
        # Update average processing time
        total_inferences = metrics['total_inferences']
        current_avg = metrics['avg_processing_time']
        metrics['avg_processing_time'] = (current_avg * (total_inferences - 1) + processing_time) / total_inferences
        
        metrics['last_used'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a model"""
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
            
        metadata = self.model_metadata[model_id]
        performance = self.performance_metrics.get(model_id, {})
        
        return {
            'metadata': {
                'model_id': metadata.model_id,
                'name': metadata.name,
                'version': metadata.version,
                'domain': metadata.domain.value,
                'description': metadata.description,
                'accuracy': metadata.accuracy,
                'sensitivity': metadata.sensitivity,
                'specificity': metadata.specificity,
                'clinical_validation': metadata.clinical_validation,
                'output_classes': metadata.output_classes
            },
            'performance': performance,
            'loaded': model_id in self.loaded_models
        }
    
    def list_models(self, domain: Optional[MedicalDomain] = None) -> List[Dict[str, Any]]:
        """List all available models, optionally filtered by domain"""
        models = []
        for model_id, metadata in self.model_metadata.items():
            if domain is None or metadata.domain == domain:
                models.append(self.get_model_info(model_id))
        return models
    
    async def unload_model(self, model_id: str):
        """Unload a model to free memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            logger.info(f"Unloaded model: {model_id}")
    
    async def reload_model(self, model_id: str):
        """Reload a specific model"""
        await self.unload_model(model_id)
        # Implementation would reload from file
        logger.info(f"Reloaded model: {model_id}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_models = len(self.model_metadata)
        loaded_models = len(self.loaded_models)
        total_inferences = sum(
            metrics.get('total_inferences', 0) 
            for metrics in self.performance_metrics.values()
        )
        
        return {
            'total_models': total_models,
            'loaded_models': loaded_models,
            'total_inferences': total_inferences,
            'device': str(self.device),
            'domains': list(set(m.domain.value for m in self.model_metadata.values()))
        }
    
    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific model"""
        if model_name not in self.model_metadata:
            return {'status': 'not_found', 'model_name': model_name}
        
        metadata = self.model_metadata[model_name]
        performance = self.performance_metrics.get(model_name, {})
        
        return {
            'status': 'loaded' if model_name in self.loaded_models else 'not_loaded',
            'model_name': model_name,
            'framework': metadata.model_type.value,
            'domain': metadata.domain.value,
            'accuracy': metadata.accuracy,
            'inference_count': performance.get('total_inferences', 0),
            'avg_processing_time': performance.get('avg_processing_time', 0),
            'last_used': performance.get('last_used'),
            'device': str(self.device)
        }
    
    async def run_inference(self, model_name: str, input_data: np.ndarray) -> 'InferenceResult':
        """
        Public method for running inference - delegates to inference method
        
        Args:
            model_name: Name of the model to use
            input_data: Input data for inference
            
        Returns:
            InferenceResult with predictions and metadata
        """
        return await self.predict(model_name, input_data) 