"""
Configuration management for Ophira AI system.

Handles environment variables, database connections, API keys,
and other configuration settings.
"""

import os
from typing import Optional, List, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # MongoDB settings
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "ophira_medical"
    
    # PostgreSQL settings
    postgresql_url: str = "postgresql://postgres:postgres@localhost:5432/ophira_db"
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    
    class Config:
        env_prefix = "DATABASE_"
        extra = "ignore"


class AIModelSettings(BaseSettings):
    """AI model configuration settings."""
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = Field(default="claude-3-opus-20240229", env="ANTHROPIC_MODEL")
    
    # HuggingFace settings
    huggingface_token: Optional[str] = None
    
    # Local model settings
    local_model_path: str = Field(default="./models", env="LOCAL_MODEL_PATH")
    primary_llm_model: str = Field(default="meta-llama/Llama-2-7b-chat-hf", env="PRIMARY_LLM_MODEL")
    
    # VLLM Medical Models Configuration
    vllm_server_url: str = Field(default="http://localhost:8001", env="VLLM_SERVER_URL")
    medical_llm_model: str = Field(default="microsoft/DialoGPT-medium", env="MEDICAL_LLM_MODEL")
    retinal_vision_model: str = Field(default="microsoft/swin-base-patch4-window7-224", env="RETINAL_VISION_MODEL")
    eye_segmentation_model: str = Field(default="facebook/detr-resnet-50-panoptic", env="EYE_SEGMENTATION_MODEL")
    ophthalmology_model: str = Field(default="google/vit-base-patch16-224", env="OPHTHALMOLOGY_MODEL")
    cardiovascular_model: str = Field(default="EleutherAI/gpt-neox-20b", env="CARDIOVASCULAR_MODEL")
    metabolic_analysis_model: str = Field(default="google/flan-t5-large", env="METABOLIC_ANALYSIS_MODEL")
    
    # Specialized Medical Model URLs
    retinal_analysis_endpoint: str = Field(default="http://localhost:8002/analyze-retina", env="RETINAL_ANALYSIS_ENDPOINT")
    eye_tracking_endpoint: str = Field(default="http://localhost:8003/eye-tracking", env="EYE_TRACKING_ENDPOINT")
    wavefront_analysis_endpoint: str = Field(default="http://localhost:8004/wavefront", env="WAVEFRONT_ANALYSIS_ENDPOINT")
    
    # Model serving settings
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    
    # Medical model specific settings
    medical_confidence_threshold: float = Field(default=0.8, env="MEDICAL_CONFIDENCE_THRESHOLD")
    enable_medical_validation: bool = Field(default=True, env="ENABLE_MEDICAL_VALIDATION")
    medical_model_timeout: int = Field(default=30, env="MEDICAL_MODEL_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields


class SensorSettings(BaseSettings):
    """Sensor configuration settings."""
    
    # NIR Camera settings
    nir_camera_enabled: bool = Field(default=True, env="NIR_CAMERA_ENABLED")
    nir_camera_device_id: int = Field(default=0, env="NIR_CAMERA_DEVICE_ID")
    nir_camera_resolution: str = Field(default="1920x1080", env="NIR_CAMERA_RESOLUTION")
    
    # Heart rate monitor settings
    heart_rate_enabled: bool = Field(default=True, env="HEART_RATE_ENABLED")
    heart_rate_device: str = Field(default="polar_h10", env="HEART_RATE_DEVICE")
    
    # PPG sensor settings
    ppg_enabled: bool = Field(default=True, env="PPG_ENABLED")
    ppg_sampling_rate: int = Field(default=250, env="PPG_SAMPLING_RATE")
    
    # Simulation mode
    simulation_mode: bool = Field(default=True, env="SIMULATION_MODE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    allowed_origins: str = Field(default="http://localhost:3000", env="ALLOWED_ORIGINS")

    def get_allowed_origins_list(self) -> List[str]:
        """Parse allowed origins string into list"""
        if isinstance(self.allowed_origins, str):
            return [origin.strip() for origin in self.allowed_origins.split(',')]
        return [self.allowed_origins]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default="logs/ophira.log", env="LOG_FILE")
    log_rotation: str = Field(default="1 day", env="LOG_ROTATION")
    log_retention: str = Field(default="30 days", env="LOG_RETENTION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields


class ApplicationSettings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = "Ophira AI Medical System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Health check settings
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Emergency settings
    emergency_contact_enabled: bool = Field(default=False, env="EMERGENCY_CONTACT_ENABLED")
    emergency_phone_number: Optional[str] = Field(default=None, env="EMERGENCY_PHONE_NUMBER")
    emergency_email: Optional[str] = Field(default=None, env="EMERGENCY_EMAIL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    
    app: ApplicationSettings = ApplicationSettings()
    database: DatabaseSettings = DatabaseSettings()
    ai_models: AIModelSettings = AIModelSettings()
    sensors: SensorSettings = SensorSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields


# Create a global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def validate_required_settings():
    """Validate that required settings are present."""
    settings = get_settings()
    
    # Check if at least one AI provider is configured
    if not any([
        settings.ai_models.openai_api_key,
        settings.ai_models.anthropic_api_key,
        settings.ai_models.huggingface_token
    ]):
        raise ValueError("At least one AI provider API key must be configured")
    
    return True


# Validate settings on import
try:
    validate_required_settings()
except ValueError as e:
    print(f"Configuration warning: {e}")

# Create the global settings instance for import
settings = get_settings() 