"""
Logging configuration for Ophira AI system.

Provides structured logging with rotation, filtering, and
integration with monitoring systems.
"""

import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
from .config import settings


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None):
    """
    Set up logging configuration for the Ophira AI system.
    
    Args:
        log_level: Override the default log level
        log_file: Override the default log file path
    
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Use settings or provided overrides
    level = log_level or settings.logging.log_level
    file_path = log_file or settings.logging.log_file
    
    # Console handler with color coding
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        diagnose=True,
        backtrace=True,
    )
    
    # File handler with rotation if file path is specified
    if file_path:
        # Ensure log directory exists
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            file_path,
            level=level,
            format=settings.logging.log_format,
            rotation=settings.logging.log_rotation,
            retention=settings.logging.log_retention,
            compression="zip",
            diagnose=True,
            backtrace=True,
        )
    
    # Add JSON handler for structured logging (useful for monitoring)
    if not settings.app.debug:
        logger.add(
            "logs/ophira_structured.jsonl",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            serialize=True,
            rotation="1 day",
            retention="30 days",
        )
    
    # Configure specific loggers
    configure_medical_logger()
    configure_agent_logger()
    configure_sensor_logger()
    configure_memory_logger()
    
    logger.info(f"Logging initialized - Level: {level}, File: {file_path}")
    
    # Return the configured logger
    return logger


def configure_medical_logger():
    """Configure specialized logger for medical events."""
    medical_logger = logger.bind(component="medical")
    
    # Add specific file for medical events
    logger.add(
        "logs/medical_events.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | MEDICAL | {extra[component]} | {message}",
        filter=lambda record: record["extra"].get("component") == "medical",
        rotation="1 day",
        retention="1 year",  # Keep medical logs longer
        compression="zip",
    )


def configure_agent_logger():
    """Configure specialized logger for agent communications."""
    agent_logger = logger.bind(component="agent")
    
    # Add specific file for agent events
    logger.add(
        "logs/agent_communications.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | AGENT | {extra[component]} | {extra[agent_type]} | {message}",
        filter=lambda record: record["extra"].get("component") == "agent",
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )


def configure_sensor_logger():
    """Configure specialized logger for sensor data."""
    sensor_logger = logger.bind(component="sensor")
    
    # Add specific file for sensor events
    logger.add(
        "logs/sensor_data.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | SENSOR | {extra[component]} | {extra[sensor_type]} | {message}",
        filter=lambda record: record["extra"].get("component") == "sensor",
        rotation="6 hours",  # More frequent rotation for sensor data
        retention="7 days",
        compression="zip",
    )


def configure_memory_logger():
    """Configure specialized logger for memory operations."""
    memory_logger = logger.bind(component="memory")
    
    # Add specific file for memory events
    logger.add(
        "logs/memory_operations.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | MEMORY | {extra[component]} | {extra[operation]} | {message}",
        filter=lambda record: record["extra"].get("component") == "memory",
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )


class MedicalLogger:
    """Specialized logger for medical events with HIPAA considerations."""
    
    def __init__(self):
        self.logger = logger.bind(component="medical")
    
    def log_health_measurement(self, user_id: str, sensor_type: str, measurement_type: str, 
                             value: float, timestamp: str, is_critical: bool = False):
        """Log a health measurement."""
        level = "CRITICAL" if is_critical else "INFO"
        
        # Note: In production, consider anonymizing user_id for HIPAA compliance
        self.logger.log(
            level,
            f"Health measurement recorded",
            extra={
                "user_id": user_id,
                "sensor_type": sensor_type,
                "measurement_type": measurement_type,
                "value": value,
                "timestamp": timestamp,
                "is_critical": is_critical
            }
        )
    
    def log_analysis_result(self, user_id: str, analysis_type: str, result: str, 
                          confidence: float, agent_name: str):
        """Log an analysis result from a medical agent."""
        self.logger.info(
            f"Medical analysis completed",
            extra={
                "user_id": user_id,
                "analysis_type": analysis_type,
                "result": result,
                "confidence": confidence,
                "agent_name": agent_name
            }
        )
    
    def log_emergency_event(self, user_id: str, event_type: str, severity: str, 
                          action_taken: str, response_time: float):
        """Log an emergency event."""
        self.logger.critical(
            f"Emergency event detected",
            extra={
                "user_id": user_id,
                "event_type": event_type,
                "severity": severity,
                "action_taken": action_taken,
                "response_time": response_time
            }
        )


class AgentLogger:
    """Specialized logger for agent communications and decisions."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.logger = logger.bind(component="agent", agent_type=agent_type)
    
    # Standard logging methods for compatibility
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    # Specialized agent methods
    def log_message_sent(self, recipient: str, message_type: str, content_summary: str):
        """Log a message sent to another agent."""
        self.logger.debug(
            f"Message sent to {recipient}",
            extra={
                "recipient": recipient,
                "message_type": message_type,
                "content_summary": content_summary
            }
        )
    
    def log_decision(self, decision_type: str, reasoning: str, confidence: float):
        """Log an agent decision."""
        self.logger.info(
            f"Decision made: {decision_type}",
            extra={
                "decision_type": decision_type,
                "reasoning": reasoning,
                "confidence": confidence
            }
        )
    
    def log_error(self, error_type: str, error_message: str, context: dict):
        """Log an agent error."""
        self.logger.error(
            f"Agent error: {error_type}",
            extra={
                "error_type": error_type,
                "error_message": error_message,
                "context": context
            }
        )


class SensorLogger:
    """Specialized logger for sensor data and operations."""
    
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.logger = logger.bind(component="sensor", sensor_type=sensor_type)
    
    def log_data_collection(self, data_points: int, collection_duration: float, 
                          quality_score: float):
        """Log sensor data collection."""
        self.logger.info(
            f"Data collected from {self.sensor_type}",
            extra={
                "data_points": data_points,
                "collection_duration": collection_duration,
                "quality_score": quality_score
            }
        )
    
    def log_calibration(self, calibration_type: str, success: bool, 
                       calibration_values: dict):
        """Log sensor calibration."""
        level = "INFO" if success else "WARNING"
        self.logger.log(
            level,
            f"Sensor calibration: {calibration_type}",
            extra={
                "calibration_type": calibration_type,
                "success": success,
                "calibration_values": calibration_values
            }
        )
    
    def log_anomaly(self, anomaly_type: str, severity: str, details: dict):
        """Log sensor anomaly detection."""
        self.logger.warning(
            f"Sensor anomaly detected: {anomaly_type}",
            extra={
                "anomaly_type": anomaly_type,
                "severity": severity,
                "details": details
            }
        )


class MemoryLogger:
    """Specialized logger for memory operations."""
    
    def __init__(self):
        self.logger = logger.bind(component="memory")
    
    def log_storage_operation(self, operation: str, memory_type: str, 
                            record_count: int, operation_time: float):
        """Log memory storage operation."""
        self.logger.info(
            f"Memory {operation} completed",
            extra={
                "operation": operation,
                "memory_type": memory_type,
                "record_count": record_count,
                "operation_time": operation_time
            }
        )
    
    def log_retrieval_operation(self, query_type: str, results_count: int, 
                              retrieval_time: float, relevance_score: float):
        """Log memory retrieval operation."""
        self.logger.info(
            f"Memory retrieval completed",
            extra={
                "query_type": query_type,
                "results_count": results_count,
                "retrieval_time": retrieval_time,
                "relevance_score": relevance_score
            }
        )
    
    def log_consolidation(self, records_processed: int, records_consolidated: int, 
                         consolidation_time: float):
        """Log memory consolidation operation."""
        self.logger.info(
            f"Memory consolidation completed",
            extra={
                "records_processed": records_processed,
                "records_consolidated": records_consolidated,
                "consolidation_time": consolidation_time
            }
        )


# Create global logger instances
medical_logger = MedicalLogger()
memory_logger = MemoryLogger()


def get_logger(name: str):
    """Get a logger instance with the specified name."""
    return logger.bind(component=name)


def get_agent_logger(agent_type: str) -> AgentLogger:
    """Get an agent logger for a specific agent type."""
    return AgentLogger(agent_type)


def get_sensor_logger(sensor_type: str) -> SensorLogger:
    """Get a sensor logger for a specific sensor type."""
    return SensorLogger(sensor_type) 