"""
Sensor Control Agent for Ophira AI System

Manages data collection from various sensors, calibrates sensors, validates data quality,
schedules regular data collection and special measurements, and reports data collection
status to the orchestrator.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .base import BaseAgent, AgentMessage, MessageType, MessagePriority, AgentCapability
from ..sensors.base_sensor import BaseSensor, SensorStatus, SensorData
from ..sensors.usb_camera_sensor import USBCameraSensor
from ..sensors.vl53l4cx_tof_sensor import VL53L4CXToFSensor
from ..core.config import settings
from ..core.logging import get_agent_logger


class SensorScheduleType(Enum):
    """Types of sensor data collection schedules."""
    CONTINUOUS = "continuous"
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"
    EVENT_TRIGGERED = "event_triggered"
    EMERGENCY = "emergency"


class DataQualityLevel(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class SensorSchedule:
    """Schedule configuration for sensor data collection."""
    sensor_id: str
    schedule_type: SensorScheduleType
    interval_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    trigger_conditions: List[str] = field(default_factory=list)
    priority: int = 1  # 1-5, higher is more important
    enabled: bool = True
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None


@dataclass
class DataQualityAssessment:
    """Assessment of sensor data quality."""
    sensor_id: str
    timestamp: datetime
    quality_level: DataQualityLevel
    quality_score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorDataBatch:
    """Batch of sensor data with quality assessment."""
    batch_id: str
    sensor_ids: List[str]
    data_points: List[SensorData]
    quality_assessments: List[DataQualityAssessment]
    collection_started: datetime
    collection_completed: datetime
    batch_quality_score: float
    is_emergency: bool = False


class SensorControlAgent(BaseAgent):
    """
    Sensor Control Agent - Manages all sensor operations.
    
    Responsibilities:
    - Manage data collection from various sensors
    - Calibrate sensors and validate data quality
    - Schedule regular data collection and special measurements
    - Report data collection status to the orchestrator
    - Handle emergency sensor protocols
    - Optimize sensor performance and battery life
    """
    
    def __init__(self):
        """Initialize the Sensor Control Agent."""
        super().__init__(
            agent_id="sensor_control_001",
            name="Sensor Controller",
            agent_type="sensor_control",
            capabilities=[
                AgentCapability.SENSOR_CONTROL,
                AgentCapability.DATA_PROCESSING,
                AgentCapability.VALIDATION
            ]
        )
        
        # Sensor management
        self.sensors: Dict[str, BaseSensor] = {}
        self.sensor_schedules: Dict[str, SensorSchedule] = {}
        self.active_collections: Dict[str, asyncio.Task] = {}
        
        # Data quality
        self.quality_assessments: List[DataQualityAssessment] = []
        self.quality_thresholds = {
            DataQualityLevel.EXCELLENT: 0.9,
            DataQualityLevel.GOOD: 0.8,
            DataQualityLevel.ACCEPTABLE: 0.6,
            DataQualityLevel.POOR: 0.4,
            DataQualityLevel.UNUSABLE: 0.0
        }
        
        # Data storage
        self.recent_data: Dict[str, List[SensorData]] = {}
        self.data_batches: List[SensorDataBatch] = []
        
        # Calibration
        self.calibration_intervals = {
            "camera": timedelta(hours=24),
            "tof": timedelta(hours=12),
            "heart_rate": timedelta(hours=6),
            "temperature": timedelta(hours=4)
        }
        self.last_calibrations: Dict[str, datetime] = {}
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._quality_monitor_task: Optional[asyncio.Task] = None
        self._calibration_task: Optional[asyncio.Task] = None
        
        self.logger.info("Sensor Control Agent initialized")
    
    async def _start_agent_tasks(self):
        """Start background tasks for sensor management."""
        await super()._start_agent_tasks()
        
        # Initialize default sensors
        await self._initialize_sensors()
        
        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._quality_monitor_task = asyncio.create_task(self._quality_monitor_loop())
        self._calibration_task = asyncio.create_task(self._calibration_loop())
        
        self.logger.info("Sensor Control Agent tasks started")
    
    async def _cleanup(self):
        """Cleanup sensor control agent."""
        # Cancel background tasks
        for task in [self._scheduler_task, self._quality_monitor_task, self._calibration_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Disconnect all sensors
        for sensor in self.sensors.values():
            try:
                await sensor.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting sensor {sensor.sensor_id}: {e}")
        
        await super()._cleanup()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages."""
        content = message.content
        message_type = message.message_type
        
        if message_type == MessageType.REQUEST:
            if "collect_data" in content:
                await self._handle_data_collection_request(message)
            elif "calibrate_sensor" in content:
                await self._handle_calibration_request(message)
            elif "schedule_collection" in content:
                await self._handle_schedule_request(message)
            elif "get_sensor_status" in content:
                await self._handle_status_request(message)
            elif "emergency_collection" in content:
                await self._handle_emergency_collection(message)
        
        elif message_type == MessageType.COMMAND:
            if content.get("action") == "start_continuous":
                await self._start_continuous_collection(content.get("sensor_ids", []))
            elif content.get("action") == "stop_collection":
                await self._stop_collection(content.get("sensor_ids", []))
            elif content.get("action") == "recalibrate_all":
                await self._recalibrate_all_sensors()
    
    async def _initialize_sensors(self):
        """Initialize and connect to available sensors."""
        try:
            # Initialize USB Camera
            camera_sensor = USBCameraSensor(
                sensor_id="usb_camera_primary",
                config=settings.sensors.camera
            )
            if await camera_sensor.connect():
                self.sensors["usb_camera_primary"] = camera_sensor
                await self._setup_default_schedule("usb_camera_primary", SensorScheduleType.ON_DEMAND)
                self.logger.info("USB Camera sensor initialized")
            
            # Initialize VL53L4CX ToF Sensor
            tof_sensor = VL53L4CXToFSensor(
                sensor_id="vl53l4cx_tof_primary",
                config=settings.sensors.tof
            )
            if await tof_sensor.connect():
                self.sensors["vl53l4cx_tof_primary"] = tof_sensor
                await self._setup_default_schedule("vl53l4cx_tof_primary", SensorScheduleType.PERIODIC, 1.0)
                self.logger.info("VL53L4CX ToF sensor initialized")
            
            # TODO: Add other sensors (heart rate, temperature, etc.)
            
        except Exception as e:
            self.logger.error(f"Error initializing sensors: {e}")
    
    async def _setup_default_schedule(
        self,
        sensor_id: str,
        schedule_type: SensorScheduleType,
        interval: Optional[float] = None
    ):
        """Set up default collection schedule for a sensor."""
        schedule = SensorSchedule(
            sensor_id=sensor_id,
            schedule_type=schedule_type,
            interval_seconds=interval,
            priority=3
        )
        
        if schedule_type == SensorScheduleType.PERIODIC and interval:
            schedule.next_execution = datetime.now() + timedelta(seconds=interval)
        
        self.sensor_schedules[sensor_id] = schedule
    
    async def _handle_data_collection_request(self, message: AgentMessage):
        """Handle request to collect data from sensors."""
        content = message.content
        sensor_ids = content.get("sensor_ids", list(self.sensors.keys()))
        collection_type = content.get("collection_type", "single")
        urgency = content.get("urgency", MessagePriority.NORMAL)
        
        if collection_type == "single":
            data_batch = await self._collect_single_batch(sensor_ids)
        elif collection_type == "continuous":
            await self._start_continuous_collection(sensor_ids)
            data_batch = None
        else:
            data_batch = None
        
        # Send response with collected data
        response = AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=urgency,
            content={
                "status": "completed" if data_batch else "started",
                "data_batch": data_batch.__dict__ if data_batch else None,
                "sensor_count": len(sensor_ids),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.send_message(response)
    
    async def _collect_single_batch(self, sensor_ids: List[str]) -> SensorDataBatch:
        """Collect a single batch of data from specified sensors."""
        batch_id = f"batch_{datetime.now().timestamp()}"
        collection_started = datetime.now()
        
        data_points = []
        quality_assessments = []
        
        for sensor_id in sensor_ids:
            sensor = self.sensors.get(sensor_id)
            if not sensor:
                self.logger.warning(f"Sensor {sensor_id} not found")
                continue
            
            try:
                # Collect data
                sensor_data = await sensor.read_single()
                if sensor_data:
                    data_points.append(sensor_data)
                    
                    # Store recent data
                    if sensor_id not in self.recent_data:
                        self.recent_data[sensor_id] = []
                    self.recent_data[sensor_id].append(sensor_data)
                    
                    # Keep only recent data (last 100 points)
                    if len(self.recent_data[sensor_id]) > 100:
                        self.recent_data[sensor_id] = self.recent_data[sensor_id][-100:]
                    
                    # Assess data quality
                    quality_assessment = await self._assess_data_quality(sensor_data)
                    quality_assessments.append(quality_assessment)
                
            except Exception as e:
                self.logger.error(f"Error collecting data from {sensor_id}: {e}")
                
                # Create poor quality assessment for failed collection
                quality_assessment = DataQualityAssessment(
                    sensor_id=sensor_id,
                    timestamp=datetime.now(),
                    quality_level=DataQualityLevel.UNUSABLE,
                    quality_score=0.0,
                    issues=[f"Collection failed: {str(e)}"],
                    recommendations=["Check sensor connection", "Retry calibration"]
                )
                quality_assessments.append(quality_assessment)
        
        collection_completed = datetime.now()
        
        # Calculate batch quality score
        if quality_assessments:
            batch_quality_score = sum(qa.quality_score for qa in quality_assessments) / len(quality_assessments)
        else:
            batch_quality_score = 0.0
        
        # Create data batch
        data_batch = SensorDataBatch(
            batch_id=batch_id,
            sensor_ids=sensor_ids,
            data_points=data_points,
            quality_assessments=quality_assessments,
            collection_started=collection_started,
            collection_completed=collection_completed,
            batch_quality_score=batch_quality_score
        )
        
        self.data_batches.append(data_batch)
        
        # Keep only recent batches
        if len(self.data_batches) > 50:
            self.data_batches = self.data_batches[-50:]
        
        self.logger.info(
            f"Collected data batch {batch_id}",
            extra={
                "batch_id": batch_id,
                "sensor_count": len(sensor_ids),
                "data_points": len(data_points),
                "quality_score": batch_quality_score
            }
        )
        
        return data_batch
    
    async def _assess_data_quality(self, sensor_data: SensorData) -> DataQualityAssessment:
        """Assess the quality of collected sensor data."""
        issues = []
        recommendations = []
        quality_score = sensor_data.quality_score
        
        # Check confidence level
        if sensor_data.confidence < 0.8:
            issues.append(f"Low confidence level: {sensor_data.confidence:.2f}")
            recommendations.append("Consider recalibrating sensor")
            quality_score *= 0.9
        
        # Check if data is marked as critical but has low quality
        if sensor_data.is_critical and quality_score < 0.8:
            issues.append("Critical data with low quality score")
            recommendations.append("Immediate sensor inspection required")
        
        # Sensor-specific quality checks
        if hasattr(sensor_data, 'distance_mm'):  # ToF sensor
            if sensor_data.distance_mm < 10 or sensor_data.distance_mm > 4000:
                issues.append("Distance reading out of normal range")
                recommendations.append("Check sensor positioning")
                quality_score *= 0.8
        
        if hasattr(sensor_data, 'frame'):  # Camera sensor
            if hasattr(sensor_data, 'focus_score') and sensor_data.focus_score < 10:
                issues.append("Poor image focus")
                recommendations.append("Adjust camera position or clean lens")
                quality_score *= 0.7
        
        # Determine quality level
        quality_level = DataQualityLevel.UNUSABLE
        for level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if quality_score >= threshold:
                quality_level = level
                break
        
        return DataQualityAssessment(
            sensor_id=sensor_data.sensor_id,
            timestamp=datetime.now(),
            quality_level=quality_level,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "original_quality_score": sensor_data.quality_score,
                "confidence": sensor_data.confidence,
                "sensor_type": sensor_data.sensor_type
            }
        )
    
    async def _start_continuous_collection(self, sensor_ids: List[str]):
        """Start continuous data collection from specified sensors."""
        for sensor_id in sensor_ids:
            if sensor_id in self.active_collections:
                continue  # Already collecting
            
            sensor = self.sensors.get(sensor_id)
            if not sensor:
                continue
            
            # Start continuous collection task
            task = asyncio.create_task(self._continuous_collection_loop(sensor_id))
            self.active_collections[sensor_id] = task
            
            self.logger.info(f"Started continuous collection for {sensor_id}")
    
    async def _continuous_collection_loop(self, sensor_id: str):
        """Continuous data collection loop for a sensor."""
        sensor = self.sensors.get(sensor_id)
        if not sensor:
            return
        
        try:
            # Get collection interval from schedule
            schedule = self.sensor_schedules.get(sensor_id)
            interval = schedule.interval_seconds if schedule else 1.0
            
            while True:
                try:
                    # Collect data
                    sensor_data = await sensor.read_single()
                    if sensor_data:
                        # Store data
                        if sensor_id not in self.recent_data:
                            self.recent_data[sensor_id] = []
                        self.recent_data[sensor_id].append(sensor_data)
                        
                        # Assess quality
                        quality_assessment = await self._assess_data_quality(sensor_data)
                        self.quality_assessments.append(quality_assessment)
                        
                        # Check for quality issues
                        if quality_assessment.quality_level in [DataQualityLevel.POOR, DataQualityLevel.UNUSABLE]:
                            await self._handle_quality_issue(sensor_id, quality_assessment)
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in continuous collection for {sensor_id}: {e}")
                    await asyncio.sleep(interval * 2)  # Wait longer after error
        
        except asyncio.CancelledError:
            self.logger.info(f"Continuous collection stopped for {sensor_id}")
        except Exception as e:
            self.logger.error(f"Continuous collection failed for {sensor_id}: {e}")
        finally:
            # Clean up
            if sensor_id in self.active_collections:
                del self.active_collections[sensor_id]
    
    async def _stop_collection(self, sensor_ids: List[str]):
        """Stop continuous collection for specified sensors."""
        for sensor_id in sensor_ids:
            if sensor_id in self.active_collections:
                task = self.active_collections[sensor_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.active_collections[sensor_id]
                self.logger.info(f"Stopped continuous collection for {sensor_id}")
    
    async def _handle_quality_issue(self, sensor_id: str, quality_assessment: DataQualityAssessment):
        """Handle detected quality issues."""
        self.logger.warning(
            f"Quality issue detected for {sensor_id}",
            extra={
                "sensor_id": sensor_id,
                "quality_level": quality_assessment.quality_level.value,
                "quality_score": quality_assessment.quality_score,
                "issues": quality_assessment.issues
            }
        )
        
        # Notify other agents about quality issues
        notification = AgentMessage(
            sender_id=self.id,
            recipient_id="*",  # Broadcast
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.HIGH,
            content={
                "type": "sensor_quality_issue",
                "sensor_id": sensor_id,
                "quality_assessment": quality_assessment.__dict__,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.send_message(notification)
    
    async def _scheduler_loop(self):
        """Background task for managing sensor schedules."""
        while True:
            try:
                current_time = datetime.now()
                
                for sensor_id, schedule in self.sensor_schedules.items():
                    if not schedule.enabled:
                        continue
                    
                    # Check if scheduled collection is due
                    if (schedule.schedule_type == SensorScheduleType.PERIODIC and
                        schedule.next_execution and
                        current_time >= schedule.next_execution):
                        
                        # Execute scheduled collection
                        await self._execute_scheduled_collection(sensor_id, schedule)
                        
                        # Update next execution time
                        if schedule.interval_seconds:
                            schedule.next_execution = current_time + timedelta(seconds=schedule.interval_seconds)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(30)
    
    async def _execute_scheduled_collection(self, sensor_id: str, schedule: SensorSchedule):
        """Execute a scheduled data collection."""
        try:
            data_batch = await self._collect_single_batch([sensor_id])
            schedule.last_execution = datetime.now()
            
            self.logger.debug(f"Executed scheduled collection for {sensor_id}")
            
        except Exception as e:
            self.logger.error(f"Error in scheduled collection for {sensor_id}: {e}")
    
    async def _quality_monitor_loop(self):
        """Background task for monitoring data quality trends."""
        while True:
            try:
                # Clean up old quality assessments (keep last 1000)
                if len(self.quality_assessments) > 1000:
                    self.quality_assessments = self.quality_assessments[-1000:]
                
                # Analyze quality trends
                await self._analyze_quality_trends()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in quality monitor loop: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_quality_trends(self):
        """Analyze quality trends and detect patterns."""
        if len(self.quality_assessments) < 10:
            return
        
        # Group assessments by sensor
        sensor_assessments = {}
        for assessment in self.quality_assessments[-100:]:  # Last 100 assessments
            if assessment.sensor_id not in sensor_assessments:
                sensor_assessments[assessment.sensor_id] = []
            sensor_assessments[assessment.sensor_id].append(assessment)
        
        # Check for declining quality trends
        for sensor_id, assessments in sensor_assessments.items():
            if len(assessments) >= 10:
                recent_scores = [a.quality_score for a in assessments[-10:]]
                older_scores = [a.quality_score for a in assessments[-20:-10]] if len(assessments) >= 20 else []
                
                if older_scores:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    older_avg = sum(older_scores) / len(older_scores)
                    
                    # If quality has declined by more than 20%
                    if recent_avg < older_avg * 0.8:
                        await self._trigger_quality_alert(sensor_id, recent_avg, older_avg)
    
    async def _trigger_quality_alert(self, sensor_id: str, recent_avg: float, older_avg: float):
        """Trigger alert for declining sensor quality."""
        self.logger.warning(
            f"Quality decline detected for {sensor_id}",
            extra={
                "sensor_id": sensor_id,
                "recent_avg": recent_avg,
                "older_avg": older_avg,
                "decline_percent": ((older_avg - recent_avg) / older_avg) * 100
            }
        )
        
        # Schedule recalibration
        await self._schedule_sensor_calibration(sensor_id)
    
    async def _calibration_loop(self):
        """Background task for sensor calibration management."""
        while True:
            try:
                current_time = datetime.now()
                
                for sensor_id, sensor in self.sensors.items():
                    # Check if calibration is due
                    last_calibration = self.last_calibrations.get(sensor_id)
                    sensor_type = sensor.sensor_type
                    
                    calibration_interval = self.calibration_intervals.get(sensor_type, timedelta(hours=24))
                    
                    if (not last_calibration or 
                        current_time - last_calibration > calibration_interval):
                        
                        await self._calibrate_sensor(sensor_id)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in calibration loop: {e}")
                await asyncio.sleep(3600)
    
    async def _calibrate_sensor(self, sensor_id: str):
        """Calibrate a specific sensor."""
        sensor = self.sensors.get(sensor_id)
        if not sensor:
            return
        
        try:
            self.logger.info(f"Starting calibration for {sensor_id}")
            
            success = await sensor.calibrate()
            
            if success:
                self.last_calibrations[sensor_id] = datetime.now()
                self.logger.info(f"Calibration completed for {sensor_id}")
            else:
                self.logger.warning(f"Calibration failed for {sensor_id}")
            
        except Exception as e:
            self.logger.error(f"Error calibrating {sensor_id}: {e}")
    
    async def _schedule_sensor_calibration(self, sensor_id: str):
        """Schedule immediate calibration for a sensor."""
        # Remove from last calibrations to force recalibration
        if sensor_id in self.last_calibrations:
            del self.last_calibrations[sensor_id]
        
        # Trigger immediate calibration
        await self._calibrate_sensor(sensor_id)
    
    async def _recalibrate_all_sensors(self):
        """Recalibrate all sensors."""
        for sensor_id in self.sensors.keys():
            await self._calibrate_sensor(sensor_id) 