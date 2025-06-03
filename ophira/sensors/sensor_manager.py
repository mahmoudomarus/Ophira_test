"""
Sensor Manager for Ophira AI Medical System
Coordinates and manages all sensor operations
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from loguru import logger

from .base_sensor import BaseSensor, SensorData, SensorStatus
from .heart_rate_sensor import HeartRateSensor
from .ppg_sensor import PPGSensor
from .nir_camera import NIRCamera
from .temperature_sensor import TemperatureSensor
from .blood_pressure_sensor import BloodPressureSensor


class SensorManager:
    """
    Central manager for all sensor operations in Ophira AI system
    Handles sensor registration, data collection, and analysis coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sensors: Dict[str, BaseSensor] = {}
        self.active_readings = {}
        self.data_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        self._health_check_task = None
        self._coordinated_reading_task = None
        self.reading_interval = self.config.get("reading_interval", 2.0)
        
    async def initialize(self):
        """Initialize sensor manager and register default sensors"""
        logger.info("🔧 Initializing Sensor Manager...")
        
        # Register default sensors based on config
        sensor_configs = self.config.get("sensors", {})
        
        # Heart Rate Sensor
        if sensor_configs.get("heart_rate", {}).get("enabled", True):
            hr_config = sensor_configs.get("heart_rate", {})
            hr_sensor = HeartRateSensor("hr_001", hr_config)
            await self.register_sensor(hr_sensor)
        
        # PPG Sensor
        if sensor_configs.get("ppg", {}).get("enabled", True):
            ppg_config = sensor_configs.get("ppg", {})
            ppg_sensor = PPGSensor("ppg_001", ppg_config)
            await self.register_sensor(ppg_sensor)
        
        # NIR Camera
        if sensor_configs.get("nir_camera", {}).get("enabled", True):
            nir_config = sensor_configs.get("nir_camera", {})
            nir_sensor = NIRCamera("nir_001", nir_config)
            await self.register_sensor(nir_sensor)
        
        # Temperature Sensor
        if sensor_configs.get("temperature", {}).get("enabled", True):
            temp_config = sensor_configs.get("temperature", {})
            temp_sensor = TemperatureSensor("temp_001", temp_config)
            await self.register_sensor(temp_sensor)
        
        # Blood Pressure Sensor
        if sensor_configs.get("blood_pressure", {}).get("enabled", True):
            bp_config = sensor_configs.get("blood_pressure", {})
            bp_sensor = BloodPressureSensor("bp_001", bp_config)
            await self.register_sensor(bp_sensor)
        
        # Start health monitoring
        await self.start_health_monitoring()
        
        # Start coordinated readings
        await self.start_coordinated_readings()
        
        logger.info(f"✅ Sensor Manager initialized with {len(self.sensors)} sensors")
    
    async def register_sensor(self, sensor: BaseSensor) -> bool:
        """Register a sensor with the manager"""
        try:
            # Connect to sensor
            connected = await sensor.connect()
            if connected:
                self.sensors[sensor.sensor_id] = sensor
                logger.info(f"📡 Registered sensor: {sensor.sensor_id} ({sensor.sensor_type})")
                return True
            else:
                logger.warning(f"⚠️ Failed to connect to sensor: {sensor.sensor_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Error registering sensor {sensor.sensor_id}: {e}")
            return False
    
    async def unregister_sensor(self, sensor_id: str) -> bool:
        """Unregister and disconnect a sensor"""
        if sensor_id in self.sensors:
            sensor = self.sensors[sensor_id]
            await sensor.disconnect()
            del self.sensors[sensor_id]
            logger.info(f"📡 Unregistered sensor: {sensor_id}")
            return True
        return False
    
    def add_data_callback(self, callback: Callable[[SensorData], None]):
        """Add callback for sensor data events"""
        self.data_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, SensorData], None]):
        """Add callback for sensor alerts"""
        self.alert_callbacks.append(callback)
    
    async def get_all_sensor_data(self) -> Dict[str, List[SensorData]]:
        """Get recent data from all sensors"""
        all_data = {}
        for sensor_id, sensor in self.sensors.items():
            recent_data = sensor.get_recent_data(count=10)
            all_data[sensor_id] = recent_data
        return all_data
    
    async def get_sensor_data(self, sensor_id: str, count: int = 10) -> List[SensorData]:
        """Get recent data from specific sensor"""
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].get_recent_data(count)
        return []
    
    async def read_all_sensors(self) -> Dict[str, SensorData]:
        """Read current data from all sensors simultaneously"""
        readings = {}
        tasks = []
        sensor_ids = []
        
        for sensor_id, sensor in self.sensors.items():
            if sensor.status == SensorStatus.CONNECTED:
                tasks.append(sensor.read_single())
                sensor_ids.append(sensor_id)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sensor_id, result in zip(sensor_ids, results):
                if isinstance(result, SensorData):
                    readings[sensor_id] = result
                    # Trigger callbacks
                    await self._trigger_data_callbacks(result)
                    # Check for alerts
                    await self._check_for_alerts(result)
        
        return readings
    
    async def trigger_health_analysis(self) -> Dict[str, Any]:
        """Trigger comprehensive health analysis using all available sensor data"""
        logger.info("🔬 Triggering comprehensive health analysis...")
        
        # Get recent data from all sensors
        all_data = await self.get_all_sensor_data()
        
        # Compile analysis data
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "sensor_count": len(self.sensors),
            "data_points": {},
            "trends": {},
            "alerts": []
        }
        
        for sensor_id, data_list in all_data.items():
            if data_list:
                latest = data_list[-1]
                analysis_data["data_points"][sensor_id] = {
                    "latest_value": latest.value,
                    "unit": latest.unit,
                    "quality_score": latest.quality_score,
                    "is_critical": latest.is_critical,
                    "timestamp": latest.timestamp.isoformat()
                }
                
                # Basic trend analysis
                if len(data_list) >= 3:
                    values = [d.value for d in data_list[-3:] if isinstance(d.value, (int, float))]
                    if len(values) >= 3:
                        trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
                        analysis_data["trends"][sensor_id] = trend
        
        return analysis_data
    
    async def start_health_monitoring(self):
        """Start continuous health monitoring"""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("💓 Started health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("💓 Stopped health monitoring")
    
    async def start_coordinated_readings(self):
        """Start coordinated sensor readings"""
        if self._coordinated_reading_task and not self._coordinated_reading_task.done():
            return
        
        self._coordinated_reading_task = asyncio.create_task(self._coordinated_reading_loop())
        logger.info("📊 Started coordinated sensor readings")
    
    async def stop_coordinated_readings(self):
        """Stop coordinated readings"""
        if self._coordinated_reading_task:
            self._coordinated_reading_task.cancel()
            try:
                await self._coordinated_reading_task
            except asyncio.CancelledError:
                pass
        logger.info("📊 Stopped coordinated sensor readings")
    
    async def _health_monitoring_loop(self):
        """Internal health monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _coordinated_reading_loop(self):
        """Internal coordinated reading loop"""
        while True:
            try:
                await self.read_all_sensors()
                await asyncio.sleep(self.reading_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordinated reading error: {e}")
                await asyncio.sleep(self.reading_interval * 2)
    
    async def _perform_health_checks(self):
        """Perform health checks on all sensors"""
        for sensor_id, sensor in self.sensors.items():
            try:
                healthy = await sensor.health_check()
                if not healthy:
                    logger.warning(f"⚠️ Health check failed for sensor: {sensor_id}")
            except Exception as e:
                logger.error(f"Health check error for {sensor_id}: {e}")
    
    async def _trigger_data_callbacks(self, data: SensorData):
        """Trigger all data callbacks"""
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
    
    async def _check_for_alerts(self, data: SensorData):
        """Check sensor data for alert conditions"""
        alert_triggered = False
        alert_message = ""
        
        # Critical data flag
        if data.is_critical:
            alert_triggered = True
            alert_message = f"Critical reading from {data.sensor_type}: {data.value} {data.unit}"
        
        # Sensor-specific alert logic
        elif data.sensor_type == "heart_rate" and isinstance(data.value, (int, float)):
            if data.value > 100 or data.value < 60:
                alert_triggered = True
                alert_message = f"Abnormal heart rate: {data.value} bpm"
        
        elif data.sensor_type == "temperature" and isinstance(data.value, (int, float)):
            if data.value > 38.5 or data.value < 36.0:
                alert_triggered = True
                alert_message = f"Abnormal temperature: {data.value}°C"
        
        elif data.sensor_type == "blood_pressure" and isinstance(data.value, dict):
            systolic = data.value.get("systolic", 0)
            diastolic = data.value.get("diastolic", 0)
            if systolic > 140 or diastolic > 90:
                alert_triggered = True
                alert_message = f"High blood pressure: {systolic}/{diastolic} mmHg"
        
        # Trigger alert callbacks
        if alert_triggered:
            logger.warning(f"🚨 ALERT: {alert_message}")
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_message, data)
                    else:
                        callback(alert_message, data)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status"""
        sensor_statuses = {}
        for sensor_id, sensor in self.sensors.items():
            sensor_statuses[sensor_id] = sensor.get_status_info()
        
        return {
            "total_sensors": len(self.sensors),
            "active_sensors": len([s for s in self.sensors.values() if s.status == SensorStatus.CONNECTED]),
            "health_monitoring": self._health_check_task is not None and not self._health_check_task.done(),
            "coordinated_readings": self._coordinated_reading_task is not None and not self._coordinated_reading_task.done(),
            "sensors": sensor_statuses,
            "config": self.config
        }
    
    async def shutdown(self):
        """Shutdown sensor manager and disconnect all sensors"""
        logger.info("🔌 Shutting down Sensor Manager...")
        
        # Stop monitoring tasks
        await self.stop_health_monitoring()
        await self.stop_coordinated_readings()
        
        # Disconnect all sensors
        for sensor_id in list(self.sensors.keys()):
            await self.unregister_sensor(sensor_id)
        
        logger.info("✅ Sensor Manager shutdown complete") 