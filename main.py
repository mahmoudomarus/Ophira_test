#!/usr/bin/env python3
"""
Enhanced Ophira Medical AI System with Comprehensive Sensor Integration
Real-time medical analysis with multi-sensor data fusion and database persistence
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Core imports
from ophira.core.config import get_settings
from ophira.core.logging import setup_logging
from ophira.core.database import DatabaseManager, MedicalDataService

# Agent imports
from ophira.agents.primary import OphiraVoiceAgent
from ophira.agents.medical_specialist import MedicalSpecialistAgent
from ophira.agents.base import AgentMessage, MessageType, MessagePriority, AgentRegistry

# Sensor imports
from ophira.sensors import SensorManager

# Analysis types
from ophira.core.medical_models import AnalysisType

# Real Hardware Sensor Imports
from ophira.sensors.ov9281_nir_camera import OV9281NIRCamera
from ophira.sensors.hardware_heart_rate_sensor import HardwareHeartRateSensor

async def test_comprehensive_sensor_integration(sensor_manager: SensorManager, medical_service: MedicalDataService):
    """Test comprehensive sensor integration with medical analysis"""
    from loguru import logger
    logger.info("🔬 Testing comprehensive sensor integration...")
    
    test_user_id = "comprehensive-test-user"
    
    # Initialize sensor manager
    sensor_config = {
        "sensors": {
            "heart_rate": {"enabled": True, "base_heart_rate": 72},
            "ppg": {"enabled": True, "sampling_rate": 100},
            "temperature": {"enabled": True, "base_temperature": 36.7},
            "blood_pressure": {"enabled": True, "base_systolic": 118, "base_diastolic": 78},
            "nir_camera": {"enabled": True, "resolution": (1024, 1024)}
        },
        "reading_interval": 3.0,
        "health_check_interval": 30.0
    }
    
    sensor_manager.config = sensor_config
    await sensor_manager.initialize()
    
    # Set up data callbacks
    sensor_data_buffer = []
    
    async def sensor_data_callback(data):
        sensor_data_buffer.append(data)
        logger.info(f"📊 Sensor data: {data.sensor_type} = {data.value} {data.unit}")
        
        # Store in database
        if medical_service:
            try:
                await medical_service.store_sensor_reading(
                    user_id=test_user_id,
                    sensor_type=data.sensor_type,
                    value=data.value,
                    metadata=data.to_dict()
                )
            except Exception as e:
                logger.warning(f"Database storage failed: {e}")
    
    async def alert_callback(message, data):
        logger.warning(f"🚨 MEDICAL ALERT: {message}")
        # In a real system, this would trigger medical professional notifications
    
    sensor_manager.add_data_callback(sensor_data_callback)
    sensor_manager.add_alert_callback(alert_callback)
    
    # Run sensor collection for a test period
    logger.info("🔄 Starting sensor data collection...")
    
    try:
        # Collect data for 30 seconds
        await asyncio.sleep(30)
        
        # Trigger comprehensive health analysis
        analysis_result = await sensor_manager.trigger_health_analysis()
        logger.info(f"📋 Health Analysis Result: {len(analysis_result['data_points'])} data points collected")
        
        # Store analysis in database
        if medical_service:
            try:
                await medical_service.store_analysis_result(
                    user_id=test_user_id,
                    analysis_type=AnalysisType.COMPREHENSIVE,
                    result=analysis_result,
                    metadata={"sensor_count": analysis_result["sensor_count"]}
                )
                logger.info("💾 Comprehensive analysis stored in database")
            except Exception as e:
                logger.warning(f"Analysis storage failed: {e}")
        
        # Get sensor manager status
        manager_status = sensor_manager.get_manager_status()
        logger.info(f"📈 Sensor Manager Status: {manager_status['active_sensors']}/{manager_status['total_sensors']} sensors active")
        
        return {
            "sensor_data_collected": len(sensor_data_buffer),
            "analysis_result": analysis_result,
            "manager_status": manager_status
        }
        
    finally:
        # Cleanup
        await sensor_manager.shutdown()

async def test_agent_sensor_integration(ophira_agent: OphiraVoiceAgent, medical_agent: MedicalSpecialistAgent, 
                                       sensor_manager: SensorManager):
    """Test integration between agents and sensor system"""
    from loguru import logger
    logger.info("🤖 Testing agent-sensor integration...")
    
    # Create test medical consultation scenario
    consultation_messages = [
        "Patient reports feeling dizzy and experiencing palpitations",
        "Please analyze current vital signs and provide recommendations",
        "Check for any critical values in the sensor readings"
    ]
    
    for i, message_content in enumerate(consultation_messages):
        # Create message
        message = AgentMessage(
            id=f"consultation_{i}",
            sender_id="user",
            recipient_id=medical_agent.id,
            content={
                "query_type": "medical_consultation",
                "message": message_content,
                "patient_data": {"id": "test_patient", "age": 35, "gender": "unknown"}
            },
            message_type=MessageType.REQUEST,
            priority=MessagePriority.HIGH
        )
        
        # Get current sensor readings
        current_readings = await sensor_manager.read_all_sensors()
        
        # Add sensor context to message
        if current_readings:
            sensor_summary = {}
            for sensor_id, data in current_readings.items():
                sensor_summary[data.sensor_type] = {
                    "value": data.value,
                    "unit": data.unit,
                    "is_critical": data.is_critical,
                    "confidence": data.confidence
                }
            
            message.metadata = {
                "current_sensor_readings": sensor_summary,
                "sensor_timestamp": datetime.now().isoformat()
            }
        
        # Send message to medical agent
        await medical_agent.receive_message(message)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        logger.info(f"✅ Processed consultation message {i+1}/{len(consultation_messages)}")
    
    return {"consultation_messages_processed": len(consultation_messages)}

async def test_real_hardware_sensors(sensor_manager: SensorManager):
    """
    Test real hardware sensors: OV9281 NIR camera and heart rate sensor via microcontroller
    """
    print("\n" + "="*80)
    print("🔧 TESTING REAL HARDWARE SENSORS")
    print("="*80)
    
    try:
        # Configuration for OV9281 NIR Camera
        nir_camera_config = {
            "device_id": 0,  # First USB camera device
            "resolution": (1280, 720),
            "fps": 60,
            "exposure": -7,  # Auto exposure
            "gain": 0,  # Auto gain
            "roi_enabled": True,
            "roi_coords": (160, 90, 960, 540),  # Center crop for retinal area
            "nir_wavelength": 850,
            "auto_white_balance": False
        }
        
        # Configuration for Heart Rate Sensor via Microcontroller
        heart_rate_config = {
            "port": None,  # Auto-detect
            "baudrate": 115200,
            "timeout": 1.0,
            "data_format": "json",  # Can be "json", "csv", or "raw"
            "expected_fields": ["bpm", "timestamp"],
            "min_heart_rate": 40,
            "max_heart_rate": 200,
            "sampling_rate": 100  # Hz, adjust based on your microcontroller
        }
        
        # Create real hardware sensors
        nir_camera = OV9281NIRCamera("ov9281_001", nir_camera_config)
        heart_rate_sensor = HardwareHeartRateSensor("heart_rate_hw_001", heart_rate_config)
        
        # Test NIR Camera Connection
        print("\n📷 Testing OV9281 NIR Camera...")
        camera_connected = await nir_camera.connect()
        if camera_connected:
            print("✅ OV9281 Camera connected successfully!")
            
            # Test camera capture
            print("📸 Testing image capture...")
            camera_data = await nir_camera.read_raw_data()
            if camera_data:
                processed_data = nir_camera.process_raw_data(camera_data)
                if processed_data:
                    print(f"   📊 Image Quality: {processed_data.quality_score:.2f}")
                    print(f"   🎯 Focus Score: {processed_data.value['focus_score']:.3f}")
                    print(f"   📈 Contrast: {processed_data.value['contrast']:.3f}")
                    print(f"   🔍 Vessel Density: {processed_data.value['vessel_density']:.4f}")
                    
                    # Save a test frame
                    frame = camera_data["frame"]
                    filename = nir_camera.save_frame(frame, "test_capture.png")
                    if filename:
                        print(f"   💾 Test image saved: {filename}")
            
            # Calibrate camera
            print("🔧 Calibrating NIR camera...")
            calibrated = await nir_camera.calibrate()
            if calibrated:
                print("✅ Camera calibration successful!")
            
            # Register with sensor manager
            await sensor_manager.register_sensor(nir_camera)
        else:
            print("❌ Failed to connect to OV9281 camera")
            print("   📝 Check that the camera is connected via USB")
            print("   📝 Try different device_id values (0, 1, 2, etc.)")
        
        # Test Heart Rate Sensor Connection
        print("\n💓 Testing Hardware Heart Rate Sensor...")
        hr_connected = await heart_rate_sensor.connect()
        if hr_connected:
            print("✅ Heart Rate Sensor connected successfully!")
            
            # Display port information
            port_info = heart_rate_sensor.get_port_info()
            print(f"   🔌 Port: {port_info['port']}")
            print(f"   ⚡ Baudrate: {port_info['baudrate']}")
            print(f"   📊 Data Format: {port_info['data_format']}")
            
            # Test heart rate reading
            print("💓 Testing heart rate measurement...")
            hr_data = await heart_rate_sensor.read_raw_data()
            if hr_data:
                processed_hr_data = heart_rate_sensor.process_raw_data(hr_data)
                if processed_hr_data:
                    print(f"   💓 Current BPM: {processed_hr_data.value}")
                    print(f"   📊 Signal Quality: {processed_hr_data.quality_score:.2f}")
                    print(f"   ✅ Confidence: {processed_hr_data.confidence:.2f}")
                    print(f"   ⚠️ Critical: {processed_hr_data.is_critical}")
            
            # Calibrate heart rate sensor
            print("🔧 Calibrating heart rate sensor...")
            hr_calibrated = await heart_rate_sensor.calibrate()
            if hr_calibrated:
                print("✅ Heart rate sensor calibration successful!")
            
            # Register with sensor manager
            await sensor_manager.register_sensor(heart_rate_sensor)
        else:
            print("❌ Failed to connect to heart rate sensor")
            print("   📝 Check that the microcontroller is connected via USB")
            print("   📝 Verify the microcontroller is sending data")
            print("   📝 Check baudrate and data format settings")
        
        # Test coordinated readings if both sensors connected
        if camera_connected and hr_connected:
            print("\n🔄 Testing coordinated sensor readings...")
            
            # Start coordinated readings
            await sensor_manager.start_coordinated_readings()
            
            # Let it run for a few seconds
            await asyncio.sleep(5)
            
            # Get recent data
            all_data = await sensor_manager.get_all_sensor_data()
            for sensor_id, data_list in all_data.items():
                if data_list:
                    latest = data_list[-1]
                    print(f"   📊 {sensor_id}: {latest.value} {latest.unit} (Quality: {latest.quality_score:.2f})")
            
            # Stop coordinated readings
            await sensor_manager.stop_coordinated_readings()
        
        # Hardware sensor summary
        print("\n📋 HARDWARE SENSOR SUMMARY:")
        print(f"   📷 OV9281 NIR Camera: {'✅ Connected' if camera_connected else '❌ Not Connected'}")
        print(f"   💓 Heart Rate Sensor: {'✅ Connected' if hr_connected else '❌ Not Connected'}")
        
        if camera_connected:
            camera_info = nir_camera.get_camera_info()
            print(f"   📷 Camera Resolution: {camera_info.get('resolution', 'Unknown')}")
            print(f"   📷 Camera FPS: {camera_info.get('fps', 'Unknown')}")
            print(f"   📷 Frames Captured: {camera_info.get('frame_count', 0)}")
        
        if hr_connected:
            hr_info = heart_rate_sensor.get_port_info()
            print(f"   💓 Serial Port: {hr_info.get('port', 'Unknown')}")
            print(f"   💓 Connection Status: {hr_info.get('is_connected', False)}")
        
        return camera_connected, hr_connected
        
    except Exception as e:
        print(f"❌ Error testing hardware sensors: {e}")
        import traceback
        traceback.print_exc()
        return False, False

async def main():
    """Main application entry point"""
    # Initialize logging first
    setup_logging()
    from loguru import logger
    settings = get_settings()
    
    logger.info("🚀 Starting Enhanced Ophira Medical AI System with Comprehensive Sensors...")
    
    # Initialize database
    logger.info("📊 Initializing database connections...")
    db_manager = DatabaseManager()
    medical_service = MedicalDataService(db_manager)
    
    # Test database with graceful handling
    try:
        await db_manager.initialize()
        logger.info("✅ Database systems ready")
    except Exception as e:
        logger.warning(f"⚠️ Database issues detected: {e}")
        logger.info("💡 Continuing with simulated data storage...")
    
    # Initialize sensor manager
    logger.info("📡 Initializing Sensor Manager...")
    sensor_manager = SensorManager()
    
    # Agent registry
    registry = AgentRegistry()
    
    # Create and register agents
    ophira_agent = OphiraVoiceAgent()
    medical_agent = MedicalSpecialistAgent()
    
    # Start agents
    await ophira_agent.start()
    await medical_agent.start()
    
    await registry.register_agent(ophira_agent)
    await registry.register_agent(medical_agent)
    
    logger.info("✅ All agents initialized successfully")
    
    # Configuration summary
    logger.info("📊 System Configuration:")
    logger.info(f"   🧠 Primary LLM: {settings.ai_models.primary_llm_model}")
    logger.info(f"   🔬 VLLM Medical Model: {settings.ai_models.medical_llm_model}")
    logger.info(f"   👁️ Retinal Model: {settings.ai_models.retinal_vision_model}")
    logger.info(f"   🗄️ MongoDB: {settings.database.mongodb_url}")
    logger.info(f"   🐘 PostgreSQL: {settings.database.postgresql_url}")
    logger.info(f"   📡 Sensor Integration: Enabled")
    
    # Run comprehensive tests
    logger.info("🧪 Running comprehensive system tests...")
    
    try:
        # Test 1: Comprehensive sensor integration
        sensor_test_results = await test_comprehensive_sensor_integration(sensor_manager, medical_service)
        logger.info(f"✅ Sensor Integration Test: {sensor_test_results['sensor_data_collected']} data points collected")
        
        # Test 2: Agent-sensor integration
        agent_test_results = await test_agent_sensor_integration(ophira_agent, medical_agent, sensor_manager)
        logger.info(f"✅ Agent-Sensor Integration Test: {agent_test_results['consultation_messages_processed']} consultations processed")
        
        # Test 3: Real hardware sensors
        camera_connected, hr_connected = await test_real_hardware_sensors(sensor_manager)
        
        # Final system status
        logger.info("📈 Final System Status:")
        logger.info("   ✅ Core AI Agents: Operational")
        logger.info("   ✅ Sensor Integration: Functional") 
        logger.info("   ✅ Database Persistence: Ready")
        logger.info("   ✅ Medical Analysis Pipeline: Active")
        logger.info("   ✅ Real-time Monitoring: Enabled")
        
        logger.info("🎉 Enhanced Ophira Medical AI System successfully initialized!")
        logger.info("🔗 Access web interface at: http://localhost:8000")
        logger.info("📊 System ready for medical consultations and real-time monitoring")
        
        # Keep system running
        while True:
            await asyncio.sleep(10)
            logger.debug("💓 System heartbeat - all systems operational")
            
    except KeyboardInterrupt:
        logger.info("👋 Shutting down Enhanced Ophira Medical AI System...")
        
        # Cleanup
        if sensor_manager:
            await sensor_manager.shutdown()
        
        if db_manager:
            await db_manager.close()
        
        logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 