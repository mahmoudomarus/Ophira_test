"""
Enhanced FastAPI Web Interface for Ophira AI Medical System with Session Management
"""
import asyncio
import json
import uuid
import time
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

from ophira.agents.primary import OphiraVoiceAgent
from ophira.agents.medical_specialist import MedicalSpecialistAgent, AnalysisType
from ophira.agents.sensor_control import SensorControlAgent
from ophira.agents.supervisory import SupervisoryAgent
from ophira.agents.base import AgentRegistry, AgentMessage, MessageType, MessagePriority
from ophira.core.config import get_settings
from ophira.core.logging import setup_logging
from ophira.core.database import (
    initialize_databases, close_databases, medical_data_service, create_test_user,
    get_optimized_db_manager, DatabaseConnectionError, DatabaseRetryExhaustedError
)
from ophira.core.session_manager import (
    session_manager, SessionType, SessionStatus, SecurityLevel,
    create_user_session, get_session, validate_session
)
from ophira.sensors.calibration import get_calibration_status
from ophira.medical_ai.medical_specialist_enhanced import EnhancedMedicalSpecialistAgent
from ophira.medical_ai.models import AIModelManager

# Initialize logging and settings
logger = setup_logging()
settings = get_settings()

# Security
security = HTTPBearer(auto_error=False)

# Enhanced WebSocket Connection Manager
class StableWebSocketManager:
    """Enhanced WebSocket connection manager with health monitoring and error recovery"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        self.max_reconnect_attempts = 3
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self, session_id: str, websocket: WebSocket) -> bool:
        """Connect a WebSocket with enhanced error handling"""
        try:
            await websocket.accept()
            
            # Store connection
            self.connections[session_id] = websocket
            self.connection_metadata[session_id] = {
                "connected_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "message_count": 0,
                "error_count": 0,
                "client_info": websocket.headers.get("user-agent", "unknown")
            }
            
            # Start heartbeat monitoring
            self.heartbeat_tasks[session_id] = asyncio.create_task(
                self._heartbeat_task(websocket, session_id)
            )
            
            logger.info(f"✅ WebSocket connected for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed for session {session_id}: {e}")
            return False
    
    async def disconnect(self, session_id: str, reason: str = "normal"):
        """Disconnect a WebSocket gracefully"""
        try:
            # Cancel heartbeat task
            if session_id in self.heartbeat_tasks:
                self.heartbeat_tasks[session_id].cancel()
                del self.heartbeat_tasks[session_id]
            
            # Close connection
            if session_id in self.connections:
                websocket = self.connections[session_id]
                try:
                    await websocket.close()
                except Exception:
                    pass  # Connection might already be closed
                del self.connections[session_id]
            
            # Clean up metadata
            if session_id in self.connection_metadata:
                metadata = self.connection_metadata[session_id]
                duration = (datetime.now() - metadata["connected_at"]).total_seconds()
                logger.info(f"🔌 WebSocket disconnected for session {session_id} "
                          f"(duration: {duration:.1f}s, reason: {reason})")
                del self.connection_metadata[session_id]
                
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect for {session_id}: {e}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send message with error handling and retry logic"""
        if session_id not in self.connections:
            logger.warning(f"No WebSocket connection for session {session_id}")
            return False
        
        websocket = self.connections[session_id]
        
        try:
            await websocket.send_text(json.dumps(message))
            
            # Update metadata
            if session_id in self.connection_metadata:
                self.connection_metadata[session_id]["message_count"] += 1
                self.connection_metadata[session_id]["last_heartbeat"] = datetime.now()
            
            return True
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during send for session {session_id}")
            await self.disconnect(session_id, "client_disconnect")
            return False
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {session_id}: {e}")
            
            # Update error count
            if session_id in self.connection_metadata:
                self.connection_metadata[session_id]["error_count"] += 1
                
                # Disconnect if too many errors
                if self.connection_metadata[session_id]["error_count"] > 5:
                    await self.disconnect(session_id, "too_many_errors")
            
            return False
    
    async def broadcast(self, message: Dict[str, Any], exclude_sessions: List[str] = None):
        """Broadcast message to all connected clients"""
        exclude_sessions = exclude_sessions or []
        
        tasks = []
        for session_id in list(self.connections.keys()):
            if session_id not in exclude_sessions:
                tasks.append(self.send_message(session_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _heartbeat_task(self, websocket: WebSocket, session_id: str):
        """Maintain WebSocket connection health with heartbeat"""
        try:
            while session_id in self.connections:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check if connection is still alive
                try:
                    # Use send_text with a ping message instead of ping()
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                    # Update last heartbeat
                    if session_id in self.connection_metadata:
                        self.connection_metadata[session_id]["last_heartbeat"] = datetime.now()
                        
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected during heartbeat for session {session_id}")
                    await self.disconnect(session_id, "client_disconnect")
                    break
                except Exception as e:
                    logger.warning(f"Heartbeat failed for session {session_id}: {e}")
                    await self.disconnect(session_id, "heartbeat_failed")
                    break
                    
        except asyncio.CancelledError:
            logger.debug(f"Heartbeat task cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Heartbeat task error for session {session_id}: {e}")
            await self.disconnect(session_id, "heartbeat_error")
    
    async def handle_connection_error(self, session_id: str, error: Exception):
        """Handle WebSocket connection errors gracefully"""
        logger.error(f"WebSocket error for session {session_id}: {error}")
        await self.disconnect(session_id, f"error: {type(error).__name__}")
        
        # Notify monitoring system (could be extended to send alerts)
        await self._notify_connection_failure(session_id, str(error))
    
    async def _notify_connection_failure(self, session_id: str, error_message: str):
        """Notify monitoring system of connection failures"""
        # This could be extended to send alerts to monitoring systems
        logger.warning(f"Connection failure notification: {session_id} - {error_message}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for monitoring"""
        total_connections = len(self.connections)
        total_messages = sum(
            metadata.get("message_count", 0) 
            for metadata in self.connection_metadata.values()
        )
        total_errors = sum(
            metadata.get("error_count", 0) 
            for metadata in self.connection_metadata.values()
        )
        
        return {
            "active_connections": total_connections,
            "total_messages_sent": total_messages,
            "total_errors": total_errors,
            "connections": {
                session_id: {
                    "connected_duration": (datetime.now() - metadata["connected_at"]).total_seconds(),
                    "message_count": metadata["message_count"],
                    "error_count": metadata["error_count"],
                    "last_heartbeat": metadata["last_heartbeat"].isoformat()
                }
                for session_id, metadata in self.connection_metadata.items()
            }
        }

# Create WebSocket manager instance
websocket_manager = StableWebSocketManager()

# API Models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: Optional[str] = None
    device_info: Optional[Dict[str, str]] = None

class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    session_type: str
    status: str
    created_at: str
    expires_at: str
    warnings: List[Dict[str, Any]] = []

class HealthAnalysisRequest(BaseModel):
    analysis_types: List[str] = ["general_health"]
    sensor_data: Optional[Dict[str, Any]] = None
    emergency: bool = False

# Global medical AI components
enhanced_medical_specialist: Optional[EnhancedMedicalSpecialistAgent] = None
ai_model_manager: Optional[AIModelManager] = None

# Add new Pydantic models after existing models (around line 250)

class MedicalAnalysisRequest(BaseModel):
    """Request for medical AI analysis"""
    patient_id: str
    analysis_type: str  # "retinal", "cardiovascular", "comprehensive"
    image_data: Optional[str] = None  # Base64 encoded image
    sensor_data: Optional[Dict[str, Any]] = None
    patient_context: Optional[Dict[str, Any]] = None
    emergency: bool = False

class RetinalAnalysisRequest(BaseModel):
    """Request for retinal analysis"""
    patient_id: str
    image_data: str  # Base64 encoded fundus image
    patient_context: Optional[Dict[str, Any]] = None

class MedicalAnalysisResponse(BaseModel):
    """Response from medical AI analysis"""
    session_id: str
    analysis_type: str
    risk_level: str
    confidence: float
    emergency_alert: bool
    recommendations: List[str]
    report_available: bool
    processing_time: float

class ClinicalReportResponse(BaseModel):
    """Clinical report response"""
    report_id: str
    patient_id: str
    session_id: str
    report_type: str
    executive_summary: str
    formatted_report: Optional[str] = None
    generated_timestamp: str

# FastAPI app
app = FastAPI(
    title="Ophira AI Medical System",
    description="Advanced medical monitoring and analysis system with multi-agent architecture",
    version="1.0.0"
)

# Enhanced Error Handling Middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Enhanced error handling middleware with comprehensive error recovery"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to headers for tracking
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        
        # Log successful requests
        process_time = time.time() - start_time
        logger.debug(f"Request {request_id} completed in {process_time:.3f}s")
        
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Request timeout for {request.url} (ID: {request_id})")
        return JSONResponse(
            status_code=408,
            content={
                "error": "Request timeout",
                "message": "The request took too long to process",
                "retry_after": 5,
                "request_id": request_id
            }
        )
        
    except DatabaseConnectionError as e:
        logger.error(f"Database connection error for {request.url} (ID: {request_id}): {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable",
                "message": "Database connection issue - please try again",
                "retry_after": 30,
                "request_id": request_id
            }
        )
        
    except DatabaseRetryExhaustedError as e:
        logger.critical(f"Database retry exhausted for {request.url} (ID: {request_id}): {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service unavailable",
                "message": "Database service is currently unavailable",
                "retry_after": 60,
                "request_id": request_id
            }
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (they're handled by FastAPI)
        raise e
        
    except Exception as e:
        logger.exception(f"Unhandled error for {request.url} (ID: {request_id}): {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "support_info": "Please contact support with this request ID"
            }
        )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ophira_agent: Optional[OphiraVoiceAgent] = None
medical_agent: Optional[MedicalSpecialistAgent] = None
sensor_agent: Optional[SensorControlAgent] = None
supervisory_agent: Optional[SupervisoryAgent] = None
test_user_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with Phase 2 medical AI initialization"""
    global ophira_agent, medical_agent, sensor_agent, supervisory_agent
    global enhanced_medical_specialist, ai_model_manager
    
    try:
        logger.info("🚀 Starting Ophira AI Medical Monitoring System (Phase 2)")
        
        # Initialize databases
        await initialize_databases()
        logger.info("✅ Databases initialized")
        
        # Initialize AI model manager
        ai_model_manager = AIModelManager()
        await ai_model_manager.initialize()
        logger.info("✅ AI Model Manager initialized")
        
        # Initialize enhanced medical specialist
        enhanced_medical_specialist = EnhancedMedicalSpecialistAgent()
        await enhanced_medical_specialist.initialize()
        logger.info("✅ Enhanced Medical Specialist initialized")
        
        # Initialize original agents
        ophira_agent = OphiraVoiceAgent()
        medical_agent = MedicalSpecialistAgent()
        sensor_agent = SensorControlAgent()
        supervisory_agent = SupervisoryAgent()
        
        # Start all agents (BaseAgent uses start() method, not initialize())
        await ophira_agent.start()
        await medical_agent.start()
        await sensor_agent.start()
        await supervisory_agent.start()
        
        logger.info("✅ All agents started")
        
        # Note: Supervisor monitoring starts automatically with agent tasks
        # No need to call start_monitoring() separately
        logger.info("✅ Supervisor monitoring started automatically")
        
        # Create test user
        try:
            test_user = await create_test_user()
            logger.info(f"✅ Test user created: {test_user['user_id']}")
        except Exception as e:
            logger.warning(f"Test user creation failed: {e}")
        
        logger.info("🎉 Ophira AI System (Phase 2) startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown with enhanced error handling."""
    global ophira_agent, medical_agent, sensor_agent, supervisory_agent
    
    try:
        logger.info("🔄 Shutting down Ophira AI Web Interface...")
        
        # Stop agents gracefully
        shutdown_tasks = []
        
        if ophira_agent:
            shutdown_tasks.append(("Primary Voice Agent", ophira_agent.stop()))
        if medical_agent:
            shutdown_tasks.append(("Medical Specialist Agent", medical_agent.stop()))
        if sensor_agent:
            shutdown_tasks.append(("Sensor Control Agent", sensor_agent.stop()))
        if supervisory_agent:
            shutdown_tasks.append(("Supervisory Agent", supervisory_agent.stop()))
        
        if shutdown_tasks:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *[task for _, task in shutdown_tasks],
                        return_exceptions=True
                    ),
                    timeout=10.0
                )
                
                for i, (agent_name, _) in enumerate(shutdown_tasks):
                    if i < len(results) and isinstance(results[i], Exception):
                        logger.warning(f"⚠️ Error stopping {agent_name}: {results[i]}")
                    else:
                        logger.info(f"✅ {agent_name} stopped")
                        
            except asyncio.TimeoutError:
                logger.warning("⚠️ Agent shutdown timeout")
        
        # Stop session manager
        try:
            await session_manager.stop()
            logger.info("✅ Session manager stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping session manager: {e}")
        
        # Close database connections
        try:
            db_manager = get_optimized_db_manager()
            await db_manager.close()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing database connections: {e}")
        
        logger.info("🔒 Ophira AI Web Interface shutdown complete")
        
    except Exception as e:
        logger.error(f"💥 Error during shutdown: {e}")

# Session management endpoints

@app.post("/api/login", response_model=SessionInfo)
async def login(request: Request, login_data: UserLogin):
    """Create a new user session."""
    try:
        # Get client information
        client_host = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # For demo purposes, accept any username
        user_id = login_data.username or "demo_user"
        
        # Create user session
        session_data = await create_user_session(
            user_id=user_id,
            session_type=SessionType.USER_WEB,
            ip_address=client_host,
            user_agent=user_agent,
            device_fingerprint=json.dumps(login_data.device_info) if login_data.device_info else None
        )
        
        # Get any session warnings
        warnings = await session_manager.get_session_warnings(session_data.session_id)
        
        return SessionInfo(
            session_id=session_data.session_id,
            user_id=session_data.user_id,
            session_type=session_data.session_type.value,
            status=session_data.status.value,
            created_at=session_data.created_at.isoformat(),
            expires_at=session_data.expires_at.isoformat(),
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logout")
async def logout(session_id: str):
    """Terminate a user session."""
    try:
        await session_manager.terminate_session(session_id, "user_logout")
        return {"status": "logged_out"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    try:
        session_data = await get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        warnings = await session_manager.get_session_warnings(session_id)
        
        return SessionInfo(
            session_id=session_data.session_id,
            user_id=session_data.user_id,
            session_type=session_data.session_type.value,
            status=session_data.status.value,
            created_at=session_data.created_at.isoformat(),
            expires_at=session_data.expires_at.isoformat(),
            warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/{session_id}/refresh")
async def refresh_session(session_id: str):
    """Refresh a session to extend its lifetime."""
    try:
        success = await session_manager.refresh_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or invalid")
        
        return {"status": "refreshed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session dependency
async def get_current_session(session_id: Optional[str] = None):
    """Dependency to get and validate current session."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Session ID required")
    
    is_valid = await validate_session(session_id)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    session_data = await get_session(session_id)
    return session_data

# Health analysis endpoints

@app.post("/api/chat")
async def chat_endpoint(
    message_data: ChatMessage,
    session_data = Depends(get_current_session)
):
    """Enhanced chat endpoint with session validation."""
    try:
        # Refresh session activity
        await session_manager.refresh_session(session_data.session_id)
        
        # Store user message in database
        try:
            await medical_data_service.store_conversation(
                session_data.user_id, 
                session_data.session_id, 
                message_data.message, 
                "user"
            )
        except Exception as e:
            logger.warning(f"Failed to store chat message: {e}")
        
        # Check for emergency keywords
        emergency_keywords = ["emergency", "help", "urgent", "chest pain", "can't breathe", "dizzy"]
        is_emergency = any(keyword in message_data.message.lower() for keyword in emergency_keywords)
        
        if is_emergency:
            # Activate emergency mode for session
            await session_manager.activate_emergency_mode(
                session_data.session_id,
                {"trigger": "chat_emergency", "keywords": emergency_keywords}
            )
        
        # Process with Ophira
        agent_message = AgentMessage(
            sender_id="web_interface",
            recipient_id=ophira_agent.id,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.CRITICAL if is_emergency else MessagePriority.NORMAL,
            content={
                "user_input": message_data.message,
                "user_id": session_data.user_id,
                "session_id": session_data.session_id,
                "interface": "web",
                "emergency": is_emergency
            }
        )
        
        await ophira_agent.receive_message(agent_message)
        
        # For now, return a simple response
        # In production, this would wait for the agent's response
        response_text = "I'm processing your request. Please wait while I analyze your health data..."
        if is_emergency:
            response_text = "🚨 I've detected this may be an emergency. I'm analyzing your situation immediately and will provide urgent guidance."
        
        # Store AI response
        try:
            await medical_data_service.store_conversation(
                session_data.user_id,
                session_data.session_id,
                response_text,
                "ai"
            )
        except Exception as e:
            logger.warning(f"Failed to store AI response: {e}")
        
        return {
            "response": response_text,
            "emergency_mode": is_emergency,
            "session_warnings": await session_manager.get_session_warnings(session_data.session_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/health/analyze")
async def analyze_health(
    analysis_request: HealthAnalysisRequest,
    session_data = Depends(get_current_session)
):
    """Request comprehensive health analysis."""
    try:
        # Refresh session activity
        await session_manager.refresh_session(session_data.session_id)
        
        if analysis_request.emergency:
            # Activate emergency mode
            await session_manager.activate_emergency_mode(
                session_data.session_id,
                {"trigger": "health_analysis_emergency", "analysis_types": analysis_request.analysis_types}
            )
        
        # Request analysis from supervisory agent
        agent_message = AgentMessage(
            sender_id="web_interface",
            recipient_id=supervisory_agent.id,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.CRITICAL if analysis_request.emergency else MessagePriority.HIGH,
            content={
                "coordinate_analysis": True,
                "analysis_types": analysis_request.analysis_types,
                "user_id": session_data.user_id,
                "session_id": session_data.session_id,
                "input_data": analysis_request.sensor_data or {},
                "emergency": analysis_request.emergency
            }
        )
        
        await supervisory_agent.receive_message(agent_message)
        
        return {
            "status": "analysis_requested",
            "analysis_types": analysis_request.analysis_types,
            "emergency_mode": analysis_request.emergency,
            "estimated_completion": (datetime.now().timestamp() + 300) * 1000,  # 5 minutes
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors/status")
async def get_sensor_status(session_data = Depends(get_current_session)):
    """Get current sensor status with graceful error handling."""
    try:
        # Refresh session activity
        await session_manager.refresh_session(session_data.session_id)
        
        # Initialize default sensor status
        sensor_status = {
            "sensors": {},
            "timestamp": datetime.now().isoformat(),
            "connection_status": "checking",
            "errors": []
        }
        
        # Try to get actual sensor status from sensor control agent
        try:
            if sensor_agent:
                status_message = AgentMessage(
                    sender_id="web_interface",
                    recipient_id=sensor_agent.id,
                    message_type=MessageType.REQUEST,
                    priority=MessagePriority.NORMAL,
                    content={
                        "get_sensor_status": True,
                        "user_id": session_data.user_id,
                        "session_id": session_data.session_id
                    }
                )
                
                await sensor_agent.receive_message(status_message)
                
                # For now, return mock status with error handling
                sensor_status["sensors"] = {
                    "usb_camera_primary": {
                        "status": "disconnected",
                        "last_data": None,
                        "quality": "unknown",
                        "error": "Camera not connected"
                    },
                    "vl53l4cx_tof_primary": {
                        "status": "disconnected", 
                        "last_data": None,
                        "quality": "unknown",
                        "error": "ToF sensor not connected"
                    },
                    "heart_rate_sensor": {
                        "status": "disconnected",
                        "last_data": None,
                        "quality": "unknown", 
                        "error": "Heart rate sensor not connected"
                    }
                }
                sensor_status["connection_status"] = "no_sensors_connected"
                sensor_status["errors"].append("No physical sensors are currently connected")
                
        except Exception as sensor_error:
            logger.warning(f"Sensor agent communication error: {sensor_error}")
            sensor_status["connection_status"] = "agent_error"
            sensor_status["errors"].append(f"Sensor agent communication failed: {str(sensor_error)}")
        
        return sensor_status
        
    except Exception as e:
        logger.error(f"Sensor status error: {e}")
        # Return error status instead of raising exception
        return {
            "sensors": {},
            "timestamp": datetime.now().isoformat(),
            "connection_status": "error",
            "errors": [f"Failed to get sensor status: {str(e)}"]
        }

# Voice API Endpoints
@app.post("/api/voice/start")
async def start_voice_session(
    request: Request,
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Start a voice interaction session."""
    try:
        # Get session ID from header or query parameter
        if not session_id:
            session_id = request.query_params.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=401, detail="Session ID required in X-Session-ID header or session_id query parameter")
        
        # Validate session
        session_data = await get_current_session(session_id)
        
        # Refresh session activity
        await session_manager.refresh_session(session_data.session_id)
        
        # Notify the voice agent about voice session start
        if ophira_agent:
            voice_message = AgentMessage(
                sender_id="web_interface",
                recipient_id=ophira_agent.id,
                message_type=MessageType.REQUEST,
                priority=MessagePriority.NORMAL,
                content={
                    "voice_session_start": True,
                    "user_id": session_data.user_id,
                    "session_id": session_id,
                    "interface": "voice",
                    "capabilities": ["speech_recognition", "text_to_speech", "real_time_processing"]
                }
            )
            
            await ophira_agent.receive_message(voice_message)
        
        # Log voice session start
        logger.info(f"Voice session started for user {session_data.user_id}, session {session_id}")
        
        return {
            "success": True,
            "message": "Voice session started successfully",
            "session_id": session_id,
            "voice_capabilities": {
                "speech_recognition": True,
                "text_to_speech": True,
                "real_time_processing": True,
                "emergency_detection": True,
                "multi_language": False  # Can be enhanced later
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Voice session start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start voice session: {str(e)}")

@app.post("/api/voice/stop")
async def stop_voice_session(
    request: Request,
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Stop the current voice interaction session."""
    try:
        # Get session ID from header or query parameter
        if not session_id:
            session_id = request.query_params.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=401, detail="Session ID required in X-Session-ID header or session_id query parameter")
        
        # Validate session
        session_data = await get_current_session(session_id)
        
        # Refresh session activity
        await session_manager.refresh_session(session_data.session_id)
        
        # Notify the voice agent about voice session stop
        if ophira_agent:
            voice_message = AgentMessage(
                sender_id="web_interface",
                recipient_id=ophira_agent.id,
                message_type=MessageType.REQUEST,
                priority=MessagePriority.NORMAL,
                content={
                    "voice_session_stop": True,
                    "user_id": session_data.user_id,
                    "session_id": session_id,
                    "interface": "voice"
                }
            )
            
            await ophira_agent.receive_message(voice_message)
        
        # Log voice session stop
        logger.info(f"Voice session stopped for user {session_data.user_id}, session {session_id}")
        
        return {
            "success": True,
            "message": "Voice session stopped successfully",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Voice session stop error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop voice session: {str(e)}")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket endpoint with stable connection management."""
    
    # Use the enhanced WebSocket manager
    connection_success = await websocket_manager.connect(session_id, websocket)
    if not connection_success:
        return
    
    try:
        # Validate session
        is_valid = await validate_session(session_id)
        if not is_valid:
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Invalid or expired session",
                "code": "SESSION_INVALID"
            })
            await websocket_manager.disconnect(session_id, "invalid_session")
            return
        
        session_data = await get_session(session_id)
        if not session_data:
            await websocket_manager.send_message(session_id, {
                "type": "error", 
                "message": "Session not found",
                "code": "SESSION_NOT_FOUND"
            })
            await websocket_manager.disconnect(session_id, "session_not_found")
            return
        
        # Send welcome message
        await websocket_manager.send_message(session_id, {
            "type": "welcome",
            "message": f"Welcome back, {session_data.user_id}!",
            "session_info": {
                "session_id": session_data.session_id,
                "status": session_data.status.value,
                "security_level": session_data.security_context.level.value,
                "expires_at": session_data.expires_at.isoformat()
            }
        })
        
        # Check for session warnings
        warnings = await session_manager.get_session_warnings(session_id)
        if warnings:
            await websocket_manager.send_message(session_id, {
                "type": "warnings",
                "warnings": warnings
            })
        
        # Main message loop with enhanced error handling
        while session_id in websocket_manager.connections:
            try:
                # Receive message from client with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=websocket_manager.connection_timeout
                )
                
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                message_type = message_data.get("type", "text")  # Default to text message
                
                # Refresh session activity
                await session_manager.refresh_session(session_id)
                
                # Handle different message types
                if message_type == "voice_message":
                    # Handle voice message specifically
                    user_message = message_data.get("message", "")
                    confidence = message_data.get("confidence", 1.0)
                    language = message_data.get("language", "en-US")
                    
                    logger.info(f"Voice message received from {session_data.user_id}: '{user_message}' (confidence: {confidence})")
                    
                    # Store voice message with metadata
                    try:
                        await medical_data_service.store_conversation(
                            session_data.user_id, 
                            session_id, 
                            user_message, 
                            "user",
                            metadata={
                                "source": "voice",
                                "confidence": confidence,
                                "language": language,
                                "timestamp": message_data.get("timestamp", datetime.now().isoformat())
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store voice message: {e}")
                
                else:
                    # Handle regular text message
                    try:
                        await medical_data_service.store_conversation(
                            session_data.user_id, session_id, user_message, "user"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store WebSocket message: {e}")
                        # Continue processing even if storage fails
                
                # Check for emergency
                emergency_keywords = ["emergency", "help", "urgent", "chest pain", "can't breathe", "dizzy"]
                is_emergency = any(keyword in user_message.lower() for keyword in emergency_keywords)
                
                if is_emergency:
                    try:
                        await session_manager.activate_emergency_mode(
                            session_id,
                            {"trigger": "websocket_emergency", "message": user_message}
                        )
                        
                        await websocket_manager.send_message(session_id, {
                            "type": "emergency_activated",
                            "message": "🚨 Emergency mode activated. Processing your request with highest priority."
                        })
                    except Exception as e:
                        logger.error(f"Failed to activate emergency mode: {e}")
                        # Continue processing even if emergency activation fails
                
                # Process with Ophira (with error handling)
                try:
                    if ophira_agent:
                        # Prepare content based on message type
                        agent_content = {
                            "user_input": user_message,
                            "user_id": session_data.user_id,
                            "session_id": session_id,
                            "interface": "websocket",
                            "emergency": is_emergency
                        }
                        
                        # Add voice-specific metadata if it's a voice message
                        if message_type == "voice_message":
                            agent_content.update({
                                "source": "voice",
                                "confidence": message_data.get("confidence", 1.0),
                                "language": message_data.get("language", "en-US"),
                                "voice_timestamp": message_data.get("timestamp"),
                                "interface": "voice_websocket"
                            })
                        
                        agent_message = AgentMessage(
                            sender_id="web_interface",
                            recipient_id=ophira_agent.id,
                            message_type=MessageType.REQUEST,
                            priority=MessagePriority.CRITICAL if is_emergency else MessagePriority.NORMAL,
                            content=agent_content
                        )
                        
                        await ophira_agent.receive_message(agent_message)
                    else:
                        logger.warning("Ophira agent not available")
                        await websocket_manager.send_message(session_id, {
                            "type": "error",
                            "message": "AI agent temporarily unavailable",
                            "code": "AGENT_UNAVAILABLE"
                        })
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing message with Ophira: {e}")
                    await websocket_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Error processing your request",
                        "code": "PROCESSING_ERROR"
                    })
                    continue
                
                # Send acknowledgment
                await websocket_manager.send_message(session_id, {
                    "type": "processing",
                    "message": "Processing your request...",
                    "emergency": is_emergency,
                    "timestamp": datetime.now().isoformat()
                })
                
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket timeout for session {session_id}")
                await websocket_manager.send_message(session_id, {
                    "type": "timeout",
                    "message": "Connection timeout - please send a message to keep the connection alive"
                })
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected for session {session_id}")
                break
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received from session {session_id}: {e}")
                await websocket_manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid message format",
                    "code": "INVALID_JSON"
                })
                
            except Exception as e:
                logger.error(f"Unexpected WebSocket error for session {session_id}: {e}")
                await websocket_manager.handle_connection_error(session_id, e)
                break
                
    except Exception as e:
        logger.error(f"Critical WebSocket error for session {session_id}: {e}")
        await websocket_manager.handle_connection_error(session_id, e)
        
    finally:
        # Ensure cleanup
        await websocket_manager.disconnect(session_id, "connection_ended")

# Add WebSocket monitoring endpoint
@app.get("/api/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics for monitoring."""
    try:
        stats = websocket_manager.get_connection_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get WebSocket statistics")

# Add monitoring dashboard endpoints after the existing endpoints

@app.get("/api/system/health")
async def get_system_health():
    """Get comprehensive system health status for monitoring dashboard."""
    try:
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Database health
        try:
            db_manager = get_optimized_db_manager()
            db_health = await db_manager.health_check()
            health_report["components"]["database"] = {
                "status": "healthy" if db_health["status"] == "healthy" else "unhealthy",
                "details": db_health
            }
        except Exception as e:
            health_report["components"]["database"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Sensor calibration status
        try:
            calibration_status = await get_calibration_status()
            expired_sensors = sum(
                1 for status in calibration_status.values() 
                if status.get("is_expired", True)
            )
            health_report["components"]["sensors"] = {
                "status": "healthy" if expired_sensors == 0 else "warning",
                "total_sensors": len(calibration_status),
                "expired_sensors": expired_sensors,
                "details": calibration_status
            }
        except Exception as e:
            health_report["components"]["sensors"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Agent status
        agent_status = {
            "ophira": ophira_agent is not None,
            "medical": medical_agent is not None,
            "sensor": sensor_agent is not None,
            "supervisory": supervisory_agent is not None
        }
        active_agents = sum(agent_status.values())
        health_report["components"]["agents"] = {
            "status": "healthy" if active_agents >= 3 else "warning",
            "active_agents": active_agents,
            "total_agents": len(agent_status),
            "details": agent_status
        }
        
        # WebSocket connections
        try:
            ws_stats = websocket_manager.get_connection_stats()
            health_report["components"]["websockets"] = {
                "status": "healthy",
                "active_connections": ws_stats["active_connections"],
                "total_messages": ws_stats["total_messages_sent"],
                "total_errors": ws_stats["total_errors"]
            }
        except Exception as e:
            health_report["components"]["websockets"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Session manager
        try:
            # Check if session manager is running
            session_healthy = hasattr(session_manager, '_running') and session_manager._running
            health_report["components"]["sessions"] = {
                "status": "healthy" if session_healthy else "warning",
                "manager_running": session_healthy
            }
        except Exception as e:
            health_report["components"]["sessions"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_report["components"].values()]
        if "error" in component_statuses:
            health_report["overall_status"] = "error"
        elif "warning" in component_statuses:
            health_report["overall_status"] = "warning"
        else:
            health_report["overall_status"] = "healthy"
        
        return health_report
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
        )

@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get detailed system performance metrics."""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "database": {},
            "websockets": {},
            "calibration": {},
            "performance": {}
        }
        
        # Database metrics
        try:
            db_manager = get_optimized_db_manager()
            db_health = await db_manager.health_check()
            # Extract metrics-like information from health check
            metrics["database"] = {
                "status": db_health.get("status", "unknown"),
                "response_time_ms": db_health.get("response_time", 0),
                "connections": db_health.get("connections", {}),
                "errors": db_health.get("errors", [])
            }
        except Exception as e:
            metrics["database"] = {"error": str(e)}
        
        # WebSocket metrics
        try:
            ws_stats = websocket_manager.get_connection_stats()
            metrics["websockets"] = ws_stats
        except Exception as e:
            metrics["websockets"] = {"error": str(e)}
        
        # Calibration metrics
        try:
            calibration_status = await get_calibration_status()
            metrics["calibration"] = {
                "total_sensors": len(calibration_status),
                "calibrated_sensors": sum(
                    1 for status in calibration_status.values()
                    if status.get("status") == "calibrated"
                ),
                "expired_sensors": sum(
                    1 for status in calibration_status.values()
                    if status.get("is_expired", True)
                ),
                "average_quality": sum(
                    status.get("quality_score", 0) or 0
                    for status in calibration_status.values()
                ) / len(calibration_status) if calibration_status else 0
            }
        except Exception as e:
            metrics["calibration"] = {"error": str(e)}
        
        # Performance metrics (simplified)
        import psutil
        try:
            metrics["performance"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except Exception:
            metrics["performance"] = {
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_percent": 0
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"System metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect system metrics")

@app.post("/api/system/calibrate")
async def trigger_sensor_calibration():
    """Trigger sensor calibration process."""
    try:
        from ophira.sensors.calibration import calibrate_all_sensors
        
        logger.info("🔧 Manual sensor calibration triggered via API")
        
        # Run calibration in background
        calibration_results = await calibrate_all_sensors()
        
        # Process results
        successful = sum(
            1 for result in calibration_results.values()
            if result.status == "calibrated"
        )
        total = len(calibration_results)
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "results": {
                "total_sensors": total,
                "successful_calibrations": successful,
                "success_rate": successful / total if total > 0 else 0,
                "details": {
                    sensor_id: {
                        "status": result.status,
                        "quality_score": result.quality_score,
                        "duration": result.calibration_duration,
                        "recommendations": result.recommendations
                    }
                    for sensor_id, result in calibration_results.items()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Manual calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@app.get("/api/system/stability-test")
async def run_stability_test():
    """Run quick stability test and return results."""
    try:
        from test_phase1_stability import run_quick_stability_check
        
        logger.info("🧪 Quick stability test triggered via API")
        
        # Run quick stability check
        test_results = await run_quick_stability_check()
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results
        }
        
    except Exception as e:
        logger.error(f"Stability test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stability test failed: {str(e)}")

@app.get("/api/system/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data."""
    try:
        # Collect all monitoring data
        health_data = await get_system_health()
        metrics_data = await get_system_metrics()
        
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "health": health_data,
            "metrics": metrics_data,
            "alerts": [],
            "recommendations": []
        }
        
        # Generate alerts based on health status
        if health_data["overall_status"] == "error":
            dashboard["alerts"].append({
                "level": "critical",
                "message": "System has critical errors requiring immediate attention",
                "timestamp": datetime.now().isoformat()
            })
        elif health_data["overall_status"] == "warning":
            dashboard["alerts"].append({
                "level": "warning", 
                "message": "System has warnings that should be addressed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate recommendations
        if health_data["components"]["database"]["status"] != "healthy":
            dashboard["recommendations"].append("Check database connections and performance")
        
        if health_data["components"]["sensors"]["status"] != "healthy":
            dashboard["recommendations"].append("Recalibrate expired sensors")
        
        if health_data["components"]["agents"]["status"] != "healthy":
            dashboard["recommendations"].append("Restart failed agents")
        
        if not dashboard["recommendations"]:
            dashboard["recommendations"].append("System is operating normally")
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Dashboard data collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect dashboard data")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ophira AI - Medical Monitoring System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #e9ecef;
            }
            .header h1 {
                color: #2c3e50;
                margin: 0;
                font-size: 2.5em;
            }
            .header p {
                color: #7f8c8d;
                margin: 10px 0 0 0;
                font-size: 1.1em;
            }
            .status-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #28a745;
            }
            .warning-card {
                background: #fff3cd;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #ffc107;
            }
            .emergency-card {
                background: #f8d7da;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #dc3545;
            }
            .session-info {
                background: #e3f2fd;
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                font-family: monospace;
                font-size: 0.9em;
            }
            .btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                margin: 5px;
                transition: background 0.3s;
            }
            .btn:hover {
                background: #0056b3;
            }
            .btn-success { background: #28a745; }
            .btn-success:hover { background: #1e7e34; }
            .btn-warning { background: #ffc107; color: #333; }
            .btn-warning:hover { background: #e0a800; }
            .btn-danger { background: #dc3545; }
            .btn-danger:hover { background: #c82333; }
            #messages {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 15px;
                margin: 20px 0;
                background: #fafafa;
                border-radius: 8px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
            }
            .user-message {
                background: #007bff;
                color: white;
                margin-left: 20%;
            }
            .ai-message {
                background: #e9ecef;
                margin-right: 20%;
            }
            .system-message {
                background: #fff3cd;
                text-align: center;
                font-style: italic;
            }
            input[type="text"] {
                width: calc(100% - 100px);
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 1em;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }
            @media (max-width: 768px) {
                .grid { grid-template-columns: 1fr; }
                .container { margin: 10px; padding: 15px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Ophira AI</h1>
                <p>Advanced Medical Monitoring & Analysis System</p>
            </div>
            
            <div id="session-status" class="status-card">
                <h3>🔐 Session Status</h3>
                <p>Not logged in</p>
                <div class="session-info" id="session-info" style="display: none;"></div>
            </div>
            
            <div id="warnings" style="display: none;" class="warning-card">
                <h3>⚠️ System Warnings</h3>
                <ul id="warning-list"></ul>
            </div>
            
            <div class="grid">
                <div>
                    <h3>🎯 Quick Actions</h3>
                    <button class="btn btn-success" onclick="login()">Login / Start Session</button>
                    <button class="btn" onclick="requestHealthCheck()">Health Check</button>
                    <button class="btn btn-warning" onclick="requestAnalysis()">Full Analysis</button>
                    <button class="btn btn-danger" onclick="emergency()">🚨 Emergency</button>
                </div>
                <div>
                    <h3>📊 System Status</h3>
                    <div id="system-status">
                        <p>✅ Agents: Online</p>
                        <p>✅ Sensors: Connected</p>
                        <p>✅ Database: Active</p>
                        <p>✅ Session Manager: Running</p>
                    </div>
                </div>
            </div>
            
            <div>
                <h3>💬 Chat with Ophira</h3>
                <div id="messages"></div>
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="messageInput" placeholder="Ask Ophira about your health..." 
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button class="btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            let currentSession = null;
            let websocket = null;
            
            // Login function
            async function login() {
                try {
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            username: 'demo_user',
                            device_info: {
                                platform: navigator.platform,
                                userAgent: navigator.userAgent
                            }
                        })
                    });
                    
                    const session = await response.json();
                    currentSession = session;
                    updateSessionDisplay();
                    connectWebSocket();
                    
                } catch (error) {
                    console.error('Login error:', error);
                    addMessage('system', 'Login failed: ' + error.message);
                }
            }
            
            function updateSessionDisplay() {
                const statusEl = document.getElementById('session-status');
                const infoEl = document.getElementById('session-info');
                
                if (currentSession) {
                    statusEl.className = 'status-card';
                    statusEl.innerHTML = `
                        <h3>🔐 Session Active</h3>
                        <p>User: ${currentSession.user_id}</p>
                        <p>Status: ${currentSession.status}</p>
                        <p>Expires: ${new Date(currentSession.expires_at).toLocaleString()}</p>
                    `;
                    
                    infoEl.style.display = 'block';
                    infoEl.textContent = `Session ID: ${currentSession.session_id}`;
                    
                    // Show warnings if any
                    if (currentSession.warnings && currentSession.warnings.length > 0) {
                        showWarnings(currentSession.warnings);
                    }
                }
            }
            
            function showWarnings(warnings) {
                const warningsEl = document.getElementById('warnings');
                const listEl = document.getElementById('warning-list');
                
                listEl.innerHTML = '';
                warnings.forEach(warning => {
                    const li = document.createElement('li');
                    li.textContent = `${warning.type}: ${warning.message}`;
                    listEl.appendChild(li);
                });
                
                warningsEl.style.display = 'block';
            }
            
            function connectWebSocket() {
                if (!currentSession) return;
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                websocket = new WebSocket(`${protocol}//${window.location.host}/ws/${currentSession.session_id}`);
                
                websocket.onopen = () => {
                    addMessage('system', 'Connected to Ophira AI');
                };
                
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                
                websocket.onclose = () => {
                    addMessage('system', 'Disconnected from Ophira AI');
                };
                
                websocket.onerror = (error) => {
                    addMessage('system', 'Connection error: ' + error.message);
                };
            }
            
            function handleWebSocketMessage(data) {
                switch (data.type) {
                    case 'welcome':
                        addMessage('ai', data.message);
                        break;
                    case 'warnings':
                        showWarnings(data.warnings);
                        break;
                    case 'emergency_activated':
                        addMessage('system', data.message);
                        document.body.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
                        break;
                    case 'processing':
                        addMessage('ai', data.message);
                        break;
                    case 'error':
                        addMessage('system', `Error: ${data.message}`);
                        break;
                }
            }
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message || !websocket) return;
                
                addMessage('user', message);
                websocket.send(JSON.stringify({ message: message }));
                input.value = '';
            }
            
            function addMessage(type, text) {
                const messagesEl = document.getElementById('messages');
                const messageEl = document.createElement('div');
                messageEl.className = `message ${type}-message`;
                messageEl.textContent = text;
                messagesEl.appendChild(messageEl);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }
            
            // Quick action functions
            async function requestHealthCheck() {
                if (websocket) {
                    websocket.send(JSON.stringify({ 
                        message: "Hi Ophira, can you check my health status?" 
                    }));
                } else {
                    addMessage('system', 'Please login first');
                }
            }
            
            async function requestAnalysis() {
                if (!currentSession) {
                    addMessage('system', 'Please login first');
                    return;
                }
                
                try {
                    const response = await fetch('/api/health/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            analysis_types: ['general_health', 'ophthalmology', 'cardiovascular'],
                            emergency: false
                        })
                    });
                    
                    const result = await response.json();
                    addMessage('ai', 'Comprehensive health analysis started. This may take a few minutes...');
                    
                } catch (error) {
                    addMessage('system', 'Analysis request failed: ' + error.message);
                }
            }
            
            function emergency() {
                if (websocket) {
                    websocket.send(JSON.stringify({ 
                        message: "EMERGENCY - I need immediate help!" 
                    }));
                } else {
                    addMessage('system', 'Please login first to access emergency features');
                }
            }
            
            // Auto-login for demo
            window.onload = () => {
                setTimeout(login, 1000);
            };
        </script>
    </body>
    </html>
    """

# Add Phase 2 Medical AI endpoints after existing endpoints (around line 1400)

@app.post("/api/medical/analyze", response_model=MedicalAnalysisResponse)
async def start_medical_analysis(
    analysis_request: MedicalAnalysisRequest,
    session_data = Depends(get_current_session)
):
    """
    Start comprehensive medical AI analysis
    """
    try:
        if not enhanced_medical_specialist:
            raise HTTPException(status_code=503, detail="Medical AI system not available")
        
        # Prepare input data
        input_data = {}
        
        # Handle image data (base64 to numpy array)
        if analysis_request.image_data:
            import base64
            import cv2
            
            # Decode base64 image
            image_bytes = base64.b64decode(analysis_request.image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if analysis_request.analysis_type in ["retinal", "comprehensive"]:
                input_data['retinal_image'] = image
        
        # Add sensor data if provided
        if analysis_request.sensor_data:
            input_data.update(analysis_request.sensor_data)
        
        start_time = time.time()
        
        # Start medical analysis
        session_id = await enhanced_medical_specialist.start_medical_analysis(
            patient_id=analysis_request.patient_id,
            analysis_type=analysis_request.analysis_type,
            input_data=input_data
        )
        
        processing_time = time.time() - start_time
        
        # Get session status
        status = await enhanced_medical_specialist.get_session_status(session_id)
        
        # Send WebSocket notification
        await websocket_manager.send_message(session_data["session_id"], {
            "type": "medical_analysis_complete",
            "session_id": session_id,
            "analysis_type": analysis_request.analysis_type,
            "risk_level": status.get("risk_level", "unknown"),
            "emergency_alert": status.get("emergency_alert", False)
        })
        
        logger.info(f"Medical analysis completed: {session_id} ({processing_time:.2f}s)")
        
        return MedicalAnalysisResponse(
            session_id=session_id,
            analysis_type=analysis_request.analysis_type,
            risk_level=status.get("risk_level", "unknown"),
            confidence=0.85,  # Would be extracted from actual results
            emergency_alert=status.get("emergency_alert", False),
            recommendations=["Comprehensive analysis completed"],
            report_available=status.get("has_report", False),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Medical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/medical/retinal-analysis")
async def analyze_retinal_image(
    analysis_request: RetinalAnalysisRequest,
    session_data = Depends(get_current_session)
):
    """
    Specialized retinal image analysis endpoint
    """
    try:
        if not enhanced_medical_specialist:
            raise HTTPException(status_code=503, detail="Medical AI system not available")
        
        # Decode and process retinal image
        import base64
        import cv2
        
        image_bytes = base64.b64decode(analysis_request.image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        retinal_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Start retinal analysis
        session_id = await enhanced_medical_specialist.start_medical_analysis(
            patient_id=analysis_request.patient_id,
            analysis_type="retinal",
            input_data={'retinal_image': retinal_image}
        )
        
        # Get detailed results
        status = await enhanced_medical_specialist.get_session_status(session_id)
        
        return {
            "session_id": session_id,
            "status": "completed",
            "risk_level": status.get("risk_level", "unknown"),
            "emergency_alert": status.get("emergency_alert", False),
            "processing_time": status.get("duration", 0.0),
            "report_available": status.get("has_report", False)
        }
        
    except Exception as e:
        logger.error(f"Retinal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retinal analysis failed: {str(e)}")

@app.get("/api/medical/session/{session_id}")
async def get_medical_session_status(
    session_id: str,
    session_data = Depends(get_current_session)
):
    """
    Get medical analysis session status
    """
    try:
        if not enhanced_medical_specialist:
            raise HTTPException(status_code=503, detail="Medical AI system not available")
        
        status = await enhanced_medical_specialist.get_session_status(session_id)
        
        if status['status'] == 'not_found':
            raise HTTPException(status_code=404, detail="Session not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@app.get("/api/medical/report/{session_id}", response_model=ClinicalReportResponse)
async def get_clinical_report(
    session_id: str,
    session_data = Depends(get_current_session)
):
    """
    Get clinical report for medical analysis session
    """
    try:
        if not enhanced_medical_specialist:
            raise HTTPException(status_code=503, detail="Medical AI system not available")
        
        report_data = await enhanced_medical_specialist.get_clinical_report(session_id)
        
        if not report_data:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Generate formatted report
        try:
            from ophira.medical_ai.medical_specialist_enhanced import ClinicalReportGenerator, ClinicalReport
            
            report_generator = ClinicalReportGenerator()
            
            # Convert dict to ClinicalReport object (simplified)
            formatted_report = f"""
# MEDICAL AI ANALYSIS REPORT

**Report ID:** {report_data['report_id']}
**Patient:** {report_data['patient_id']}
**Generated:** {report_data['generated_timestamp']}

## EXECUTIVE SUMMARY
{report_data['executive_summary']}

## RISK ASSESSMENT
- **Risk Level:** {report_data['risk_assessment'].get('risk_level', 'unknown')}
- **Emergency Status:** {'YES' if report_data['emergency_alerts'] else 'NO'}
- **AI Confidence:** {report_data['ai_confidence']:.1%}

## RECOMMENDATIONS
{chr(10).join([f"• {rec['description']}" for rec in report_data['recommendations']])}

---
*This report was generated by Ophira AI Medical Analysis System.*
            """
            
        except Exception as e:
            logger.warning(f"Report formatting failed: {e}")
            formatted_report = None
        
        return ClinicalReportResponse(
            report_id=report_data['report_id'],
            patient_id=report_data['patient_id'],
            session_id=report_data['session_id'],
            report_type=report_data['report_type'],
            executive_summary=report_data['executive_summary'],
            formatted_report=formatted_report,
            generated_timestamp=report_data['generated_timestamp']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clinical report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get clinical report: {str(e)}")

@app.get("/api/medical/models/status")
async def get_medical_models_status(session_data = Depends(get_current_session)):
    """
    Get status of medical AI models
    """
    try:
        if not ai_model_manager:
            raise HTTPException(status_code=503, detail="AI Model Manager not available")
        
        # Get model information
        models = ai_model_manager.list_models()
        system_stats = ai_model_manager.get_system_stats()
        
        return {
            "status": "operational",
            "total_models": len(models),
            "loaded_models": system_stats.get("loaded_models", 0),
            "total_inferences": system_stats.get("total_inferences", 0),
            "device": system_stats.get("device", "unknown"),
            "models": [
                {
                    "model_id": model["metadata"]["model_id"],
                    "name": model["metadata"]["name"],
                    "domain": model["metadata"]["domain"],
                    "accuracy": model["metadata"]["accuracy"],
                    "loaded": model["loaded"]
                }
                for model in models
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/api/medical/emergency")
async def handle_medical_emergency(
    emergency_data: Dict[str, Any],
    session_data = Depends(get_current_session)
):
    """
    Handle medical emergency alerts
    """
    try:
        if not enhanced_medical_specialist:
            raise HTTPException(status_code=503, detail="Medical AI system not available")
        
        # Process emergency data
        emergency_response = {
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "severity": "critical",
            "status": "acknowledged",
            "recommended_actions": [
                "Contact emergency medical services immediately",
                "Ensure patient safety and comfort",
                "Monitor vital signs continuously"
            ]
        }
        
        # Send emergency notification via WebSocket
        await websocket_manager.broadcast({
            "type": "medical_emergency",
            "alert": emergency_response,
            "session_id": session_data["session_id"]
        })
        
        logger.critical(f"Medical emergency processed: {emergency_response['alert_id']}")
        
        return emergency_response
        
    except Exception as e:
        logger.error(f"Emergency handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency handling failed: {str(e)}")

@app.get("/api/medical/dashboard")
async def get_medical_dashboard(session_data = Depends(get_current_session)):
    """
    Get medical AI dashboard data
    """
    try:
        dashboard_data = {
            "system_status": "operational",
            "active_sessions": 0,
            "total_analyses_today": 0,
            "emergency_alerts": 0,
            "model_performance": {},
            "recent_activities": []
        }
        
        # Get enhanced medical specialist data
        if enhanced_medical_specialist:
            dashboard_data["active_sessions"] = len(enhanced_medical_specialist.active_sessions)
            
            model_status = await enhanced_medical_specialist.get_model_status()
            if model_status.get("model_manager") == "initialized":
                dashboard_data["system_status"] = "operational"
                dashboard_data["model_performance"] = model_status.get("system_stats", {})
        
        # Get AI model manager data
        if ai_model_manager:
            system_stats = ai_model_manager.get_system_stats()
            dashboard_data["total_analyses_today"] = system_stats.get("total_inferences", 0)
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get medical dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get medical dashboard: {str(e)}")

# Missing API endpoints that were causing 404 errors
@app.get("/api/vital-signs")
async def get_vital_signs(session_data = Depends(get_current_session)):
    """Get current vital signs data."""
    try:
        # Return mock vital signs data (in a real implementation, this would come from sensors)
        vital_signs = {
            "heart_rate": {"value": 72, "unit": "bpm", "status": "normal", "timestamp": datetime.now().isoformat()},
            "blood_pressure": {"systolic": 120, "diastolic": 80, "unit": "mmHg", "status": "normal", "timestamp": datetime.now().isoformat()},
            "temperature": {"value": 36.8, "unit": "°C", "status": "normal", "timestamp": datetime.now().isoformat()},
            "oxygen_saturation": {"value": 98, "unit": "%", "status": "normal", "timestamp": datetime.now().isoformat()},
            "respiratory_rate": {"value": 16, "unit": "bpm", "status": "normal", "timestamp": datetime.now().isoformat()}
        }
        
        return {
            "vital_signs": vital_signs,
            "session_id": session_data.session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Vital signs error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get vital signs: {str(e)}")

@app.get("/api/alerts")
async def get_alerts(session_data = Depends(get_current_session)):
    """Get current medical alerts."""
    try:
        # Return mock alerts data (in a real implementation, this would come from the alert system)
        alerts = [
            {
                "id": "alert_001",
                "type": "info",
                "severity": "low",
                "title": "System Connected",
                "message": "All sensors are functioning normally",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            }
        ]
        
        return {
            "alerts": alerts,
            "session_id": session_data.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_count": len(alerts),
            "unacknowledged_count": len([a for a in alerts if not a.get("acknowledged", False)])
        }
        
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "web_interface:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    ) 