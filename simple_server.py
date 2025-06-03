#!/usr/bin/env python3
"""
Simplified FastAPI Server for Ophira AI Frontend Testing
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import uuid

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

# FastAPI app
app = FastAPI(
    title="Ophira AI Medical System - Demo",
    description="Simplified demo server for frontend integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (for demo)
sessions = {}
active_connections = {}

@app.post("/api/login", response_model=SessionInfo)
async def login(request: Request, login_data: UserLogin):
    """Create a new demo session."""
    session_id = str(uuid.uuid4())
    user_id = login_data.username or "demo_user"
    
    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "session_type": "demo",
        "status": "active",
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=2),
        "warnings": []
    }
    
    sessions[session_id] = session_data
    
    return SessionInfo(
        session_id=session_id,
        user_id=user_id,
        session_type="demo",
        status="active",
        created_at=session_data["created_at"].isoformat(),
        expires_at=session_data["expires_at"].isoformat(),
        warnings=[]
    )

@app.post("/api/logout")
async def logout(session_id: str):
    """Terminate a session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "logged_out"}

@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return SessionInfo(
        session_id=session["session_id"],
        user_id=session["user_id"],
        session_type=session["session_type"],
        status=session["status"],
        created_at=session["created_at"].isoformat(),
        expires_at=session["expires_at"].isoformat(),
        warnings=session["warnings"]
    )

@app.post("/api/chat")
async def chat_endpoint(message_data: ChatMessage):
    """Demo chat endpoint."""
    # Simulate AI response
    responses = [
        "I'm analyzing your health data. Everything looks normal.",
        "I recommend staying hydrated and getting regular exercise.",
        "Your vital signs are within normal ranges.",
        "I'm monitoring your health parameters continuously.",
        "Please let me know if you experience any symptoms."
    ]
    
    import random
    response = random.choice(responses)
    
    return {
        "response": response,
        "emergency_mode": False,
        "session_warnings": [],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/health/analyze")
async def analyze_health(analysis_request: HealthAnalysisRequest):
    """Demo health analysis."""
    return {
        "status": "analysis_requested",
        "analysis_types": analysis_request.analysis_types,
        "emergency_mode": analysis_request.emergency,
        "estimated_completion": (datetime.now().timestamp() + 300) * 1000,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/sensors/status")
async def get_sensor_status():
    """Demo sensor status."""
    return {
        "sensors": {
            "usb_camera_primary": {
                "status": "connected",
                "last_data": datetime.now().isoformat(),
                "quality": "good"
            },
            "vl53l4cx_tof_primary": {
                "status": "connected", 
                "last_data": datetime.now().isoformat(),
                "quality": "excellent"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/vital-signs")
async def get_vital_signs(session_id: str):
    """Demo vital signs."""
    import random
    return {
        "heart_rate": random.randint(60, 100),
        "blood_pressure_systolic": random.randint(110, 140),
        "blood_pressure_diastolic": random.randint(70, 90),
        "temperature": round(random.uniform(36.1, 37.2), 1),
        "spo2": random.randint(95, 100),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/alerts")
async def get_alerts(session_id: str):
    """Demo medical alerts."""
    return {
        "alerts": [],
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Demo WebSocket endpoint."""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": f"Connected to Ophira AI Demo Server!",
            "session_info": {
                "session_id": session_id,
                "status": "active"
            }
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            # Check for emergency keywords
            emergency_keywords = ["emergency", "help", "urgent", "chest pain", "can't breathe", "dizzy"]
            is_emergency = any(keyword in user_message.lower() for keyword in emergency_keywords)
            
            if is_emergency:
                await websocket.send_json({
                    "type": "emergency_activated",
                    "message": "🚨 Emergency mode activated in demo. In a real system, this would alert medical professionals."
                })
            
            # Send processing acknowledgment
            await websocket.send_json({
                "type": "processing",
                "message": "Processing your request in demo mode...",
                "emergency": is_emergency,
                "timestamp": datetime.now().isoformat()
            })
            
            # Simulate AI response after short delay
            await asyncio.sleep(1)
            
            # Demo responses
            demo_responses = [
                "Demo: I'm analyzing your health parameters.",
                "Demo: Your vital signs appear normal.",
                "Demo: I recommend regular check-ups with your healthcare provider.",
                "Demo: Please monitor your symptoms and report any changes.",
                "Demo: I'm here to help with your health monitoring needs."
            ]
            
            import random
            ai_response = random.choice(demo_responses)
            
            await websocket.send_json({
                "type": "ai_response",
                "message": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "An error occurred in demo mode",
                "code": "DEMO_ERROR"
            })
        except:
            pass

@app.get("/api/system/status")
async def system_status():
    """Demo system status."""
    return {
        "status": "demo_mode",
        "agents": {
            "primary": "active",
            "medical": "active", 
            "sensor": "active",
            "supervisory": "active"
        },
        "database": "demo",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Ophira AI Demo Server", "status": "running", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    ) 