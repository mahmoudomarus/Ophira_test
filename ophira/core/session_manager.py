"""
Session Management System for Ophira AI

Handles user sessions, AI agent sessions, authentication, security warnings,
and multi-user concurrent access with medical data protection.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import jwt
from passlib.context import CryptContext
import secrets

from .config import settings
from .logging import get_logger
from .database import medical_data_service


class SessionType(Enum):
    """Types of sessions in the system."""
    USER_WEB = "user_web"
    USER_MOBILE = "user_mobile"  
    USER_VOICE = "user_voice"
    AI_AGENT = "ai_agent"
    ADMIN = "admin"
    EMERGENCY = "emergency"


class SessionStatus(Enum):
    """Session status states."""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"
    EMERGENCY_MODE = "emergency_mode"


class SecurityLevel(Enum):
    """Security levels for different session types."""
    PUBLIC = 1
    BASIC = 2
    MEDICAL = 3
    EMERGENCY = 4
    ADMIN = 5


@dataclass
class SessionSecurityContext:
    """Security context for a session."""
    level: SecurityLevel
    permissions: Set[str] = field(default_factory=set)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    mfa_verified: bool = False
    last_security_check: Optional[datetime] = None
    failed_attempts: int = 0
    is_suspicious: bool = False


@dataclass
class SessionData:
    """Core session data structure."""
    session_id: str
    user_id: str
    session_type: SessionType
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    security_context: SessionSecurityContext
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # AI Agent specific data
    agent_id: Optional[str] = None
    agent_context: Dict[str, Any] = field(default_factory=dict)
    
    # Medical context
    active_health_session: bool = False
    emergency_mode: bool = False
    medical_warnings: List[str] = field(default_factory=list)
    
    def is_active(self) -> bool:
        """Check if session is active and not expired."""
        return (
            self.status == SessionStatus.ACTIVE and 
            datetime.utcnow() < self.expires_at
        )
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() >= self.expires_at
    
    def time_until_expiry(self) -> timedelta:
        """Get time remaining until session expires."""
        return max(timedelta(0), self.expires_at - datetime.utcnow())


class SessionManager:
    """
    Comprehensive session management for Ophira AI system.
    
    Handles:
    - User authentication and session creation
    - AI agent session management  
    - Security monitoring and warnings
    - Medical data access control
    - Emergency session protocols
    - Concurrent user management
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.logger = get_logger("session_manager")
        
        # Session storage
        self.active_sessions: Dict[str, SessionData] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self.agent_sessions: Dict[str, str] = {}  # agent_id -> session_id
        
        # Security
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = settings.security.secret_key
        
        # Session configuration
        self.session_timeouts = {
            SessionType.USER_WEB: timedelta(hours=8),
            SessionType.USER_MOBILE: timedelta(hours=24),
            SessionType.USER_VOICE: timedelta(hours=2),
            SessionType.AI_AGENT: timedelta(days=1),
            SessionType.ADMIN: timedelta(hours=4),
            SessionType.EMERGENCY: timedelta(hours=1)
        }
        
        # Security monitoring
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_activities: Dict[str, List[Dict[str, Any]]] = {}
        
        # Medical session warnings
        self.medical_warnings = {
            "concurrent_access": "Multiple users are accessing medical data simultaneously",
            "emergency_override": "Emergency protocols have been activated",
            "ai_analysis_active": "AI is currently analyzing sensitive medical data", 
            "data_inconsistency": "Potential inconsistency detected in medical data",
            "privacy_concern": "Unusual access pattern detected for medical records",
            "session_hijack": "Potential session hijacking attempt detected"
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._security_monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info("Session Manager initialized")
    
    async def start(self):
        """Start background session management tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._security_monitor_task = asyncio.create_task(self._security_monitor_loop())
        self.logger.info("Session Manager background tasks started")
    
    async def stop(self):
        """Stop session manager and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._security_monitor_task:
            self._security_monitor_task.cancel()
        
        # Terminate all sessions gracefully
        for session_id in list(self.active_sessions.keys()):
            await self.terminate_session(session_id, "system_shutdown")
        
        self.logger.info("Session Manager stopped")
    
    async def create_user_session(
        self,
        user_id: str,
        session_type: SessionType,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_fingerprint: Optional[str] = None
    ) -> SessionData:
        """Create a new user session with security validation."""
        
        # Generate secure session ID
        session_id = self._generate_secure_session_id()
        
        # Determine security level
        security_level = self._determine_security_level(session_type)
        
        # Create security context
        security_context = SessionSecurityContext(
            level=security_level,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            last_security_check=datetime.utcnow()
        )
        
        # Set permissions based on user and session type
        security_context.permissions = await self._get_user_permissions(user_id, session_type)
        
        # Calculate expiry time
        timeout = self.session_timeouts.get(session_type, timedelta(hours=8))
        expires_at = datetime.utcnow() + timeout
        
        # Create session data
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            session_type=session_type,
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=expires_at,
            security_context=security_context
        )
        
        # Store session
        self.active_sessions[session_id] = session_data
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Check for concurrent session warnings
        await self._check_concurrent_sessions(user_id)
        
        self.logger.info(
            f"Created {session_type.value} session for user {user_id}",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "session_type": session_type.value,
                "expires_at": expires_at.isoformat(),
                "security_level": security_level.value
            }
        )
        
        return session_data
    
    async def create_agent_session(
        self,
        agent_id: str,
        agent_context: Dict[str, Any] = None
    ) -> SessionData:
        """Create a session for an AI agent."""
        
        session_id = self._generate_secure_session_id()
        
        security_context = SessionSecurityContext(
            level=SecurityLevel.MEDICAL,
            permissions={"agent_communication", "medical_analysis", "sensor_access"}
        )
        
        expires_at = datetime.utcnow() + self.session_timeouts[SessionType.AI_AGENT]
        
        session_data = SessionData(
            session_id=session_id,
            user_id=f"agent_{agent_id}",
            session_type=SessionType.AI_AGENT,
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=expires_at,
            security_context=security_context,
            agent_id=agent_id,
            agent_context=agent_context or {}
        )
        
        self.active_sessions[session_id] = session_data
        self.agent_sessions[agent_id] = session_id
        
        self.logger.info(f"Created AI agent session for {agent_id}", extra={"session_id": session_id})
        
        return session_data
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID with expiry check."""
        session = self.active_sessions.get(session_id)
        
        if not session:
            return None
        
        if session.is_expired():
            await self.expire_session(session_id)
            return None
        
        return session
    
    async def validate_session(self, session_id: str) -> bool:
        """Validate if a session is active and valid."""
        session = await self.get_session(session_id)
        return session is not None and session.is_active()
    
    async def refresh_session(self, session_id: str) -> bool:
        """Refresh session activity and extend expiry if needed."""
        session = self.active_sessions.get(session_id)
        
        if not session or session.status != SessionStatus.ACTIVE:
            return False
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        
        # Extend expiry if session is more than halfway through timeout
        timeout = self.session_timeouts.get(session.session_type, timedelta(hours=8))
        time_remaining = session.time_until_expiry()
        
        if time_remaining < timeout / 2:
            session.expires_at = datetime.utcnow() + timeout
            self.logger.debug(f"Extended session {session_id} expiry")
        
        # Perform security check
        await self._perform_security_check(session)
        
        return True
    
    async def terminate_session(self, session_id: str, reason: str = "user_logout"):
        """Terminate a session."""
        session = self.active_sessions.get(session_id)
        
        if not session:
            return
        
        # Update session status
        session.status = SessionStatus.TERMINATED
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Clean up user session tracking
        if session.user_id in self.user_sessions:
            if session_id in self.user_sessions[session.user_id]:
                self.user_sessions[session.user_id].remove(session_id)
            
            if not self.user_sessions[session.user_id]:
                del self.user_sessions[session.user_id]
        
        # Clean up agent session tracking
        if session.agent_id and session.agent_id in self.agent_sessions:
            del self.agent_sessions[session.agent_id]
        
        self.logger.info(
            f"Terminated session {session_id}",
            extra={
                "session_id": session_id,
                "user_id": session.user_id,
                "reason": reason,
                "session_duration": (datetime.utcnow() - session.created_at).total_seconds()
            }
        )
    
    async def expire_session(self, session_id: str):
        """Expire a session due to timeout."""
        await self.terminate_session(session_id, "expired")
    
    async def suspend_session(self, session_id: str, reason: str):
        """Suspend a session due to security concerns."""
        session = self.active_sessions.get(session_id)
        
        if session:
            session.status = SessionStatus.SUSPENDED
            session.security_context.is_suspicious = True
            
            self.logger.warning(
                f"Suspended session {session_id}",
                extra={"session_id": session_id, "reason": reason}
            )
    
    async def activate_emergency_mode(self, session_id: str, emergency_context: Dict[str, Any]):
        """Activate emergency mode for a session."""
        session = self.active_sessions.get(session_id)
        
        if session:
            session.emergency_mode = True
            session.status = SessionStatus.EMERGENCY_MODE
            session.security_context.level = SecurityLevel.EMERGENCY
            session.metadata["emergency_context"] = emergency_context
            
            # Extend session during emergency
            session.expires_at = datetime.utcnow() + self.session_timeouts[SessionType.EMERGENCY]
            
            self.logger.critical(
                f"Emergency mode activated for session {session_id}",
                extra={"session_id": session_id, "emergency_context": emergency_context}
            )
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all active sessions for a user."""
        session_ids = self.user_sessions.get(user_id, [])
        sessions = []
        
        for session_id in session_ids[:]:  # Copy list to avoid modification during iteration
            session = await self.get_session(session_id)
            if session:
                sessions.append(session)
            else:
                # Clean up expired session reference
                self.user_sessions[user_id].remove(session_id)
        
        return sessions
    
    async def get_agent_session(self, agent_id: str) -> Optional[SessionData]:
        """Get the active session for an AI agent."""
        session_id = self.agent_sessions.get(agent_id)
        if session_id:
            return await self.get_session(session_id)
        return None
    
    async def issue_warning(self, session_id: str, warning_type: str, details: str = ""):
        """Issue a warning for a session."""
        session = self.active_sessions.get(session_id)
        
        if session:
            warning_message = self.medical_warnings.get(warning_type, warning_type)
            if details:
                warning_message += f": {details}"
            
            session.medical_warnings.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": warning_type,
                "message": warning_message,
                "details": details
            })
            
            self.logger.warning(
                f"Session warning issued: {warning_type}",
                extra={
                    "session_id": session_id,
                    "warning_type": warning_type,
                    "details": details
                }
            )
    
    async def get_session_warnings(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all warnings for a session."""
        session = self.active_sessions.get(session_id)
        return session.medical_warnings if session else []
    
    def _generate_secure_session_id(self) -> str:
        """Generate a cryptographically secure session ID."""
        return secrets.token_urlsafe(32)
    
    def _determine_security_level(self, session_type: SessionType) -> SecurityLevel:
        """Determine security level based on session type."""
        level_mapping = {
            SessionType.USER_WEB: SecurityLevel.MEDICAL,
            SessionType.USER_MOBILE: SecurityLevel.MEDICAL,
            SessionType.USER_VOICE: SecurityLevel.MEDICAL,
            SessionType.AI_AGENT: SecurityLevel.MEDICAL,
            SessionType.ADMIN: SecurityLevel.ADMIN,
            SessionType.EMERGENCY: SecurityLevel.EMERGENCY
        }
        return level_mapping.get(session_type, SecurityLevel.BASIC)
    
    async def _get_user_permissions(self, user_id: str, session_type: SessionType) -> Set[str]:
        """Get permissions for a user based on their role and session type."""
        # Base permissions for medical system
        base_permissions = {
            "view_own_data", "basic_health_check", "conversation"
        }
        
        # Add permissions based on session type
        if session_type == SessionType.USER_WEB:
            base_permissions.update({"upload_data", "view_reports", "manage_profile"})
        elif session_type == SessionType.USER_MOBILE:
            base_permissions.update({"sensor_access", "notifications", "emergency_contact"})
        elif session_type == SessionType.USER_VOICE:
            base_permissions.update({"voice_commands", "hands_free_operation"})
        elif session_type == SessionType.AI_AGENT:
            base_permissions.update({"medical_analysis", "cross_reference", "agent_communication"})
        elif session_type == SessionType.ADMIN:
            base_permissions.update({"admin_access", "system_config", "user_management"})
        
        # TODO: Fetch additional permissions from database based on user role
        
        return base_permissions
    
    async def _check_concurrent_sessions(self, user_id: str):
        """Check for concurrent sessions and issue warnings if needed."""
        user_sessions = await self.get_user_sessions(user_id)
        
        if len(user_sessions) > 1:
            for session in user_sessions:
                await self.issue_warning(
                    session.session_id,
                    "concurrent_access",
                    f"User has {len(user_sessions)} active sessions"
                )
    
    async def _perform_security_check(self, session: SessionData):
        """Perform security checks on a session."""
        current_time = datetime.utcnow()
        
        # Check if security check is due
        if (session.security_context.last_security_check and 
            current_time - session.security_context.last_security_check < timedelta(minutes=15)):
            return
        
        session.security_context.last_security_check = current_time
        
        # Check for suspicious activity patterns
        # TODO: Implement more sophisticated security checks
        
        # Check session duration
        session_duration = current_time - session.created_at
        max_duration = timedelta(hours=24)
        
        if session_duration > max_duration:
            await self.issue_warning(
                session.session_id,
                "privacy_concern",
                f"Session has been active for {session_duration}"
            )
    
    async def _cleanup_loop(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_sessions = []
                
                # Find expired sessions
                for session_id, session in self.active_sessions.items():
                    if session.is_expired():
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self.expire_session(session_id)
                
                if expired_sessions:
                    self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                # Sleep for 1 minute before next cleanup
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _security_monitor_loop(self):
        """Background task to monitor security threats."""
        while True:
            try:
                # Monitor for suspicious patterns
                await self._monitor_failed_logins()
                await self._monitor_session_anomalies()
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in security monitor loop: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_failed_logins(self):
        """Monitor for suspicious login patterns."""
        # Clean up old failed attempts (older than 1 hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for user_id in list(self.failed_login_attempts.keys()):
            self.failed_login_attempts[user_id] = [
                attempt for attempt in self.failed_login_attempts[user_id]
                if attempt > cutoff_time
            ]
            
            if not self.failed_login_attempts[user_id]:
                del self.failed_login_attempts[user_id]
    
    async def _monitor_session_anomalies(self):
        """Monitor sessions for anomalous behavior."""
        for session_id, session in self.active_sessions.items():
            # Check for sessions that haven't been active for a while
            time_since_activity = datetime.utcnow() - session.last_activity
            
            if time_since_activity > timedelta(hours=2) and session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.IDLE
                self.logger.info(f"Session {session_id} marked as idle")


# Global session manager instance
session_manager = SessionManager()


async def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return session_manager


# Convenience functions
async def create_user_session(**kwargs) -> SessionData:
    """Create a user session."""
    return await session_manager.create_user_session(**kwargs)


async def get_session(session_id: str) -> Optional[SessionData]:
    """Get a session by ID."""
    return await session_manager.get_session(session_id)


async def validate_session(session_id: str) -> bool:
    """Validate a session."""
    return await session_manager.validate_session(session_id) 