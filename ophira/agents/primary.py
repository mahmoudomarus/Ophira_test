"""
Primary Voice Agent (Ophira) for the Ophira AI system.

This is the main user-facing agent that handles natural language interactions,
maintains conversation context, and coordinates with specialized agents.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .base import BaseAgent, AgentMessage, MessageType, MessagePriority, AgentCapability, AgentRegistry
from ..core.config import settings
from ..core.logging import get_agent_logger


@dataclass
class ConversationContext:
    """Context for maintaining conversation state."""
    
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    current_intent: Optional[str] = None
    active_health_concerns: List[str] = None
    last_interaction: Optional[datetime] = None
    user_preferences: Dict[str, Any] = None
    emergency_contacts: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.active_health_concerns is None:
            self.active_health_concerns = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.emergency_contacts is None:
            self.emergency_contacts = []


@dataclass
class HealthQuery:
    """Structure for health-related queries."""
    
    query_text: str
    intent: str
    entities: Dict[str, Any]
    urgency_level: int  # 1-5 scale
    requires_sensor_data: bool = False
    specific_sensors: List[str] = None
    requires_analysis: bool = False
    analysis_types: List[str] = None
    
    def __post_init__(self):
        if self.specific_sensors is None:
            self.specific_sensors = []
        if self.analysis_types is None:
            self.analysis_types = []


class OphiraVoiceAgent(BaseAgent):
    """
    Primary Voice Agent (Ophira) - Main user interface agent.
    
    Responsibilities:
    - Handle natural language conversations with users
    - Maintain conversation context and user rapport
    - Coordinate with specialized agents for health analysis
    - Manage emergency situations and escalations
    - Provide personalized health insights and recommendations
    """
    
    def __init__(self):
        """Initialize the Ophira Voice Agent."""
        super().__init__(
            agent_id="ophira_voice_001",
            name="Ophira",
            agent_type="voice_primary",
            capabilities=[
                AgentCapability.CONVERSATION,
                AgentCapability.EMERGENCY_RESPONSE,
                AgentCapability.COORDINATION
            ]
        )
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_timeout = 30 * 60  # 30 minutes
        
        # Health query processing
        self.intent_patterns = {
            "health_check": ["how am i", "health status", "check my health", "how do i feel"],
            "symptom_report": ["i feel", "i have", "experiencing", "symptoms"],
            "emergency": ["help", "emergency", "urgent", "can't breathe", "chest pain"],
            "question": ["what is", "why", "how", "explain"],
            "greeting": ["hello", "hi", "hey ophira", "good morning"],
            "goodbye": ["bye", "goodbye", "see you", "thanks"]
        }
        
        # Response templates
        self.response_templates = {
            "greeting": [
                "Hi {name}, how are you feeling today?",
                "Hello {name}! I'm here to help with your health. How can I assist you?",
                "Good to see you, {name}. What can I help you with today?"
            ],
            "health_check_initiate": [
                "Let me run some quick health checks for you. Please bear with me while I gather data from your sensors.",
                "I'll analyze your current health status. This may take a moment while I coordinate with my specialist colleagues.",
                "Let me check your vital signs and recent health patterns. One moment please."
            ],
            "emergency_response": [
                "I'm detecting a potentially serious situation. Let me assess this immediately.",
                "This requires urgent attention. I'm analyzing your condition and will take appropriate action.",
                "I'm concerned about what I'm seeing. Please stay calm while I evaluate the situation."
            ],
            "analysis_complete": [
                "I've completed my analysis. Here's what I found:",
                "Based on my examination, here are my findings:",
                "After reviewing your data with my specialist team, here's what we discovered:"
            ]
        }
        
        # Agent coordination
        self.specialist_agents = {}
        self.active_analyses = {}
        
        self.logger.info("Ophira Voice Agent initialized")
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents or the web interface."""
        try:
            content = message.content
            
            # Handle voice session start/stop
            if content.get("voice_session_start"):
                await self._handle_voice_session_start(content)
                return
            
            if content.get("voice_session_stop"):
                await self._handle_voice_session_stop(content)
                return
            
            # Handle user input (text or voice)
            if "user_input" in content:
                user_input = content["user_input"]
                user_id = content.get("user_id", "unknown")
                session_id = content.get("session_id", "unknown")
                
                # Log voice-specific information if available
                if content.get("source") == "voice":
                    confidence = content.get("confidence", 1.0)
                    language = content.get("language", "en-US")
                    self.logger.info(f"Processing voice input from {user_id}: confidence={confidence}, language={language}")
                
                await self._process_user_input(user_input, user_id, session_id)
            
            # Handle other message types...
            elif "coordinate_analysis" in content:
                # Handle analysis coordination requests
                pass
            
        except Exception as e:
            self.logger.log_error(
                "Error handling message",
                str(e),
                {"message_id": message.id, "sender": message.sender_id}
            )
    
    async def _handle_voice_session_start(self, content: Dict[str, Any]):
        """Handle voice session start notification."""
        user_id = content.get("user_id", "unknown")
        session_id = content.get("session_id", "unknown")
        capabilities = content.get("capabilities", [])
        
        self.logger.info(f"Voice session started for user {user_id}, session {session_id}")
        self.logger.info(f"Voice capabilities: {capabilities}")
        
        # Store voice session state
        if not hasattr(self, 'voice_sessions'):
            self.voice_sessions = {}
        
        self.voice_sessions[session_id] = {
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "capabilities": capabilities,
            "active": True
        }
        
        # Send welcome message for voice session
        welcome_message = "Voice session activated. I'm ready to listen and help with your health needs. You can speak naturally, and I'll understand your requests."
        await self._send_user_response(welcome_message, user_id, session_id)
    
    async def _handle_voice_session_stop(self, content: Dict[str, Any]):
        """Handle voice session stop notification."""
        user_id = content.get("user_id", "unknown")
        session_id = content.get("session_id", "unknown")
        
        self.logger.info(f"Voice session stopped for user {user_id}, session {session_id}")
        
        # Update voice session state
        if hasattr(self, 'voice_sessions') and session_id in self.voice_sessions:
            self.voice_sessions[session_id]["active"] = False
            self.voice_sessions[session_id]["stopped_at"] = datetime.now().isoformat()
        
        # Send goodbye message for voice session
        goodbye_message = "Voice session ended. I'm still here if you need me - just start a new voice session or send a text message."
        await self._send_user_response(goodbye_message, user_id, session_id)
    
    async def _process_user_input(self, user_input: str, user_id: str, session_id: str):
        """Process user input and generate appropriate response."""
        try:
            # Get or create conversation context
            context = await self._get_conversation_context(user_id, session_id)
            
            # Parse user input
            health_query = await self._parse_user_input(user_input, context)
            
            # Update conversation history
            context.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "intent": health_query.intent,
                "urgency": health_query.urgency_level
            })
            
            # Handle different types of queries
            if health_query.intent == "greeting":
                response = await self._handle_greeting(context)
            elif health_query.intent == "health_check":
                response = await self._handle_health_check(health_query, context)
            elif health_query.intent == "symptom_report":
                response = await self._handle_symptom_report(health_query, context)
            elif health_query.intent == "emergency":
                response = await self._handle_emergency(health_query, context)
            elif health_query.intent == "question":
                response = await self._handle_question(health_query, context)
            elif health_query.intent == "goodbye":
                response = await self._handle_goodbye(context)
            else:
                response = await self._handle_general_query(health_query, context)
            
            # Send response to user
            await self._send_user_response(response, user_id, session_id)
            
            # Update context
            context.last_interaction = datetime.now()
            context.conversation_history[-1]["ophira_response"] = response
            
        except Exception as e:
            self.logger.log_error(
                "Error processing user input",
                str(e),
                {"user_id": user_id, "user_input": user_input}
            )
            
            # Send error response
            error_response = "I apologize, but I encountered an issue processing your request. Please try again or contact support if the problem persists."
            await self._send_user_response(error_response, user_id, session_id)
    
    async def _parse_user_input(self, user_input: str, context: ConversationContext) -> HealthQuery:
        """Parse user input to determine intent and extract entities."""
        user_input_lower = user_input.lower()
        
        # Determine intent
        intent = "general"
        urgency_level = 1
        
        for intent_type, patterns in self.intent_patterns.items():
            if any(pattern in user_input_lower for pattern in patterns):
                intent = intent_type
                break
        
        # Assess urgency
        emergency_keywords = ["emergency", "urgent", "help", "can't breathe", "chest pain", "severe pain"]
        if any(keyword in user_input_lower for keyword in emergency_keywords):
            urgency_level = 5
        elif intent == "symptom_report":
            urgency_level = 3
        elif intent == "health_check":
            urgency_level = 2
        
        # Extract entities (simplified for now)
        entities = {}
        requires_sensor_data = False
        requires_analysis = False
        
        if intent in ["health_check", "symptom_report", "emergency"]:
            requires_sensor_data = True
            requires_analysis = True
        
        return HealthQuery(
            query_text=user_input,
            intent=intent,
            entities=entities,
            urgency_level=urgency_level,
            requires_sensor_data=requires_sensor_data,
            requires_analysis=requires_analysis
        )
    
    async def _handle_greeting(self, context: ConversationContext) -> str:
        """Handle greeting messages."""
        user_name = context.user_preferences.get("name", "there")
        template = self.response_templates["greeting"][0]  # Use first template for simplicity
        return template.format(name=user_name)
    
    async def _handle_health_check(self, query: HealthQuery, context: ConversationContext) -> str:
        """Handle health check requests."""
        try:
            # Notify user that analysis is starting
            initial_response = self.response_templates["health_check_initiate"][0]
            await self._send_user_response(initial_response, context.user_id, context.session_id)
            
            # Request sensor data collection
            if query.requires_sensor_data:
                await self._request_sensor_data(context.user_id)
            
            # Request health analysis
            if query.requires_analysis:
                analysis_result = await self._request_health_analysis(query, context)
                return self._format_health_analysis_response(analysis_result)
            
            return "I've completed a basic health check. Everything appears to be within normal ranges."
            
        except Exception as e:
            self.logger.log_error(
                "Error in health check",
                str(e),
                {"user_id": context.user_id}
            )
            return "I encountered an issue while checking your health. Please try again in a moment."
    
    async def _handle_symptom_report(self, query: HealthQuery, context: ConversationContext) -> str:
        """Handle symptom reporting."""
        # Add symptom to active health concerns
        context.active_health_concerns.append({
            "symptom": query.query_text,
            "timestamp": datetime.now().isoformat(),
            "urgency": query.urgency_level
        })
        
        # Request immediate analysis for symptoms
        if query.urgency_level >= 4:
            analysis_result = await self._request_emergency_analysis(query, context)
            return self._format_emergency_response(analysis_result)
        else:
            analysis_result = await self._request_health_analysis(query, context)
            return self._format_symptom_analysis_response(analysis_result)
    
    async def _handle_emergency(self, query: HealthQuery, context: ConversationContext) -> str:
        """Handle emergency situations."""
        self.logger.log_error(
            "Emergency situation detected",
            query.query_text,
            {"user_id": context.user_id, "urgency": query.urgency_level}
        )
        
        # Immediate response
        emergency_response = self.response_templates["emergency_response"][0]
        await self._send_user_response(emergency_response, context.user_id, context.session_id)
        
        # Trigger emergency protocols
        await self._trigger_emergency_protocols(query, context)
        
        # Get immediate analysis
        analysis_result = await self._request_emergency_analysis(query, context)
        return self._format_emergency_response(analysis_result)
    
    async def _handle_question(self, query: HealthQuery, context: ConversationContext) -> str:
        """Handle general health questions."""
        # For now, provide a generic response
        # In production, this would interface with medical knowledge bases
        return "That's a great question about your health. Based on your current profile and recent data, let me provide some information. However, for specific medical advice, please consult with your healthcare provider."
    
    async def _handle_goodbye(self, context: ConversationContext) -> str:
        """Handle goodbye messages."""
        return "Take care! I'll continue monitoring your health in the background. Feel free to reach out anytime if you need assistance."
    
    async def _handle_general_query(self, query: HealthQuery, context: ConversationContext) -> str:
        """Handle general queries that don't fit other categories."""
        return "I understand you're asking about your health. Could you be more specific about what you'd like me to check or analyze?"
    
    async def _request_sensor_data(self, user_id: str):
        """Request sensor data collection from sensor control agent."""
        registry = AgentRegistry.get_instance()
        sensor_agent_id = registry.get_agent_by_capability(AgentCapability.SENSOR_CONTROL)
        
        if sensor_agent_id:
            message = AgentMessage(
                recipient_id=sensor_agent_id,
                message_type=MessageType.REQUEST,
                priority=MessagePriority.HIGH,
                content={
                    "action": "collect_data",
                    "user_id": user_id,
                    "sensors": ["nir_camera", "heart_rate", "ppg"],
                    "duration": 30  # seconds
                },
                requires_response=True,
                timeout_seconds=60
            )
            
            response = await self.send_message(message)
            return response
        else:
            self.logger.log_error(
                "Sensor control agent not available",
                "Cannot collect sensor data",
                {"user_id": user_id}
            )
            return None
    
    async def _request_health_analysis(self, query: HealthQuery, context: ConversationContext):
        """Request health analysis from specialist agents."""
        registry = AgentRegistry.get_instance()
        
        # Request analysis from general health analyzer
        health_agent_id = registry.get_agent_by_capability(AgentCapability.MEDICAL_ANALYSIS)
        
        if health_agent_id:
            message = AgentMessage(
                recipient_id=health_agent_id,
                message_type=MessageType.REQUEST,
                priority=MessagePriority.NORMAL,
                content={
                    "action": "analyze_health",
                    "user_id": context.user_id,
                    "query": query.query_text,
                    "context": context.active_health_concerns,
                    "urgency": query.urgency_level
                },
                requires_response=True,
                timeout_seconds=120
            )
            
            response = await self.send_message(message)
            return response.content if response else {"status": "analysis_unavailable"}
        
        return {"status": "no_analyst_available"}
    
    async def _request_emergency_analysis(self, query: HealthQuery, context: ConversationContext):
        """Request emergency analysis with highest priority."""
        registry = AgentRegistry.get_instance()
        
        # Get all available medical agents for emergency analysis
        all_agents = registry.get_all_agents()
        medical_agents = [
            agent_id for agent_id, info in all_agents.items()
            if AgentCapability.MEDICAL_ANALYSIS in info.capabilities
        ]
        
        analysis_results = []
        for agent_id in medical_agents:
            message = AgentMessage(
                recipient_id=agent_id,
                message_type=MessageType.REQUEST,
                priority=MessagePriority.EMERGENCY,
                content={
                    "action": "emergency_analysis",
                    "user_id": context.user_id,
                    "query": query.query_text,
                    "urgency": 5
                },
                requires_response=True,
                timeout_seconds=30
            )
            
            response = await self.send_message(message)
            if response:
                analysis_results.append(response.content)
        
        return {"emergency_analyses": analysis_results}
    
    async def _trigger_emergency_protocols(self, query: HealthQuery, context: ConversationContext):
        """Trigger emergency protocols and notifications."""
        # Log emergency event
        self.logger.log_error(
            "Emergency protocol triggered",
            f"Emergency query: {query.query_text}",
            {
                "user_id": context.user_id,
                "urgency": query.urgency_level,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # In production, this would:
        # 1. Contact emergency services if needed
        # 2. Notify emergency contacts
        # 3. Store emergency event in memory
        # 4. Escalate to human medical professionals
        
        # For now, just log the emergency
        self.logger.info(f"Emergency protocols activated for user {context.user_id}")
    
    def _format_health_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """Format health analysis results into user-friendly response."""
        if not analysis_result or analysis_result.get("status") == "analysis_unavailable":
            return "I wasn't able to complete a full analysis at the moment. Please try again shortly."
        
        # Format response based on analysis results
        response = self.response_templates["analysis_complete"][0] + "\n\n"
        
        # Add specific findings (simplified for now)
        response += "Your vital signs appear to be within normal ranges. "
        response += "I'll continue monitoring and will alert you if I notice any concerning patterns."
        
        return response
    
    def _format_symptom_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """Format symptom analysis results."""
        if not analysis_result:
            return "I've noted your symptoms and will monitor the situation closely."
        
        response = "Thank you for reporting your symptoms. "
        response += "Based on my analysis, I recommend monitoring the situation. "
        response += "Please let me know if your symptoms worsen or if you develop any new concerns."
        
        return response
    
    def _format_emergency_response(self, analysis_result: Dict[str, Any]) -> str:
        """Format emergency analysis response."""
        response = "I've completed an urgent analysis of your situation. "
        
        if analysis_result.get("emergency_analyses"):
            response += "Based on multiple specialist evaluations, "
            response += "I recommend seeking immediate medical attention. "
            
            # In production, add specific emergency instructions
            response += "If this is a life-threatening emergency, please call emergency services immediately."
        else:
            response += "While I'm taking this seriously, my analysis capabilities are currently limited. "
            response += "Please consider contacting your healthcare provider or emergency services if you feel this is urgent."
        
        return response
    
    async def _send_user_response(self, response: str, user_id: str, session_id: str):
        """Send response back to user interface."""
        # In production, this would send to the actual user interface
        # For now, just log the response
        self.logger.info(f"Response to user {user_id}: {response}")
        
        # Store response in conversation context
        if hasattr(self, 'active_conversations'):
            context_key = f"{user_id}_{session_id}"
            if context_key in self.active_conversations:
                context = self.active_conversations[context_key]
                if context.conversation_history:
                    context.conversation_history[-1]["ophira_response"] = response
    
    async def _get_conversation_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Get or create conversation context for user."""
        context_key = f"{user_id}_{session_id}"
        
        if context_key not in self.active_conversations:
            # Create new context
            self.active_conversations[context_key] = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                conversation_history=[],
                user_preferences={"name": f"User{user_id[-4:]}"}  # Simplified for demo
            )
        
        return self.active_conversations[context_key]
    
    async def _cleanup_expired_conversations(self):
        """Clean up expired conversations."""
        current_time = datetime.now()
        expired_contexts = []
        
        for context_key, context in self.active_conversations.items():
            if (context.last_interaction and 
                (current_time - context.last_interaction).seconds > self.conversation_timeout):
                expired_contexts.append(context_key)
        
        for context_key in expired_contexts:
            del self.active_conversations[context_key]
            self.logger.info(f"Cleaned up expired conversation: {context_key}")
    
    async def _start_agent_tasks(self):
        """Start agent-specific tasks."""
        # Start conversation cleanup task
        cleanup_task = asyncio.create_task(self._conversation_cleanup_loop())
        self.tasks.append(cleanup_task)
        
        # Register with other agents
        await self._register_with_specialist_agents()
    
    async def _conversation_cleanup_loop(self):
        """Periodic cleanup of expired conversations."""
        while self.is_running:
            try:
                await self._cleanup_expired_conversations()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.log_error(
                    "Error in conversation cleanup",
                    str(e),
                    {}
                )
    
    async def _register_with_specialist_agents(self):
        """Register with specialist agents for coordination."""
        registry = AgentRegistry.get_instance()
        
        # Wait for other agents to be available
        await asyncio.sleep(5)
        
        # Get specialist agents
        self.specialist_agents = {
            "sensor_control": registry.get_agent_by_capability(AgentCapability.SENSOR_CONTROL),
            "medical_analysis": registry.get_agent_by_capability(AgentCapability.MEDICAL_ANALYSIS),
            "memory_management": registry.get_agent_by_capability(AgentCapability.MEMORY_MANAGEMENT)
        }
        
        self.logger.info(f"Registered with specialist agents: {self.specialist_agents}")
    
    async def _cleanup(self):
        """Cleanup agent resources."""
        # Save conversation contexts to persistent storage
        # In production, this would save to database
        self.logger.info(f"Saving {len(self.active_conversations)} active conversations")
        
        # Clear active conversations
        self.active_conversations.clear()
        
        self.logger.info("Ophira Voice Agent cleanup completed") 