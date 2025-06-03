"""
Base agent classes and communication system for Ophira AI.

Provides the foundation for the hierarchical agent architecture
with message passing, lifecycle management, and agent registry.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from loguru import logger

from ..core.logging import get_agent_logger


class AgentStatus(Enum):
    """Agent status enumeration."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessageType(Enum):
    """Message type enumeration."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    content: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    timeout_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "context": self.context,
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
            "timeout_seconds": self.timeout_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            context=data["context"],
            correlation_id=data.get("correlation_id"),
            requires_response=data.get("requires_response", False),
            timeout_seconds=data.get("timeout_seconds")
        )


class AgentCapability(Enum):
    """Agent capability enumeration."""
    CONVERSATION = "conversation"
    MEDICAL_ANALYSIS = "medical_analysis"
    SENSOR_CONTROL = "sensor_control"
    MEMORY_MANAGEMENT = "memory_management"
    EMERGENCY_RESPONSE = "emergency_response"
    DATA_PROCESSING = "data_processing"
    REPORT_GENERATION = "report_generation"
    COORDINATION = "coordination"
    VALIDATION = "validation"


@dataclass
class AgentInfo:
    """Agent information structure."""
    
    id: str
    name: str
    agent_type: str
    status: AgentStatus
    capabilities: List[AgentCapability]
    load_level: float = 0.0  # 0.0 to 1.0
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents in the Ophira system."""
    
    def __init__(self, agent_id: str, name: str, agent_type: str, 
                 capabilities: List[AgentCapability]):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            agent_type: Type of agent (e.g., "voice", "medical", "sensor")
            capabilities: List of capabilities this agent provides
        """
        self.id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = AgentStatus.INACTIVE
        
        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Performance metrics
        self.load_level = 0.0
        self.last_heartbeat = None
        self.message_count = 0
        self.error_count = 0
        
        # Logging
        self.logger = get_agent_logger(agent_type)
        
        # Task management
        self.tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        self.logger.info(f"Agent {self.name} ({self.id}) initialized")
    
    def _setup_message_handlers(self):
        """Setup default message handlers."""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.ERROR] = self._handle_error
        self.message_handlers[MessageType.COMMAND] = self._handle_command
    
    async def start(self):
        """Start the agent."""
        if self.is_running:
            return
        
        self.is_running = True
        self.status = AgentStatus.ACTIVE
        
        # Start message processing task
        message_task = asyncio.create_task(self._process_messages())
        self.tasks.append(message_task)
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.tasks.append(heartbeat_task)
        
        # Start agent-specific tasks
        await self._start_agent_tasks()
        
        self.logger.info(f"Agent {self.name} started")
    
    async def stop(self):
        """Stop the agent."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.status = AgentStatus.SHUTDOWN
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        
        # Cleanup
        await self._cleanup()
        
        self.logger.info(f"Agent {self.name} stopped")
    
    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Send a message to another agent.
        
        Args:
            message: Message to send
            
        Returns:
            Response message if requires_response is True
        """
        message.sender_id = self.id
        
        self.logger.log_message_sent(
            message.recipient_id,
            message.message_type.value,
            str(message.content)[:100]
        )
        
        # Register agent registry to handle message routing
        registry = AgentRegistry.get_instance()
        
        if message.requires_response:
            # Create future for response
            future = asyncio.Future()
            self.pending_responses[message.id] = future
            
            # Send message
            await registry.route_message(message)
            
            # Wait for response with timeout
            try:
                timeout = message.timeout_seconds or 30
                response = await asyncio.wait_for(future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self.logger.error(
                    "Message timeout",
                    f"No response received for message {message.id}",
                    {"message_id": message.id, "recipient": message.recipient_id}
                )
                self.pending_responses.pop(message.id, None)
                return None
        else:
            # Send message without waiting for response
            await registry.route_message(message)
            return None
    
    async def receive_message(self, message: AgentMessage):
        """
        Receive a message from another agent.
        
        Args:
            message: Received message
        """
        await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Process messages from the queue."""
        while self.is_running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Update metrics
                self.message_count += 1
                
                # Handle response messages
                if (message.correlation_id and 
                    message.correlation_id in self.pending_responses):
                    future = self.pending_responses.pop(message.correlation_id)
                    if not future.done():
                        future.set_result(message)
                    continue
                
                # Handle other message types
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    try:
                        await handler(message)
                    except Exception as e:
                        self.error_count += 1
                        self.logger.log_error(
                            "Message handling error",
                            str(e),
                            {"message_id": message.id, "message_type": message.message_type.value}
                        )
                else:
                    # No handler found - use default processing
                    await self._handle_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.error_count += 1
                self.logger.log_error(
                    "Message processing error",
                    str(e),
                    {}
                )
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat signals."""
        while self.is_running:
            try:
                self.last_heartbeat = datetime.now()
                
                # Update load level
                self.load_level = min(1.0, self.message_queue.qsize() / 100)
                
                # Register with agent registry
                registry = AgentRegistry.get_instance()
                await registry.update_agent_info(self.get_agent_info())
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.log_error(
                    "Heartbeat error",
                    str(e),
                    {}
                )
    
    def get_agent_info(self) -> AgentInfo:
        """Get current agent information."""
        return AgentInfo(
            id=self.id,
            name=self.name,
            agent_type=self.agent_type,
            status=self.status,
            capabilities=self.capabilities,
            load_level=self.load_level,
            last_heartbeat=self.last_heartbeat,
            metadata={
                "message_count": self.message_count,
                "error_count": self.error_count,
                "queue_size": self.message_queue.qsize()
            }
        )
    
    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle heartbeat message."""
        # Respond with current status
        response = AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            correlation_id=message.id,
            content={"status": self.status.value, "load_level": self.load_level}
        )
        await self.send_message(response)
    
    async def _handle_error(self, message: AgentMessage):
        """Handle error message."""
        self.logger.log_error(
            "Received error message",
            message.content.get("error", "Unknown error"),
            message.context
        )
    
    async def _handle_command(self, message: AgentMessage):
        """Handle command message."""
        command = message.content.get("command")
        
        if command == "status":
            response = AgentMessage(
                sender_id=self.id,
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                correlation_id=message.id,
                content={"agent_info": self.get_agent_info().__dict__}
            )
            await self.send_message(response)
        elif command == "stop":
            await self.stop()
        else:
            await self._handle_custom_command(message)
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _start_agent_tasks(self):
        """Start agent-specific tasks. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _cleanup(self):
        """Cleanup agent resources. Must be implemented by subclasses."""
        pass
    
    async def _handle_custom_command(self, message: AgentMessage):
        """Handle custom commands. Can be overridden by subclasses."""
        self.logger.log_error(
            "Unknown command",
            message.content.get("command", ""),
            message.context
        )


class AgentRegistry:
    """Registry for managing agents and message routing."""
    
    _instance: Optional["AgentRegistry"] = None
    
    def __init__(self):
        """Initialize agent registry."""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_info: Dict[str, AgentInfo] = {}
        self.message_bus: asyncio.Queue = asyncio.Queue()
        self.routing_table: Dict[str, str] = {}  # capability -> agent_id
        self.logger = get_agent_logger("registry")
    
    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Get singleton instance of agent registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def register_agent(self, agent: BaseAgent):
        """Register an agent with the registry."""
        self.agents[agent.id] = agent
        self.agent_info[agent.id] = agent.get_agent_info()
        
        # Update routing table
        for capability in agent.capabilities:
            self.routing_table[capability.value] = agent.id
        
        self.logger.info(f"Agent registered: {agent.name} ({agent.id})")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent from the registry."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.agent_info.pop(agent_id, None)
            
            # Remove from routing table
            for capability in agent.capabilities:
                if self.routing_table.get(capability.value) == agent_id:
                    self.routing_table.pop(capability.value, None)
            
            self.logger.info(f"Agent unregistered: {agent.name} ({agent_id})")
    
    async def route_message(self, message: AgentMessage):
        """Route a message to the appropriate agent."""
        recipient_id = message.recipient_id
        
        if recipient_id in self.agents:
            agent = self.agents[recipient_id]
            await agent.receive_message(message)
        else:
            self.logger.log_error(
                "Message routing failed",
                f"Agent {recipient_id} not found",
                {"message_id": message.id, "recipient_id": recipient_id}
            )
    
    async def update_agent_info(self, agent_info: AgentInfo):
        """Update agent information."""
        self.agent_info[agent_info.id] = agent_info
    
    def get_agent_by_capability(self, capability: AgentCapability) -> Optional[str]:
        """Get agent ID by capability."""
        return self.routing_table.get(capability.value)
    
    def get_all_agents(self) -> Dict[str, AgentInfo]:
        """Get information for all registered agents."""
        return self.agent_info.copy()
    
    def get_agents_by_type(self, agent_type: str) -> List[AgentInfo]:
        """Get agents by type."""
        return [info for info in self.agent_info.values() 
                if info.agent_type == agent_type]
    
    def get_healthy_agents(self) -> List[AgentInfo]:
        """Get agents that are currently healthy."""
        now = datetime.now()
        healthy_agents = []
        
        for info in self.agent_info.values():
            if (info.status == AgentStatus.ACTIVE and 
                info.last_heartbeat and 
                (now - info.last_heartbeat).seconds < 60):  # 1 minute timeout
                healthy_agents.append(info)
        
        return healthy_agents 