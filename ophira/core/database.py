"""
Database models and connection management for Ophira AI medical system.

Supports MongoDB for document storage, PostgreSQL for relational data,
and Redis for caching and real-time data.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from enum import Enum

import motor.motor_asyncio
import asyncpg
import aioredis
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from .config import get_settings
from .logging import setup_logging

# Initialize logging
logger = setup_logging()
settings = get_settings()

Base = declarative_base()

# Enums
class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SensorType(str, Enum):
    NIR_CAMERA = "nir_camera"
    PPG_SENSOR = "ppg_sensor"
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE = "blood_pressure"
    TEMPERATURE = "temperature"

# Custom Exception Classes
class DatabaseConnectionError(Exception):
    """Database connection related errors"""
    pass

class DatabaseRetryExhaustedError(Exception):
    """All database retry attempts exhausted"""
    pass

# PostgreSQL Models (SQLAlchemy)
class User(Base):
    """User account information"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255))
    date_of_birth = Column(DateTime)
    medical_record_number = Column(String(50), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    medical_analyses = relationship("MedicalAnalysis", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    health_records = relationship("HealthRecord", back_populates="user")

class MedicalAnalysis(Base):
    """Medical analysis results"""
    __tablename__ = "medical_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    analysis_type = Column(String(100), nullable=False)  # retinal, glaucoma, cardiovascular
    status = Column(String(20), default=AnalysisStatus.PENDING)
    confidence_score = Column(Float)
    risk_level = Column(String(20))
    findings = Column(JSONB)  # List of findings
    recommendations = Column(JSONB)  # List of recommendations
    image_path = Column(String(500))  # Path to analyzed image
    model_used = Column(String(100))  # AI model that performed analysis
    processing_time = Column(Float)  # Time taken for analysis in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="medical_analyses")

class Conversation(Base):
    """Chat conversations with Ophira AI"""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    """Individual messages in conversations"""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    sender_type = Column(String(20), nullable=False)  # user, ai, system
    content = Column(Text, nullable=False)
    message_metadata = Column(JSONB)  # Additional message metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class HealthRecord(Base):
    """Health monitoring data"""
    __tablename__ = "health_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    sensor_type = Column(String(50), nullable=False)
    measurement_type = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_critical = Column(Boolean, default=False)
    notes = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="health_records")

# Enhanced Database Connection Manager with Optimizations
class OptimizedDatabaseManager:
    """
    Enhanced database manager with connection pooling, retry mechanisms,
    health monitoring, and optimized async operations.
    """
    
    def __init__(self):
        # Connection pools with proper sizing
        self.mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.mongo_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[aioredis.ConnectionPool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Legacy SQLAlchemy support
        self.postgres_engine = None
        self.postgres_session = None
        
        # Connection health monitoring
        self.health_check_interval = 30  # seconds
        self.connection_timeouts = {
            'mongo': 10.0,
            'postgres': 15.0,
            'redis': 5.0
        }
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.exponential_backoff = True
        
        # Connection status tracking
        self.connection_status = {
            'mongo': False,
            'postgres': False,
            'redis': False
        }
        
        # Performance metrics
        self.connection_metrics = {
            'mongo': {'connections': 0, 'errors': 0, 'avg_response_time': 0.0},
            'postgres': {'connections': 0, 'errors': 0, 'avg_response_time': 0.0},
            'redis': {'connections': 0, 'errors': 0, 'avg_response_time': 0.0}
        }
        
        # Health check task
        self._health_check_task = None
        
    async def initialize(self):
        """Initialize all database connections with enhanced error handling"""
        logger.info("🔧 Starting optimized database initialization...")
        
        # Initialize connections concurrently for faster startup
        initialization_tasks = [
            self._initialize_mongodb_optimized(),
            self._initialize_postgresql_optimized(),
            self._initialize_redis_optimized()
        ]
        
        # Execute with timeout and gather results
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*initialization_tasks, return_exceptions=True),
                timeout=30.0
            )
            
            # Process results
            success_count = sum(1 for result in results if result is True)
            logger.info(f"✅ Database initialization complete: {success_count}/3 connections successful")
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            return success_count >= 2  # At least 2 databases should be connected
            
        except asyncio.TimeoutError:
            logger.error("❌ Database initialization timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    async def _initialize_mongodb_optimized(self) -> bool:
        """Enhanced MongoDB initialization with connection pooling"""
        try:
            logger.info("🔌 Connecting to MongoDB...")
            
            # Enhanced MongoDB client with optimized settings
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.database.mongodb_url,
                maxPoolSize=20,        # Maximum connections in pool
                minPoolSize=5,         # Minimum connections in pool
                maxIdleTimeMS=30000,   # Close connections after 30 seconds of inactivity
                waitQueueTimeoutMS=5000,  # Wait timeout for connection from pool
                serverSelectionTimeoutMS=10000,  # Server selection timeout
                connectTimeoutMS=10000,   # Connection timeout
                socketTimeoutMS=20000,    # Socket timeout
                retryWrites=True,         # Enable retry writes
                retryReads=True,          # Enable retry reads
                appName="ophira_medical_system"
            )
            
            self.mongo_db = self.mongo_client[settings.database.mongodb_database]
            
            # Test connection with timeout
            start_time = time.time()
            await asyncio.wait_for(
                self.mongo_client.admin.command('ping'),
                timeout=self.connection_timeouts['mongo']
            )
            response_time = time.time() - start_time
            
            self.connection_status['mongo'] = True
            self.connection_metrics['mongo']['connections'] += 1
            self.connection_metrics['mongo']['avg_response_time'] = response_time
            
            logger.info(f"✅ MongoDB connected successfully (response time: {response_time:.3f}s)")
            
            # Create indexes for performance
            await self._create_mongo_indexes_optimized()
            
            return True
            
        except Exception as e:
            self.connection_status['mongo'] = False
            self.connection_metrics['mongo']['errors'] += 1
            await self._handle_db_connection_error('mongodb', e)
            return False
    
    async def _initialize_postgresql_optimized(self) -> bool:
        """Enhanced PostgreSQL with optimized pool settings"""
        try:
            logger.info("🔌 Connecting to PostgreSQL...")
            
            start_time = time.time()
            
            # Create optimized connection pool
            self.postgres_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="ophira_db",
                user="postgres",
                password="postgres",
                min_size=5,           # Minimum connections in pool
                max_size=20,          # Maximum connections in pool
                max_queries=50000,    # Queries per connection before recycling
                max_inactive_connection_lifetime=3600,  # 1 hour
                command_timeout=30,   # Command timeout
                server_settings={
                    'application_name': 'ophira_medical_system',
                    'jit': 'off',     # Disable JIT for predictable performance
                    'log_statement': 'none'  # Reduce log verbosity
                }
            )
            
            # Test connection
            async with self.postgres_pool.acquire() as connection:
                await connection.execute("SELECT 1")
            
            response_time = time.time() - start_time
            self.connection_status['postgres'] = True
            self.connection_metrics['postgres']['connections'] += 1
            self.connection_metrics['postgres']['avg_response_time'] = response_time
            
            logger.info(f"✅ PostgreSQL connected successfully (response time: {response_time:.3f}s)")
            
            # Create SQLAlchemy engine for ORM operations (legacy support)
            self.postgres_engine = create_engine(
                "postgresql://postgres:postgres@localhost:5432/ophira_db",
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create tables
            Base.metadata.create_all(self.postgres_engine)
            
            # Create session factory
            self.postgres_session = sessionmaker(
                bind=self.postgres_engine,
                autoflush=True,
                autocommit=False
            )
            
            return True
            
        except Exception as e:
            self.connection_status['postgres'] = False
            self.connection_metrics['postgres']['errors'] += 1
            await self._handle_db_connection_error('postgresql', e)
            return False
    
    async def _initialize_redis_optimized(self) -> bool:
        """Enhanced Redis with connection pooling"""
        try:
            logger.info("🔌 Connecting to Redis...")
            
            start_time = time.time()
            
            # Create async Redis connection with simplified settings
            self.redis_client = aioredis.from_url(
                settings.database.redis_url,
                max_connections=20,          # Maximum connections in pool
                retry_on_timeout=True,       # Retry on timeout
                socket_timeout=5.0,          # Socket operation timeout
                socket_connect_timeout=10.0, # Connection timeout
                decode_responses=True        # Auto-decode responses
            )
            
            # Test connection with timeout
            await asyncio.wait_for(
                self.redis_client.ping(),
                timeout=self.connection_timeouts['redis']
            )
            
            response_time = time.time() - start_time
            self.connection_status['redis'] = True
            self.connection_metrics['redis']['connections'] += 1
            self.connection_metrics['redis']['avg_response_time'] = response_time
            
            logger.info(f"✅ Redis connected successfully (response time: {response_time:.3f}s)")
            
            return True
            
        except Exception as e:
            self.connection_status['redis'] = False
            self.connection_metrics['redis']['errors'] += 1
            await self._handle_db_connection_error('redis', e)
            return False
    
    async def _handle_db_connection_error(self, db_type: str, error: Exception):
        """Centralized database connection error handling with retry logic"""
        logger.error(f"Database connection failed ({db_type}): {error}")
        
        if not isinstance(error, (asyncio.TimeoutError, ConnectionError, OSError)):
            logger.error(f"Non-recoverable error for {db_type}: {type(error).__name__}")
            return False
        
        # Implement exponential backoff retry
        for attempt in range(self.max_retries):
            if self.exponential_backoff:
                delay = self.retry_delay * (2 ** attempt)
            else:
                delay = self.retry_delay * (attempt + 1)
            
            logger.info(f"🔄 Retrying {db_type} connection in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
            await asyncio.sleep(delay)
            
            try:
                if db_type == 'postgresql':
                    success = await self._initialize_postgresql_optimized()
                elif db_type == 'redis':
                    success = await self._initialize_redis_optimized()
                elif db_type == 'mongodb':
                    success = await self._initialize_mongodb_optimized()
                else:
                    logger.error(f"Unknown database type: {db_type}")
                    return False
                
                if success:
                    logger.info(f"✅ {db_type} reconnection successful")
                    return True
                    
            except Exception as retry_error:
                logger.warning(f"Retry {attempt + 1} failed for {db_type}: {retry_error}")
                self.connection_metrics[db_type]['errors'] += 1
        
        logger.critical(f"❌ All retry attempts failed for {db_type}")
        raise DatabaseRetryExhaustedError(f"Failed to connect to {db_type} after {self.max_retries} attempts")
    
    async def _create_mongo_indexes_optimized(self):
        """Create MongoDB indexes for better performance with error handling"""
        try:
            # Enhanced indexes for medical data
            index_tasks = [
                # Medical images collection
                self.mongo_db.medical_images.create_index([
                    ("user_id", 1), ("created_at", -1)
                ], background=True),
                self.mongo_db.medical_images.create_index([
                    ("analysis_type", 1), ("status", 1)
                ], background=True),
                self.mongo_db.medical_images.create_index([
                    ("created_at", -1)
                ], background=True),
                
                # Sensor data collection
                self.mongo_db.sensor_data.create_index([
                    ("user_id", 1), ("timestamp", -1)
                ], background=True),
                self.mongo_db.sensor_data.create_index([
                    ("sensor_type", 1), ("timestamp", -1)
                ], background=True),
                self.mongo_db.sensor_data.create_index([
                    ("is_critical", 1), ("timestamp", -1)
                ], background=True),
                
                # Agent interactions collection
                self.mongo_db.agent_interactions.create_index([
                    ("user_id", 1), ("timestamp", -1)
                ], background=True),
                self.mongo_db.agent_interactions.create_index([
                    ("session_id", 1), ("timestamp", -1)
                ], background=True),
                
                # Health analyses collection
                self.mongo_db.health_analyses.create_index([
                    ("user_id", 1), ("analysis_date", -1)
                ], background=True),
                self.mongo_db.health_analyses.create_index([
                    ("analysis_type", 1), ("status", 1)
                ], background=True)
            ]
            
            # Execute index creation concurrently
            await asyncio.gather(*index_tasks, return_exceptions=True)
            logger.info("✅ MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB index creation failed: {e}")
    
    async def _start_health_monitoring(self):
        """Start periodic health monitoring of database connections"""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("🔍 Database health monitoring started")
    
    async def _health_check_loop(self):
        """Periodic health check for all database connections"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                logger.info("🔍 Database health monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all database connections"""
        health_tasks = []
        
        # MongoDB health check
        if self.mongo_client:
            health_tasks.append(self._check_mongo_health())
        
        # PostgreSQL health check
        if self.postgres_pool:
            health_tasks.append(self._check_postgres_health())
        
        # Redis health check
        if self.redis_client:
            health_tasks.append(self._check_redis_health())
        
        if health_tasks:
            results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            # Log health status
            for i, result in enumerate(results):
                db_names = ['MongoDB', 'PostgreSQL', 'Redis']
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ {db_names[i]} health check failed: {result}")
                elif result:
                    logger.debug(f"✅ {db_names[i]} health check passed")
    
    async def _check_mongo_health(self) -> bool:
        """Check MongoDB connection health"""
        try:
            start_time = time.time()
            await asyncio.wait_for(
                self.mongo_client.admin.command('ping'),
                timeout=5.0
            )
            response_time = time.time() - start_time
            self.connection_metrics['mongo']['avg_response_time'] = response_time
            self.connection_status['mongo'] = True
            return True
        except Exception as e:
            self.connection_status['mongo'] = False
            self.connection_metrics['mongo']['errors'] += 1
            logger.warning(f"MongoDB health check failed: {e}")
            # Attempt reconnection
            asyncio.create_task(self._initialize_mongodb_optimized())
            return False
    
    async def _check_postgres_health(self) -> bool:
        """Check PostgreSQL connection health"""
        try:
            start_time = time.time()
            async with self.postgres_pool.acquire() as connection:
                await asyncio.wait_for(
                    connection.execute("SELECT 1"),
                    timeout=5.0
                )
            response_time = time.time() - start_time
            self.connection_metrics['postgres']['avg_response_time'] = response_time
            self.connection_status['postgres'] = True
            return True
        except Exception as e:
            self.connection_status['postgres'] = False
            self.connection_metrics['postgres']['errors'] += 1
            logger.warning(f"PostgreSQL health check failed: {e}")
            # Attempt reconnection
            asyncio.create_task(self._initialize_postgresql_optimized())
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connection health"""
        try:
            start_time = time.time()
            await asyncio.wait_for(
                self.redis_client.ping(),
                timeout=5.0
            )
            response_time = time.time() - start_time
            self.connection_metrics['redis']['avg_response_time'] = response_time
            self.connection_status['redis'] = True
            return True
        except Exception as e:
            self.connection_status['redis'] = False
            self.connection_metrics['redis']['errors'] += 1
            logger.warning(f"Redis health check failed: {e}")
            # Attempt reconnection
            asyncio.create_task(self._initialize_redis_optimized())
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive database health status"""
        return {
            "status": self.connection_status,
            "metrics": self.connection_metrics,
            "last_check": datetime.now().isoformat(),
            "overall_health": all(self.connection_status.values())
        }
    
    async def close(self):
        """Close all database connections gracefully"""
        logger.info("🔒 Closing database connections...")
        
        # Cancel health monitoring
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close connections
        close_tasks = []
        
        if self.mongo_client:
            close_tasks.append(self._close_mongo())
        
        if self.postgres_pool:
            close_tasks.append(self._close_postgres())
        
        if self.redis_client:
            close_tasks.append(self._close_redis())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        logger.info("🔒 All database connections closed")
    
    async def _close_mongo(self):
        """Close MongoDB connection"""
        try:
            self.mongo_client.close()
            self.connection_status['mongo'] = False
        except Exception as e:
            logger.error(f"Error closing MongoDB: {e}")
    
    async def _close_postgres(self):
        """Close PostgreSQL connection pool"""
        try:
            await self.postgres_pool.close()
            self.connection_status['postgres'] = False
        except Exception as e:
            logger.error(f"Error closing PostgreSQL: {e}")
    
    async def _close_redis(self):
        """Close Redis connection"""
        try:
            await self.redis_client.close()
            self.connection_status['redis'] = False
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")

# Create optimized database manager instance
_optimized_db_manager: Optional[OptimizedDatabaseManager] = None

def get_optimized_db_manager() -> OptimizedDatabaseManager:
    """Get or create the optimized database manager instance"""
    global _optimized_db_manager
    if _optimized_db_manager is None:
        _optimized_db_manager = OptimizedDatabaseManager()
    return _optimized_db_manager

# Legacy database manager for backward compatibility
class DatabaseManager:
    """Legacy database manager - kept for backward compatibility"""
    
    def __init__(self):
        self.mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.mongo_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_engine = None
        self.postgres_session = None
        
    async def initialize(self):
        """Initialize all database connections"""
        await self._initialize_mongodb()
        await self._initialize_postgresql()
        await self._initialize_redis()
        logger.info("✅ All database connections initialized successfully")
    
    async def _initialize_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.database.mongodb_url)
            self.mongo_db = self.mongo_client[settings.database.mongodb_database]
            
            # Test connection with timeout
            await asyncio.wait_for(self.mongo_client.admin.command('ping'), timeout=5.0)
            logger.info("✅ MongoDB connected successfully")
            
            # Create indexes
            await self._create_mongo_indexes()
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            logger.info("💡 Install and start MongoDB: sudo systemctl start mongod")
            # Don't raise - allow system to work without MongoDB
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection"""
        try:
            # Create connection pool with timeout
            self.postgres_pool = await asyncio.wait_for(
                asyncpg.create_pool(
                    host="localhost",
                    port=5432,
                    database="ophira_db",
                    user="postgres",
                    password="postgres",
                    min_size=1,
                    max_size=5
                ),
                timeout=5.0
            )
            
            # Test connection
            async with self.postgres_pool.acquire() as connection:
                await connection.execute("SELECT 1")
            
            logger.info("✅ PostgreSQL connected successfully")
            
            # Create SQLAlchemy engine for ORM operations
            self.postgres_engine = create_engine(
                "postgresql://postgres:postgres@localhost:5432/ophira_db",
                echo=False
            )
            
            # Create tables
            Base.metadata.create_all(self.postgres_engine)
            
            # Create session factory
            self.postgres_session = sessionmaker(bind=self.postgres_engine)
            
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL connection failed: {e}")
            logger.info("💡 Install and start PostgreSQL: sudo systemctl start postgresql")
            # Don't raise - allow system to work without PostgreSQL
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(settings.database.redis_url)
            
            # Test connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=5.0)
            logger.info("✅ Redis connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.info("💡 Install and start Redis: sudo systemctl start redis")
            # Don't raise - allow system to work without Redis
    
    async def _create_mongo_indexes(self):
        """Create MongoDB indexes for better performance"""
        try:
            # Medical images collection indexes
            await self.mongo_db.medical_images.create_index([("user_id", 1), ("created_at", -1)])
            await self.mongo_db.medical_images.create_index([("analysis_type", 1)])
            await self.mongo_db.medical_images.create_index([("created_at", -1)])
            
            # Sensor data collection indexes
            await self.mongo_db.sensor_data.create_index([("user_id", 1), ("timestamp", -1)])
            await self.mongo_db.sensor_data.create_index([("sensor_type", 1), ("timestamp", -1)])
            
            # Agent interactions collection indexes
            await self.mongo_db.agent_interactions.create_index([("user_id", 1), ("timestamp", -1)])
            await self.mongo_db.agent_interactions.create_index([("session_id", 1)])
            
            logger.info("✅ MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB index creation failed: {e}")
    
    async def close(self):
        """Close all database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("🔒 All database connections closed")

# Create a global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# Medical Data Service
class MedicalDataService:
    """Service for managing medical data operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    async def store_medical_image(self, user_id: str, image_data: bytes, 
                                metadata: Dict[str, Any]) -> str:
        """Store medical image in MongoDB"""
        if not self.db.mongo_client:
            logger.warning("MongoDB not available, skipping image storage")
            return f"simulated_image_id_{int(time.time())}"
            
        try:
            image_doc = {
                "user_id": user_id,
                "image_data": image_data,
                "metadata": metadata,
                "created_at": datetime.utcnow(),
                "analysis_status": AnalysisStatus.PENDING
            }
            
            result = await self.db.mongo_db.medical_images.insert_one(image_doc)
            logger.info(f"📸 Medical image stored: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to store medical image: {e}")
            return f"fallback_image_id_{int(time.time())}"
    
    async def store_analysis_result(self, user_id: str, analysis_data: Dict[str, Any]) -> str:
        """Store medical analysis result in PostgreSQL"""
        if not self.db.postgres_pool:
            logger.warning("PostgreSQL not available, skipping analysis storage")
            return f"simulated_analysis_id_{int(time.time())}"
            
        try:
            # Use asyncpg for direct SQL
            async with self.db.postgres_pool.acquire() as connection:
                query = """
                    INSERT INTO medical_analyses 
                    (user_id, analysis_type, status, confidence_score, risk_level, 
                     findings, recommendations, model_used, processing_time)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """
                
                result = await connection.fetchval(
                    query,
                    uuid.UUID(user_id),
                    analysis_data.get("analysis_type"),
                    analysis_data.get("status", AnalysisStatus.COMPLETED),
                    analysis_data.get("confidence_score"),
                    analysis_data.get("risk_level"),
                    json.dumps(analysis_data.get("findings", [])),
                    json.dumps(analysis_data.get("recommendations", [])),
                    analysis_data.get("processing_time")
                )
                
                logger.info(f"🔬 Analysis result stored: {result}")
                return str(result)
                
        except Exception as e:
            logger.error(f"❌ Failed to store analysis result: {e}")
            return f"fallback_analysis_id_{int(time.time())}"
    
    async def store_sensor_data(self, user_id: str, sensor_data: Dict[str, Any]):
        """Store sensor data in MongoDB for time-series analysis"""
        if not self.db.mongo_client:
            logger.warning("MongoDB not available, skipping sensor data storage")
            return
            
        try:
            sensor_doc = {
                "user_id": user_id,
                "sensor_type": sensor_data.get("sensor_type"),
                "measurement_type": sensor_data.get("measurement_type"),
                "value": sensor_data.get("value"),
                "unit": sensor_data.get("unit"),
                "timestamp": datetime.utcnow(),
                "metadata": sensor_data.get("metadata", {}),
                "is_critical": sensor_data.get("is_critical", False)
            }
            
            await self.db.mongo_db.sensor_data.insert_one(sensor_doc)
            
            # Also cache recent data in Redis for real-time access
            if self.db.redis_client:
                cache_key = f"sensor_data:{user_id}:{sensor_data.get('sensor_type')}"
                await self.db.redis_client.lpush(cache_key, json.dumps(sensor_doc, default=str))
                await self.db.redis_client.ltrim(cache_key, 0, 99)  # Keep last 100 readings
                await self.db.redis_client.expire(cache_key, 3600)  # 1 hour TTL
            
            logger.info(f"📊 Sensor data stored: {sensor_data.get('sensor_type')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store sensor data: {e}")
    
    async def store_conversation(self, user_id: str, session_id: str, 
                               message: str, sender_type: str) -> str:
        """Store conversation message"""
        if not self.db.postgres_pool:
            logger.warning("PostgreSQL not available, skipping conversation storage")
            return f"simulated_message_id_{int(time.time())}"
            
        try:
            async with self.db.postgres_pool.acquire() as connection:
                # First, ensure conversation exists
                conv_query = """
                    INSERT INTO conversations (user_id, session_id, title, created_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, session_id) DO NOTHING
                    RETURNING id
                """
                
                conv_id = await connection.fetchval(
                    conv_query,
                    uuid.UUID(user_id),
                    session_id,
                    f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                    datetime.utcnow()
                )
                
                if not conv_id:
                    # Get existing conversation
                    conv_id = await connection.fetchval(
                        "SELECT id FROM conversations WHERE user_id = $1 AND session_id = $2",
                        uuid.UUID(user_id), session_id
                    )
                
                # Store message
                msg_query = """
                    INSERT INTO messages (conversation_id, sender_type, content, created_at)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """
                
                msg_id = await connection.fetchval(
                    msg_query,
                    conv_id,
                    sender_type,
                    message,
                    datetime.utcnow()
                )
                
                logger.info(f"💬 Message stored: {msg_id}")
                return str(msg_id)
                
        except Exception as e:
            logger.error(f"❌ Failed to store conversation: {e}")
            return f"fallback_message_id_{int(time.time())}"
    
    async def get_user_analysis_history(self, user_id: str, 
                                      analysis_type: Optional[str] = None) -> List[Dict]:
        """Get user's analysis history"""
        try:
            async with self.db.postgres_pool.acquire() as connection:
                if analysis_type:
                    query = """
                        SELECT * FROM medical_analyses 
                        WHERE user_id = $1 AND analysis_type = $2
                        ORDER BY created_at DESC
                        LIMIT 50
                    """
                    rows = await connection.fetch(query, uuid.UUID(user_id), analysis_type)
                else:
                    query = """
                        SELECT * FROM medical_analyses 
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                        LIMIT 50
                    """
                    rows = await connection.fetch(query, uuid.UUID(user_id))
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get analysis history: {e}")
            return []
    
    async def get_recent_sensor_data(self, user_id: str, sensor_type: str, 
                                   limit: int = 100) -> List[Dict]:
        """Get recent sensor data from cache or database"""
        try:
            # Try Redis cache first
            if self.db.redis_client:
                cache_key = f"sensor_data:{user_id}:{sensor_type}"
                cached_data = await self.db.redis_client.lrange(cache_key, 0, limit-1)
                if cached_data:
                    return [json.loads(data) for data in cached_data]
            
            # Fallback to MongoDB
            cursor = self.db.mongo_db.sensor_data.find(
                {"user_id": user_id, "sensor_type": sensor_type}
            ).sort("timestamp", -1).limit(limit)
            
            return await cursor.to_list(length=limit)
            
        except Exception as e:
            logger.error(f"❌ Failed to get sensor data: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager()
medical_data_service = MedicalDataService(db_manager)

async def initialize_databases():
    """Initialize all database connections"""
    await db_manager.initialize()

async def close_databases():
    """Close all database connections"""
    await db_manager.close()

# Utility functions
async def create_test_user() -> str:
    """Create a test user for development"""
    try:
        async with db_manager.postgres_pool.acquire() as connection:
            user_id = await connection.fetchval(
                """
                INSERT INTO users (email, username, full_name, medical_record_number)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (email) DO UPDATE SET updated_at = NOW()
                RETURNING id
                """,
                "test@ophira.ai",
                "testuser",
                "Test User",
                "MRN001"
            )
            
            logger.info(f"👤 Test user created/updated: {user_id}")
            return str(user_id)
            
    except Exception as e:
        logger.error(f"❌ Failed to create test user: {e}")
        return "test-user-id" 