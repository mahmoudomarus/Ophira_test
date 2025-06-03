# Phase 1: Stability Implementation Plan (1-2 Days)

## Overview
This document outlines the technical implementation plan for achieving system stability across three critical areas: backend optimization, hardware calibration, and comprehensive integration testing.

---

## 🎯 **Objective 1: Backend Stability - Database Connections & Error Handling**

### Current Issues Identified
- Database connection pooling needs optimization
- Error handling in async operations needs improvement  
- Memory management for sensor data streams
- WebSocket connection stability under load

### Implementation Tasks

#### **1.1 Database Connection Optimization**

**Priority: HIGH** | **Time: 4-6 hours**

**Problems to Fix:**
- Single connection instances without pooling
- No connection retry mechanisms
- Blocking operations in async contexts
- Memory leaks in long-running connections

**Solutions:**

```python
# Enhanced Database Manager with Connection Pooling
class OptimizedDatabaseManager:
    def __init__(self):
        # Connection pools with proper sizing
        self.mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        
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
        
    async def _initialize_postgresql_optimized(self):
        """Enhanced PostgreSQL with optimized pool settings"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="ophira_db", 
                user="postgres",
                password="postgres",
                min_size=5,      # Increased minimum connections
                max_size=20,     # Increased maximum connections  
                max_queries=50000,  # Queries per connection before recycling
                max_inactive_connection_lifetime=3600,  # 1 hour
                command_timeout=30,  # Command timeout
                server_settings={
                    'application_name': 'ophira_medical_system',
                    'jit': 'off'  # Disable JIT for predictable performance
                }
            )
        except Exception as e:
            await self._handle_db_connection_error('postgresql', e)
    
    async def _initialize_redis_optimized(self):
        """Enhanced Redis with connection pooling"""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                settings.database.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_timeout=5.0,
                socket_connect_timeout=10.0,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        except Exception as e:
            await self._handle_db_connection_error('redis', e)
    
    async def _handle_db_connection_error(self, db_type: str, error: Exception):
        """Centralized database connection error handling"""
        logger.error(f"Database connection failed ({db_type}): {error}")
        
        for attempt in range(self.max_retries):
            await asyncio.sleep(self.retry_delay * (attempt + 1))
            try:
                if db_type == 'postgresql':
                    await self._initialize_postgresql_optimized()
                elif db_type == 'redis': 
                    await self._initialize_redis_optimized()
                elif db_type == 'mongodb':
                    await self._initialize_mongodb_optimized()
                logger.info(f"✅ {db_type} reconnection successful")
                return True
            except Exception as retry_error:
                logger.warning(f"Retry {attempt + 1} failed for {db_type}: {retry_error}")
        
        logger.critical(f"❌ All retry attempts failed for {db_type}")
        return False
```

**Files to Modify:**
- `ophira/core/database.py` - Add connection pooling and retry logic
- `ophira/core/config.py` - Add database pool configuration settings
- `web_interface.py` - Implement proper connection management in FastAPI

#### **1.2 Async Error Handling Enhancement**

**Priority: HIGH** | **Time: 3-4 hours**

**Current Issues:**
- Unhandled exceptions in WebSocket connections
- Memory leaks in long-running async tasks
- Inconsistent error responses across API endpoints

**Solutions:**

```python
# Enhanced Error Handling Middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except asyncio.TimeoutError:
        logger.error(f"Request timeout: {request.url}")
        return JSONResponse(
            status_code=408,
            content={"error": "Request timeout", "retry_after": 5}
        )
    except DatabaseConnectionError as e:
        logger.error(f"Database error: {e}")
        return JSONResponse(
            status_code=503, 
            content={"error": "Service temporarily unavailable", "retry_after": 30}
        )
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": str(uuid.uuid4())}
        )

# WebSocket Connection Management
class StableWebSocketManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.heartbeat_interval = 30
        self.connection_timeout = 60
        
    async def heartbeat_task(self, websocket: WebSocket, session_id: str):
        """Maintain WebSocket connection health"""
        try:
            while session_id in self.connections:
                await websocket.ping()
                await asyncio.sleep(self.heartbeat_interval)
        except Exception as e:
            logger.warning(f"Heartbeat failed for {session_id}: {e}")
            await self.disconnect(session_id)
    
    async def handle_connection_error(self, session_id: str, error: Exception):
        """Handle WebSocket connection errors gracefully"""
        logger.error(f"WebSocket error for {session_id}: {error}")
        await self.disconnect(session_id)
        
        # Notify monitoring system
        await self.notify_connection_failure(session_id, str(error))
```

#### **1.3 Memory Management Optimization**

**Priority: MEDIUM** | **Time: 2-3 hours**

**Solutions:**
- Implement data streaming for large sensor data
- Add automatic cleanup for expired sessions
- Optimize image processing memory usage

---

## 🔧 **Objective 2: Hardware Calibration - Fine-tune Camera & Sensors**

### Current Sensor Status
- **NIR Camera (OV9281)**: Basic implementation exists, needs calibration routine
- **Heart Rate Sensor**: Hardware interface ready, needs signal processing optimization
- **PPG Sensor**: Simulation working, needs real hardware integration
- **Temperature/BP Sensors**: Simulation only

### Implementation Tasks

#### **2.1 NIR Camera Calibration Enhancement**

**Priority: HIGH** | **Time: 4-5 hours**

**Current Issues:**
- Basic calibration without adaptive parameters
- No automatic white balance for NIR
- Image quality assessment needs improvement

**Solutions:**

```python
class EnhancedNIRCalibration:
    def __init__(self, camera_sensor):
        self.camera = camera_sensor
        self.calibration_frames = 50  # Increased sample size
        self.roi_analysis_zones = 9   # 3x3 grid analysis
        
    async def comprehensive_calibration(self) -> Dict[str, Any]:
        """Advanced calibration routine for optimal NIR imaging"""
        
        # Step 1: Environment Assessment
        environment_data = await self._assess_environment()
        
        # Step 2: Auto-Exposure Optimization
        optimal_exposure = await self._optimize_exposure()
        
        # Step 3: Gain Calibration  
        optimal_gain = await self._optimize_gain()
        
        # Step 4: Focus Calibration
        focus_metrics = await self._calibrate_focus()
        
        # Step 5: Vessel Detection Calibration
        vessel_params = await self._calibrate_vessel_detection()
        
        return {
            "environment": environment_data,
            "exposure": optimal_exposure,
            "gain": optimal_gain, 
            "focus": focus_metrics,
            "vessel_detection": vessel_params,
            "calibration_quality": await self._assess_calibration_quality()
        }
    
    async def _optimize_exposure(self) -> float:
        """Auto-exposure optimization for NIR"""
        best_exposure = -7
        best_score = 0
        
        # Test exposure range
        for exposure in [-9, -8, -7, -6, -5, -4]:
            self.camera.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            await asyncio.sleep(0.2)  # Allow camera to adjust
            
            # Capture test frames
            frames = []
            for _ in range(5):
                ret, frame = self.camera.camera.read()
                if ret:
                    frames.append(frame)
            
            if frames:
                score = self._calculate_exposure_score(frames)
                if score > best_score:
                    best_score = score
                    best_exposure = exposure
        
        return best_exposure
    
    def _calculate_exposure_score(self, frames: List[np.ndarray]) -> float:
        """Calculate optimal exposure score"""
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(avg_frame.shape) == 3:
            gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = avg_frame
        
        # Score factors
        mean_intensity = np.mean(gray)
        contrast = np.std(gray)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Avoid overexposure (clipping)
        overexposed_pixels = np.sum(gray > 250) / gray.size
        underexposed_pixels = np.sum(gray < 10) / gray.size
        
        # Optimal score balances intensity, contrast, and exposure
        intensity_score = 1 - abs(mean_intensity - 128) / 128  # Target ~50% intensity
        contrast_score = min(contrast / 50, 1.0)  # Good contrast without noise
        exposure_penalty = (overexposed_pixels + underexposed_pixels) * 2
        
        return max(0, (intensity_score * 0.4 + contrast_score * 0.4) - exposure_penalty)
```

#### **2.2 Heart Rate Sensor Signal Processing**

**Priority: HIGH** | **Time: 3-4 hours**

**Current Issues:**
- Basic peak detection algorithm
- No adaptive filtering
- Signal quality assessment too simple

**Solutions:**

```python
class AdvancedHeartRateProcessor:
    def __init__(self):
        self.sampling_rate = 100  # Hz
        self.butter_filter = None
        self.adaptive_threshold = True
        self.signal_buffer_size = 1000
        
    def setup_signal_processing(self):
        """Initialize signal processing components"""
        # Butterworth bandpass filter for heart rate signals (0.5-4 Hz)
        from scipy.signal import butter
        low_freq = 0.5 / (self.sampling_rate / 2)   # 30 BPM
        high_freq = 4.0 / (self.sampling_rate / 2)  # 240 BPM
        self.butter_filter = butter(4, [low_freq, high_freq], btype='band')
    
    def process_signal_advanced(self, raw_signal: List[float]) -> Dict[str, Any]:
        """Advanced heart rate signal processing"""
        
        # Apply bandpass filter
        filtered_signal = self._apply_filter(raw_signal)
        
        # Adaptive peak detection
        peaks = self._detect_peaks_adaptive(filtered_signal)
        
        # Calculate heart rate with confidence
        bpm, confidence = self._calculate_bpm_with_confidence(peaks)
        
        # Signal quality metrics
        quality_metrics = self._assess_signal_quality(filtered_signal, peaks)
        
        return {
            "bpm": bpm,
            "confidence": confidence,
            "signal_quality": quality_metrics,
            "peaks_detected": len(peaks),
            "signal_strength": np.std(filtered_signal)
        }
    
    def _detect_peaks_adaptive(self, signal: np.ndarray) -> List[int]:
        """Adaptive peak detection with dynamic thresholding"""
        # Calculate dynamic threshold based on signal characteristics
        window_size = self.sampling_rate * 2  # 2-second window
        peaks = []
        
        for i in range(window_size, len(signal) - window_size):
            window = signal[i-window_size:i+window_size]
            threshold = np.mean(window) + 0.6 * np.std(window)
            
            # Peak detection with refractory period
            if (signal[i] > threshold and 
                signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and
                (not peaks or i - peaks[-1] > self.sampling_rate * 0.3)):  # 300ms refractory
                peaks.append(i)
        
        return peaks
```

#### **2.3 Sensor Synchronization**

**Priority: MEDIUM** | **Time: 2-3 hours**

**Solutions:**
- Implement timestamp synchronization across sensors
- Add sensor data fusion algorithms
- Create sensor health monitoring

---

## 🧪 **Objective 3: Complete Integration Testing**

### Current Testing Status
- Basic integration tests exist in `test_integration.py`
- Limited to API endpoint testing
- No performance or load testing
- Missing sensor simulation validation

### Implementation Tasks

#### **3.1 Enhanced Integration Test Suite**

**Priority: HIGH** | **Time: 4-5 hours**

**Current Issues:**
- Tests don't cover sensor hardware
- No WebSocket stress testing  
- Missing medical workflow validation
- No performance benchmarking

**Solutions:**

```python
class ComprehensiveIntegrationTester:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.ws_url = "ws://localhost:8001"
        self.test_results = []
        self.performance_metrics = {}
        
    async def run_full_test_suite(self):
        """Execute complete integration test suite"""
        
        # Phase 1: Basic Connectivity
        await self.test_basic_connectivity()
        
        # Phase 2: Database Integration
        await self.test_database_operations()
        
        # Phase 3: Sensor Integration  
        await self.test_sensor_integration()
        
        # Phase 4: Medical Workflow
        await self.test_medical_workflows()
        
        # Phase 5: Performance & Load
        await self.test_performance()
        
        # Phase 6: Error Handling
        await self.test_error_scenarios()
        
        # Generate comprehensive report
        return await self.generate_test_report()
    
    async def test_sensor_integration(self):
        """Test real sensor hardware integration"""
        
        # Test NIR Camera
        camera_result = await self._test_nir_camera_full()
        
        # Test Heart Rate Sensor
        hr_result = await self._test_heart_rate_sensor_full()
        
        # Test Sensor Synchronization
        sync_result = await self._test_sensor_synchronization()
        
        # Test Sensor Data Processing Pipeline
        pipeline_result = await self._test_data_pipeline()
        
        return {
            "camera": camera_result,
            "heart_rate": hr_result, 
            "synchronization": sync_result,
            "data_pipeline": pipeline_result
        }
    
    async def _test_nir_camera_full(self) -> Dict[str, Any]:
        """Comprehensive NIR camera testing"""
        
        test_results = {
            "connection": False,
            "calibration": False,
            "image_capture": False,
            "image_quality": 0.0,
            "vessel_detection": False,
            "focus_accuracy": 0.0,
            "processing_time": 0.0
        }
        
        try:
            # Test camera connection
            camera = OV9281NIRCamera("test_camera", {
                "device_id": 0,
                "resolution": (1280, 720),
                "fps": 60
            })
            
            start_time = time.time()
            connected = await camera.connect()
            test_results["connection"] = connected
            
            if connected:
                # Test calibration
                calibrated = await camera.calibrate()
                test_results["calibration"] = calibrated
                
                # Test image capture and processing
                for i in range(10):  # Multiple samples
                    raw_data = await camera.read_raw_data()
                    if raw_data:
                        processed = camera.process_raw_data(raw_data)
                        if processed:
                            test_results["image_capture"] = True
                            test_results["image_quality"] += processed.quality_score
                            
                            # Test vessel detection
                            if processed.value.get("vessel_density", 0) > 0:
                                test_results["vessel_detection"] = True
                
                test_results["image_quality"] /= 10  # Average
                test_results["processing_time"] = time.time() - start_time
                
                await camera.disconnect()
        
        except Exception as e:
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_performance(self):
        """Performance and load testing"""
        
        # Test concurrent WebSocket connections
        ws_performance = await self._test_websocket_load()
        
        # Test API response times under load
        api_performance = await self._test_api_load()
        
        # Test database performance
        db_performance = await self._test_database_performance()
        
        # Test memory usage
        memory_usage = await self._test_memory_usage()
        
        return {
            "websocket": ws_performance,
            "api": api_performance, 
            "database": db_performance,
            "memory": memory_usage
        }
    
    async def _test_websocket_load(self) -> Dict[str, Any]:
        """Test WebSocket connection under load"""
        
        # Test parameters
        concurrent_connections = 20
        messages_per_connection = 100
        message_interval = 0.1  # seconds
        
        results = {
            "connections_successful": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "average_latency": 0.0,
            "errors": []
        }
        
        async def websocket_client(client_id: int):
            try:
                # Login and get session
                session_response = await self._create_test_session(f"load_test_user_{client_id}")
                if not session_response:
                    return
                
                session_id = session_response["session_id"]
                
                # Connect WebSocket
                async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                    results["connections_successful"] += 1
                    
                    # Send messages and measure latency
                    latencies = []
                    for i in range(messages_per_connection):
                        start_time = time.time()
                        
                        message = {
                            "type": "chat_message",
                            "message": f"Test message {i} from client {client_id}",
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await websocket.send(json.dumps(message))
                        results["messages_sent"] += 1
                        
                        # Wait for response
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            latency = time.time() - start_time
                            latencies.append(latency)
                            results["messages_received"] += 1
                        except asyncio.TimeoutError:
                            results["errors"].append(f"Timeout on client {client_id} message {i}")
                        
                        await asyncio.sleep(message_interval)
                    
                    # Calculate average latency for this client
                    if latencies:
                        results["average_latency"] += np.mean(latencies)
            
            except Exception as e:
                results["errors"].append(f"Client {client_id} error: {str(e)}")
        
        # Run concurrent WebSocket clients
        tasks = [websocket_client(i) for i in range(concurrent_connections)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate final average latency
        if results["connections_successful"] > 0:
            results["average_latency"] /= results["connections_successful"]
        
        return results
```

#### **3.2 Medical Workflow Validation**

**Priority: HIGH** | **Time: 3-4 hours**

**Solutions:**
- Test complete patient monitoring workflow
- Validate emergency response scenarios
- Test medical data accuracy and consistency
- Verify HIPAA compliance in data handling

#### **3.3 Performance Benchmarking**

**Priority: MEDIUM** | **Time: 2-3 hours**

**Solutions:**
- Establish performance baselines
- Load testing for concurrent users
- Memory usage profiling
- Database query optimization validation

---

## 📋 **Implementation Checklist**

### Day 1 Tasks (8 hours)
- [ ] **Database Connection Optimization** (4 hours)
  - [ ] Implement connection pooling for PostgreSQL
  - [ ] Add Redis connection pool
  - [ ] Implement retry mechanisms
  - [ ] Add connection health monitoring

- [ ] **NIR Camera Calibration** (4 hours)
  - [ ] Implement comprehensive calibration routine
  - [ ] Add auto-exposure optimization
  - [ ] Enhance image quality assessment
  - [ ] Add vessel detection calibration

### Day 2 Tasks (8 hours)
- [ ] **Error Handling Enhancement** (3 hours)
  - [ ] Add middleware for error handling
  - [ ] Implement WebSocket connection management
  - [ ] Add memory management optimization

- [ ] **Heart Rate Signal Processing** (3 hours)
  - [ ] Implement advanced signal filtering
  - [ ] Add adaptive peak detection
  - [ ] Enhance signal quality assessment

- [ ] **Integration Testing** (2 hours)
  - [ ] Complete sensor integration tests
  - [ ] Add performance benchmarking
  - [ ] Create comprehensive test report

---

## 🎯 **Success Metrics**

### Backend Stability
- [ ] Database connection uptime > 99.5%
- [ ] Average API response time < 100ms
- [ ] Memory usage stable over 24 hours
- [ ] Zero unhandled exceptions in production

### Hardware Calibration  
- [ ] NIR camera image quality score > 0.8
- [ ] Heart rate accuracy within ±3 BPM
- [ ] Sensor synchronization drift < 50ms
- [ ] Calibration routine completion < 30 seconds

### Integration Testing
- [ ] 100% test coverage for critical paths
- [ ] Load testing with 50+ concurrent users
- [ ] All medical workflows validated
- [ ] Performance benchmarks documented

---

## 🚀 **Next Steps After Phase 1**

1. **Phase 2: Feature Enhancement** - Add advanced medical analysis
2. **Phase 3: Security Hardening** - Implement comprehensive security measures  
3. **Phase 4: Scalability** - Optimize for production deployment
4. **Phase 5: Medical Certification** - Prepare for medical device compliance

This plan ensures a stable foundation for the Ophira AI medical monitoring system while maintaining focus on the core objectives of reliability, accuracy, and performance. 