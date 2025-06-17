'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Camera,
  CameraOff,
  Activity,
  Wifi,
  WifiOff,
  Battery,
  Signal,
  Eye,
  Heart,
  Thermometer,
  Gauge,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  Play,
  Pause,
  Settings
} from 'lucide-react';
import { SensorReading, SensorStatus, VitalSigns, HealthTrend } from '@/types';
import { useSessionStore } from '@/stores/sessionStore';
import { useMedicalStore } from '@/stores/medicalStore';
import { formatDistanceToNow } from 'date-fns';

interface SensorPanelProps {
  className?: string;
}

export function SensorPanel({ className = '' }: SensorPanelProps) {
  const [activeTab, setActiveTab] = useState<'camera' | 'sensors' | 'trends'>('camera');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d' | '30d'>('1h');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const { sessionInfo } = useSessionStore();
  const { 
    sensorReadings, 
    sensorStatuses, 
    vitalSigns, 
    healthTrends,
    fetchSensorStatus,
    fetchDashboardData 
  } = useMedicalStore();

  useEffect(() => {
    if (sessionInfo?.session_id) {
      fetchSensorStatus(sessionInfo.session_id);
      fetchDashboardData(sessionInfo.session_id);
    }
  }, [sessionInfo?.session_id, fetchSensorStatus, fetchDashboardData]);

  const startCamera = async () => {
    try {
      setCameraError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const getSensorIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'heart_rate':
      case 'heart':
        return <Heart className="h-4 w-4 text-heart-500" />;
      case 'camera':
        return <Camera className="h-4 w-4 text-blue-500" />;
      case 'distance':
      case 'tof':
        return <Gauge className="h-4 w-4 text-purple-500" />;
      case 'temperature':
        return <Thermometer className="h-4 w-4 text-orange-500" />;
      default:
        return <Activity className="h-4 w-4 text-gray-500" />;
    }
  };

  const getSensorStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'text-success-600 bg-success-100';
      case 'disconnected':
        return 'text-gray-600 bg-gray-100';
      case 'error':
        return 'text-heart-600 bg-heart-100';
      case 'calibrating':
        return 'text-warning-600 bg-warning-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp className="h-4 w-4 text-success-600" />;
      case 'declining':
        return <TrendingDown className="h-4 w-4 text-heart-600" />;
      case 'stable':
        return <Minus className="h-4 w-4 text-gray-600" />;
      default:
        return <Minus className="h-4 w-4 text-gray-600" />;
    }
  };

  const getMockChartData = (metric: string, timeRange: string) => {
    // Mock data generation for demonstration
    const points = timeRange === '1h' ? 12 : timeRange === '6h' ? 36 : timeRange === '24h' ? 96 : 168;
    const data = [];
    
    for (let i = 0; i < points; i++) {
      const baseValue = metric === 'heart_rate' ? 75 : 
                       metric === 'temperature' ? 36.5 : 98;
      const variation = Math.random() * 10 - 5;
      data.push({
        time: i,
        value: baseValue + variation
      });
    }
    
    return data;
  };

  const renderCameraTab = () => (
    <div className="space-y-4">
      {/* Camera Feed */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Camera Feed</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={isCameraActive ? stopCamera : startCamera}
                className={`p-2 rounded-lg transition-colors ${
                  isCameraActive 
                    ? 'bg-heart-100 text-heart-600 hover:bg-heart-200' 
                    : 'bg-medical-100 text-medical-600 hover:bg-medical-200'
                }`}
              >
                {isCameraActive ? <CameraOff className="h-4 w-4" /> : <Camera className="h-4 w-4" />}
              </button>
              <button className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors">
                <Settings className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="relative bg-gray-900 h-64">
          {isCameraActive ? (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full"
                style={{ display: 'none' }}
              />
              
              {/* Camera overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent">
                <div className="absolute bottom-4 left-4 flex items-center space-x-2">
                  <div className="w-2 h-2 bg-heart-500 rounded-full animate-pulse"></div>
                  <span className="text-white text-sm">Recording</span>
                </div>
                
                <div className="absolute top-4 right-4">
                  <div className="bg-black/50 rounded-lg px-2 py-1">
                    <span className="text-white text-xs">720p</span>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-400">
                {cameraError ? (
                  <>
                    <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-heart-500" />
                    <p className="text-sm">{cameraError}</p>
                    <button 
                      onClick={startCamera}
                      className="mt-2 px-3 py-1 bg-medical-600 text-white text-xs rounded hover:bg-medical-700 transition-colors"
                    >
                      Retry
                    </button>
                  </>
                ) : (
                  <>
                    <Camera className="h-8 w-8 mx-auto mb-2" />
                    <p className="text-sm">Camera not started</p>
                    <p className="text-xs">Click the camera button to begin</p>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Computer Vision Analysis */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Analysis</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-medical-50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-2">
              <Eye className="h-4 w-4 text-medical-600" />
              <span className="text-sm font-medium text-medical-900">Facial Analysis</span>
            </div>
            <div className="text-sm text-gray-600">
              <div>Pain Level: <span className="font-medium">Low (2/10)</span></div>
              <div>Emotion: <span className="font-medium">Neutral</span></div>
              <div>Alertness: <span className="font-medium">Alert</span></div>
            </div>
          </div>

          <div className="bg-success-50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="h-4 w-4 text-success-600" />
              <span className="text-sm font-medium text-success-900">Gait Analysis</span>
            </div>
            <div className="text-sm text-gray-600">
              <div>Stability: <span className="font-medium">Good</span></div>
              <div>Fall Risk: <span className="font-medium">Low</span></div>
              <div>Posture: <span className="font-medium">Normal</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSensorsTab = () => (
    <div className="space-y-4">
      {/* Sensor Status Overview */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Sensor Status</h3>
          <button 
            onClick={() => fetchSensorStatus(sessionInfo?.session_id || '')}
            className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-3">
          {sensorStatuses.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                <WifiOff className="h-8 w-8 text-gray-400" />
              </div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">No Sensors Connected</h4>
              <p className="text-gray-500 mb-4 max-w-sm mx-auto">
                Physical sensors are not currently connected to the system. 
                The system is running in demo mode with simulated data.
              </p>
              <button
                onClick={() => fetchSensorStatus(sessionInfo?.session_id || '')}
                className="px-4 py-2 bg-medical-blue text-white rounded-lg hover:bg-medical-blue-dark transition-colors"
              >
                Retry Connection
              </button>
            </div>
          ) : (
            sensorStatuses.map((sensor) => (
              <div key={sensor.sensor_id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getSensorIcon(sensor.type)}
                  <div>
                    <div className="font-medium text-gray-900">{sensor.name}</div>
                    <div className="text-sm text-gray-500">
                      {sensor.type}
                      {sensor.error_message && (
                        <div className="text-red-600 text-xs mt-1">
                          Error: {sensor.error_message}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  {sensor.battery_level !== undefined && (
                    <div className="flex items-center space-x-1 text-xs text-gray-600">
                      <Battery className="h-3 w-3" />
                      <span>{sensor.battery_level}%</span>
                    </div>
                  )}
                  
                  {sensor.signal_strength !== undefined && (
                    <div className="flex items-center space-x-1 text-xs text-gray-600">
                      <Signal className="h-3 w-3" />
                      <span>{sensor.signal_strength}%</span>
                    </div>
                  )}
                  
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getSensorStatusColor(sensor.status)}`}>
                    {sensor.status}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Real-time Readings */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Live Readings</h3>
        
        <div className="space-y-3">
          {sensorReadings.length === 0 ? (
            <div className="text-center text-gray-500 py-4">
              <Activity className="h-6 w-6 mx-auto mb-2" />
              <p className="text-sm">No sensor data available</p>
            </div>
          ) : (
            sensorReadings.slice(-5).map((reading, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getSensorIcon(reading.sensor_type)}
                  <div>
                    <div className="font-medium text-gray-900 capitalize">
                      {reading.sensor_type.replace('_', ' ')}
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatDistanceToNow(new Date(reading.timestamp), { addSuffix: true })}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="font-bold text-gray-900">
                    {typeof reading.value === 'number' 
                      ? reading.value.toFixed(1) 
                      : typeof reading.value === 'string' 
                        ? reading.value 
                        : String(reading.value)
                    }
                    {reading.unit && ` ${reading.unit}`}
                  </div>
                  <div className={`text-xs font-medium ${
                    reading.quality === 'excellent' ? 'text-success-600' :
                    reading.quality === 'good' ? 'text-blue-600' :
                    reading.quality === 'fair' ? 'text-warning-600' : 'text-heart-600'
                  }`}>
                    {reading.quality} quality
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );

  const renderTrendsTab = () => (
    <div className="space-y-4">
      {/* Time Range Selector */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Health Trends</h3>
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>

        {/* Health Trend Cards */}
        <div className="space-y-4">
          {['heart_rate', 'temperature', 'blood_pressure'].map((metric) => {
            const trend = healthTrends.find(t => t.metric === metric);
            const mockData = getMockChartData(metric, selectedTimeRange);
            
            return (
              <div key={metric} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {getSensorIcon(metric)}
                    <span className="font-medium text-gray-900 capitalize">
                      {metric.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(trend?.trend || 'stable')}
                    <span className={`text-sm font-medium ${
                      trend?.trend === 'improving' ? 'text-success-600' :
                      trend?.trend === 'declining' ? 'text-heart-600' : 'text-gray-600'
                    }`}>
                      {trend?.trend || 'stable'}
                    </span>
                  </div>
                </div>

                {/* Simple Chart Representation */}
                <div className="h-20 bg-gray-50 rounded relative overflow-hidden">
                  <div className="absolute inset-0 flex items-end justify-between px-2 pb-2">
                    {mockData.slice(-20).map((point, index) => (
                      <div
                        key={index}
                        className="bg-medical-600 rounded-t"
                        style={{
                          height: `${Math.max(10, (point.value / (metric === 'heart_rate' ? 100 : 40)) * 60)}%`,
                          width: '3px'
                        }}
                      />
                    ))}
                  </div>
                  
                  <div className="absolute top-2 left-2 text-xs text-gray-500">
                    {mockData[mockData.length - 1]?.value.toFixed(1)}
                    {metric === 'heart_rate' ? ' BPM' : metric === 'temperature' ? '°C' : ' mmHg'}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`bg-gray-50 flex flex-col ${className}`}>
      {/* Panel Header */}
      <div className="p-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900">Sensor Monitoring</h2>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 bg-white">
        <nav className="flex space-x-0">
          {[
            { key: 'camera', label: 'Camera', icon: Camera },
            { key: 'sensors', label: 'Sensors', icon: Activity },
            { key: 'trends', label: 'Trends', icon: TrendingUp }
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key as any)}
              className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 border-b-2 font-medium text-sm transition-colors ${
                activeTab === key
                  ? 'border-medical-600 text-medical-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'camera' && renderCameraTab()}
        {activeTab === 'sensors' && renderSensorsTab()}
        {activeTab === 'trends' && renderTrendsTab()}
      </div>
    </div>
  );
} 