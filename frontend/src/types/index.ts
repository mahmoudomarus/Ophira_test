import { ReactNode } from 'react';

// Session Management Types
export interface SessionInfo {
  session_id: string;
  user_id: string;
  session_type: string;
  status: string;
  created_at: string;
  expires_at: string;
  warnings: SessionWarning[];
}

export interface SessionWarning {
  type: string;
  message: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

// Medical Data Types
export interface VitalSigns {
  heart_rate?: number;
  blood_pressure?: {
    systolic: number;
    diastolic: number;
  };
  temperature?: number;
  respiratory_rate?: number;
  oxygen_saturation?: number;
  timestamp: string;
}

export interface SensorReading {
  sensor_type: 'heart_rate' | 'distance' | 'camera' | 'microphone';
  value: number | string | object;
  unit?: string;
  timestamp: string;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
  confidence?: number;
}

export interface MedicalAlert {
  id: string;
  type: 'warning' | 'critical' | 'emergency';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  vital_sign?: keyof VitalSigns;
  current_value?: number;
  normal_range?: [number, number];
  recommendations?: string[];
}

// Chat & Communication Types
export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'ai' | 'system';
  timestamp: string;
  session_id: string;
  message_type?: 'text' | 'voice' | 'image' | 'analysis_result';
  metadata?: {
    agent_id?: string;
    analysis_type?: string;
    confidence?: number;
    processing_time?: number;
  };
}

export interface VoiceTranscription {
  text: string;
  confidence: number;
  language: string;
  timestamp: string;
}

// Agent Types
export interface AgentStatus {
  agent_id: string;
  name: string;
  type: 'primary' | 'medical' | 'sensor' | 'supervisory';
  status: 'online' | 'offline' | 'busy' | 'error';
  last_activity: string;
  capabilities: string[];
  current_task?: string;
  performance_metrics?: {
    response_time: number;
    accuracy: number;
    uptime: number;
  };
}

// Medical Analysis Types
export interface HealthAnalysis {
  analysis_id: string;
  user_id: string;
  timestamp: string;
  analysis_type: string[];
  results: {
    overall_score: number;
    risk_level: 'low' | 'moderate' | 'high' | 'critical';
    findings: AnalysisFinding[];
    recommendations: string[];
    confidence: number;
  };
  vital_signs_analysis?: VitalSignsAnalysis;
  predictive_insights?: PredictiveInsight[];
}

export interface AnalysisFinding {
  category: string;
  description: string;
  severity: 'normal' | 'attention' | 'concern' | 'urgent';
  confidence: number;
  supporting_data?: any;
}

export interface VitalSignsAnalysis {
  heart_rate_analysis?: {
    current: number;
    trend: 'stable' | 'increasing' | 'decreasing';
    variability: number;
    status: 'normal' | 'bradycardia' | 'tachycardia';
  };
  blood_pressure_analysis?: {
    systolic: number;
    diastolic: number;
    classification: string;
    risk_factors: string[];
  };
  respiratory_analysis?: {
    rate: number;
    pattern: string;
    efficiency: number;
  };
}

export interface PredictiveInsight {
  prediction_type: string;
  timeframe: string;
  probability: number;
  description: string;
  risk_factors: string[];
  preventive_measures: string[];
}

// Sensor Status Types
export interface SensorStatus {
  sensor_id: string;
  name: string;
  type: string;
  status: 'connected' | 'disconnected' | 'error' | 'calibrating';
  last_reading?: string;
  battery_level?: number;
  signal_strength?: number;
  calibration_status?: 'calibrated' | 'needs_calibration' | 'calibrating';
  error_message?: string;
}

// Dashboard Types
export interface DashboardData {
  session_info: SessionInfo;
  vital_signs: VitalSigns;
  sensor_readings: SensorReading[];
  recent_alerts: MedicalAlert[];
  agent_statuses: AgentStatus[];
  health_trends: HealthTrend[];
}

export interface HealthTrend {
  metric: keyof VitalSigns;
  data_points: Array<{
    timestamp: string;
    value: number;
  }>;
  trend: 'improving' | 'stable' | 'declining';
  period: '1h' | '6h' | '24h' | '7d' | '30d';
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
  };
}

// UI Component Types
export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: 'vital_signs' | 'alert' | 'agent_status' | 'chat_message' | 'sensor_update' | 'ping' | 'pong';
  data: any;
  timestamp: string;
  session_id: string;
}

// Voice Interface Types
export interface VoiceCommand {
  command: string;
  confidence: number;
  intent: string;
  entities: Record<string, any>;
}

export interface SpeechSynthesisOptions {
  text: string;
  voice?: string;
  rate?: number;
  pitch?: number;
  volume?: number;
}

// Emergency Types
export interface EmergencyProtocol {
  trigger: string;
  severity: 'high' | 'critical';
  actions: EmergencyAction[];
  contacts: EmergencyContact[];
}

export interface EmergencyAction {
  action: string;
  description: string;
  priority: number;
  automated: boolean;
}

export interface EmergencyContact {
  name: string;
  role: string;
  phone: string;
  email?: string;
  notification_method: 'sms' | 'call' | 'email' | 'app';
}

// Chart and Visualization Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
  color?: string;
}

export interface ChartConfig {
  type: 'line' | 'bar' | 'area' | 'scatter';
  title: string;
  xAxis: string;
  yAxis: string;
  data: ChartDataPoint[];
  options?: Record<string, any>;
} 