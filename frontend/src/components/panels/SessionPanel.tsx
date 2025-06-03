'use client';

import { useState, useEffect } from 'react';
import { 
  User, 
  Calendar,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  Heart,
  Thermometer,
  Eye,
  Shield,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  Bell,
  BellOff
} from 'lucide-react';
import { SessionInfo, MedicalAlert, VitalSigns, SessionWarning } from '@/types';
import { useSessionStore } from '@/stores/sessionStore';
import { useMedicalStore } from '@/stores/medicalStore';
import { formatDistanceToNow, format } from 'date-fns';

interface SessionPanelProps {
  className?: string;
}

export function SessionPanel({ className = '' }: SessionPanelProps) {
  const [activeTab, setActiveTab] = useState<'session' | 'alerts' | 'history'>('session');
  const [expandedAlert, setExpandedAlert] = useState<string | null>(null);
  
  const { sessionInfo, warnings } = useSessionStore();
  const { 
    alerts, 
    vitalSigns, 
    acknowledgeAlert, 
    dismissAlert,
    fetchDashboardData 
  } = useMedicalStore();

  useEffect(() => {
    if (sessionInfo?.session_id) {
      fetchDashboardData(sessionInfo.session_id);
    }
  }, [sessionInfo?.session_id, fetchDashboardData]);

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical':
      case 'emergency':
        return <AlertTriangle className="h-4 w-4 text-heart-600" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-warning-600" />;
      default:
        return <CheckCircle className="h-4 w-4 text-success-600" />;
    }
  };

  const getVitalSignStatus = (value: number | undefined, ranges: { normal: [number, number], warning: [number, number] }) => {
    if (!value) return 'unknown';
    if (value >= ranges.normal[0] && value <= ranges.normal[1]) return 'normal';
    if (value >= ranges.warning[0] && value <= ranges.warning[1]) return 'warning';
    return 'critical';
  };

  const renderSessionTab = () => (
    <div className="space-y-6">
      {/* Patient Profile */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-12 h-12 bg-medical-100 rounded-full flex items-center justify-center">
            <User className="h-6 w-6 text-medical-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {sessionInfo?.user_id || 'Demo Patient'}
            </h3>
            <p className="text-sm text-gray-500">Patient ID: {sessionInfo?.session_id?.slice(0, 8)}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Session Type:</span>
            <div className="font-medium text-gray-900 capitalize">
              {sessionInfo?.session_type || 'Standard'}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Status:</span>
            <div className={`font-medium capitalize ${
              sessionInfo?.status === 'active' ? 'text-success-600' : 'text-gray-600'
            }`}>
              {sessionInfo?.status || 'Unknown'}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Started:</span>
            <div className="font-medium text-gray-900">
              {sessionInfo?.created_at ? format(new Date(sessionInfo.created_at), 'MMM d, HH:mm') : 'Unknown'}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Duration:</span>
            <div className="font-medium text-gray-900">
              {sessionInfo?.created_at ? formatDistanceToNow(new Date(sessionInfo.created_at)) : 'Unknown'}
            </div>
          </div>
        </div>
      </div>

      {/* Current Vital Signs */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center space-x-2 mb-4">
          <Activity className="h-5 w-5 text-medical-600" />
          <h3 className="text-lg font-semibold text-gray-900">Current Vitals</h3>
        </div>

        <div className="space-y-3">
          {/* Heart Rate */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <Heart className="h-5 w-5 text-heart-500" />
              <span className="font-medium text-gray-900">Heart Rate</span>
            </div>
            <div className="text-right">
              <div className={`text-lg font-bold ${
                vitalSigns?.heart_rate 
                  ? getVitalSignStatus(vitalSigns.heart_rate, { normal: [60, 100], warning: [50, 120] }) === 'normal' 
                    ? 'text-success-600' 
                    : 'text-heart-600'
                  : 'text-gray-400'
              }`}>
                {vitalSigns?.heart_rate || '--'} BPM
              </div>
            </div>
          </div>

          {/* Blood Pressure */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <Activity className="h-5 w-5 text-medical-600" />
              <span className="font-medium text-gray-900">Blood Pressure</span>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-gray-900">
                {vitalSigns?.blood_pressure 
                  ? `${vitalSigns.blood_pressure.systolic}/${vitalSigns.blood_pressure.diastolic}`
                  : '--/--'
                } mmHg
              </div>
            </div>
          </div>

          {/* Temperature */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <Thermometer className="h-5 w-5 text-blue-500" />
              <span className="font-medium text-gray-900">Temperature</span>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-gray-900">
                {vitalSigns?.temperature || '--'}°C
              </div>
            </div>
          </div>

          {/* Oxygen Saturation */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <Eye className="h-5 w-5 text-blue-600" />
              <span className="font-medium text-gray-900">SpO₂</span>
            </div>
            <div className="text-right">
              <div className={`text-lg font-bold ${
                vitalSigns?.oxygen_saturation && vitalSigns.oxygen_saturation >= 95 
                  ? 'text-success-600' 
                  : 'text-heart-600'
              }`}>
                {vitalSigns?.oxygen_saturation || '--'}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="space-y-2">
          <button className="w-full p-3 text-left bg-medical-50 hover:bg-medical-100 rounded-lg transition-colors">
            <div className="flex items-center space-x-3">
              <RefreshCw className="h-4 w-4 text-medical-600" />
              <span className="font-medium text-medical-900">Refresh Data</span>
            </div>
          </button>
          <button className="w-full p-3 text-left bg-success-50 hover:bg-success-100 rounded-lg transition-colors">
            <div className="flex items-center space-x-3">
              <Shield className="h-4 w-4 text-success-600" />
              <span className="font-medium text-success-900">Start Monitoring</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );

  const renderAlertsTab = () => (
    <div className="space-y-4">
      {/* Session Warnings */}
      {warnings.length > 0 && (
        <div className="bg-warning-50 border border-warning-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-warning-900 mb-3">Session Warnings</h3>
          <div className="space-y-2">
            {warnings.map((warning, index) => (
              <div key={index} className="flex items-start space-x-3 p-2 bg-white rounded border border-warning-100">
                <AlertTriangle className="h-4 w-4 text-warning-600 mt-0.5" />
                <div className="flex-1">
                  <div className="font-medium text-warning-900">{warning.type}</div>
                  <div className="text-sm text-warning-700">{warning.message}</div>
                  <div className="text-xs text-warning-600 mt-1">
                    {formatDistanceToNow(new Date(warning.timestamp), { addSuffix: true })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Medical Alerts */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Medical Alerts</h3>
        </div>
        
        <div className="max-h-96 overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="p-6 text-center text-gray-500">
              <CheckCircle className="h-8 w-8 mx-auto mb-2 text-success-600" />
              <p>No active alerts</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-100">
              {alerts.map((alert) => (
                <div key={alert.id} className="p-4">
                  <div className="flex items-start space-x-3">
                    {getAlertIcon(alert.type)}
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-gray-900">{alert.title}</h4>
                        <button
                          onClick={() => setExpandedAlert(
                            expandedAlert === alert.id ? null : alert.id
                          )}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          {expandedAlert === alert.id ? 
                            <ChevronDown className="h-4 w-4" /> : 
                            <ChevronRight className="h-4 w-4" />
                          }
                        </button>
                      </div>
                      
                      <p className="text-sm text-gray-600 mt-1">{alert.message}</p>
                      
                      <div className="text-xs text-gray-500 mt-2">
                        {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                      </div>

                      {expandedAlert === alert.id && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          {alert.recommendations && (
                            <div className="mb-3">
                              <h5 className="text-sm font-medium text-gray-900 mb-1">Recommendations:</h5>
                              <ul className="text-sm text-gray-600 space-y-1">
                                {alert.recommendations.map((rec, idx) => (
                                  <li key={idx} className="flex items-start space-x-1">
                                    <span className="text-gray-400">•</span>
                                    <span>{rec}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          <div className="flex space-x-2">
                            {!alert.acknowledged && (
                              <button
                                onClick={() => acknowledgeAlert(alert.id)}
                                className="px-3 py-1 text-xs bg-medical-600 text-white rounded hover:bg-medical-700 transition-colors"
                              >
                                Acknowledge
                              </button>
                            )}
                            <button
                              onClick={() => dismissAlert(alert.id)}
                              className="px-3 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                            >
                              Dismiss
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderHistoryTab = () => (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Session History</h3>
        <div className="text-center text-gray-500 py-8">
          <Clock className="h-8 w-8 mx-auto mb-2" />
          <p>Session history will appear here</p>
          <p className="text-sm">Previous sessions and their data</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className={`bg-gray-50 border-r border-gray-200 flex flex-col ${className}`}>
      {/* Panel Header */}
      <div className="p-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900">Patient Dashboard</h2>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 bg-white">
        <nav className="flex space-x-0">
          {[
            { key: 'session', label: 'Session', icon: User },
            { key: 'alerts', label: 'Alerts', icon: Bell, badge: alerts.filter(a => !a.acknowledged).length },
            { key: 'history', label: 'History', icon: Calendar }
          ].map(({ key, label, icon: Icon, badge }) => (
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
              {badge !== undefined && badge > 0 && (
                <span className="bg-heart-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {badge}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'session' && renderSessionTab()}
        {activeTab === 'alerts' && renderAlertsTab()}
        {activeTab === 'history' && renderHistoryTab()}
      </div>
    </div>
  );
} 