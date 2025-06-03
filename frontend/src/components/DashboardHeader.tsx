'use client';

import { useState } from 'react';
import { 
  User, 
  Activity, 
  Wifi, 
  WifiOff, 
  AlertTriangle, 
  Shield,
  Settings,
  LogOut,
  RefreshCw
} from 'lucide-react';
import { SessionInfo } from '@/types';
import { useSessionStore } from '@/stores/sessionStore';
import { formatDistanceToNow } from 'date-fns';

interface DashboardHeaderProps {
  sessionInfo: SessionInfo | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';
  isConnected: boolean;
}

export function DashboardHeader({ 
  sessionInfo, 
  connectionStatus, 
  isConnected 
}: DashboardHeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const { logout, refreshSession } = useSessionStore();

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-success-600" />;
      case 'connecting':
      case 'reconnecting':
        return <RefreshCw className="h-4 w-4 text-warning-600 animate-spin" />;
      case 'disconnected':
      case 'error':
        return <WifiOff className="h-4 w-4 text-heart-600" />;
      default:
        return <WifiOff className="h-4 w-4 text-gray-400" />;
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'reconnecting':
        return 'Reconnecting...';
      case 'disconnected':
        return 'Disconnected';
      case 'error':
        return 'Connection Error';
      default:
        return 'Unknown';
    }
  };

  const getSessionTime = () => {
    if (!sessionInfo?.created_at) return '';
    try {
      return formatDistanceToNow(new Date(sessionInfo.created_at), { addSuffix: true });
    } catch {
      return '';
    }
  };

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left: Logo and Title */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-medical rounded-xl flex items-center justify-center">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Ophira AI</h1>
              <p className="text-sm text-gray-500">Medical Monitoring System</p>
            </div>
          </div>
        </div>

        {/* Center: Connection Status */}
        <div className="flex items-center space-x-6">
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            {getConnectionIcon()}
            <span className={`text-sm font-medium ${
              isConnected ? 'text-success-600' : 'text-heart-600'
            }`}>
              {getConnectionText()}
            </span>
          </div>

          {/* Warnings */}
          {sessionInfo?.warnings && sessionInfo.warnings.length > 0 && (
            <div className="flex items-center space-x-2 text-warning-600">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm font-medium">
                {sessionInfo.warnings.length} Warning{sessionInfo.warnings.length !== 1 ? 's' : ''}
              </span>
            </div>
          )}

          {/* System Status */}
          <div className="flex items-center space-x-2 text-success-600">
            <Shield className="h-4 w-4" />
            <span className="text-sm font-medium">System Active</span>
          </div>
        </div>

        {/* Right: User Menu */}
        <div className="relative">
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="w-8 h-8 bg-medical-100 rounded-full flex items-center justify-center">
              <User className="h-4 w-4 text-medical-600" />
            </div>
            <div className="text-left">
              <div className="text-sm font-medium text-gray-900">
                {sessionInfo?.user_id || 'Demo User'}
              </div>
              <div className="text-xs text-gray-500">
                {getSessionTime()}
              </div>
            </div>
          </button>

          {/* User Menu Dropdown */}
          {showUserMenu && (
            <div className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
              <div className="px-4 py-3 border-b border-gray-100">
                <div className="text-sm font-medium text-gray-900">Session Details</div>
                <div className="text-xs text-gray-500 mt-1">
                  ID: {sessionInfo?.session_id?.slice(0, 8)}...
                </div>
                <div className="text-xs text-gray-500">
                  Status: {sessionInfo?.status}
                </div>
              </div>
              
              <button
                onClick={() => {
                  refreshSession();
                  setShowUserMenu(false);
                }}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center space-x-2"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Refresh Session</span>
              </button>
              
              <button
                onClick={() => setShowUserMenu(false)}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center space-x-2"
              >
                <Settings className="h-4 w-4" />
                <span>Settings</span>
              </button>
              
              <div className="border-t border-gray-100 my-1"></div>
              
              <button
                onClick={() => {
                  logout();
                  setShowUserMenu(false);
                }}
                className="w-full text-left px-4 py-2 text-sm text-heart-600 hover:bg-heart-50 flex items-center space-x-2"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
} 