'use client';

import { useEffect } from 'react';
import { DashboardHeader } from '@/components/DashboardHeader';
import { SessionPanel } from '@/components/panels/SessionPanel';
import { ChatPanel } from '@/components/panels/ChatPanel';
import { SensorPanel } from '@/components/panels/SensorPanel';
import { useSessionStore } from '@/stores/sessionStore';
import { useOphiraWebSocket } from '@/hooks/useOphiraWebSocket';

export default function DashboardPage() {
  const { sessionInfo, isAuthenticated, isLoading, isInitializing, initializeSession } = useSessionStore();
  
  // Initialize WebSocket connection with session data
  const {
    isConnected,
    connectionStatus,
    lastMessage,
    error: wsError,
    sendMessage,
    reconnect
  } = useOphiraWebSocket({
    enabled: !!sessionInfo?.session_id, // Re-enable WebSocket when session exists
    sessionId: sessionInfo?.session_id,
    autoReconnect: false, // Keep auto-reconnect disabled for stability
    maxReconnectAttempts: 1, // Limit to just 1 attempt
    reconnectInterval: 10000 // 10 second intervals if any reconnection happens
  });

  useEffect(() => {
    // Initialize session on component mount
    if (!isAuthenticated && !isLoading && !isInitializing) {
      initializeSession();
    }
  }, [isAuthenticated, isLoading, isInitializing]);

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-medical-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Initializing Ophira AI</h2>
          <p className="text-gray-600">Setting up your medical monitoring session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <DashboardHeader 
        sessionInfo={sessionInfo}
        connectionStatus={connectionStatus}
        isConnected={isConnected}
        onReconnect={reconnect}
      />

      {/* Main Dashboard */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Session Management */}
        <SessionPanel className="w-1/3 min-w-[320px] max-w-[400px]" />
        
        {/* Middle Panel: Chat Interface */}
        <ChatPanel className="flex-1 min-w-[400px]" />
        
        {/* Right Panel: Sensor Data */}
        <SensorPanel className="w-1/3 min-w-[320px] max-w-[400px]" />
      </div>
    </div>
  );
} 