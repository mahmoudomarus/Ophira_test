import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketEvent } from '@/types';
import { useMedicalStore } from '@/stores/medicalStore';
import { createWebSocketConnection } from '@/lib/api';
import toast from 'react-hot-toast';

interface UseOphiraWebSocketOptions {
  enabled?: boolean;
  sessionId?: string;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface WebSocketState {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';
  lastMessage: WebSocketEvent | null;
  error: string | null;
}

export function useOphiraWebSocket({
  enabled = true,
  sessionId,
  autoReconnect = true,
  reconnectInterval = 5000,
  maxReconnectAttempts = 3
}: UseOphiraWebSocketOptions = {}) {
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    connectionStatus: 'disconnected',
    lastMessage: null,
    error: null
  });

  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isManuallyClosedRef = useRef(false);
  const isConnectingRef = useRef(false);

  // Medical store actions
  const {
    updateVitalSigns,
    addSensorReading,
    updateSensorStatus,
    addAlert,
    addHealthAnalysis
  } = useMedicalStore();

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data: WebSocketEvent = JSON.parse(event.data);
      
      setState(prev => ({
        ...prev,
        lastMessage: data
      }));

      // Route messages to appropriate store actions
      switch (data.type) {
        case 'ping':
          // Respond to server ping with pong
          if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
            websocketRef.current.send(JSON.stringify({
              type: 'pong',
              timestamp: new Date().toISOString()
            }));
          }
          break;
          
        case 'vital_signs':
          if (data.data) {
            updateVitalSigns(data.data);
          }
          break;
          
        case 'sensor_update':
          if (data.data) {
            if (data.data.reading) {
              addSensorReading(data.data.reading);
            }
            if (data.data.status) {
              updateSensorStatus(data.data.status);
            }
          }
          break;
          
        case 'alert':
          if (data.data) {
            addAlert(data.data);
            
            // Show toast notification for alerts
            const alertType = data.data.type || 'info';
            const message = data.data.message || 'New medical alert';
            
            if (alertType === 'critical' || alertType === 'emergency') {
              toast.error(message, {
                duration: 10000,
                icon: '🚨'
              });
            } else if (alertType === 'warning') {
              toast.error(message, {
                duration: 6000,
                icon: '⚠️'
              });
            } else {
              toast(message, {
                duration: 4000,
                icon: 'ℹ️'
              });
            }
          }
          break;
          
        case 'chat_message':
          // Handle chat messages (could be routed to a chat store)
          console.log('Chat message received:', data.data);
          break;
          
        case 'agent_status':
          // Handle agent status updates
          console.log('Agent status update:', data.data);
          break;
          
        default:
          console.log('Unknown WebSocket message type:', data.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [updateVitalSigns, addSensorReading, updateSensorStatus, addAlert, addHealthAnalysis]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled || !sessionId || isConnectingRef.current) return;

    // Close existing connection
    if (websocketRef.current) {
      websocketRef.current.close();
    }

    isConnectingRef.current = true;
    setState(prev => ({
      ...prev,
      connectionStatus: 'connecting',
      error: null
    }));

    try {
      const ws = createWebSocketConnection(sessionId);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        isConnectingRef.current = false;
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionStatus: 'connected',
          error: null
        }));
        
        // Reset reconnection attempts on successful connection
        reconnectAttemptsRef.current = 0;
        
        // Only show success toast if we were previously disconnected
        if (reconnectAttemptsRef.current > 0) {
          toast.success('Reconnected to Ophira AI', {
            duration: 2000,
            icon: '🔗'
          });
        }
      };
      
      ws.onmessage = handleMessage;
      
      ws.onerror = (event: Event) => {
        console.error('WebSocket error:', event);
        isConnectingRef.current = false;
        
        // Only update state if we're not already in an error state to prevent loops
        setState(prev => {
          if (prev.connectionStatus !== 'error') {
            return {
              ...prev,
              error: 'WebSocket connection error',
              connectionStatus: 'error'
            };
          }
          return prev;
        });
      };
      
      ws.onclose = (event: CloseEvent) => {
        console.log('WebSocket closed:', event.code, event.reason);
        isConnectingRef.current = false;
        
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionStatus: 'disconnected'
        }));

        // Only attempt reconnection if conditions are met
        if (!isManuallyClosedRef.current && 
            autoReconnect && 
            enabled && 
            sessionId &&
            reconnectAttemptsRef.current < maxReconnectAttempts &&
            event.code !== 1000) { // Don't reconnect on normal closure
          
          setState(prev => ({
            ...prev,
            connectionStatus: 'reconnecting'
          }));
          
          // Clear any existing timeout
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current += 1;
            console.log(`Attempting to reconnect... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
            connect();
          }, reconnectInterval * Math.pow(2, reconnectAttemptsRef.current)); // Exponential backoff
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.error('Max reconnection attempts reached');
          setState(prev => ({
            ...prev,
            error: 'Failed to reconnect after maximum attempts',
            connectionStatus: 'error'
          }));
        }
      };
      
      websocketRef.current = ws;
      isManuallyClosedRef.current = false;
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      isConnectingRef.current = false;
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to connect',
        connectionStatus: 'error'
      }));
    }
  }, [enabled, sessionId, handleMessage, autoReconnect, maxReconnectAttempts, reconnectInterval]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    isManuallyClosedRef.current = true;
    isConnectingRef.current = false;
    
    // Clear reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Close WebSocket connection
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    
    setState(prev => ({
      ...prev,
      isConnected: false,
      connectionStatus: 'disconnected'
    }));
  }, []);

  // Send message through WebSocket
  const sendMessage = useCallback((message: any) => {
    if (websocketRef.current && state.isConnected) {
      try {
        websocketRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    }
    return false;
  }, [state.isConnected]);

  // Effect to handle connection lifecycle
  useEffect(() => {
    if (enabled && sessionId) {
      connect();
    } else {
      disconnect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [enabled, sessionId, connect, disconnect]);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    reconnect: () => {
      reconnectAttemptsRef.current = 0;
      connect();
    }
  };
} 