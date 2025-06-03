import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  message?: string;
  data?: any;
  timestamp?: string;
  emergency?: boolean;
  session_info?: any;
  warnings?: any[];
}

export interface UseWebSocketOptions {
  sessionId: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface UseWebSocketReturn {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  sendMessage: (message: any) => void;
  disconnect: () => void;
  reconnect: () => void;
  lastMessage: WebSocketMessage | null;
}

export const useWebSocket = (options: UseWebSocketOptions): UseWebSocketReturn => {
  const {
    sessionId,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 5,
  } = options;

  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const mountedRef = useRef(true);

  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.NEXT_PUBLIC_WS_BASE_URL?.replace(/^https?:\/\//, '') || 'localhost:8001';
    return `${protocol}//${host}/ws/${sessionId}`;
  }, [sessionId]);

  const connect = useCallback(() => {
    if (!sessionId || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    try {
      setConnecting(true);
      setError(null);

      const wsUrl = getWebSocketUrl();
      console.log('🔗 Connecting to WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = (event) => {
        console.log('✅ WebSocket connected');
        setConnected(true);
        setConnecting(false);
        setError(null);
        reconnectAttemptsRef.current = 0;
        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', message);
          
          setLastMessage(message);
          onMessage?.(message);
        } catch (err) {
          console.error('❌ Failed to parse WebSocket message:', err);
          setError('Failed to parse message from server');
        }
      };

      ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        setConnected(false);
        setConnecting(false);
        wsRef.current = null;
        onDisconnect?.();

        // Auto-reconnect logic
        if (autoReconnect && mountedRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          console.log(`🔄 Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, reconnectInterval);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError('Max reconnection attempts reached');
        }
      };

      ws.onerror = (event) => {
        console.error('❌ WebSocket error:', event);
        setError('WebSocket connection error');
        setConnecting(false);
        onError?.(event);
      };

    } catch (err) {
      console.error('❌ Failed to create WebSocket connection:', err);
      setError('Failed to create WebSocket connection');
      setConnecting(false);
    }
  }, [
    sessionId,
    getWebSocketUrl,
    onConnect,
    onMessage,
    onDisconnect,
    onError,
    autoReconnect,
    maxReconnectAttempts,
    reconnectInterval,
  ]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        const messageString = JSON.stringify(message);
        console.log('📤 Sending WebSocket message:', message);
        wsRef.current.send(messageString);
      } catch (err) {
        console.error('❌ Failed to send WebSocket message:', err);
        setError('Failed to send message');
      }
    } else {
      console.warn('⚠️ WebSocket not connected, cannot send message');
      setError('WebSocket not connected');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      console.log('🔌 Manually disconnecting WebSocket');
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }

    setConnected(false);
    setConnecting(false);
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(connect, 1000);
  }, [disconnect, connect]);

  // Connect when sessionId changes or component mounts
  useEffect(() => {
    if (sessionId) {
      connect();
    }

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [sessionId, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    connected,
    connecting,
    error,
    sendMessage,
    disconnect,
    reconnect,
    lastMessage,
  };
}; 