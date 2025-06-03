import { useState, useRef, useCallback, useEffect } from 'react';
import { useWebSocket, WebSocketMessage } from './useWebSocket';

// Type declarations for Speech API
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  onstart: ((event: Event) => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: ((event: Event) => void) | null;
}

export interface VoiceAgentOptions {
  sessionId: string;
  onTranscript?: (transcript: string, isFinal: boolean) => void;
  onResponse?: (response: string) => void;
  onError?: (error: string) => void;
  onStatusChange?: (status: VoiceStatus) => void;
  language?: string;
  continuous?: boolean;
}

export type VoiceStatus = 'idle' | 'listening' | 'processing' | 'speaking' | 'error';

export interface VoiceAgentReturn {
  status: VoiceStatus;
  isListening: boolean;
  isSpeaking: boolean;
  transcript: string;
  error: string | null;
  startListening: () => void;
  stopListening: () => void;
  speak: (text: string) => void;
  stopSpeaking: () => void;
  sendVoiceMessage: (message: string) => void;
  connected: boolean;
}

export const useVoiceAgent = (options: VoiceAgentOptions): VoiceAgentReturn => {
  const {
    sessionId,
    onTranscript,
    onResponse,
    onError,
    onStatusChange,
    language = 'en-US',
    continuous = true,
  } = options;

  const [status, setStatus] = useState<VoiceStatus>('idle');
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthesisRef = useRef<SpeechSynthesisUtterance | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket connection for real-time communication
  const { connected, sendMessage } = useWebSocket({
    sessionId,
    onMessage: handleWebSocketMessage,
    onConnect: () => {
      console.log('🎤 Voice agent connected to WebSocket');
    },
    onDisconnect: () => {
      console.log('🎤 Voice agent disconnected from WebSocket');
    },
    onError: (error) => {
      console.error('🎤 Voice agent WebSocket error:', error);
      setError('WebSocket connection failed');
      updateStatus('error');
    },
  });

  function handleWebSocketMessage(message: WebSocketMessage) {
    console.log('🎤 Voice agent received message:', message);
    
    switch (message.type) {
      case 'ai_response':
        if (message.message) {
          onResponse?.(message.message);
          speak(message.message);
        }
        break;
      case 'processing':
        updateStatus('processing');
        break;
      case 'emergency_activated':
        if (message.message) {
          speak(`Emergency mode activated. ${message.message}`);
        }
        break;
      case 'error':
        setError(message.message || 'Unknown error');
        updateStatus('error');
        break;
    }
  }

  const updateStatus = useCallback((newStatus: VoiceStatus) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);

  // Initialize Speech Recognition
  useEffect(() => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const SpeechRecognitionConstructor = window.webkitSpeechRecognition || window.SpeechRecognition;
      const recognition = new SpeechRecognitionConstructor() as SpeechRecognition;
      
      recognition.continuous = continuous;
      recognition.interimResults = true;
      recognition.lang = language;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        console.log('🎤 Speech recognition started');
        setIsListening(true);
        updateStatus('listening');
        setError(null);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) {
            finalTranscript += result[0].transcript;
          } else {
            interimTranscript += result[0].transcript;
          }
        }

        const currentTranscript = finalTranscript || interimTranscript;
        setTranscript(currentTranscript);
        onTranscript?.(currentTranscript, !!finalTranscript);

        if (finalTranscript) {
          console.log('🎤 Final transcript:', finalTranscript);
          sendVoiceMessage(finalTranscript);
          
          // Clear transcript after sending
          setTimeout(() => setTranscript(''), 1000);
        }
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('🎤 Speech recognition error:', event.error);
        setError(`Speech recognition error: ${event.error}`);
        setIsListening(false);
        updateStatus('error');
        onError?.(`Speech recognition error: ${event.error}`);
      };

      recognition.onend = () => {
        console.log('🎤 Speech recognition ended');
        setIsListening(false);
        if (status === 'listening') {
          updateStatus('idle');
        }
      };

      recognitionRef.current = recognition;
    } else {
      setError('Speech recognition not supported in this browser');
      updateStatus('error');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [language, continuous, onTranscript, onError, status, updateStatus]);

  const startListening = useCallback(() => {
    if (!recognitionRef.current) {
      setError('Speech recognition not available');
      return;
    }

    if (isListening) {
      console.log('🎤 Already listening');
      return;
    }

    try {
      recognitionRef.current.start();
    } catch (error) {
      console.error('🎤 Failed to start speech recognition:', error);
      setError('Failed to start speech recognition');
      updateStatus('error');
    }
  }, [isListening, updateStatus]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  }, [isListening]);

  const speak = useCallback((text: string) => {
    if (!text) return;

    // Stop any current speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 0.8;

    utterance.onstart = () => {
      console.log('🔊 Speech synthesis started');
      setIsSpeaking(true);
      updateStatus('speaking');
    };

    utterance.onend = () => {
      console.log('🔊 Speech synthesis ended');
      setIsSpeaking(false);
      updateStatus('idle');
    };

    utterance.onerror = (event) => {
      console.error('🔊 Speech synthesis error:', event.error);
      setIsSpeaking(false);
      setError(`Speech synthesis error: ${event.error}`);
      updateStatus('error');
    };

    synthesisRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  }, [updateStatus]);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    if (status === 'speaking') {
      updateStatus('idle');
    }
  }, [status, updateStatus]);

  const sendVoiceMessage = useCallback((message: string) => {
    if (!connected) {
      setError('Not connected to voice service');
      return;
    }

    console.log('🎤 Sending voice message:', message);
    updateStatus('processing');
    
    sendMessage({
      type: 'voice_message',
      message: message,
      timestamp: new Date().toISOString(),
      session_id: sessionId,
    });
  }, [connected, sendMessage, sessionId, updateStatus]);

  // Auto-stop listening after period of silence
  useEffect(() => {
    if (isListening && transcript) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      timeoutRef.current = setTimeout(() => {
        if (isListening && transcript) {
          console.log('🎤 Auto-stopping due to silence');
          stopListening();
        }
      }, 3000); // Stop after 3 seconds of silence
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [isListening, transcript, stopListening]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      window.speechSynthesis.cancel();
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return {
    status,
    isListening,
    isSpeaking,
    transcript,
    error,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    sendVoiceMessage,
    connected,
  };
}; 