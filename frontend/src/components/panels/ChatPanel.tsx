'use client';

import React, { useState, useRef, useEffect } from 'react';
import { 
  Send,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  MessageSquare,
  Bot,
  User,
  Loader2,
  Phone,
  PhoneOff,
  Zap,
  Sparkles
} from 'lucide-react';
import { ChatMessage, VoiceTranscription } from '@/types';
import { useSessionStore } from '@/stores/sessionStore';
import { formatDistanceToNow } from 'date-fns';
import { api } from '@/lib/api';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useVoiceAgent } from '@/hooks/useVoiceAgent';

interface ChatPanelProps {
  className?: string;
}

export function ChatPanel({ className = '' }: ChatPanelProps) {
  const { sessionInfo } = useSessionStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [isVoiceConnected, setIsVoiceConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceTranscription, setVoiceTranscription] = useState<VoiceTranscription | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // WebSocket connection for real-time chat (temporarily disabled for stability)
  const { connected: wsConnected, sendMessage: sendWSMessage } = useWebSocket({
    sessionId: '', // Temporarily disable by passing empty sessionId
    autoReconnect: false, // Disable auto-reconnect
    maxReconnectAttempts: 0, // No reconnect attempts
    onMessage: (message) => {
      console.log('💬 Chat received WebSocket message:', message);
      
      switch (message.type) {
        case 'welcome':
          addSystemMessage("Welcome to Ophira AI Medical Monitoring. I'm here to assist with your health monitoring and answer any medical questions you may have.");
          break;
        case 'ai_response':
          setIsLoading(false);
          addMessage(message.message || '', 'ai', message.data?.confidence);
          break;
        case 'processing':
          setIsLoading(true);
          break;
        case 'emergency_activated':
          addSystemMessage(`🚨 ${message.message || 'Emergency mode activated'}`);
          break;
        case 'error':
          setIsLoading(false);
          addSystemMessage(`Error: ${message.message || 'Unknown error'}`);
          break;
      }
    },
    onConnect: () => {
      console.log('💬 Chat WebSocket connected');
      // Removed system message to prevent spam
    },
    onDisconnect: () => {
      console.log('💬 Chat WebSocket disconnected');
      // Removed system message to prevent spam
    },
  });

  // Voice Agent integration
  const {
    status: voiceStatus,
    isListening: voiceListening,
    isSpeaking: voiceSpeaking,
    transcript: voiceTranscript,
    error: voiceError,
    startListening: startVoiceListening,
    stopListening: stopVoiceListening,
    connected: voiceConnected,
  } = useVoiceAgent({
    sessionId: sessionInfo?.session_id || '',
    onTranscript: (transcript, isFinal) => {
      if (isFinal) {
        setInputText(transcript);
      }
    },
    onResponse: (response) => {
      console.log('🎤 Voice response received:', response);
    },
    onError: (error) => {
      console.error('🎤 Voice error:', error);
      addSystemMessage(`Voice error: ${error}`);
    },
    onStatusChange: (status) => {
      console.log('🎤 Voice status changed:', status);
    },
  });

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Add welcome message on component mount
    if (messages.length === 0) {
      addSystemMessage("Welcome to Ophira AI Medical Monitoring. I'm here to assist with your health monitoring and answer any medical questions you may have.");
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const addMessage = (content: string, sender: 'user' | 'ai' | 'system', metadata?: any) => {
    const newMessage: ChatMessage = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      content,
      sender,
      timestamp: new Date().toISOString(),
      session_id: sessionInfo?.session_id || 'demo',
      message_type: 'text',
      metadata
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const addSystemMessage = (content: string) => {
    addMessage(content, 'system');
  };

  const sendTextMessage = async () => {
    if (!inputText.trim() || !sessionInfo?.session_id) return;

    const userMessage = inputText.trim();
    setInputText('');
    addMessage(userMessage, 'user');
    setIsLoading(true);

    try {
      // Send via WebSocket for real-time response
      if (wsConnected) {
        sendWSMessage({
          type: 'chat_message',
          message: userMessage,
          session_id: sessionInfo.session_id,
          timestamp: new Date().toISOString(),
        });
      } else {
        // Fallback to REST API
        const response = await api.sendChatMessage(userMessage, sessionInfo.session_id);
        setIsLoading(false);
        
        if (response.success && response.data) {
          addMessage(response.data.content || response.data.message, 'ai', {
            agent_id: response.data.agent_id,
            confidence: response.data.confidence,
            processing_time: response.data.processing_time
          });
        } else {
          addMessage("I apologize, but I'm having trouble processing your request right now. Please try again.", 'ai');
        }
      }
    } catch (error) {
      setIsLoading(false);
      console.error('Failed to send message:', error);
      addSystemMessage("Failed to send message. Please try again.");
    }
  };

  const startVoiceSession = async () => {
    if (!sessionInfo?.session_id) return;

    try {
      setIsVoiceActive(true);
      const response = await api.startVoiceSession(sessionInfo.session_id);
      
      if (response.success) {
        setIsVoiceConnected(true);
        addSystemMessage("Voice session started. You can now speak to interact with Ophira AI.");
        
        // Initialize Web Speech API for voice recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          startVoiceListening();
        }
      }
    } catch (error) {
      console.error('Error starting voice session:', error);
      addSystemMessage("Failed to start voice session. Please try again.");
      setIsVoiceActive(false);
    }
  };

  const stopVoiceSession = async () => {
    if (!sessionInfo?.session_id) return;

    try {
      await api.stopVoiceSession(sessionInfo.session_id);
      setIsVoiceConnected(false);
      setIsVoiceActive(false);
      setIsListening(false);
      addSystemMessage("Voice session ended.");
    } catch (error) {
      console.error('Error stopping voice session:', error);
    }
  };

  const startSpeechRecognition = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      addSystemMessage("Speech recognition is not supported in your browser.");
      return;
    }

    // @ts-ignore - Web Speech API types
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      if (finalTranscript) {
        setVoiceTranscription({
          text: finalTranscript,
          confidence: event.results[event.resultIndex][0].confidence || 0.9,
          language: 'en-US',
          timestamp: new Date().toISOString()
        });
        
        // Send the transcribed message
        addMessage(finalTranscript, 'user', { source: 'voice' });
        setInputText(finalTranscript);
        setTimeout(() => sendTextMessage(), 100);
      }
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (isVoiceConnected) {
        // Restart recognition if still in voice mode
        setTimeout(() => recognition.start(), 100);
      }
    };

    recognition.start();
  };

  const toggleMute = () => {
    setIsSpeaking(!isSpeaking);
    // Implement text-to-speech toggle logic here
  };

  const getMessageIcon = (sender: string) => {
    switch (sender) {
      case 'user':
        return <User className="h-4 w-4" />;
      case 'ai':
        return <Bot className="h-4 w-4" />;
      case 'system':
        return <Sparkles className="h-4 w-4" />;
      default:
        return <MessageSquare className="h-4 w-4" />;
    }
  };

  const getMessageStyle = (sender: string) => {
    switch (sender) {
      case 'user':
        return 'bg-medical-600 text-white ml-auto';
      case 'ai':
        return 'bg-white border border-gray-200 text-gray-900';
      case 'system':
        return 'bg-gray-100 text-gray-700 text-center';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  const toggleVoiceListening = () => {
    if (voiceListening) {
      stopVoiceListening();
    } else {
      startVoiceListening();
    }
  };

  const getVoiceButtonColor = () => {
    switch (voiceStatus) {
      case 'listening': return 'bg-green-600 text-white animate-pulse';
      case 'processing': return 'bg-yellow-600 text-white';
      case 'speaking': return 'bg-blue-600 text-white animate-pulse';
      case 'error': return 'bg-red-600 text-white';
      default: return 'bg-medical-600 text-white hover:bg-medical-700';
    }
  };

  const getConnectionStatus = () => {
    if (wsConnected && voiceConnected) return 'Connected';
    if (wsConnected) return 'Chat Connected';
    if (voiceConnected) return 'Voice Connected';
    return 'Disconnected';
  };

  const getConnectionColor = () => {
    if (wsConnected && voiceConnected) return 'text-success-600';
    if (wsConnected || voiceConnected) return 'text-yellow-600';
    return 'text-heart-600';
  };

  return (
    <div className={`bg-white border-r border-gray-200 flex flex-col ${className}`}>
      {/* Panel Header */}
      <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-medical-50 to-blue-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-medical-600 rounded-lg flex items-center justify-center">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Ophira AI Assistant</h2>
              <p className="text-sm text-gray-500">Medical Monitoring & Consultation</p>
            </div>
          </div>

          {/* Voice Controls */}
          <div className="flex items-center space-x-2">
            {isVoiceConnected && (
              <button
                onClick={toggleMute}
                className={`p-2 rounded-lg transition-colors ${
                  isSpeaking 
                    ? 'bg-success-100 text-success-600 hover:bg-success-200' 
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
                title={isSpeaking ? 'Mute AI Voice' : 'Unmute AI Voice'}
              >
                {isSpeaking ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
              </button>
            )}

            <button
              onClick={toggleVoiceListening}
              className={`p-2 rounded-lg transition-colors ${getVoiceButtonColor()}`}
              title={voiceListening ? 'Stop Voice Input' : 'Start Voice Input'}
              disabled={!sessionInfo?.session_id}
            >
              {voiceListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            </button>
          </div>
        </div>

        {/* Voice Status Indicator */}
        {voiceStatus !== 'idle' && (
          <div className="mt-2 text-sm">
            <span className="text-gray-600">Voice Status: </span>
            <span className={`font-medium ${
              voiceStatus === 'listening' ? 'text-green-600' :
              voiceStatus === 'processing' ? 'text-yellow-600' :
              voiceStatus === 'speaking' ? 'text-blue-600' :
              voiceStatus === 'error' ? 'text-red-600' : 'text-gray-600'
            }`}>
              {voiceStatus.charAt(0).toUpperCase() + voiceStatus.slice(1)}
            </span>
            {voiceTranscript && (
              <div className="text-xs text-gray-500 mt-1">
                Transcript: "{voiceTranscript}"
              </div>
            )}
          </div>
        )}
        
        {/* Error Display */}
        {voiceError && (
          <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
            {voiceError}
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 
              message.sender === 'system' ? 'justify-center' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg shadow-sm ${getMessageStyle(message.sender)}`}
            >
              <div className="flex items-start space-x-2">
                {message.sender !== 'user' && message.sender !== 'system' && (
                  <div className="flex-shrink-0 mt-0.5">
                    {getMessageIcon(message.sender)}
                  </div>
                )}
                <div className="flex-1">
                  <div className="text-sm leading-relaxed">{message.content}</div>
                  
                  {message.metadata && (
                    <div className="mt-2 text-xs opacity-75">
                      {message.metadata.agent_id && (
                        <span className="mr-2">Agent: {message.metadata.agent_id}</span>
                      )}
                      {message.metadata.confidence && (
                        <span className="mr-2">
                          Confidence: {Math.round(message.metadata.confidence * 100)}%
                        </span>
                      )}
                    </div>
                  )}
                  
                  <div className="text-xs opacity-60 mt-1">
                    {formatDistanceToNow(new Date(message.timestamp), { addSuffix: true })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 p-3 rounded-lg shadow-sm">
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-medical-600" />
                <span className="text-sm text-gray-600">Ophira AI is thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <div className="flex items-end space-x-2">
          {/* Voice Recording Button */}
          <button
            onClick={toggleVoiceListening}
            disabled={isLoading}
            className={`p-3 rounded-lg transition-colors ${getVoiceButtonColor()}`}
          >
            <Mic className="h-5 w-5" />
          </button>

          {/* Text Input */}
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={voiceListening ? "Listening..." : "Type your message..."}
              disabled={isLoading}
              className="w-full p-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>

          {/* Send Button */}
          <button
            onClick={sendTextMessage}
            disabled={!inputText.trim() || isLoading}
            className="p-3 bg-medical-600 text-white rounded-lg hover:bg-medical-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </div>

        {/* Quick Actions */}
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            onClick={() => setInputText("What are my current vital signs?")}
            className="px-3 py-1 text-xs bg-white border border-gray-300 text-gray-700 rounded-full hover:bg-gray-50 transition-colors"
          >
            Check Vitals
          </button>
          <button
            onClick={() => setInputText("Are there any health concerns I should know about?")}
            className="px-3 py-1 text-xs bg-white border border-gray-300 text-gray-700 rounded-full hover:bg-gray-50 transition-colors"
          >
            Health Status
          </button>
          <button
            onClick={() => setInputText("Start a new health analysis")}
            className="px-3 py-1 text-xs bg-white border border-gray-300 text-gray-700 rounded-full hover:bg-gray-50 transition-colors"
          >
            New Analysis
          </button>
          <button
            onClick={() => setInputText("Emergency - I need immediate help")}
            className="px-3 py-1 text-xs bg-heart-100 border border-heart-300 text-heart-700 rounded-full hover:bg-heart-200 transition-colors"
          >
            🚨 Emergency
          </button>
        </div>
      </div>
    </div>
  );
} 