# Ophira AI Medical Monitoring Frontend

A modern, responsive web interface for the Ophira AI Advanced Medical Monitoring System built with Next.js 14, TypeScript, and Tailwind CSS.

## Features

### 🏥 Medical Dashboard
- **3-Panel Interface**: Session management, AI chat, and sensor monitoring
- **Real-time Vital Signs**: Live heart rate, blood pressure, temperature, and SpO₂ monitoring
- **Medical Alerts**: Critical, warning, and informational notifications with acknowledgment
- **Health Trends**: Historical data visualization with multiple time ranges (1h-30d)

### 🤖 AI Assistant
- **Conversational Interface**: Natural language interaction with medical AI
- **Voice Integration**: Web Speech API for hands-free operation
- **Quick Actions**: Pre-defined medical queries and emergency protocols
- **Agent Metadata**: Confidence scores and processing time display

### 📹 Computer Vision
- **Live Camera Feed**: Real-time video streaming with privacy controls
- **Facial Analysis**: Pain level assessment and emotion detection
- **Gait Analysis**: Fall risk evaluation and posture monitoring
- **AI Overlays**: Visual indicators and analysis results

### 📊 Sensor Monitoring
- **Multi-sensor Support**: Heart rate, ToF distance, camera, microphone
- **Connection Status**: Real-time sensor health and calibration status
- **Data Quality**: Signal strength and battery level monitoring
- **Historical Charts**: Simple trend visualization

## Technology Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS with medical color palette
- **State Management**: Zustand for lightweight state management
- **Real-time**: WebSocket integration for live data
- **Icons**: Lucide React for consistent iconography
- **Date Handling**: date-fns for time formatting

## Project Structure

```
frontend/
├── src/
│   ├── app/                 # Next.js App Router
│   │   ├── globals.css     # Global styles and medical classes
│   │   ├── layout.tsx      # Root layout with medical theming
│   │   └── page.tsx        # Main dashboard page
│   ├── components/         # React components
│   │   ├── panels/         # Main dashboard panels
│   │   │   ├── SessionPanel.tsx    # Left: Patient info & alerts
│   │   │   ├── ChatPanel.tsx       # Center: AI conversation
│   │   │   └── SensorPanel.tsx     # Right: Camera & sensors
│   │   └── DashboardHeader.tsx     # Top navigation
│   ├── hooks/              # Custom React hooks
│   │   └── useOphiraWebSocket.ts   # WebSocket management
│   ├── lib/                # Utilities
│   │   └── api.ts          # API client for FastAPI backend
│   ├── stores/             # Zustand state stores
│   │   ├── sessionStore.ts # Session management
│   │   └── medicalStore.ts # Medical data state
│   └── types/              # TypeScript definitions
│       └── index.ts        # All medical and UI types
├── public/                 # Static assets
├── package.json           # Dependencies and scripts
├── tailwind.config.js     # Medical color system
├── next.config.js         # API proxy configuration
└── tsconfig.json          # TypeScript configuration
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Running Ophira AI backend (FastAPI server on port 8000)

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env.local
   ```
   
   Edit `.env.local` with your API endpoints:
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   NEXT_PUBLIC_WS_BASE_URL=ws://localhost:8000
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**: Navigate to [http://localhost:3000](http://localhost:3000)

### Production Build

```bash
npm run build
npm run start
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI backend URL | `http://localhost:8000` |
| `NEXT_PUBLIC_WS_BASE_URL` | WebSocket server URL | `ws://localhost:8000` |

### Medical Color System

The frontend uses a specialized medical color palette:

- **Medical Blue**: Primary interface elements (`medical-50` to `medical-900`)
- **Heart Red**: Critical alerts and heart rate (`heart-50` to `heart-900`)
- **Success Green**: Normal readings and confirmations
- **Warning Orange**: Caution states and warnings

## API Integration

### FastAPI Backend

The frontend communicates with the Ophira AI FastAPI backend through:

- **REST API**: Session management, data queries, configurations
- **WebSocket**: Real-time sensor data, alerts, and AI responses
- **File Upload**: Image and audio data for AI analysis

### Key Endpoints

```typescript
// Session Management
POST /api/v1/session/login
GET  /api/v1/session/{session_id}
POST /api/v1/session/{session_id}/logout

// Medical Data
GET  /api/v1/medical/vitals/{session_id}
GET  /api/v1/medical/alerts/{session_id}
POST /api/v1/medical/analysis/{session_id}

// Sensor Management
GET  /api/v1/sensors/status/{session_id}
POST /api/v1/sensors/calibrate/{sensor_id}

// AI Chat
POST /api/v1/chat/message
POST /api/v1/chat/voice/start
POST /api/v1/chat/voice/stop
```

## Real-time Features

### WebSocket Events

```typescript
interface WebSocketEvent {
  type: 'vital_signs' | 'alert' | 'agent_status' | 'chat_message' | 'sensor_update';
  data: any;
  timestamp: string;
  session_id: string;
}
```

### Auto-reconnection

WebSocket connections include:
- Automatic reconnection with exponential backoff
- Connection status indicators
- Graceful error handling
- Session validation

## Development

### Code Style

- **TypeScript**: Strict mode enabled with comprehensive types
- **Component Structure**: Functional components with hooks
- **State Management**: Zustand stores for different domains
- **Error Handling**: Try-catch blocks with user-friendly messages

### Medical Data Types

All medical data follows standardized interfaces:

```typescript
interface VitalSigns {
  heart_rate?: number;
  blood_pressure?: { systolic: number; diastolic: number };
  temperature?: number;
  respiratory_rate?: number;
  oxygen_saturation?: number;
  timestamp: string;
}
```

### Responsive Design

- **Mobile-first**: Tailwind's responsive breakpoints
- **Panel Layout**: Flexible 3-column layout that stacks on mobile
- **Touch-friendly**: Large buttons and touch targets for medical use

## Testing

### Browser Compatibility

- **Modern Browsers**: Chrome 90+, Firefox 88+, Safari 14+
- **WebRTC**: Camera access requires HTTPS in production
- **WebSocket**: Native support in all modern browsers

### Accessibility

- **ARIA Labels**: Screen reader support for medical data
- **Keyboard Navigation**: Full keyboard accessibility
- **Color Contrast**: WCAG 2.1 AA compliance for medical interfaces

## Deployment

### Production Considerations

1. **HTTPS Required**: Camera access and secure medical data
2. **CORS Configuration**: Ensure backend allows frontend domain
3. **WebSocket Proxy**: Configure reverse proxy for WebSocket connections
4. **Error Monitoring**: Implement logging for medical data errors

### Docker Deployment

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Contributing

1. Follow TypeScript strict mode
2. Use medical-appropriate color classes
3. Implement proper error handling for medical data
4. Test WebSocket reconnection scenarios
5. Ensure mobile responsiveness

## License

Part of the Ophira AI Advanced Medical Monitoring System.

## Support

For technical issues or medical compliance questions, refer to the main Ophira AI documentation. 