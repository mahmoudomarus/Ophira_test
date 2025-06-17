import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { SessionInfo, SessionWarning } from '@/types';
import { ophiraApi } from '@/lib/api';

interface SessionState {
  // State
  sessionInfo: SessionInfo | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isInitializing: boolean;
  error: string | null;
  warnings: SessionWarning[];
  lastActivity: Date | null;

  // Actions
  initializeSession: () => Promise<void>;
  login: (username: string, deviceInfo?: Record<string, string>) => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
  updateLastActivity: () => void;
  clearError: () => void;
  acknowledgeWarning: (warningId: string) => void;
}

export const useSessionStore = create<SessionState>()(
  devtools(
    (set, get) => ({
      // Initial state
      sessionInfo: null,
      isAuthenticated: false,
      isLoading: false,
      isInitializing: false,
      error: null,
      warnings: [],
      lastActivity: null,

      // Initialize session (for auto-login)
      initializeSession: async () => {
        const currentState = get();
        
        // Prevent multiple concurrent initializations
        if (currentState.isInitializing || currentState.isAuthenticated) {
          return;
        }
        
        set({ isLoading: true, isInitializing: true, error: null });
        
        try {
          // Try to create a demo session
          await get().login('demo_user', {
            platform: navigator?.platform || 'unknown',
            userAgent: navigator?.userAgent || 'unknown'
          });
        } catch (error) {
          console.error('Failed to initialize session:', error);
          set({ 
            error: error instanceof Error ? error.message : 'Failed to initialize session',
            isLoading: false,
            isInitializing: false
          });
        } finally {
          set({ isInitializing: false });
        }
      },

      // Login with username and device info
      login: async (username: string, deviceInfo?: Record<string, string>) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await ophiraApi.login(username, deviceInfo);
          
          if (response.success && response.data) {
            set({
              sessionInfo: response.data,
              isAuthenticated: true,
              isLoading: false,
              warnings: response.data.warnings || [],
              lastActivity: new Date(),
              error: null
            });
          } else {
            throw new Error(response.error || 'Login failed');
          }
        } catch (error) {
          console.error('Login failed:', error);
          set({
            error: error instanceof Error ? error.message : 'Login failed',
            isLoading: false,
            isAuthenticated: false,
            sessionInfo: null
          });
          throw error;
        }
      },

      // Logout and clear session
      logout: async () => {
        const { sessionInfo } = get();
        
        if (sessionInfo?.session_id) {
          try {
            await ophiraApi.logout(sessionInfo.session_id);
          } catch (error) {
            console.error('Logout request failed:', error);
          }
        }

        set({
          sessionInfo: null,
          isAuthenticated: false,
          warnings: [],
          lastActivity: null,
          error: null
        });
      },

      // Refresh current session
      refreshSession: async () => {
        const { sessionInfo } = get();
        
        if (!sessionInfo?.session_id) {
          set({ error: 'No active session to refresh' });
          return;
        }

        try {
          const response = await ophiraApi.refreshSession(sessionInfo.session_id);
          
          if (response.success && response.data) {
            set({
              sessionInfo: response.data,
              warnings: response.data.warnings || [],
              lastActivity: new Date(),
              error: null
            });
          } else {
            throw new Error(response.error || 'Session refresh failed');
          }
        } catch (error) {
          console.error('Session refresh failed:', error);
          set({ error: error instanceof Error ? error.message : 'Session refresh failed' });
          
          // If refresh fails, try to re-authenticate
          await get().logout();
        }
      },

      // Update last activity timestamp
      updateLastActivity: () => {
        set({ lastActivity: new Date() });
      },

      // Clear error state
      clearError: () => {
        set({ error: null });
      },

      // Acknowledge a warning
      acknowledgeWarning: (warningId: string) => {
        set((state) => ({
          warnings: state.warnings.filter(w => w.type !== warningId)
        }));
      }
    }),
    {
      name: 'ophira-session-store',
      partialize: (state: SessionState) => ({
        sessionInfo: state.sessionInfo,
        isAuthenticated: state.isAuthenticated,
        lastActivity: state.lastActivity
      })
    }
  )
);

// Auto-refresh session every 5 minutes (temporarily disabled to prevent issues)
/*
if (typeof window !== 'undefined') {
  setInterval(() => {
    const store = useSessionStore.getState();
    if (store.isAuthenticated && store.sessionInfo) {
      store.refreshSession().catch(console.error);
    }
  }, 5 * 60 * 1000);
}
*/ 