import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  VitalSigns, 
  SensorReading, 
  SensorStatus, 
  MedicalAlert, 
  HealthAnalysis,
  HealthTrend,
  DashboardData
} from '@/types';
import { ophiraApi } from '@/lib/api';

interface MedicalState {
  // State
  vitalSigns: VitalSigns | null;
  sensorReadings: SensorReading[];
  sensorStatuses: SensorStatus[];
  alerts: MedicalAlert[];
  healthAnalysis: HealthAnalysis[];
  healthTrends: HealthTrend[];
  dashboardData: DashboardData | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  updateVitalSigns: (vitalSigns: VitalSigns) => void;
  addSensorReading: (reading: SensorReading) => void;
  updateSensorStatus: (status: SensorStatus) => void;
  addAlert: (alert: MedicalAlert) => void;
  acknowledgeAlert: (alertId: string) => void;
  dismissAlert: (alertId: string) => void;
  addHealthAnalysis: (analysis: HealthAnalysis) => void;
  updateHealthTrends: (trends: HealthTrend[]) => void;
  fetchDashboardData: (sessionId: string) => Promise<void>;
  fetchSensorStatus: (sessionId: string) => Promise<void>;
  requestHealthAnalysis: (sessionId: string, analysisTypes: string[]) => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

export const useMedicalStore = create<MedicalState>()(
  devtools(
    (set, get) => ({
      // Initial state
      vitalSigns: null,
      sensorReadings: [],
      sensorStatuses: [],
      alerts: [],
      healthAnalysis: [],
      healthTrends: [],
      dashboardData: null,
      isLoading: false,
      error: null,

      // Update vital signs
      updateVitalSigns: (vitalSigns: VitalSigns) => {
        set((state) => ({
          vitalSigns,
          sensorReadings: [
            ...state.sensorReadings.slice(-99), // Keep last 100 readings
            {
              sensor_type: 'heart_rate',
              value: vitalSigns.heart_rate || 0,
              unit: 'bpm',
              timestamp: vitalSigns.timestamp,
              quality: 'good',
              confidence: 0.95
            }
          ]
        }));
      },

      // Add sensor reading
      addSensorReading: (reading: SensorReading) => {
        set((state) => ({
          sensorReadings: [
            ...state.sensorReadings.slice(-99), // Keep last 100 readings
            reading
          ]
        }));
      },

      // Update sensor status
      updateSensorStatus: (status: SensorStatus) => {
        set((state) => ({
          sensorStatuses: [
            ...state.sensorStatuses.filter(s => s.sensor_id !== status.sensor_id),
            status
          ]
        }));
      },

      // Add medical alert
      addAlert: (alert: MedicalAlert) => {
        set((state) => ({
          alerts: [alert, ...state.alerts].slice(0, 50) // Keep last 50 alerts
        }));
      },

      // Acknowledge alert
      acknowledgeAlert: (alertId: string) => {
        set((state) => ({
          alerts: state.alerts.map(alert =>
            alert.id === alertId
              ? { ...alert, acknowledged: true }
              : alert
          )
        }));
      },

      // Dismiss alert
      dismissAlert: (alertId: string) => {
        set((state) => ({
          alerts: state.alerts.filter(alert => alert.id !== alertId)
        }));
      },

      // Add health analysis
      addHealthAnalysis: (analysis: HealthAnalysis) => {
        set((state) => ({
          healthAnalysis: [analysis, ...state.healthAnalysis].slice(0, 20) // Keep last 20 analyses
        }));
      },

      // Update health trends
      updateHealthTrends: (trends: HealthTrend[]) => {
        set({ healthTrends: trends });
      },

      // Fetch dashboard data
      fetchDashboardData: async (sessionId: string) => {
        set({ isLoading: true, error: null });

        try {
          // Fetch multiple data sources
          const [vitalSigns, sensorStatus, alerts] = await Promise.all([
            ophiraApi.getVitalSigns(sessionId),
            ophiraApi.getSensorStatus(sessionId),
            ophiraApi.getAlerts(sessionId)
          ]);

          // Update state with fetched data
          if (vitalSigns.success && vitalSigns.data) {
            get().updateVitalSigns(vitalSigns.data);
          }

          if (sensorStatus.success && sensorStatus.data) {
            sensorStatus.data.forEach((status: SensorStatus) => {
              get().updateSensorStatus(status);
            });
          }

          if (alerts.success && alerts.data) {
            set({ alerts: alerts.data });
          }

          set({ isLoading: false });
        } catch (error) {
          console.error('Failed to fetch dashboard data:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to fetch dashboard data',
            isLoading: false
          });
        }
      },

      // Fetch sensor status
      fetchSensorStatus: async (sessionId: string) => {
        try {
          const response = await ophiraApi.getSensorStatus(sessionId);
          
          if (response.success && response.data) {
            response.data.forEach((status: SensorStatus) => {
              get().updateSensorStatus(status);
            });
          }
        } catch (error) {
          console.error('Failed to fetch sensor status:', error);
          set({ error: error instanceof Error ? error.message : 'Failed to fetch sensor status' });
        }
      },

      // Request health analysis
      requestHealthAnalysis: async (sessionId: string, analysisTypes: string[]) => {
        set({ isLoading: true, error: null });

        try {
          const response = await ophiraApi.requestHealthAnalysis(sessionId, {
            analysis_types: analysisTypes
          });

          if (response.success && response.data) {
            get().addHealthAnalysis(response.data);
          }

          set({ isLoading: false });
        } catch (error) {
          console.error('Health analysis request failed:', error);
          set({
            error: error instanceof Error ? error.message : 'Health analysis failed',
            isLoading: false
          });
        }
      },

      // Clear error state
      clearError: () => {
        set({ error: null });
      },

      // Reset all medical data
      reset: () => {
        set({
          vitalSigns: null,
          sensorReadings: [],
          sensorStatuses: [],
          alerts: [],
          healthAnalysis: [],
          healthTrends: [],
          dashboardData: null,
          isLoading: false,
          error: null
        });
      }
    }),
    {
      name: 'ophira-medical-store'
    }
  )
); 