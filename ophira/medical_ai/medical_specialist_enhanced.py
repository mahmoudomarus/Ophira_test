"""
Enhanced Medical Specialist Agent - Phase 2
==========================================

Advanced medical AI agent that integrates all medical analysis capabilities:
- Retinal analysis coordination
- Cardiovascular assessment
- Metabolic health evaluation
- Clinical report generation
- Risk assessment and recommendations
"""

import asyncio
import logging
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import json
import numpy as np
from enum import Enum

from ..agents.base import BaseAgent, AgentCapability
from ..core.logging import setup_logging
from .models import AIModelManager, InferenceResult, MedicalDomain
from .retinal_analysis import RetinalAnalysisManager, RetinalAnalysisResult

logger = setup_logging()

@dataclass
class PatientProfile:
    """Comprehensive patient profile for medical analysis"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_medications: List[str]
    vital_signs: Dict[str, float]
    risk_factors: Dict[str, Any]
    previous_analyses: List[Dict[str, Any]]
    emergency_contacts: List[Dict[str, str]]
    preferences: Dict[str, Any]

@dataclass
class MedicalAnalysisSession:
    """Medical analysis session data"""
    session_id: str
    patient_profile: PatientProfile
    analysis_type: str
    input_data: Dict[str, Any]
    results: Dict[str, Any]
    recommendations: List[str]
    risk_level: str
    follow_up_required: bool
    emergency_alert: bool
    timestamp: str
    duration: float

@dataclass
class ClinicalReport:
    """Comprehensive clinical report"""
    report_id: str
    patient_id: str
    session_id: str
    report_type: str
    executive_summary: str
    detailed_findings: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    follow_up_plan: Dict[str, Any]
    emergency_alerts: List[str]
    ai_confidence: float
    physician_review_required: bool
    generated_timestamp: str
    report_data: Dict[str, Any]

class EnhancedMedicalSpecialistAgent(BaseAgent):
    """
    Enhanced Medical Specialist Agent with AI-powered Analysis
    
    Coordinates all medical AI models and provides comprehensive health analysis
    """
    
    def __init__(self, agent_id: str = "medical_specialist_enhanced"):
        # Initialize BaseAgent with required parameters
        super().__init__(
            agent_id=agent_id,
            name="Enhanced Medical Specialist",
            agent_type="medical_specialist_enhanced",
            capabilities=[
                AgentCapability.MEDICAL_ANALYSIS,
                AgentCapability.REPORT_GENERATION,
                AgentCapability.DATA_PROCESSING,
                AgentCapability.EMERGENCY_RESPONSE
            ]
        )
        
        self.model_manager: Optional[AIModelManager] = None
        self.retinal_manager: Optional[RetinalAnalysisManager] = None
        
        # Session management
        self.active_sessions: Dict[str, MedicalAnalysisSession] = {}
        self.patient_profiles: Dict[str, PatientProfile] = {}
        
        # Clinical thresholds
        self.emergency_thresholds = {
            'overall_risk': 0.8,
            'dr_severe': True,
            'glaucoma_critical': True,
            'cardiac_emergency': True
        }
        
        # Report generation settings
        self.auto_generate_reports = True
        self.require_physician_review = True
        
        logger.info("Enhanced Medical Specialist Agent initialized")
    
    async def initialize(self):
        """Initialize the enhanced medical specialist agent"""
        try:
            # Initialize AI model manager
            self.model_manager = AIModelManager()
            await self.model_manager.initialize()
            
            # Initialize specialized analysis managers
            self.retinal_manager = RetinalAnalysisManager(self.model_manager)
            
            # Load existing patient profiles
            await self._load_patient_profiles()
            
            logger.info("Enhanced Medical Specialist Agent fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Medical Specialist Agent: {e}")
            raise
    
    async def start_medical_analysis(self, patient_id: str, analysis_type: str,
                                   input_data: Dict[str, Any]) -> str:
        """
        Start a comprehensive medical analysis session
        
        Args:
            patient_id: Patient identifier
            analysis_type: Type of analysis ('retinal', 'cardiovascular', 'comprehensive')
            input_data: Analysis input data
            
        Returns:
            Session ID for tracking the analysis
        """
        try:
            session_id = f"session_{int(time.time() * 1000)}"
            
            # Get or create patient profile
            patient_profile = await self._get_patient_profile(patient_id)
            
            # Create analysis session
            session = MedicalAnalysisSession(
                session_id=session_id,
                patient_profile=patient_profile,
                analysis_type=analysis_type,
                input_data=input_data,
                results={},
                recommendations=[],
                risk_level="pending",
                follow_up_required=False,
                emergency_alert=False,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                duration=0.0
            )
            
            self.active_sessions[session_id] = session
            
            # Start analysis based on type
            await self._execute_analysis(session)
            
            # Generate clinical report if enabled
            if self.auto_generate_reports:
                await self._generate_clinical_report(session)
            
            logger.info(f"Medical analysis session {session_id} completed for patient {patient_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start medical analysis: {e}")
            raise
    
    async def _execute_analysis(self, session: MedicalAnalysisSession):
        """Execute the medical analysis based on session type"""
        start_time = time.time()
        
        try:
            analysis_type = session.analysis_type
            input_data = session.input_data
            patient_context = await self._build_patient_context(session.patient_profile)
            
            if analysis_type == "retinal":
                await self._execute_retinal_analysis(session, input_data, patient_context)
            elif analysis_type == "cardiovascular":
                await self._execute_cardiovascular_analysis(session, input_data, patient_context)
            elif analysis_type == "comprehensive":
                await self._execute_comprehensive_analysis(session, input_data, patient_context)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            session.duration = time.time() - start_time
            
            # Assess overall risk and generate recommendations
            await self._assess_overall_risk(session)
            
        except Exception as e:
            logger.error(f"Analysis execution failed: {e}")
            session.results['error'] = str(e)
            session.risk_level = "error"
    
    async def _execute_retinal_analysis(self, session: MedicalAnalysisSession,
                                       input_data: Dict[str, Any],
                                       patient_context: Dict[str, Any]):
        """Execute comprehensive retinal analysis"""
        if 'retinal_image' not in input_data:
            raise ValueError("Retinal image required for retinal analysis")
        
        retinal_image = input_data['retinal_image']
        
        # Run comprehensive retinal analysis
        retinal_result = await self.retinal_manager.comprehensive_analysis(
            retinal_image, patient_context
        )
        
        session.results['retinal_analysis'] = asdict(retinal_result)
        
        # Extract key findings
        session.results['key_findings'] = {
            'diabetic_retinopathy': {
                'classification': max(retinal_result.diabetic_retinopathy.predictions, 
                                    key=retinal_result.diabetic_retinopathy.predictions.get),
                'confidence': retinal_result.diabetic_retinopathy.confidence,
                'risk_level': retinal_result.diabetic_retinopathy.risk_level
            },
            'glaucoma_risk': {
                'classification': max(retinal_result.glaucoma_risk.predictions,
                                    key=retinal_result.glaucoma_risk.predictions.get),
                'confidence': retinal_result.glaucoma_risk.confidence,
                'risk_level': retinal_result.glaucoma_risk.risk_level
            },
            'macular_health': retinal_result.macular_assessment,
            'image_quality': asdict(retinal_result.image_quality)
        }
        
        # Set emergency alert if needed
        if retinal_result.emergency_referral:
            session.emergency_alert = True
            session.results['emergency_reason'] = "Urgent ophthalmology referral required"
        
        # Add recommendations
        session.recommendations.extend(retinal_result.recommendations)
        session.follow_up_required = retinal_result.overall_risk_score > 0.5
    
    async def _execute_cardiovascular_analysis(self, session: MedicalAnalysisSession,
                                             input_data: Dict[str, Any],
                                             patient_context: Dict[str, Any]):
        """Execute cardiovascular analysis (placeholder for future implementation)"""
        # This would be implemented with cardiovascular AI models
        session.results['cardiovascular_analysis'] = {
            'status': 'placeholder',
            'message': 'Cardiovascular analysis will be implemented in future versions'
        }
        
        logger.info("Cardiovascular analysis placeholder executed")
    
    async def _execute_comprehensive_analysis(self, session: MedicalAnalysisSession,
                                            input_data: Dict[str, Any],
                                            patient_context: Dict[str, Any]):
        """Execute comprehensive multi-modal analysis"""
        # Execute multiple analysis types
        analysis_tasks = []
        
        if 'retinal_image' in input_data:
            analysis_tasks.append(
                self._execute_retinal_analysis(session, input_data, patient_context)
            )
        
        if 'ecg_data' in input_data or 'heart_rate_data' in input_data:
            analysis_tasks.append(
                self._execute_cardiovascular_analysis(session, input_data, patient_context)
            )
        
        # Execute all analyses in parallel
        if analysis_tasks:
            await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Combine results for comprehensive assessment
        await self._combine_analysis_results(session)
    
    async def _combine_analysis_results(self, session: MedicalAnalysisSession):
        """Combine results from multiple analyses for comprehensive assessment"""
        results = session.results
        combined_findings = {}
        combined_risk_scores = []
        
        # Combine retinal analysis
        if 'retinal_analysis' in results:
            retinal_data = results['retinal_analysis']
            combined_findings['retinal'] = {
                'overall_risk': retinal_data['overall_risk_score'],
                'emergency_referral': retinal_data['emergency_referral'],
                'key_conditions': {
                    'diabetic_retinopathy': retinal_data['diabetic_retinopathy']['risk_level'],
                    'glaucoma': retinal_data['glaucoma_risk']['risk_level']
                }
            }
            combined_risk_scores.append(retinal_data['overall_risk_score'])
        
        # Combine cardiovascular analysis (when implemented)
        if 'cardiovascular_analysis' in results:
            # Placeholder for cardiovascular data combination
            pass
        
        # Calculate overall combined risk
        if combined_risk_scores:
            combined_risk = np.mean(combined_risk_scores)
            session.results['combined_risk_score'] = combined_risk
            session.results['combined_findings'] = combined_findings
    
    async def _assess_overall_risk(self, session: MedicalAnalysisSession):
        """Assess overall risk level and determine follow-up needs"""
        results = session.results
        
        # Determine risk level based on results
        risk_indicators = []
        
        if 'retinal_analysis' in results:
            retinal_risk = results['retinal_analysis']['overall_risk_score']
            risk_indicators.append(retinal_risk)
            
            # Check for emergency conditions
            if results['retinal_analysis']['emergency_referral']:
                session.emergency_alert = True
        
        if 'combined_risk_score' in results:
            risk_indicators.append(results['combined_risk_score'])
        
        # Calculate overall risk
        if risk_indicators:
            overall_risk = max(risk_indicators)  # Use highest risk
        else:
            overall_risk = 0.0
        
        # Set risk level categories
        if overall_risk >= 0.8:
            session.risk_level = "critical"
            session.emergency_alert = True
        elif overall_risk >= 0.6:
            session.risk_level = "high"
            session.follow_up_required = True
        elif overall_risk >= 0.3:
            session.risk_level = "medium"
            session.follow_up_required = True
        else:
            session.risk_level = "low"
        
        session.results['overall_risk_assessment'] = {
            'risk_score': overall_risk,
            'risk_level': session.risk_level,
            'emergency_alert': session.emergency_alert,
            'follow_up_required': session.follow_up_required
        }
    
    async def _build_patient_context(self, profile: PatientProfile) -> Dict[str, Any]:
        """Build patient context for AI analysis"""
        return {
            'patient_id': profile.patient_id,
            'age': profile.age,
            'gender': profile.gender,
            'medical_history': profile.medical_history,
            'current_medications': profile.current_medications,
            'vital_signs': profile.vital_signs,
            'risk_factors': profile.risk_factors,
            
            # Extract specific risk factors for medical AI
            'diabetes_duration_years': profile.risk_factors.get('diabetes_duration_years', 0),
            'hba1c': profile.vital_signs.get('hba1c', 7.0),
            'hypertension': 'hypertension' in profile.medical_history,
            'family_history_glaucoma': profile.risk_factors.get('family_history_glaucoma', False),
            'myopia': 'myopia' in profile.medical_history,
            'steroid_use': any('steroid' in med.lower() for med in profile.current_medications)
        }
    
    async def _get_patient_profile(self, patient_id: str) -> PatientProfile:
        """Get or create patient profile"""
        if patient_id in self.patient_profiles:
            return self.patient_profiles[patient_id]
        
        # Create default profile if not exists
        default_profile = PatientProfile(
            patient_id=patient_id,
            age=50,
            gender="unknown",
            medical_history=[],
            current_medications=[],
            vital_signs={},
            risk_factors={},
            previous_analyses=[],
            emergency_contacts=[],
            preferences={}
        )
        
        self.patient_profiles[patient_id] = default_profile
        return default_profile
    
    async def _load_patient_profiles(self):
        """Load existing patient profiles"""
        # Placeholder for loading patient profiles from database
        # In production, this would load from secure patient database
        logger.info("Patient profiles loaded")
    
    async def _generate_clinical_report(self, session: MedicalAnalysisSession) -> str:
        """Generate comprehensive clinical report"""
        try:
            report_id = f"report_{session.session_id}"
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(session)
            
            # Extract detailed findings
            detailed_findings = session.results
            
            # Build risk assessment
            risk_assessment = {
                'overall_risk_score': session.results.get('overall_risk_assessment', {}).get('risk_score', 0.0),
                'risk_level': session.risk_level,
                'emergency_status': session.emergency_alert,
                'follow_up_required': session.follow_up_required
            }
            
            # Format recommendations
            formatted_recommendations = []
            for i, rec in enumerate(session.recommendations):
                formatted_recommendations.append({
                    'priority': 'high' if session.emergency_alert else 'medium',
                    'category': 'clinical_action',
                    'description': rec,
                    'timeline': 'immediate' if session.emergency_alert else 'routine'
                })
            
            # Create follow-up plan
            follow_up_plan = await self._create_follow_up_plan(session)
            
            # Emergency alerts
            emergency_alerts = []
            if session.emergency_alert:
                emergency_alerts.append("URGENT MEDICAL ATTENTION REQUIRED")
                if 'emergency_reason' in session.results:
                    emergency_alerts.append(session.results['emergency_reason'])
            
            # Calculate AI confidence
            ai_confidence = await self._calculate_ai_confidence(session)
            
            # Create clinical report
            report = ClinicalReport(
                report_id=report_id,
                patient_id=session.patient_profile.patient_id,
                session_id=session.session_id,
                report_type=session.analysis_type,
                executive_summary=executive_summary,
                detailed_findings=detailed_findings,
                risk_assessment=risk_assessment,
                recommendations=formatted_recommendations,
                follow_up_plan=follow_up_plan,
                emergency_alerts=emergency_alerts,
                ai_confidence=ai_confidence,
                physician_review_required=self.require_physician_review or session.emergency_alert,
                generated_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                report_data=asdict(session)
            )
            
            # Store report
            session.results['clinical_report'] = asdict(report)
            
            logger.info(f"Clinical report {report_id} generated successfully")
            return report_id
            
        except Exception as e:
            logger.error(f"Failed to generate clinical report: {e}")
            raise
    
    async def _generate_executive_summary(self, session: MedicalAnalysisSession) -> str:
        """Generate executive summary for clinical report"""
        summary_parts = []
        
        # Patient info
        profile = session.patient_profile
        summary_parts.append(f"Patient: {profile.patient_id} ({profile.age}yo {profile.gender})")
        
        # Analysis type
        summary_parts.append(f"Analysis: {session.analysis_type.title()} Assessment")
        
        # Key findings
        if 'retinal_analysis' in session.results:
            retinal_data = session.results['retinal_analysis']
            dr_class = retinal_data['diabetic_retinopathy']['predictions']
            dr_main = max(dr_class, key=dr_class.get)
            
            glaucoma_class = retinal_data['glaucoma_risk']['predictions']
            glaucoma_main = max(glaucoma_class, key=glaucoma_class.get)
            
            summary_parts.append(f"Diabetic Retinopathy: {dr_main}")
            summary_parts.append(f"Glaucoma Risk: {glaucoma_main}")
        
        # Risk level
        summary_parts.append(f"Overall Risk: {session.risk_level.upper()}")
        
        # Emergency status
        if session.emergency_alert:
            summary_parts.append("⚠️ EMERGENCY REFERRAL REQUIRED")
        
        return " | ".join(summary_parts)
    
    async def _create_follow_up_plan(self, session: MedicalAnalysisSession) -> Dict[str, Any]:
        """Create follow-up plan based on analysis results"""
        plan = {
            'immediate_actions': [],
            'short_term_follow_up': [],
            'long_term_monitoring': [],
            'patient_education': []
        }
        
        if session.emergency_alert:
            plan['immediate_actions'].append({
                'action': 'Emergency ophthalmology referral',
                'timeline': 'Within 24-48 hours',
                'priority': 'critical'
            })
        
        if 'retinal_analysis' in session.results:
            retinal_data = session.results['retinal_analysis']
            follow_up_timeline = retinal_data.get('follow_up_timeline', '12 months')
            
            plan['short_term_follow_up'].append({
                'action': 'Ophthalmology follow-up',
                'timeline': follow_up_timeline,
                'priority': 'high' if session.follow_up_required else 'routine'
            })
            
            plan['long_term_monitoring'].append({
                'action': 'Regular diabetic eye screening',
                'frequency': 'Annual' if session.risk_level == 'low' else 'Every 6 months',
                'priority': 'routine'
            })
        
        plan['patient_education'].extend([
            'Importance of regular eye examinations',
            'Diabetes management and glycemic control',
            'Recognition of vision changes requiring immediate attention'
        ])
        
        return plan
    
    async def _calculate_ai_confidence(self, session: MedicalAnalysisSession) -> float:
        """Calculate overall AI confidence in the analysis"""
        confidence_scores = []
        
        if 'retinal_analysis' in session.results:
            retinal_data = session.results['retinal_analysis']
            dr_confidence = retinal_data['diabetic_retinopathy']['confidence']
            glaucoma_confidence = retinal_data['glaucoma_risk']['confidence']
            confidence_scores.extend([dr_confidence, glaucoma_confidence])
        
        if confidence_scores:
            return np.mean(confidence_scores)
        else:
            return 0.5  # Default confidence
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of analysis session"""
        if session_id not in self.active_sessions:
            return {'status': 'not_found'}
        
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'status': 'completed',
            'patient_id': session.patient_profile.patient_id,
            'analysis_type': session.analysis_type,
            'risk_level': session.risk_level,
            'emergency_alert': session.emergency_alert,
            'follow_up_required': session.follow_up_required,
            'duration': session.duration,
            'timestamp': session.timestamp,
            'has_report': 'clinical_report' in session.results
        }
    
    async def get_clinical_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get clinical report for session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return session.results.get('clinical_report')
    
    async def handle_voice_command(self, command: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice commands for medical analysis"""
        try:
            command_lower = command.lower()
            
            if "start analysis" in command_lower or "begin analysis" in command_lower:
                return {
                    'response': "I'm ready to begin medical analysis. Please provide the patient information and analysis type.",
                    'action': 'request_analysis_data',
                    'requires_input': True
                }
            
            elif "emergency" in command_lower or "urgent" in command_lower:
                return {
                    'response': "Emergency protocol activated. What type of urgent medical analysis is needed?",
                    'action': 'emergency_mode',
                    'priority': 'critical'
                }
            
            elif "report" in command_lower:
                if 'session_id' in session_context:
                    report = await self.get_clinical_report(session_context['session_id'])
                    if report:
                        return {
                            'response': f"Clinical report available for session {session_context['session_id']}",
                            'action': 'provide_report',
                            'data': report
                        }
                
                return {
                    'response': "No clinical report found. Please specify a session ID or start a new analysis.",
                    'action': 'request_session_info'
                }
            
            elif "status" in command_lower:
                if 'session_id' in session_context:
                    status = await self.get_session_status(session_context['session_id'])
                    return {
                        'response': f"Session status: {status['status']}, Risk level: {status.get('risk_level', 'unknown')}",
                        'action': 'provide_status',
                        'data': status
                    }
                
                return {
                    'response': "Please specify which analysis session you'd like status for.",
                    'action': 'request_session_info'
                }
            
            else:
                return {
                    'response': "I can help with medical analysis, emergency protocols, reports, and status updates. What would you like to do?",
                    'action': 'provide_help',
                    'available_commands': ['start analysis', 'emergency', 'report', 'status']
                }
                
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
            return {
                'response': "I encountered an error processing your request. Please try again.",
                'action': 'error',
                'error': str(e)
            }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all medical AI models"""
        if not self.model_manager:
            return {'status': 'not_initialized'}
        
        return {
            'model_manager': 'initialized',
            'models': self.model_manager.list_models(),
            'system_stats': self.model_manager.get_system_stats()
        }
    
    async def shutdown(self):
        """Shutdown the enhanced medical specialist agent"""
        try:
            # Save active sessions
            await self._save_active_sessions()
            
            # Close model connections
            if self.model_manager:
                logger.info("Closing AI model connections...")
            
            logger.info("Enhanced Medical Specialist Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _save_active_sessions(self):
        """Save active sessions for persistence"""
        # In production, save to database
        logger.info(f"Saving {len(self.active_sessions)} active sessions")
    
    # Abstract method implementations required by BaseAgent
    async def _handle_message(self, message):
        """Handle incoming messages from other agents"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type.value == "medical_analysis_request":
                # Handle medical analysis requests
                analysis_type = content.get('analysis_type', 'comprehensive')
                patient_id = content.get('patient_id')
                input_data = content.get('input_data', {})
                
                if patient_id:
                    session_id = await self.start_medical_analysis(
                        patient_id, analysis_type, input_data
                    )
                    
                    # Send response
                    response_message = type(message)(
                        sender_id=self.id,
                        recipient_id=message.sender_id,
                        message_type=type(message.message_type)("response"),
                        content={
                            'session_id': session_id,
                            'status': 'analysis_started'
                        },
                        correlation_id=message.id
                    )
                    await self.send_message(response_message)
                    
            elif message_type.value == "voice_command":
                # Handle voice commands
                command = content.get('command', '')
                session_context = content.get('session_context', {})
                
                response = await self.handle_voice_command(command, session_context)
                
                # Send response
                response_message = type(message)(
                    sender_id=self.id,
                    recipient_id=message.sender_id,
                    message_type=type(message.message_type)("response"),
                    content=response,
                    correlation_id=message.id
                )
                await self.send_message(response_message)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _start_agent_tasks(self):
        """Start agent-specific background tasks"""
        try:
            # Start session monitoring task
            session_task = asyncio.create_task(self._monitor_active_sessions())
            self.tasks.append(session_task)
            
            # Start model health monitoring
            model_task = asyncio.create_task(self._monitor_model_health())
            self.tasks.append(model_task)
            
            logger.info("Enhanced Medical Specialist Agent tasks started")
            
        except Exception as e:
            logger.error(f"Error starting agent tasks: {e}")
    
    async def _cleanup(self):
        """Cleanup resources and save state"""
        try:
            # Save active sessions
            await self._save_active_sessions()
            
            # Stop background tasks
            for task in getattr(self, 'tasks', []):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close model manager
            if self.model_manager:
                # Model manager cleanup would go here
                pass
            
            logger.info("Enhanced Medical Specialist Agent cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _monitor_active_sessions(self):
        """Monitor active sessions for timeouts and status updates"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for session timeouts (1 hour timeout)
                expired_sessions = []
                for session_id, session in self.active_sessions.items():
                    session_age = current_time - time.mktime(
                        time.strptime(session.timestamp, '%Y-%m-%d %H:%M:%S')
                    )
                    
                    if session_age > 3600:  # 1 hour
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    del self.active_sessions[session_id]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _monitor_model_health(self):
        """Monitor AI model health and performance"""
        while self.is_running:
            try:
                if self.model_manager:
                    stats = self.model_manager.get_system_stats()
                    logger.debug(f"Model manager stats: {stats}")
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in model health monitoring: {e}")
                await asyncio.sleep(300)  # Wait before retrying

class ClinicalReportGenerator:
    """
    Dedicated Clinical Report Generation System
    
    Generates professional medical reports from AI analysis results
    """
    
    def __init__(self):
        self.report_templates = self._load_report_templates()
        logger.info("Clinical Report Generator initialized")
    
    def _load_report_templates(self) -> Dict[str, str]:
        """Load clinical report templates"""
        return {
            'retinal': """
# OPHTHALMOLOGIC AI ANALYSIS REPORT

**Patient:** {patient_id}
**Date:** {timestamp}
**Analysis Type:** Retinal Assessment

## EXECUTIVE SUMMARY
{executive_summary}

## FINDINGS
### Diabetic Retinopathy Assessment
- **Classification:** {dr_classification}
- **Confidence:** {dr_confidence:.1%}
- **Risk Level:** {dr_risk_level}

### Glaucoma Screening
- **Assessment:** {glaucoma_classification}
- **Confidence:** {glaucoma_confidence:.1%}
- **Cup-to-Disc Ratio:** {cup_disc_ratio:.2f}

### Image Quality
- **Overall Score:** {image_quality_score:.2f}/1.0
- **Notes:** {image_quality_notes}

## RISK ASSESSMENT
- **Overall Risk Score:** {overall_risk:.2f}/1.0
- **Risk Category:** {risk_level}
- **Emergency Referral:** {emergency_referral}

## RECOMMENDATIONS
{recommendations}

## FOLLOW-UP PLAN
{follow_up_plan}

---
*This report was generated by Ophira AI Medical Analysis System. Physician review recommended for all clinical decisions.*
            """,
            
            'comprehensive': """
# COMPREHENSIVE MEDICAL AI ANALYSIS REPORT

**Patient:** {patient_id}
**Date:** {timestamp}
**Analysis Type:** Multi-Modal Assessment

## EXECUTIVE SUMMARY
{executive_summary}

## DETAILED FINDINGS
{detailed_findings}

## INTEGRATED RISK ASSESSMENT
{risk_assessment}

## CLINICAL RECOMMENDATIONS
{recommendations}

## FOLLOW-UP CARE PLAN
{follow_up_plan}

## AI ANALYSIS METADATA
- **AI Confidence:** {ai_confidence:.1%}
- **Analysis Duration:** {duration:.2f} seconds
- **Models Used:** {models_used}

---
*This comprehensive report integrates multiple AI analysis systems. Clinical correlation and physician oversight required.*
            """
        }
    
    async def generate_formatted_report(self, report: ClinicalReport) -> str:
        """Generate formatted clinical report"""
        try:
            template = self.report_templates.get(report.report_type, self.report_templates['comprehensive'])
            
            # Prepare template variables
            variables = {
                'patient_id': report.patient_id,
                'timestamp': report.generated_timestamp,
                'executive_summary': report.executive_summary,
                'overall_risk': report.risk_assessment.get('overall_risk_score', 0.0),
                'risk_level': report.risk_assessment.get('risk_level', 'unknown'),
                'emergency_referral': 'YES' if report.emergency_alerts else 'NO',
                'ai_confidence': report.ai_confidence,
                'duration': 0.0,  # Would be extracted from report_data
                'models_used': 'Retinal Analysis AI, Glaucoma Detection AI'
            }
            
            # Add specific findings for retinal reports
            if report.report_type == 'retinal' and 'retinal_analysis' in report.detailed_findings:
                retinal_data = report.detailed_findings['retinal_analysis']
                
                dr_predictions = retinal_data['diabetic_retinopathy']['predictions']
                dr_classification = max(dr_predictions, key=dr_predictions.get)
                
                glaucoma_predictions = retinal_data['glaucoma_risk']['predictions']
                glaucoma_classification = max(glaucoma_predictions, key=glaucoma_predictions.get)
                
                variables.update({
                    'dr_classification': dr_classification,
                    'dr_confidence': retinal_data['diabetic_retinopathy']['confidence'],
                    'dr_risk_level': retinal_data['diabetic_retinopathy']['risk_level'],
                    'glaucoma_classification': glaucoma_classification,
                    'glaucoma_confidence': retinal_data['glaucoma_risk']['confidence'],
                    'cup_disc_ratio': retinal_data['landmarks']['cup_to_disc_ratio'],
                    'image_quality_score': retinal_data['image_quality']['overall_score'],
                    'image_quality_notes': retinal_data['image_quality']['quality_notes']
                })
            
            # Format recommendations
            rec_text = "\n".join([f"• {rec['description']}" for rec in report.recommendations])
            variables['recommendations'] = rec_text
            
            # Format follow-up plan
            follow_up_text = ""
            if report.follow_up_plan:
                for category, actions in report.follow_up_plan.items():
                    if actions:
                        follow_up_text += f"\n**{category.replace('_', ' ').title()}:**\n"
                        if isinstance(actions, list):
                            for action in actions:
                                if isinstance(action, dict):
                                    follow_up_text += f"• {action.get('action', action)}\n"
                                else:
                                    follow_up_text += f"• {action}\n"
            variables['follow_up_plan'] = follow_up_text
            
            # Format other complex fields
            variables['detailed_findings'] = json.dumps(report.detailed_findings, indent=2)
            variables['risk_assessment'] = json.dumps(report.risk_assessment, indent=2)
            
            # Generate report
            formatted_report = template.format(**variables)
            
            return formatted_report
            
        except Exception as e:
            logger.error(f"Failed to generate formatted report: {e}")
            return f"Error generating report: {str(e)}" 