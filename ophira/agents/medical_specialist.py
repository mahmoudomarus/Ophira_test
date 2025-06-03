"""
Medical Specialist Agent for Ophira
Specialized in ophthalmology analysis using VLLM medical models
"""
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from enum import Enum

from .base import BaseAgent, AgentMessage, MessageType, MessagePriority, AgentCapability, AgentStatus
from ..core.medical_models import (
    VLLMMedicalModelService, 
    MedicalAnalysisRequest, 
    MedicalAnalysisResult,
    AnalysisType,
    get_medical_model_service,
    analyze_medical_data
)
from ..core.config import get_settings
from ..core.logging import get_agent_logger

logger = get_agent_logger("medical_specialist")


@dataclass
class PatientData:
    """Patient data structure for medical analysis"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_medications: List[str]
    diabetes_status: bool = False
    hypertension_status: bool = False
    family_history: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.family_history is None:
            self.family_history = {
                "diabetes": False,
                "glaucoma": False,
                "macular_degeneration": False,
                "hypertension": False
            }


@dataclass
class MedicalAnalysisSession:
    """Session for tracking medical analysis progress"""
    session_id: str
    patient_id: str
    requested_analyses: List[AnalysisType]
    completed_analyses: Dict[AnalysisType, MedicalAnalysisResult]
    start_time: datetime
    priority_level: MessagePriority
    requesting_agent: str
    
    def __post_init__(self):
        if self.completed_analyses is None:
            self.completed_analyses = {}


class ClinicalSeverity(Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class RiskCategory(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ClinicalAlert:
    alert_id: str
    severity: ClinicalSeverity
    category: str
    message: str
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime
    patient_id: str
    triggered_by: Dict[str, any]

@dataclass
class VitalSignsAnalysis:
    heart_rate_analysis: Dict[str, any]
    blood_pressure_analysis: Dict[str, any]
    temperature_analysis: Dict[str, any]
    oxygen_saturation_analysis: Dict[str, any]
    overall_stability: str
    risk_indicators: List[str]
    mews_score: int  # Modified Early Warning Score

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for medical monitoring"""
    
    def __init__(self):
        self.logger = get_agent_logger("medical_specialist")
        self.risk_models = self._initialize_risk_models()
        
    def _initialize_risk_models(self):
        """Initialize various predictive models"""
        return {
            'sepsis_risk': SepsisRiskModel(),
            'cardiac_risk': CardiacRiskModel(),
            'deterioration_risk': DeteriorationRiskModel(),
            'fall_risk': FallRiskModel(),
            'medication_adherence': MedicationAdherenceModel()
        }
    
    def calculate_mews_score(self, vital_signs: Dict) -> int:
        """Calculate Modified Early Warning Score"""
        score = 0
        
        # Heart Rate scoring
        hr = vital_signs.get('heart_rate', 0)
        if hr <= 40 or hr >= 131:
            score += 3
        elif (41 <= hr <= 50) or (111 <= hr <= 130):
            score += 2
        elif (51 <= hr <= 100):
            score += 0
        elif (101 <= hr <= 110):
            score += 1
            
        # Blood Pressure scoring (systolic)
        bp_sys = vital_signs.get('blood_pressure_systolic', 0)
        if bp_sys <= 70:
            score += 3
        elif 71 <= bp_sys <= 80:
            score += 2
        elif 81 <= bp_sys <= 100:
            score += 1
        elif 101 <= bp_sys <= 199:
            score += 0
        elif bp_sys >= 200:
            score += 3
            
        # Temperature scoring
        temp = vital_signs.get('temperature', 0)
        if temp <= 35.0:
            score += 3
        elif 35.1 <= temp <= 36.0:
            score += 1
        elif 36.1 <= temp <= 37.9:
            score += 0
        elif 38.0 <= temp <= 38.9:
            score += 1
        elif temp >= 39.0:
            score += 3
            
        # Oxygen Saturation scoring
        spo2 = vital_signs.get('spo2', 0)
        if spo2 <= 91:
            score += 3
        elif 92 <= spo2 <= 93:
            score += 2
        elif 94 <= spo2 <= 95:
            score += 1
        elif spo2 >= 96:
            score += 0
            
        return score
    
    def predict_deterioration_risk(self, patient_data: Dict, time_horizon: int = 6) -> Dict:
        """Predict patient deterioration risk within specified hours"""
        try:
            vital_trends = self._analyze_vital_trends(patient_data)
            mews_score = self.calculate_mews_score(patient_data.get('current_vitals', {}))
            
            # Advanced deterioration prediction
            risk_factors = {
                'mews_score': mews_score,
                'vital_instability': vital_trends['instability_score'],
                'trend_direction': vital_trends['trend_direction'],
                'comorbidity_score': self._calculate_comorbidity_score(patient_data),
                'medication_interactions': self._check_medication_risks(patient_data)
            }
            
            # Calculate composite risk score
            risk_score = self._calculate_composite_risk(risk_factors)
            
            return {
                'risk_level': self._categorize_risk(risk_score),
                'risk_score': risk_score,
                'time_to_intervention': self._estimate_intervention_time(risk_score),
                'recommended_actions': self._generate_recommendations(risk_factors),
                'confidence': min(0.95, max(0.6, 0.8 + (mews_score * 0.02)))
            }
            
        except Exception as e:
            self.logger.error(f"Error in deterioration prediction: {str(e)}")
            return {'risk_level': 'unknown', 'error': str(e)}
    
    def _analyze_vital_trends(self, patient_data: Dict) -> Dict:
        """Analyze trends in vital signs over time"""
        vital_history = patient_data.get('vital_history', [])
        if len(vital_history) < 3:
            return {'instability_score': 0, 'trend_direction': 'stable'}
        
        # Calculate variability and trends
        hr_values = [v.get('heart_rate', 0) for v in vital_history[-10:]]
        bp_values = [v.get('blood_pressure_systolic', 0) for v in vital_history[-10:]]
        temp_values = [v.get('temperature', 0) for v in vital_history[-10:]]
        
        instability_score = (
            np.std(hr_values) / max(np.mean(hr_values), 1) +
            np.std(bp_values) / max(np.mean(bp_values), 1) +
            np.std(temp_values) / max(np.mean(temp_values), 1)
        ) * 100
        
        # Determine trend direction
        recent_avg = np.mean([hr_values[-3:], bp_values[-3:], temp_values[-3:]])
        earlier_avg = np.mean([hr_values[:3], bp_values[:3], temp_values[:3]])
        
        trend_direction = 'improving' if recent_avg < earlier_avg else 'deteriorating'
        
        return {
            'instability_score': min(100, instability_score),
            'trend_direction': trend_direction
        }

    def _calculate_comorbidity_score(self, patient_data: Dict) -> float:
        """Calculate comorbidity burden score"""
        history = patient_data.get('medical_history', {})
        comorbidities = history.get('conditions', [])
        
        # Simplified comorbidity scoring
        high_risk_conditions = ['diabetes', 'hypertension', 'cardiac', 'renal', 'respiratory']
        score = sum(1 for condition in comorbidities 
                   if any(risk in condition.lower() for risk in high_risk_conditions))
        
        return min(1.0, score / 5.0)
    
    def _check_medication_risks(self, patient_data: Dict) -> float:
        """Check for high-risk medications"""
        medications = patient_data.get('medications', [])
        high_risk_meds = ['warfarin', 'digoxin', 'insulin', 'opioids']
        
        risk_count = sum(1 for med in medications 
                        if any(risk_med in med.lower() for risk_med in high_risk_meds))
        
        return min(1.0, risk_count / 4.0)
    
    def _calculate_composite_risk(self, risk_factors: Dict) -> float:
        """Calculate composite risk score from multiple factors"""
        weights = {
            'mews_score': 0.4,
            'vital_instability': 0.2,
            'comorbidity_score': 0.2,
            'medication_interactions': 0.2
        }
        
        # Normalize MEWS score (0-15 scale to 0-1)
        normalized_mews = min(1.0, risk_factors.get('mews_score', 0) / 15.0)
        
        composite_score = (
            normalized_mews * weights['mews_score'] +
            risk_factors.get('vital_instability', 0) / 100 * weights['vital_instability'] +
            risk_factors.get('comorbidity_score', 0) * weights['comorbidity_score'] +
            risk_factors.get('medication_interactions', 0) * weights['medication_interactions']
        )
        
        return min(1.0, composite_score)
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'moderate'
        else:
            return 'low'
    
    def _estimate_intervention_time(self, risk_score: float) -> str:
        """Estimate time frame for intervention"""
        if risk_score >= 0.8:
            return 'Immediate (within 15 minutes)'
        elif risk_score >= 0.6:
            return 'Urgent (within 1 hour)'
        elif risk_score >= 0.4:
            return 'Soon (within 4 hours)'
        else:
            return 'Routine (within 24 hours)'
    
    def _generate_recommendations(self, risk_factors: Dict) -> List[str]:
        """Generate recommendations based on risk factors"""
        recommendations = []
        
        mews_score = risk_factors.get('mews_score', 0)
        if mews_score >= 6:
            recommendations.extend([
                "Immediate physician evaluation required",
                "Consider rapid response team activation",
                "Increase monitoring frequency to every 15 minutes"
            ])
        elif mews_score >= 4:
            recommendations.extend([
                "Notify physician within 30 minutes",
                "Increase monitoring to every 30 minutes",
                "Review patient condition and orders"
            ])
        
        if risk_factors.get('vital_instability', 0) > 50:
            recommendations.append("Stabilize vital signs - consider underlying causes")
        
        if risk_factors.get('medication_interactions', 0) > 0.5:
            recommendations.append("Review medication list for interactions and contraindications")
        
        return recommendations

class ClinicalDecisionSupport:
    """Advanced clinical decision support system"""
    
    def __init__(self):
        self.logger = get_agent_logger("medical_specialist")
        self.drug_database = self._load_drug_database()
        self.clinical_protocols = self._load_clinical_protocols()
        
    def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """Check for dangerous drug interactions"""
        interactions = []
        
        # Simplified drug interaction checking
        high_risk_combinations = {
            ('warfarin', 'aspirin'): {
                'severity': 'high',
                'risk': 'Increased bleeding risk',
                'recommendation': 'Monitor INR closely, consider dose adjustment'
            },
            ('digoxin', 'furosemide'): {
                'severity': 'moderate',
                'risk': 'Increased digoxin toxicity due to potassium loss',
                'recommendation': 'Monitor potassium and digoxin levels'
            }
        }
        
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                key = tuple(sorted([med1.lower(), med2.lower()]))
                if key in high_risk_combinations:
                    interactions.append({
                        'medications': [med1, med2],
                        **high_risk_combinations[key]
                    })
        
        return interactions
    
    def generate_care_recommendations(self, patient_data: Dict, analysis_results: Dict) -> List[str]:
        """Generate evidence-based care recommendations"""
        recommendations = []
        
        # Vital signs-based recommendations
        vitals = patient_data.get('current_vitals', {})
        
        if vitals.get('heart_rate', 0) > 100:
            recommendations.append("Consider tachycardia evaluation - check for fever, pain, dehydration, or cardiac issues")
        
        if vitals.get('blood_pressure_systolic', 0) > 160:
            recommendations.append("Hypertension detected - consider antihypertensive therapy and cardiovascular risk assessment")
        
        if vitals.get('spo2', 100) < 94:
            recommendations.append("Hypoxemia detected - provide supplemental oxygen, assess respiratory status")
        
        if vitals.get('temperature', 36.5) > 38.5:
            recommendations.append("Fever detected - consider infection workup, administer antipyretics if appropriate")
        
        # Risk-based recommendations
        risk_level = analysis_results.get('risk_level', 'low')
        if risk_level in ['high', 'critical']:
            recommendations.extend([
                "Increase monitoring frequency to every 15-30 minutes",
                "Consider immediate physician evaluation",
                "Ensure IV access is established",
                "Prepare for potential rapid response team activation"
            ])
        
        return recommendations

    def _load_drug_database(self) -> Dict:
        """Load drug interaction database"""
        # In a real implementation, this would load from a comprehensive database
        return {
            'interactions': {
                ('warfarin', 'aspirin'): {'severity': 'high', 'mechanism': 'Bleeding risk'},
                ('digoxin', 'furosemide'): {'severity': 'moderate', 'mechanism': 'Electrolyte imbalance'},
                ('simvastatin', 'clarithromycin'): {'severity': 'high', 'mechanism': 'Rhabdomyolysis risk'}
            },
            'contraindications': {
                'warfarin': ['active_bleeding', 'pregnancy'],
                'metformin': ['renal_failure', 'severe_heart_failure']
            }
        }
    
    def _load_clinical_protocols(self) -> Dict:
        """Load clinical care protocols"""
        return {
            'sepsis': {
                'criteria': ['fever', 'hypotension', 'altered_mental_status'],
                'actions': ['blood_cultures', 'antibiotics', 'fluids']
            },
            'cardiac_arrest': {
                'criteria': ['no_pulse', 'unconscious'],
                'actions': ['cpr', 'aed', 'epinephrine']
            },
            'stroke': {
                'criteria': ['facial_droop', 'arm_weakness', 'speech_difficulty'],
                'actions': ['ct_scan', 'neurology_consult', 'thrombolytics']
            }
        }

class MedicalSpecialistAgent(BaseAgent):
    """
    Specialized medical agent for analyzing health data using VLLM models
    Focuses on ophthalmology, cardiovascular, and metabolic analysis
    """
    
    def __init__(self):
        super().__init__(
            agent_id="medical_specialist_001",
            name="Dr. Ophira Medical Specialist",
            agent_type="medical_specialist",
            capabilities=[
                AgentCapability.MEDICAL_ANALYSIS,
                AgentCapability.DATA_PROCESSING,
                AgentCapability.REPORT_GENERATION,
                AgentCapability.VALIDATION,
                AgentCapability.EMERGENCY_RESPONSE
            ]
        )
        
        self.medical_model_service: Optional[VLLMMedicalModelService] = None
        self.active_sessions: Dict[str, MedicalAnalysisSession] = {}
        self.analysis_queue: List[MedicalAnalysisRequest] = []
        self.settings = get_settings()
        
        # Medical expertise areas
        self.specialties = {
            "ophthalmology": {
                "diabetic_retinopathy": AnalysisType.DIABETIC_RETINOPATHY,
                "glaucoma": AnalysisType.GLAUCOMA_DETECTION,
                "macular_degeneration": AnalysisType.MACULAR_DEGENERATION,
                "retinal_vessels": AnalysisType.RETINAL_VESSEL_ANALYSIS,
                "wavefront": AnalysisType.WAVEFRONT_ABERRATIONS
            },
            "cardiovascular": {
                "heart_rate_variability": AnalysisType.HEART_RATE_VARIABILITY,
                "blood_pressure": AnalysisType.BLOOD_PRESSURE_ANALYSIS
            },
            "metabolic": {
                "glucose_estimation": AnalysisType.GLUCOSE_ESTIMATION
            }
        }
        
        self.predictive_engine = PredictiveAnalyticsEngine()
        self.clinical_decision_support = ClinicalDecisionSupport()
        self.alert_thresholds = self._initialize_alert_thresholds()
    
    async def _start_agent_tasks(self):
        """Start agent-specific background tasks"""
        # Initialize medical model service
        self.medical_model_service = await get_medical_model_service()
        
        # Start analysis processing loop
        asyncio.create_task(self._process_analysis_queue())
        
        # Start session cleanup loop
        asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.log_decision("medical_specialist_startup", "Medical specialist agent initialized", 1.0)
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages for medical analysis requests"""
        try:
            if message.message_type == MessageType.REQUEST:
                await self._handle_analysis_request(message)
            elif message.message_type == MessageType.COMMAND:
                await self._handle_medical_command(message)
            else:
                await super()._handle_message(message)
                
        except Exception as e:
            logger.log_error("message_handling_error", str(e), {
                "message_id": message.id,
                "sender": message.sender_id
            })
    
    async def _handle_analysis_request(self, message: AgentMessage):
        """Handle medical analysis requests"""
        content = message.content
        
        # Extract analysis request details
        patient_id = content.get("patient_id")
        analysis_types = content.get("analysis_types", [])
        data = content.get("data")
        metadata = content.get("metadata", {})
        priority = MessagePriority(content.get("priority", MessagePriority.NORMAL.value))
        
        if not patient_id or not analysis_types or not data:
            await self._send_error_response(message, "Missing required analysis parameters")
            return
        
        # Create analysis session
        session_id = f"med_session_{patient_id}_{datetime.now().timestamp()}"
        session = MedicalAnalysisSession(
            session_id=session_id,
            patient_id=patient_id,
            requested_analyses=[AnalysisType(at) for at in analysis_types],
            completed_analyses={},
            start_time=datetime.now(),
            priority_level=priority,
            requesting_agent=message.sender_id
        )
        
        self.active_sessions[session_id] = session
        
        # Process each requested analysis
        for analysis_type_str in analysis_types:
            try:
                analysis_type = AnalysisType(analysis_type_str)
                
                # Create analysis request
                analysis_request = MedicalAnalysisRequest(
                    patient_id=patient_id,
                    analysis_type=analysis_type,
                    data=data,
                    metadata=metadata
                )
                
                # Add to processing queue with priority
                self.analysis_queue.append(analysis_request)
                
                logger.log_decision(
                    "analysis_queued",
                    f"Queued {analysis_type.value} analysis for patient {patient_id}",
                    0.9
                )
                
            except ValueError as e:
                logger.log_error("invalid_analysis_type", str(e), {"type": analysis_type_str})
        
        # Send acknowledgment response
        response = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=priority,
            content={
                "status": "accepted",
                "session_id": session_id,
                "analyses_queued": len(analysis_types),
                "estimated_completion": "2-5 minutes"
            },
            correlation_id=message.id
        )
        
        await self.send_message(response)
    
    async def _process_analysis_queue(self):
        """Process medical analysis requests from the queue"""
        while True:
            try:
                if self.analysis_queue:
                    # Sort queue by priority (emergency first)
                    self.analysis_queue.sort(key=lambda x: x.metadata.get('priority', 2), reverse=True)
                    
                    # Process next analysis
                    request = self.analysis_queue.pop(0)
                    await self._perform_medical_analysis(request)
                
                await asyncio.sleep(1)  # Brief pause between analyses
                
            except Exception as e:
                logger.log_error("analysis_queue_error", str(e), {})
                await asyncio.sleep(5)  # Longer pause on error
    
    async def _perform_medical_analysis(self, request: MedicalAnalysisRequest):
        """Perform the actual medical analysis using VLLM models"""
        start_time = datetime.now()
        
        try:
            logger.log_decision(
                "analysis_started",
                f"Starting {request.analysis_type.value} analysis for patient {request.patient_id}",
                0.9
            )
            
            # Perform analysis using medical model service
            result = await analyze_medical_data(request)
            
            # Update session with results
            session = self._find_session_for_patient(request.patient_id)
            if session:
                session.completed_analyses[request.analysis_type] = result
                
                # Check if all requested analyses are complete
                if len(session.completed_analyses) == len(session.requested_analyses):
                    await self._send_complete_analysis_report(session)
            
            # Log analysis completion
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.log_decision(
                "analysis_completed",
                f"Completed {request.analysis_type.value} analysis",
                result.confidence
            )
            
            # Send individual analysis result if high risk
            if result.risk_level in ["high", "critical"]:
                await self._send_urgent_analysis_alert(request, result, session)
        
        except Exception as e:
            logger.log_error("analysis_performance_error", str(e), {
                "patient_id": request.patient_id,
                "analysis_type": request.analysis_type.value
            })
    
    async def _send_complete_analysis_report(self, session: MedicalAnalysisSession):
        """Send complete analysis report when all analyses are finished"""
        # Generate comprehensive medical report
        report = await self._generate_medical_report(session)
        
        # Send to requesting agent
        response = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=session.requesting_agent,
            message_type=MessageType.RESPONSE,
            priority=session.priority_level,
            content={
                "status": "completed",
                "session_id": session.session_id,
                "patient_id": session.patient_id,
                "medical_report": report,
                "analysis_results": {
                    analysis_type.value: {
                        "confidence": result.confidence,
                        "risk_level": result.risk_level,
                        "findings": result.findings,
                        "recommendations": result.recommendations,
                        "requires_followup": result.requires_followup,
                        "specialist_referral": result.specialist_referral
                    }
                    for analysis_type, result in session.completed_analyses.items()
                }
            }
        )
        
        await self.send_message(response)
        
        # Clean up session
        del self.active_sessions[session.session_id]
        
        logger.log_decision(
            "complete_report_sent",
            f"Complete medical report sent for patient {session.patient_id}",
            0.95
        )
    
    async def _send_urgent_analysis_alert(self, request: MedicalAnalysisRequest, 
                                         result: MedicalAnalysisResult, 
                                         session: Optional[MedicalAnalysisSession]):
        """Send urgent alert for high-risk analysis results"""
        alert_message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=session.requesting_agent if session else "ophira_voice_agent",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.CRITICAL if result.risk_level == "critical" else MessagePriority.HIGH,
            content={
                "alert_type": "urgent_medical_finding",
                "patient_id": request.patient_id,
                "analysis_type": request.analysis_type.value,
                "risk_level": result.risk_level,
                "confidence": result.confidence,
                "findings": result.findings,
                "immediate_recommendations": result.recommendations[:3],  # Top 3 recommendations
                "specialist_referral": result.specialist_referral,
                "requires_immediate_action": result.requires_followup
            }
        )
        
        await self.send_message(alert_message)
        
        logger.log_decision(
            "urgent_alert_sent",
            f"Urgent alert sent for {request.analysis_type.value} - {result.risk_level} risk",
            result.confidence
        )
    
    async def _generate_medical_report(self, session: MedicalAnalysisSession) -> Dict[str, Any]:
        """Generate comprehensive medical report from analysis results"""
        report = {
            "patient_id": session.patient_id,
            "session_id": session.session_id,
            "analysis_date": session.start_time.isoformat(),
            "analyses_performed": len(session.completed_analyses),
            "overall_risk_assessment": "low",
            "critical_findings": [],
            "recommendations": [],
            "specialist_referrals": [],
            "follow_up_required": False,
            "summary": "",
            "detailed_results": {}
        }
        
        # Analyze results for overall assessment
        high_risk_count = 0
        critical_findings = []
        all_recommendations = set()
        referrals = set()
        
        for analysis_type, result in session.completed_analyses.items():
            # Add to detailed results
            report["detailed_results"][analysis_type.value] = {
                "confidence": result.confidence,
                "risk_level": result.risk_level,
                "findings": result.findings,
                "recommendations": result.recommendations,
                "processing_time": result.processing_time,
                "model_version": result.model_version
            }
            
            # Aggregate findings
            if result.risk_level in ["high", "critical"]:
                high_risk_count += 1
                critical_findings.extend([
                    f"{analysis_type.value}: {finding}" 
                    for finding in result.recommendations[:2]
                ])
            
            all_recommendations.update(result.recommendations)
            
            if result.specialist_referral:
                referrals.add(result.specialist_referral)
            
            if result.requires_followup:
                report["follow_up_required"] = True
        
        # Determine overall risk
        if high_risk_count >= 2:
            report["overall_risk_assessment"] = "high"
        elif high_risk_count == 1:
            report["overall_risk_assessment"] = "medium"
        
        report["critical_findings"] = critical_findings
        report["recommendations"] = list(all_recommendations)[:10]  # Top 10
        report["specialist_referrals"] = list(referrals)
        
        # Generate summary
        if high_risk_count > 0:
            report["summary"] = f"Medical analysis identified {high_risk_count} areas of concern requiring attention. Specialist consultation recommended."
        else:
            report["summary"] = "Medical analysis completed with no critical findings. Continue routine monitoring."
        
        return report
    
    async def _handle_medical_command(self, message: AgentMessage):
        """Handle medical-specific commands"""
        command = message.content.get("command")
        
        if command == "emergency_analysis":
            await self._handle_emergency_analysis(message)
        elif command == "get_session_status":
            await self._handle_session_status_request(message)
        elif command == "cancel_analysis":
            await self._handle_analysis_cancellation(message)
        else:
            await super()._handle_custom_command(message)
    
    async def _handle_emergency_analysis(self, message: AgentMessage):
        """Handle emergency medical analysis requests"""
        content = message.content
        
        # Emergency analysis gets highest priority
        emergency_request = MedicalAnalysisRequest(
            patient_id=content["patient_id"],
            analysis_type=AnalysisType(content["analysis_type"]),
            data=content["data"],
            metadata={**content.get("metadata", {}), "priority": MessagePriority.EMERGENCY.value}
        )
        
        # Insert at front of queue
        self.analysis_queue.insert(0, emergency_request)
        
        logger.log_decision(
            "emergency_analysis_queued",
            f"Emergency analysis queued for patient {content['patient_id']}",
            1.0
        )
    
    def _find_session_for_patient(self, patient_id: str) -> Optional[MedicalAnalysisSession]:
        """Find active session for a patient"""
        for session in self.active_sessions.values():
            if session.patient_id == patient_id:
                return session
        return None
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired analysis sessions"""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Sessions expire after 30 minutes
                    if current_time - session.start_time > timedelta(minutes=30):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    logger.log_decision("session_expired", f"Cleaned up expired session {session_id}", 0.8)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.log_error("session_cleanup_error", str(e), {})
                await asyncio.sleep(60)
    
    async def _send_error_response(self, original_message: AgentMessage, error_description: str):
        """Send error response to requesting agent"""
        error_response = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.ERROR,
            priority=original_message.priority,
            content={
                "error": error_description,
                "original_message_id": original_message.id
            },
            correlation_id=original_message.id
        )
        
        await self.send_message(error_response)
    
    async def _cleanup(self):
        """Cleanup agent resources"""
        if self.medical_model_service:
            await self.medical_model_service.cleanup()
        
        # Clear active sessions
        self.active_sessions.clear()
        self.analysis_queue.clear()
        
        logger.log_decision("medical_specialist_cleanup", "Medical specialist agent cleaned up", 1.0)
    
    def _initialize_alert_thresholds(self) -> Dict:
        """Initialize clinical alert thresholds"""
        return {
            'heart_rate': {'low': 50, 'high': 120, 'critical_low': 40, 'critical_high': 150},
            'blood_pressure_systolic': {'low': 90, 'high': 160, 'critical_low': 70, 'critical_high': 200},
            'temperature': {'low': 35.5, 'high': 38.0, 'critical_low': 35.0, 'critical_high': 39.5},
            'spo2': {'low': 94, 'critical_low': 90},
            'mews_score': {'moderate': 4, 'high': 6, 'critical': 8}
        }
    
    async def analyze_patient_comprehensively(self, patient_data: Dict) -> Dict:
        """Comprehensive patient analysis with advanced features"""
        try:
            analysis_start = datetime.now()
            
            # Basic vital signs analysis
            vital_analysis = self._analyze_vital_signs_advanced(patient_data)
            
            # Predictive analytics
            deterioration_risk = self.predictive_engine.predict_deterioration_risk(patient_data)
            
            # Clinical decision support
            drug_interactions = self.clinical_decision_support.check_drug_interactions(
                patient_data.get('medications', [])
            )
            
            care_recommendations = self.clinical_decision_support.generate_care_recommendations(
                patient_data, {'risk_level': deterioration_risk.get('risk_level')}
            )
            
            # Generate clinical alerts
            alerts = self._generate_clinical_alerts(patient_data, vital_analysis, deterioration_risk)
            
            # Calculate overall patient status
            overall_status = self._calculate_overall_status(vital_analysis, deterioration_risk)
            
            analysis_result = {
                'analysis_timestamp': analysis_start.isoformat(),
                'patient_id': patient_data.get('patient_id'),
                'vital_signs_analysis': {
                    'heart_rate_analysis': vital_analysis.heart_rate_analysis,
                    'blood_pressure_analysis': vital_analysis.blood_pressure_analysis,
                    'temperature_analysis': vital_analysis.temperature_analysis,
                    'oxygen_saturation_analysis': vital_analysis.oxygen_saturation_analysis,
                    'overall_stability': vital_analysis.overall_stability,
                    'risk_indicators': vital_analysis.risk_indicators,
                    'mews_score': vital_analysis.mews_score
                },
                'deterioration_risk': deterioration_risk,
                'drug_interactions': drug_interactions,
                'care_recommendations': care_recommendations,
                'clinical_alerts': [alert.__dict__ for alert in alerts],
                'overall_status': overall_status,
                'confidence_score': self._calculate_confidence_score(vital_analysis, deterioration_risk),
                'next_assessment_time': (analysis_start + timedelta(minutes=15)).isoformat()
            }
            
            self.logger.info(f"Comprehensive analysis completed for patient {patient_data.get('patient_id')}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive patient analysis: {str(e)}")
            return {'error': str(e), 'status': 'analysis_failed'}
    
    def _analyze_vital_signs_advanced(self, patient_data: Dict) -> VitalSignsAnalysis:
        """Advanced vital signs analysis with clinical interpretation"""
        vitals = patient_data.get('current_vitals', {})
        history = patient_data.get('vital_history', [])
        
        # Calculate MEWS score
        mews_score = self.predictive_engine.calculate_mews_score(vitals)
        
        # Analyze each vital sign
        hr_analysis = self._analyze_heart_rate(vitals.get('heart_rate'), history)
        bp_analysis = self._analyze_blood_pressure(vitals.get('blood_pressure_systolic'), 
                                                  vitals.get('blood_pressure_diastolic'), history)
        temp_analysis = self._analyze_temperature(vitals.get('temperature'), history)
        spo2_analysis = self._analyze_oxygen_saturation(vitals.get('spo2'), history)
        
        # Determine overall stability
        stability_indicators = [
            hr_analysis['stability'],
            bp_analysis['stability'],
            temp_analysis['stability'],
            spo2_analysis['stability']
        ]
        
        stable_count = sum(1 for s in stability_indicators if s == 'stable')
        if stable_count >= 3:
            overall_stability = 'stable'
        elif stable_count >= 2:
            overall_stability = 'unstable'
        else:
            overall_stability = 'critical'
        
        # Identify risk indicators
        risk_indicators = []
        if mews_score >= 4:
            risk_indicators.append(f"Elevated MEWS score: {mews_score}")
        if hr_analysis['risk_level'] != 'normal':
            risk_indicators.append(f"Heart rate concern: {hr_analysis['interpretation']}")
        if bp_analysis['risk_level'] != 'normal':
            risk_indicators.append(f"Blood pressure concern: {bp_analysis['interpretation']}")
        
        return VitalSignsAnalysis(
            heart_rate_analysis=hr_analysis,
            blood_pressure_analysis=bp_analysis,
            temperature_analysis=temp_analysis,
            oxygen_saturation_analysis=spo2_analysis,
            overall_stability=overall_stability,
            risk_indicators=risk_indicators,
            mews_score=mews_score
        )
    
    def _generate_clinical_alerts(self, patient_data: Dict, vital_analysis: VitalSignsAnalysis, 
                                 deterioration_risk: Dict) -> List[ClinicalAlert]:
        """Generate clinical alerts based on analysis"""
        alerts = []
        patient_id = patient_data.get('patient_id', 'unknown')
        
        # MEWS score alerts
        if vital_analysis.mews_score >= 8:
            alerts.append(ClinicalAlert(
                alert_id=f"MEWS_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                severity=ClinicalSeverity.CRITICAL,
                category="Early Warning Score",
                message=f"Critical MEWS score: {vital_analysis.mews_score}",
                recommendations=[
                    "Immediate physician evaluation required",
                    "Consider rapid response team activation",
                    "Increase monitoring to continuous"
                ],
                confidence_score=0.95,
                timestamp=datetime.now(),
                patient_id=patient_id,
                triggered_by={'mews_score': vital_analysis.mews_score}
            ))
        elif vital_analysis.mews_score >= 6:
            alerts.append(ClinicalAlert(
                alert_id=f"MEWS_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                severity=ClinicalSeverity.SEVERE,
                category="Early Warning Score",
                message=f"High MEWS score: {vital_analysis.mews_score}",
                recommendations=[
                    "Notify physician within 30 minutes",
                    "Increase monitoring frequency",
                    "Review patient condition"
                ],
                confidence_score=0.90,
                timestamp=datetime.now(),
                patient_id=patient_id,
                triggered_by={'mews_score': vital_analysis.mews_score}
            ))
        
        # Deterioration risk alerts
        risk_level = deterioration_risk.get('risk_level', 'low')
        if risk_level in ['high', 'critical']:
            severity = ClinicalSeverity.CRITICAL if risk_level == 'critical' else ClinicalSeverity.SEVERE
            alerts.append(ClinicalAlert(
                alert_id=f"RISK_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                severity=severity,
                category="Deterioration Risk",
                message=f"High risk of patient deterioration detected",
                recommendations=deterioration_risk.get('recommended_actions', []),
                confidence_score=deterioration_risk.get('confidence', 0.8),
                timestamp=datetime.now(),
                patient_id=patient_id,
                triggered_by={'risk_score': deterioration_risk.get('risk_score')}
            ))
        
        return alerts
    
    def _calculate_overall_status(self, vital_analysis: VitalSignsAnalysis, deterioration_risk: Dict) -> str:
        """Calculate overall patient status based on analysis results"""
        risk_level = deterioration_risk.get('risk_level', 'low')
        if risk_level in ['high', 'critical']:
            return 'Deteriorating'
        else:
            return 'Stable'
    
    def _calculate_confidence_score(self, vital_analysis: VitalSignsAnalysis, deterioration_risk: Dict) -> float:
        """Calculate overall confidence score based on analysis results"""
        risk_level = deterioration_risk.get('risk_level', 'low')
        if risk_level in ['high', 'critical']:
            return 0.8
        else:
            return 0.95

    def _analyze_heart_rate(self, heart_rate: Optional[float], history: List[Dict]) -> Dict:
        """Analyze heart rate with clinical interpretation"""
        if heart_rate is None:
            return {'value': None, 'interpretation': 'Not available', 'risk_level': 'unknown', 'stability': 'unknown'}
        
        interpretation = ""
        risk_level = "normal"
        stability = "stable"
        
        if heart_rate < 50:
            interpretation = "Bradycardia - potentially concerning"
            risk_level = "moderate" if heart_rate > 40 else "high"
            stability = "unstable"
        elif heart_rate > 100:
            interpretation = "Tachycardia - investigate underlying cause"
            risk_level = "moderate" if heart_rate < 120 else "high"
            stability = "unstable"
        else:
            interpretation = "Normal heart rate"
        
        # Check trend if history available
        if len(history) >= 3:
            recent_hrs = [h.get('heart_rate') for h in history[-3:] if h.get('heart_rate')]
            if len(recent_hrs) >= 2:
                trend = "increasing" if recent_hrs[-1] > recent_hrs[0] else "decreasing"
                if abs(recent_hrs[-1] - recent_hrs[0]) > 20:
                    stability = "unstable"
        
        return {
            'value': heart_rate,
            'interpretation': interpretation,
            'risk_level': risk_level,
            'stability': stability,
            'normal_range': '60-100 bpm'
        }
    
    def _analyze_blood_pressure(self, systolic: Optional[float], diastolic: Optional[float], history: List[Dict]) -> Dict:
        """Analyze blood pressure with clinical interpretation"""
        if systolic is None or diastolic is None:
            return {'systolic': systolic, 'diastolic': diastolic, 'interpretation': 'Not available', 
                   'risk_level': 'unknown', 'stability': 'unknown'}
        
        interpretation = ""
        risk_level = "normal"
        stability = "stable"
        
        # Classify blood pressure
        if systolic < 90 or diastolic < 60:
            interpretation = "Hypotension - assess for shock or medication effects"
            risk_level = "high"
            stability = "unstable"
        elif systolic >= 180 or diastolic >= 110:
            interpretation = "Hypertensive crisis - immediate evaluation needed"
            risk_level = "high"
            stability = "critical"
        elif systolic >= 160 or diastolic >= 100:
            interpretation = "Hypertension - consider antihypertensive therapy"
            risk_level = "moderate"
        elif systolic >= 140 or diastolic >= 90:
            interpretation = "Elevated blood pressure - monitor closely"
            risk_level = "low"
        else:
            interpretation = "Normal blood pressure"
        
        return {
            'systolic': systolic,
            'diastolic': diastolic,
            'interpretation': interpretation,
            'risk_level': risk_level,
            'stability': stability,
            'category': self._categorize_bp(systolic, diastolic)
        }
    
    def _analyze_temperature(self, temperature: Optional[float], history: List[Dict]) -> Dict:
        """Analyze temperature with clinical interpretation"""
        if temperature is None:
            return {'value': None, 'interpretation': 'Not available', 'risk_level': 'unknown', 'stability': 'unknown'}
        
        interpretation = ""
        risk_level = "normal"
        stability = "stable"
        
        if temperature < 35.0:
            interpretation = "Hypothermia - severe concern"
            risk_level = "high"
            stability = "critical"
        elif temperature < 36.0:
            interpretation = "Mild hypothermia - monitor closely"
            risk_level = "moderate"
            stability = "unstable"
        elif temperature > 39.0:
            interpretation = "High fever - investigate infection, consider cooling measures"
            risk_level = "high"
            stability = "unstable"
        elif temperature > 38.0:
            interpretation = "Fever - possible infection"
            risk_level = "moderate"
        else:
            interpretation = "Normal temperature"
        
        return {
            'value': temperature,
            'interpretation': interpretation,
            'risk_level': risk_level,
            'stability': stability,
            'unit': '°C',
            'normal_range': '36.1-37.9°C'
        }
    
    def _analyze_oxygen_saturation(self, spo2: Optional[float], history: List[Dict]) -> Dict:
        """Analyze oxygen saturation with clinical interpretation"""
        if spo2 is None:
            return {'value': None, 'interpretation': 'Not available', 'risk_level': 'unknown', 'stability': 'unknown'}
        
        interpretation = ""
        risk_level = "normal"
        stability = "stable"
        
        if spo2 < 90:
            interpretation = "Severe hypoxemia - immediate oxygen therapy required"
            risk_level = "high"
            stability = "critical"
        elif spo2 < 94:
            interpretation = "Hypoxemia - supplemental oxygen indicated"
            risk_level = "moderate"
            stability = "unstable"
        elif spo2 < 96:
            interpretation = "Mild hypoxemia - monitor closely"
            risk_level = "low"
        else:
            interpretation = "Normal oxygen saturation"
        
        return {
            'value': spo2,
            'interpretation': interpretation,
            'risk_level': risk_level,
            'stability': stability,
            'unit': '%',
            'normal_range': '≥96%'
        }
    
    def _categorize_bp(self, systolic: float, diastolic: float) -> str:
        """Categorize blood pressure according to clinical guidelines"""
        if systolic < 90 or diastolic < 60:
            return "Hypotension"
        elif systolic >= 180 or diastolic >= 110:
            return "Hypertensive Crisis"
        elif systolic >= 160 or diastolic >= 100:
            return "Stage 2 Hypertension"
        elif systolic >= 140 or diastolic >= 90:
            return "Stage 1 Hypertension"
        elif systolic >= 130 or diastolic >= 80:
            return "Elevated"
        else:
            return "Normal"

# Add new risk model classes
class SepsisRiskModel:
    """Sepsis risk prediction model"""
    
    def predict_risk(self, patient_data: Dict) -> float:
        # Simplified sepsis risk calculation based on qSOFA
        vitals = patient_data.get('current_vitals', {})
        
        risk_score = 0
        if vitals.get('blood_pressure_systolic', 120) <= 100:
            risk_score += 1
        if vitals.get('heart_rate', 70) >= 90:
            risk_score += 1
        if vitals.get('temperature', 36.5) >= 38.0 or vitals.get('temperature', 36.5) <= 36.0:
            risk_score += 1
            
        return min(1.0, risk_score / 3.0)

class CardiacRiskModel:
    """Cardiac event risk prediction"""
    
    def predict_risk(self, patient_data: Dict) -> float:
        vitals = patient_data.get('current_vitals', {})
        history = patient_data.get('medical_history', {})
        
        risk_factors = 0
        if vitals.get('heart_rate', 70) > 100 or vitals.get('heart_rate', 70) < 50:
            risk_factors += 1
        if vitals.get('blood_pressure_systolic', 120) > 160:
            risk_factors += 1
        if history.get('cardiac_history', False):
            risk_factors += 2
            
        return min(1.0, risk_factors / 4.0)

class DeteriorationRiskModel:
    """General patient deterioration risk"""
    
    def predict_risk(self, patient_data: Dict) -> float:
        # Implementation would use machine learning model
        # For now, simplified rule-based approach
        vital_analysis = patient_data.get('vital_analysis', {})
        mews_score = vital_analysis.get('mews_score', 0)
        
        return min(1.0, mews_score / 10.0)

class FallRiskModel:
    """Fall risk assessment"""
    
    def predict_risk(self, patient_data: Dict) -> float:
        # Simplified fall risk based on age, medications, mobility
        age = patient_data.get('age', 0)
        medications = patient_data.get('medications', [])
        
        risk_score = 0
        if age > 65:
            risk_score += 0.3
        if any('sedative' in med.lower() for med in medications):
            risk_score += 0.2
        if any('antihypertensive' in med.lower() for med in medications):
            risk_score += 0.1
            
        return min(1.0, risk_score)

class MedicationAdherenceModel:
    """Medication adherence prediction"""
    
    def predict_adherence(self, patient_data: Dict) -> float:
        # Simplified adherence prediction
        medication_count = len(patient_data.get('medications', []))
        age = patient_data.get('age', 0)
        
        base_adherence = 0.8
        if medication_count > 5:
            base_adherence -= 0.1
        if age > 75:
            base_adherence -= 0.1
            
        return max(0.3, base_adherence) 