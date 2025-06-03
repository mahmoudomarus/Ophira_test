"""
Report Organization Agent - Compiles and structures medical analysis results.

This agent is responsible for:
- Compiling findings from all specialized agents
- Structuring information for clarity and relevance  
- Prioritizing information based on medical significance
- Generating appropriate visualizations and summaries
- Creating comprehensive medical reports
- Formatting outputs for different audiences (patients, clinicians)
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from ..core.logging import get_logger
from .base import BaseAgent, AgentMessage, MessageType, MessagePriority, AgentCapability


class ReportType(Enum):
    """Types of medical reports."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    EMERGENCY = "emergency"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    PREVENTIVE = "preventive"


class ReportPriority(Enum):
    """Priority levels for report generation."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingSeverity(Enum):
    """Severity levels for medical findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NORMAL = "normal"


@dataclass
class Finding:
    """Individual medical finding."""
    finding_id: str
    source_agent: str
    finding_type: str
    severity: FindingSeverity
    confidence: float
    title: str
    description: str
    values: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReportSection:
    """Section within a medical report."""
    section_id: str
    title: str
    priority: int
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MedicalReport:
    """Comprehensive medical report."""
    report_id: str
    report_type: ReportType
    user_id: str
    session_id: str
    priority: ReportPriority
    
    # Content
    title: str
    executive_summary: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    overall_findings: List[Finding] = field(default_factory=list)
    key_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: Optional[datetime] = None
    generated_by: Set[str] = field(default_factory=set)
    data_sources: List[str] = field(default_factory=list)
    analysis_period: Optional[Dict[str, datetime]] = None
    
    # Status
    is_complete: bool = False
    requires_attention: bool = False
    emergency_flag: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ReportOrganizationAgent(BaseAgent):
    """Agent responsible for organizing and structuring medical reports."""
    
    def __init__(self):
        super().__init__(
            agent_id="report_organization",
            name="Report Organization Agent",
            agent_type="report_organization",
            capabilities=[
                AgentCapability.REPORT_GENERATION,
                AgentCapability.DATA_PROCESSING,
                AgentCapability.VALIDATION
            ]
        )
        
        self.logger = get_logger(f"Agent.{self.id}")
        
        # Report management
        self.active_reports: Dict[str, MedicalReport] = {}
        self.report_queue: List[str] = []
        self.finding_aggregation: Dict[str, List[Finding]] = {}
        
        # Report templates and configurations
        self.report_templates = {
            ReportType.SUMMARY: self._get_summary_template(),
            ReportType.DETAILED: self._get_detailed_template(),
            ReportType.EMERGENCY: self._get_emergency_template(),
            ReportType.TREND_ANALYSIS: self._get_trend_template(),
            ReportType.COMPARISON: self._get_comparison_template(),
            ReportType.PREVENTIVE: self._get_preventive_template()
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            "critical": {"confidence": 0.9, "immediate_attention": True},
            "high": {"confidence": 0.8, "attention_required": True},
            "moderate": {"confidence": 0.7, "monitoring_required": True},
            "low": {"confidence": 0.6, "informational": True},
            "normal": {"confidence": 0.5, "baseline": True}
        }
        
        # Background tasks
        self._report_processor_task: Optional[asyncio.Task] = None
        self._aggregation_monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info("Report Organization Agent initialized")
    
    async def _start_agent_tasks(self):
        """Start background tasks for report organization."""
        await super()._start_agent_tasks()
        
        # Start background tasks
        self._report_processor_task = asyncio.create_task(self._report_processor_loop())
        self._aggregation_monitor_task = asyncio.create_task(self._aggregation_monitor_loop())
        
        self.logger.info("Report Organization Agent tasks started")
    
    async def _cleanup(self):
        """Cleanup report organization agent."""
        # Cancel background tasks
        for task in [self._report_processor_task, self._aggregation_monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await super()._cleanup()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages."""
        content = message.content
        message_type = message.message_type
        
        if message_type == MessageType.REQUEST:
            if "generate_report" in content:
                await self._handle_report_generation_request(message)
            elif "add_findings" in content:
                await self._handle_findings_addition(message)
            elif "get_report_status" in content:
                await self._handle_report_status_request(message)
            elif "emergency_report" in content:
                await self._handle_emergency_report_request(message)
        
        elif message_type == MessageType.RESPONSE:
            if "analysis_result" in content:
                await self._handle_analysis_result(message)
        
        elif message_type == MessageType.NOTIFICATION:
            if content.get("type") == "findings_ready":
                await self._handle_findings_notification(message)
    
    async def _handle_report_generation_request(self, message: AgentMessage):
        """Handle request to generate a medical report."""
        content = message.content
        
        # Extract report parameters
        report_type = ReportType(content.get("report_type", "summary"))
        user_id = content.get("user_id")
        session_id = content.get("session_id")
        priority = ReportPriority(content.get("priority", "medium"))
        findings_data = content.get("findings", [])
        
        # Create report
        report = MedicalReport(
            report_id=f"report_{uuid.uuid4()}",
            report_type=report_type,
            user_id=user_id,
            session_id=session_id,
            priority=priority,
            title=self._generate_report_title(report_type, findings_data)
        )
        
        # Process any provided findings
        if findings_data:
            for finding_data in findings_data:
                finding = self._create_finding_from_data(finding_data)
                report.overall_findings.append(finding)
                report.generated_by.add(finding.source_agent)
        
        # Store report and queue for processing
        self.active_reports[report.report_id] = report
        self.report_queue.append(report.report_id)
        
        # Send response
        response = AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=message.priority,
            content={
                "status": "report_queued",
                "report_id": report.report_id,
                "estimated_completion": datetime.now() + timedelta(minutes=2),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.send_message(response)
        
        self.logger.info(f"Queued report generation for user {user_id}, type: {report_type}")
    
    async def _handle_findings_addition(self, message: AgentMessage):
        """Handle addition of new findings to existing report."""
        content = message.content
        report_id = content.get("report_id")
        findings_data = content.get("findings", [])
        
        if report_id in self.active_reports:
            report = self.active_reports[report_id]
            
            for finding_data in findings_data:
                finding = self._create_finding_from_data(finding_data)
                report.overall_findings.append(finding)
                report.generated_by.add(finding.source_agent)
            
            # Re-queue for processing with updated findings
            if report_id not in self.report_queue:
                self.report_queue.append(report_id)
            
            self.logger.info(f"Added {len(findings_data)} findings to report {report_id}")
    
    async def _handle_emergency_report_request(self, message: AgentMessage):
        """Handle urgent emergency report generation."""
        content = message.content
        
        # Create emergency report with highest priority
        report = MedicalReport(
            report_id=f"emergency_{uuid.uuid4()}",
            report_type=ReportType.EMERGENCY,
            user_id=content.get("user_id"),
            session_id=content.get("session_id"),
            priority=ReportPriority.CRITICAL,
            title="Emergency Health Assessment",
            emergency_flag=True,
            requires_attention=True
        )
        
        # Process emergency findings immediately
        findings_data = content.get("findings", [])
        for finding_data in findings_data:
            finding = self._create_finding_from_data(finding_data)
            if finding.severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]:
                report.overall_findings.append(finding)
                report.generated_by.add(finding.source_agent)
        
        # Generate emergency report immediately
        await self._generate_emergency_report(report)
        
        # Send immediate response
        response = AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.CRITICAL,
            content={
                "status": "emergency_report_generated",
                "report_id": report.report_id,
                "report": self._serialize_report(report),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.send_message(response)
        
        self.logger.critical(f"Generated emergency report for user {report.user_id}")
    
    async def _report_processor_loop(self):
        """Background loop for processing queued reports."""
        while True:
            try:
                if self.report_queue:
                    report_id = self.report_queue.pop(0)
                    
                    if report_id in self.active_reports:
                        report = self.active_reports[report_id]
                        await self._process_report(report)
                
                await asyncio.sleep(1)  # Process reports every second
                
            except Exception as e:
                self.logger.error(f"Error in report processor loop: {e}")
                await asyncio.sleep(5)
    
    async def _aggregation_monitor_loop(self):
        """Monitor and aggregate findings from multiple sources."""
        while True:
            try:
                # Check for findings that can be aggregated
                await self._check_finding_aggregation()
                
                # Clean up old completed reports
                await self._cleanup_old_reports()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in aggregation monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_report(self, report: MedicalReport):
        """Process and structure a medical report."""
        try:
            # Sort findings by severity and confidence
            report.overall_findings.sort(
                key=lambda f: (f.severity.value, f.confidence), 
                reverse=True
            )
            
            # Generate report sections based on type
            if report.report_type == ReportType.EMERGENCY:
                await self._generate_emergency_report(report)
            elif report.report_type == ReportType.SUMMARY:
                await self._generate_summary_report(report)
            elif report.report_type == ReportType.DETAILED:
                await self._generate_detailed_report(report)
            elif report.report_type == ReportType.TREND_ANALYSIS:
                await self._generate_trend_report(report)
            
            # Generate executive summary
            report.executive_summary = await self._generate_executive_summary(report)
            
            # Generate key recommendations
            report.key_recommendations = await self._generate_key_recommendations(report)
            
            # Check if report requires immediate attention
            report.requires_attention = self._requires_immediate_attention(report)
            
            # Mark as complete
            report.is_complete = True
            
            # Notify completion
            await self._notify_report_completion(report)
            
            self.logger.info(f"Completed processing report {report.report_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing report {report.report_id}: {e}")
    
    async def _generate_emergency_report(self, report: MedicalReport):
        """Generate an emergency medical report."""
        # Emergency Assessment Section
        emergency_section = ReportSection(
            section_id="emergency_assessment",
            title="Emergency Assessment",
            priority=1
        )
        
        critical_findings = [f for f in report.overall_findings 
                           if f.severity == FindingSeverity.CRITICAL]
        emergency_section.findings = critical_findings
        
        if critical_findings:
            emergency_section.summary = f"URGENT: {len(critical_findings)} critical findings requiring immediate attention."
        else:
            emergency_section.summary = "No critical emergency indicators detected."
        
        report.sections.append(emergency_section)
        
        # Immediate Actions Section
        actions_section = ReportSection(
            section_id="immediate_actions",
            title="Immediate Actions Required",
            priority=2
        )
        
        # Compile immediate recommendations
        immediate_actions = []
        for finding in critical_findings:
            immediate_actions.extend(finding.recommendations)
        
        actions_section.summary = "\n".join(f"• {action}" for action in immediate_actions[:5])
        report.sections.append(actions_section)
    
    async def _generate_summary_report(self, report: MedicalReport):
        """Generate a summary medical report."""
        # Overall Health Status
        status_section = ReportSection(
            section_id="health_status",
            title="Overall Health Status",
            priority=1
        )
        
        severity_counts = self._count_findings_by_severity(report.overall_findings)
        status_section.summary = self._format_severity_summary(severity_counts)
        report.sections.append(status_section)
        
        # Key Findings
        key_section = ReportSection(
            section_id="key_findings",
            title="Key Findings",
            priority=2
        )
        
        # Top 5 most significant findings
        key_findings = sorted(
            report.overall_findings,
            key=lambda f: (f.severity.value, f.confidence),
            reverse=True
        )[:5]
        
        key_section.findings = key_findings
        key_section.summary = f"Identified {len(key_findings)} significant findings for review."
        report.sections.append(key_section)
    
    async def _generate_detailed_report(self, report: MedicalReport):
        """Generate a detailed medical report."""
        # Group findings by type/system
        finding_groups = {}
        for finding in report.overall_findings:
            group = finding.finding_type
            if group not in finding_groups:
                finding_groups[group] = []
            finding_groups[group].append(finding)
        
        # Create section for each group
        priority = 1
        for group_name, findings in finding_groups.items():
            section = ReportSection(
                section_id=f"detailed_{group_name.lower()}",
                title=f"{group_name.title()} Analysis",
                priority=priority,
                findings=findings
            )
            
            # Generate detailed summary for this group
            section.summary = self._generate_group_summary(findings)
            report.sections.append(section)
            priority += 1
    
    async def _generate_trend_report(self, report: MedicalReport):
        """Generate a trend analysis report."""
        # Trends over time section
        trends_section = ReportSection(
            section_id="trend_analysis",
            title="Health Trends Analysis",
            priority=1
        )
        
        # Analyze trends in findings (placeholder - would use historical data)
        trends_section.summary = "Trend analysis based on recent health measurements and patterns."
        report.sections.append(trends_section)
    
    async def _generate_executive_summary(self, report: MedicalReport) -> str:
        """Generate executive summary for the report."""
        total_findings = len(report.overall_findings)
        severity_counts = self._count_findings_by_severity(report.overall_findings)
        
        summary_parts = [
            f"Health assessment completed with {total_findings} findings analyzed."
        ]
        
        if severity_counts.get("critical", 0) > 0:
            summary_parts.append(f"⚠️  {severity_counts['critical']} critical findings require immediate attention.")
        
        if severity_counts.get("high", 0) > 0:
            summary_parts.append(f"📋 {severity_counts['high']} high-priority findings for review.")
        
        if severity_counts.get("normal", 0) > 0:
            summary_parts.append(f"✅ {severity_counts['normal']} normal findings indicate good health status.")
        
        return " ".join(summary_parts)
    
    async def _generate_key_recommendations(self, report: MedicalReport) -> List[str]:
        """Generate key recommendations from all findings."""
        recommendations = []
        
        # Collect recommendations from high-severity findings
        for finding in report.overall_findings:
            if finding.severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]:
                recommendations.extend(finding.recommendations)
        
        # Remove duplicates and limit to top 5
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]
    
    def _create_finding_from_data(self, finding_data: Dict[str, Any]) -> Finding:
        """Create a Finding object from raw data."""
        return Finding(
            finding_id=finding_data.get("finding_id", f"finding_{uuid.uuid4()}"),
            source_agent=finding_data.get("source_agent", "unknown"),
            finding_type=finding_data.get("finding_type", "general"),
            severity=FindingSeverity(finding_data.get("severity", "normal")),
            confidence=finding_data.get("confidence", 0.5),
            title=finding_data.get("title", "Medical Finding"),
            description=finding_data.get("description", ""),
            values=finding_data.get("values", {}),
            recommendations=finding_data.get("recommendations", [])
        )
    
    def _count_findings_by_severity(self, findings: List[Finding]) -> Dict[str, int]:
        """Count findings by severity level."""
        counts = {}
        for finding in findings:
            severity = finding.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _format_severity_summary(self, severity_counts: Dict[str, int]) -> str:
        """Format severity counts into readable summary."""
        parts = []
        for severity, count in severity_counts.items():
            if count > 0:
                parts.append(f"{count} {severity}")
        
        return f"Assessment shows: {', '.join(parts)} findings."
    
    def _generate_group_summary(self, findings: List[Finding]) -> str:
        """Generate summary for a group of related findings."""
        if not findings:
            return "No significant findings in this category."
        
        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        severity_counts = self._count_findings_by_severity(findings)
        
        return f"Analysis of {len(findings)} findings with average confidence {avg_confidence:.2f}. " + \
               self._format_severity_summary(severity_counts)
    
    def _requires_immediate_attention(self, report: MedicalReport) -> bool:
        """Determine if report requires immediate attention."""
        for finding in report.overall_findings:
            if finding.severity == FindingSeverity.CRITICAL:
                return True
            if finding.severity == FindingSeverity.HIGH and finding.confidence > 0.8:
                return True
        return False
    
    def _serialize_report(self, report: MedicalReport) -> Dict[str, Any]:
        """Serialize report for transmission."""
        return {
            "report_id": report.report_id,
            "report_type": report.report_type.value,
            "title": report.title,
            "executive_summary": report.executive_summary,
            "sections": [
                {
                    "title": section.title,
                    "summary": section.summary,
                    "findings_count": len(section.findings),
                    "priority": section.priority
                }
                for section in report.sections
            ],
            "key_recommendations": report.key_recommendations,
            "total_findings": len(report.overall_findings),
            "requires_attention": report.requires_attention,
            "emergency_flag": report.emergency_flag,
            "created_at": report.created_at.isoformat() if report.created_at else None,
            "generated_by": list(report.generated_by)
        }
    
    async def _notify_report_completion(self, report: MedicalReport):
        """Notify interested parties of report completion."""
        # Notify supervisory agent
        notification = AgentMessage(
            sender_id=self.id,
            recipient_id="supervisory",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.HIGH if report.requires_attention else MessagePriority.NORMAL,
            content={
                "type": "report_completed",
                "report_id": report.report_id,
                "user_id": report.user_id,
                "session_id": report.session_id,
                "requires_attention": report.requires_attention,
                "emergency_flag": report.emergency_flag,
                "report_summary": self._serialize_report(report),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.send_message(notification)
    
    async def _check_finding_aggregation(self):
        """Check for findings that can be aggregated."""
        # Implementation for aggregating related findings
        # This would analyze findings across reports for patterns
        pass
    
    async def _cleanup_old_reports(self):
        """Clean up old completed reports."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        reports_to_remove = []
        for report_id, report in self.active_reports.items():
            if (report.is_complete and 
                report.created_at and 
                report.created_at < cutoff_time):
                reports_to_remove.append(report_id)
        
        for report_id in reports_to_remove:
            del self.active_reports[report_id]
            self.logger.info(f"Cleaned up old report {report_id}")
    
    def _generate_report_title(self, report_type: ReportType, findings_data: List[Dict]) -> str:
        """Generate appropriate title for report."""
        titles = {
            ReportType.SUMMARY: "Health Assessment Summary",
            ReportType.DETAILED: "Comprehensive Health Analysis",
            ReportType.EMERGENCY: "Emergency Health Assessment",
            ReportType.TREND_ANALYSIS: "Health Trends Report",
            ReportType.COMPARISON: "Comparative Health Analysis",
            ReportType.PREVENTIVE: "Preventive Health Recommendations"
        }
        
        base_title = titles.get(report_type, "Medical Report")
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        return f"{base_title} - {timestamp}"
    
    def _get_summary_template(self) -> Dict[str, Any]:
        """Get template for summary reports."""
        return {
            "sections": ["health_status", "key_findings", "recommendations"],
            "max_findings_per_section": 5,
            "include_trends": False
        }
    
    def _get_detailed_template(self) -> Dict[str, Any]:
        """Get template for detailed reports."""
        return {
            "sections": ["all_findings_by_type", "detailed_analysis", "correlations"],
            "max_findings_per_section": -1,  # No limit
            "include_trends": True
        }
    
    def _get_emergency_template(self) -> Dict[str, Any]:
        """Get template for emergency reports."""
        return {
            "sections": ["emergency_assessment", "immediate_actions", "critical_findings"],
            "max_findings_per_section": -1,
            "priority_filter": ["critical", "high"]
        }
    
    def _get_trend_template(self) -> Dict[str, Any]:
        """Get template for trend reports."""
        return {
            "sections": ["trend_analysis", "pattern_detection", "predictions"],
            "time_range": "30_days",
            "include_comparisons": True
        }
    
    def _get_comparison_template(self) -> Dict[str, Any]:
        """Get template for comparison reports."""
        return {
            "sections": ["baseline_comparison", "progress_tracking", "goal_assessment"],
            "comparison_periods": ["previous_week", "previous_month"]
        }
    
    def _get_preventive_template(self) -> Dict[str, Any]:
        """Get template for preventive reports."""
        return {
            "sections": ["risk_assessment", "preventive_measures", "lifestyle_recommendations"],
            "focus": "prevention",
            "include_education": True
        } 