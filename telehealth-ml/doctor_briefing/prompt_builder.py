"""
Telehealth ML — Prompt Builder for Doctor Briefings

Constructs structured, deterministic LLM prompts from aggregated patient data
and trend insights.  Output prompts are designed for concise clinical summaries.

Safety guardrails:
  • The prompt explicitly instructs the LLM NOT to diagnose
  • The prompt explicitly instructs the LLM NOT to recommend medication
  • A disclaimer is always appended to the output
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SAFETY_DISCLAIMER = (
    "⚕️ DISCLAIMER: This summary is AI-generated and should support, "
    "not replace, clinical judgment. All clinical decisions must be made "
    "by qualified healthcare professionals."
)

SYSTEM_PROMPT = """You are a clinical AI assistant generating concise doctor briefings for telehealth consultations.

STRICT RULES:
1. Do NOT diagnose any condition.
2. Do NOT recommend any treatment or medication.
3. Focus ONLY on observed data patterns and measured vital sign trends.
4. Highlight abnormalities factually using the provided data.
5. Keep the summary under 150 words.
6. Use professional medical terminology.
7. Be direct and concise — the doctor has 30 seconds to read this.

STRUCTURE your response in exactly 3 paragraphs:
1. **Overall Status**: Patient's general condition based on vital signs over the monitoring period.
2. **Key Abnormalities**: Specific anomalies, out-of-range readings, and concerning trends.
3. **Urgency Assessment**: Data-driven urgency level — do NOT prescribe action, only state urgency."""


class PromptBuilder:
    """
    Constructs deterministic LLM prompts from aggregated stats and trend analysis.
    """

    def __init__(self):
        logger.info("PromptBuilder initialized")

    def build(
        self,
        aggregation: dict,
        trends: dict,
        top_alerts: Optional[list] = None,
        rag_context: Optional[list] = None,
    ) -> dict:
        """
        Build a complete prompt payload for the LLM.

        Parameters
        ----------
        aggregation : dict
            Output from DataAggregator.aggregate().
        trends : dict
            Output from TrendAnalyzer.analyze().
        top_alerts : list, optional
            Top N alert dicts to include as context.
        rag_context : list, optional
            Documents retrieved from VitalsVectorStore semantic search.

        Returns
        -------
        dict with keys: system_prompt, user_prompt, metadata
        """
        patient_id = aggregation.get("patientId", "UNKNOWN")
        user_prompt = self._build_user_prompt(
            aggregation, trends, top_alerts or [], rag_context or []
        )

        logger.info("Prompt built for patient %s (%d chars)", patient_id, len(user_prompt))

        return {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "metadata": {
                "patientId": patient_id,
                "totalReadings": aggregation.get("totalReadings", 0),
                "anomalyCount": aggregation.get("anomalyCount", 0),
                "promptLength": len(user_prompt),
            },
        }

    def _build_user_prompt(
        self, agg: dict, trends: dict, alerts: list, rag_context: list
    ) -> str:
        """Assemble the user-facing data portion of the prompt."""
        sections = []

        # ── Header ──
        sections.append(
            f"Generate a concise clinical briefing for Patient {agg.get('patientId', 'UNKNOWN')}.\n"
            f"Monitoring window: {agg.get('windowStart', 'N/A')} to {agg.get('windowEnd', 'N/A')}\n"
            f"Total readings: {agg.get('totalReadings', 0)}"
        )

        # ── Vital Sign Summary ──
        vitals = agg.get("vitalRanges", {})
        if vitals:
            lines = ["VITAL SIGN RANGES:"]
            for vital, label, unit in [
                ("heartRate", "Heart Rate", "bpm"),
                ("spo2", "SpO2", "%"),
                ("systolicBP", "Systolic BP", "mmHg"),
                ("diastolicBP", "Diastolic BP", "mmHg"),
                ("temperature", "Temperature", "°C"),
            ]:
                stats = vitals.get(vital, {})
                if stats.get("min") is not None:
                    lines.append(
                        f"  {label}: min={stats['min']}{unit}, "
                        f"max={stats['max']}{unit}, mean={stats['mean']}{unit}"
                    )
            sections.append("\n".join(lines))

        # ── Risk Summary ──
        risk_dist = agg.get("riskDistribution", {})
        sections.append(
            "RISK ANALYSIS:\n"
            f"  Distribution: {json.dumps(risk_dist)}\n"
            f"  Max risk score: {agg.get('maxRiskScore', 0)}\n"
            f"  Mean risk score: {agg.get('meanRiskScore', 0)}\n"
            f"  Anomalies detected: {agg.get('anomalyCount', 0)}\n"
            f"  Peak instability at: {agg.get('peakRiskTime', 'N/A')}\n"
            f"  Fever events: {agg.get('feverEvents', 0)}"
        )

        # ── Trend Insights ──
        insights = trends.get("insights", [])
        if insights:
            lines = ["TREND INSIGHTS:"]
            for i, insight in enumerate(insights, 1):
                lines.append(f"  {i}. {insight}")

            # Add specific trend details
            hr_trend = trends.get("hrTrend", {})
            if hr_trend.get("interpretation"):
                lines.append(f"  • {hr_trend['interpretation']}")

            spo2_trend = trends.get("spo2Trend", {})
            if spo2_trend.get("interpretation"):
                lines.append(f"  • {spo2_trend['interpretation']}")

            bp_vol = trends.get("bpVolatility", {})
            if bp_vol.get("interpretation"):
                lines.append(f"  • {bp_vol['interpretation']}")

            escalation = trends.get("riskEscalation", {})
            if escalation.get("interpretation"):
                lines.append(f"  • {escalation['interpretation']}")

            sections.append("\n".join(lines))

        # ── Critical Events ──
        critical = agg.get("criticalEvents", [])
        if critical:
            lines = [f"CRITICAL EVENTS ({len(critical)} total):"]
            for event in critical[:5]:  # top 5
                lines.append(
                    f"  [{event.get('riskLevel')}] at {event.get('timestamp')}: "
                    f"{', '.join(event.get('reasons', []))}"
                )
            sections.append("\n".join(lines))

        # ── Recent Alerts ──
        if alerts:
            lines = [f"RECENT ALERTS ({len(alerts)} shown):"]
            for alert in alerts[:5]:
                lines.append(
                    f"  [{alert.get('riskLevel', 'N/A')}] "
                    f"{', '.join(alert.get('reasons', []))}"
                )
            sections.append("\n".join(lines))

        # ── RAG Retrieved Context ──
        if rag_context:
            lines = [f"RAG RETRIEVED CONTEXT (top {min(len(rag_context), 5)} semantically similar anomalies):"]
            for i, doc in enumerate(rag_context[:5], 1):
                lines.append(f"  [{i}] {doc.get('text', '')}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


    @staticmethod
    def get_safety_disclaimer() -> str:
        """Return the mandatory safety disclaimer."""
        return SAFETY_DISCLAIMER
