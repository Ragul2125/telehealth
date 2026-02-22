"""
Telehealth ML — Alert Engine

Consumes inference output and produces structured, human-readable alert
JSON objects ready for downstream consumers (notification service, GenAI
triage agent, DynamoDB persistence).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from config import RISK_LEVEL_THRESHOLDS

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Stateless alert generator.

    Takes an inference result dict and decides whether to emit an alert.
    Alerts are structured JSON payloads conforming to the platform schema.
    """

    # Minimum risk level that triggers an alert dispatch
    ALERT_DISPATCH_LEVELS = {"MODERATE", "HIGH", "CRITICAL"}

    def __init__(self):
        logger.info("AlertEngine initialized")

    def evaluate(self, inference_result: dict) -> Optional[dict]:
        """
        Evaluate an inference result and return an alert dict if warranted.

        Returns None for LOW-risk results (no alert needed).
        """
        risk_level = inference_result.get("riskLevel", "LOW")
        anomaly_detected = inference_result.get("anomalyDetected", False)

        if risk_level not in self.ALERT_DISPATCH_LEVELS and not anomaly_detected:
            logger.debug(
                "No alert for patient %s (risk=%s)",
                inference_result.get("patientId"), risk_level,
            )
            return None

        alert = self._build_alert(inference_result)
        logger.info(
            "ALERT generated for patient %s — %s",
            alert["patientId"], alert["riskLevel"],
        )
        return alert

    def evaluate_batch(self, results: list) -> list:
        """Evaluate a batch of inference results and return all generated alerts."""
        alerts = []
        for result in results:
            alert = self.evaluate(result)
            if alert is not None:
                alerts.append(alert)
        logger.info(
            "Batch alert evaluation: %d alerts from %d results",
            len(alerts), len(results),
        )
        return alerts

    # ── alert construction ────────────────────────────────────

    def _build_alert(self, result: dict) -> dict:
        """Build a structured alert payload."""
        timestamp = result.get("timestamp", datetime.now(timezone.utc).isoformat())

        return {
            "patientId": result.get("patientId", "UNKNOWN"),
            "riskLevel": result.get("riskLevel", "LOW"),
            "anomalyDetected": result.get("anomalyDetected", False),
            "reasons": result.get("reasons", []),
            "timestamp": timestamp,
            # Extended fields for DynamoDB / GenAI triage agent
            "combinedRiskScore": result.get("combinedRiskScore", 0.0),
            "vitals": result.get("vitals", {}),
            "alertType": self._classify_alert_type(result.get("reasons", [])),
            "requiresImmediateAttention": result.get("riskLevel") in ("HIGH", "CRITICAL"),
        }

    @staticmethod
    def _classify_alert_type(reasons: list) -> str:
        """Derive a primary alert category from the reasons list."""
        text = " ".join(reasons).lower()
        if "critical hypoxia" in text:
            return "RESPIRATORY_CRITICAL"
        if "hypoxia" in text:
            return "RESPIRATORY"
        if "tachycardia" in text or "bradycardia" in text:
            return "CARDIAC"
        if "hypertension" in text:
            return "CARDIOVASCULAR"
        if "fever" in text or "hypothermia" in text:
            return "THERMOREGULATORY"
        if "ml anomaly" in text:
            return "ML_DETECTED"
        return "GENERAL"

    @staticmethod
    def format_json(alert: dict) -> str:
        """Pretty-print alert as JSON string."""
        return json.dumps(alert, indent=2, default=str)
