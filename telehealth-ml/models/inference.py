"""
Telehealth ML — Inference Engine

Stateless inference module.  Loads a trained model bundle once, then
provides fast single-row predictions combining IsolationForest anomaly
scores with the rule-based risk engine.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    RISK_LEVEL_THRESHOLDS,
    SAVED_MODEL_PATH,
)
from features.feature_engineering import FeatureEngineer
from models.train import RuleBasedRiskEngine

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Production inference engine.

    • Loads model bundle from disk once at init.
    • Per-call: engineers features → IF score → rule engine → combined risk.
    • Returns DynamoDB-compatible JSON dict.
    • Target latency: < 200 ms per call.
    """

    def __init__(self, model_path=SAVED_MODEL_PATH):
        logger.info("Loading model bundle from %s", model_path)
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_means = bundle["feature_means"]
        self.feature_stds = bundle["feature_stds"]
        self.feature_columns = bundle["feature_columns"]
        self.feature_engineer = FeatureEngineer()
        self.rule_engine = RuleBasedRiskEngine()
        logger.info("InferenceEngine ready")

    # ── public API ────────────────────────────────────────────

    def predict(
        self,
        vitals: dict,
        history: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Run full inference on a single vitals reading.

        Parameters
        ----------
        vitals : dict
            Must contain: patient_id, heart_rate, spo2, systolic_bp,
            diastolic_bp, temperature.  Optionally timestamp.
        history : pd.DataFrame, optional
            Recent history for this patient (for rolling features).

        Returns
        -------
        dict : DynamoDB-compatible result with alert fields.
        """
        t0 = time.perf_counter()

        # 1. Feature engineering
        features = self.feature_engineer.transform_single(vitals, history)

        # 2. Isolation Forest prediction
        feature_vec = np.array([[features.get(c, 0.0) for c in self.feature_columns]])
        feature_vec_scaled = (feature_vec - self.feature_means) / self.feature_stds
        if_prediction = self.model.predict(feature_vec_scaled)[0]    # 1 = normal, -1 = anomaly
        if_score = self.model.decision_function(feature_vec_scaled)[0]
        is_if_anomaly = if_prediction == -1

        # 3. Rule-based evaluation
        rule_score, rule_reasons = self.rule_engine.evaluate(vitals)

        # 4. Combine scores
        #    IF anomaly score is inverted: lower = more anomalous
        #    Normalise to [0, 1] then blend 60 % ML + 40 % rule
        if_anomaly_norm = float(np.clip(1.0 - (if_score + 0.5), 0.0, 1.0))
        combined_score = 0.6 * if_anomaly_norm + 0.4 * rule_score
        combined_score = min(combined_score, 1.0)

        anomaly_detected = is_if_anomaly or rule_score > 0.2

        # 5. Determine risk level
        risk_level = self._score_to_level(combined_score)

        # 6. Compile reasons
        reasons = list(rule_reasons)
        if is_if_anomaly:
            reasons.insert(0, f"ML anomaly detector triggered (score={if_score:.3f})")
        if not reasons:
            reasons.append("All vitals within normal range")

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Inference for %s: risk=%s score=%.3f latency=%.1f ms",
            vitals.get("patient_id", "?"), risk_level, combined_score, latency_ms,
        )

        # 7. Build DynamoDB-compatible output
        timestamp = vitals.get("timestamp", datetime.now(timezone.utc).isoformat())
        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()

        return {
            "patientId": str(vitals.get("patient_id", "")),
            "timestamp": timestamp,
            "riskLevel": risk_level,
            "anomalyDetected": anomaly_detected,
            "combinedRiskScore": round(combined_score, 4),
            "isolationForestScore": round(float(if_score), 4),
            "ruleBasedScore": round(rule_score, 4),
            "reasons": reasons,
            "vitals": {
                "heartRate": vitals.get("heart_rate"),
                "spo2": vitals.get("spo2"),
                "systolicBP": vitals.get("systolic_bp"),
                "diastolicBP": vitals.get("diastolic_bp"),
                "temperature": vitals.get("temperature"),
            },
            "inferenceLatencyMs": round(latency_ms, 1),
        }

    # ── batch prediction ──────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> list:
        """
        Vectorized batch inference.
        Engineers features once in bulk, then runs model on the entire matrix.
        """
        t0 = time.perf_counter()

        # Bulk feature engineering
        enriched_df = self.feature_engineer.transform(df)

        # Vectorized IF prediction on full feature matrix
        feature_matrix = enriched_df[self.feature_columns].values.astype(float)
        feature_matrix_scaled = (feature_matrix - self.feature_means) / self.feature_stds
        if_predictions = self.model.predict(feature_matrix_scaled)       # (n,)
        if_scores = self.model.decision_function(feature_matrix_scaled)  # (n,)

        # Vectorized score computation
        if_anomaly_norm = np.clip(1.0 - (if_scores + 0.5), 0.0, 1.0)

        results = []
        for idx in range(len(enriched_df)):
            row = enriched_df.iloc[idx]
            raw_vitals = {
                "patient_id": row.get("patient_id", ""),
                "timestamp": row.get("timestamp", ""),
                "heart_rate": row.get("heart_rate", 0),
                "spo2": row.get("spo2", 0),
                "systolic_bp": row.get("systolic_bp", 0),
                "diastolic_bp": row.get("diastolic_bp", 0),
                "temperature": row.get("temperature", 0),
            }

            is_if_anomaly = if_predictions[idx] == -1
            if_score = float(if_scores[idx])
            rule_score, rule_reasons = self.rule_engine.evaluate(raw_vitals)

            combined_score = min(0.6 * float(if_anomaly_norm[idx]) + 0.4 * rule_score, 1.0)
            anomaly_detected = is_if_anomaly or rule_score > 0.2
            risk_level = self._score_to_level(combined_score)

            reasons = list(rule_reasons)
            if is_if_anomaly:
                reasons.insert(0, f"ML anomaly detector triggered (score={if_score:.3f})")
            if not reasons:
                reasons.append("All vitals within normal range")

            timestamp = raw_vitals.get("timestamp", "")
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()

            results.append({
                "patientId": str(raw_vitals["patient_id"]),
                "timestamp": str(timestamp),
                "riskLevel": risk_level,
                "anomalyDetected": anomaly_detected,
                "combinedRiskScore": round(combined_score, 4),
                "isolationForestScore": round(if_score, 4),
                "ruleBasedScore": round(rule_score, 4),
                "reasons": reasons,
                "vitals": {
                    "heartRate": raw_vitals["heart_rate"],
                    "spo2": raw_vitals["spo2"],
                    "systolicBP": raw_vitals["systolic_bp"],
                    "diastolicBP": raw_vitals["diastolic_bp"],
                    "temperature": raw_vitals["temperature"],
                },
                "inferenceLatencyMs": 0.0,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_latency = elapsed_ms / max(len(results), 1)
        for r in results:
            r["inferenceLatencyMs"] = round(avg_latency, 1)

        logger.info(
            "Batch inference complete: %d records in %.1f ms (%.2f ms/record)",
            len(results), elapsed_ms, avg_latency,
        )
        return results

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _score_to_level(score: float) -> str:
        for level in ("CRITICAL", "HIGH", "MODERATE"):
            if score >= RISK_LEVEL_THRESHOLDS[level]:
                return level
        return "LOW"
