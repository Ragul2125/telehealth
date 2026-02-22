"""
Telehealth ML — Trend Analyzer for Doctor Briefings

Analyzes temporal trends in patient vitals from inference results:
  • HR trend slope (rising/falling/stable)
  • SpO2 downward trend detection
  • BP volatility measurement
  • Risk escalation pattern (worsening/improving/stable)
  • First-half vs second-half comparison (12h vs 12h)

Produces interpretation-ready insights for the prompt builder.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Computes clinical trend insights from inference results.

    All methods operate on filtered, single-patient data and return
    human-interpretable insight dicts.
    """

    # Thresholds for classification
    SLOPE_THRESHOLD = 0.05       # per-reading slope threshold for "rising"/"falling"
    SPO2_DROP_THRESHOLD = -0.03  # SpO2 slope below this = downward trend
    BP_VOLATILITY_HIGH = 8.0     # std dev above this = high volatility
    RISK_ESCALATION_THRESHOLD = 0.05  # mean risk difference for escalation

    def __init__(self):
        logger.info("TrendAnalyzer initialized")

    def analyze(self, results: List[dict], patient_id: str) -> dict:
        """
        Full trend analysis on a patient's inference results.

        Parameters
        ----------
        results : list[dict]
            Inference results (already filtered to patient if needed).
        patient_id : str
            Patient ID.

        Returns
        -------
        dict : Structured trend insights.
        """
        if not results:
            return self._empty_trends(patient_id)

        df = pd.DataFrame(results)
        df = df[df["patientId"] == patient_id].copy()
        if df.empty:
            return self._empty_trends(patient_id)

        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)

        # Extract vitals
        vitals_df = pd.json_normalize(df["vitals"])
        for col in ["heartRate", "spo2", "systolicBP", "diastolicBP", "temperature"]:
            if col in vitals_df.columns:
                vitals_df[col] = pd.to_numeric(vitals_df[col], errors="coerce")

        n = len(df)
        mid = n // 2

        return {
            "patientId": patient_id,
            "totalReadings": n,
            "hrTrend": self._compute_vital_trend(vitals_df, "heartRate", "Heart Rate"),
            "spo2Trend": self._compute_spo2_trend(vitals_df),
            "bpVolatility": self._compute_bp_volatility(vitals_df),
            "riskEscalation": self._compute_risk_escalation(df, mid),
            "halfComparison": self._half_comparison(df, vitals_df, mid),
            "insights": self._generate_insights(df, vitals_df, mid),
        }

    # ── Vital trend (slope-based) ─────────────────────────────

    def _compute_vital_trend(self, vitals_df: pd.DataFrame, col: str, label: str) -> dict:
        if col not in vitals_df.columns or vitals_df[col].isna().all():
            return {"direction": "unknown", "slope": 0.0, "label": label}

        values = vitals_df[col].dropna().values
        if len(values) < 3:
            return {"direction": "stable", "slope": 0.0, "label": label}

        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])

        if slope > self.SLOPE_THRESHOLD:
            direction = "rising"
        elif slope < -self.SLOPE_THRESHOLD:
            direction = "falling"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope": round(slope, 4),
            "label": label,
            "interpretation": f"{label} is {direction} (slope: {slope:+.3f}/reading)",
        }

    # ── SpO2 downward trend ───────────────────────────────────

    def _compute_spo2_trend(self, vitals_df: pd.DataFrame) -> dict:
        trend = self._compute_vital_trend(vitals_df, "spo2", "SpO2")
        downward = trend["slope"] < self.SPO2_DROP_THRESHOLD
        trend["downwardTrendDetected"] = downward
        if downward:
            trend["interpretation"] = (
                f"⚠ SpO2 shows a downward trend (slope: {trend['slope']:+.4f}/reading). "
                "Potential desaturation in progress."
            )
        return trend

    # ── BP volatility ─────────────────────────────────────────

    def _compute_bp_volatility(self, vitals_df: pd.DataFrame) -> dict:
        result = {"systolicStd": 0.0, "diastolicStd": 0.0, "level": "low"}

        if "systolicBP" in vitals_df.columns:
            sys_std = float(vitals_df["systolicBP"].std())
            result["systolicStd"] = round(sys_std, 2)

        if "diastolicBP" in vitals_df.columns:
            dia_std = float(vitals_df["diastolicBP"].std())
            result["diastolicStd"] = round(dia_std, 2)

        max_std = max(result["systolicStd"], result["diastolicStd"])
        if max_std > self.BP_VOLATILITY_HIGH * 2:
            result["level"] = "critical"
        elif max_std > self.BP_VOLATILITY_HIGH:
            result["level"] = "high"
        elif max_std > self.BP_VOLATILITY_HIGH / 2:
            result["level"] = "moderate"
        else:
            result["level"] = "low"

        result["interpretation"] = (
            f"BP volatility is {result['level']} "
            f"(systolic σ={result['systolicStd']:.1f}, diastolic σ={result['diastolicStd']:.1f})"
        )
        return result

    # ── Risk escalation ───────────────────────────────────────

    def _compute_risk_escalation(self, df: pd.DataFrame, mid: int) -> dict:
        if mid == 0 or len(df) < 4:
            return {"pattern": "insufficient_data", "interpretation": "Not enough data for trend"}

        first_half_mean = float(df.iloc[:mid]["combinedRiskScore"].mean())
        second_half_mean = float(df.iloc[mid:]["combinedRiskScore"].mean())
        diff = second_half_mean - first_half_mean

        if diff > self.RISK_ESCALATION_THRESHOLD:
            pattern = "worsening"
        elif diff < -self.RISK_ESCALATION_THRESHOLD:
            pattern = "improving"
        else:
            pattern = "stable"

        return {
            "pattern": pattern,
            "firstHalfMeanRisk": round(first_half_mean, 4),
            "secondHalfMeanRisk": round(second_half_mean, 4),
            "riskDelta": round(diff, 4),
            "interpretation": (
                f"Risk trend is {pattern}: "
                f"first-half avg={first_half_mean:.3f}, "
                f"second-half avg={second_half_mean:.3f} "
                f"(Δ={diff:+.3f})"
            ),
        }

    # ── Half comparison ───────────────────────────────────────

    def _half_comparison(self, df: pd.DataFrame, vitals_df: pd.DataFrame, mid: int) -> dict:
        if mid == 0:
            return {}

        first_vitals = vitals_df.iloc[:mid]
        second_vitals = vitals_df.iloc[mid:]

        comparison = {}
        for col, label in [
            ("heartRate", "HR"), ("spo2", "SpO2"),
            ("systolicBP", "SysBP"), ("temperature", "Temp"),
        ]:
            if col in vitals_df.columns:
                first_mean = float(first_vitals[col].mean()) if not first_vitals[col].isna().all() else 0
                second_mean = float(second_vitals[col].mean()) if not second_vitals[col].isna().all() else 0
                comparison[label] = {
                    "firstHalfMean": round(first_mean, 2),
                    "secondHalfMean": round(second_mean, 2),
                    "change": round(second_mean - first_mean, 2),
                }

        # Anomaly frequency comparison
        first_anomalies = df.iloc[:mid]["anomalyDetected"].apply(
            lambda x: x if isinstance(x, bool) else str(x).lower() == "true"
        ).sum()
        second_anomalies = df.iloc[mid:]["anomalyDetected"].apply(
            lambda x: x if isinstance(x, bool) else str(x).lower() == "true"
        ).sum()
        comparison["anomalies"] = {
            "firstHalf": int(first_anomalies),
            "secondHalf": int(second_anomalies),
        }

        return comparison

    # ── Synthesized insights ──────────────────────────────────

    def _generate_insights(self, df: pd.DataFrame, vitals_df: pd.DataFrame, mid: int) -> List[str]:
        insights = []

        # Check for sustained anomalies
        anomaly_mask = df["anomalyDetected"].apply(
            lambda x: x if isinstance(x, bool) else str(x).lower() == "true"
        )
        anomaly_pct = anomaly_mask.mean() * 100
        if anomaly_pct > 15:
            insights.append(
                f"High anomaly rate: {anomaly_pct:.0f}% of readings flagged as anomalous"
            )

        # Check for SpO2 dips
        if "spo2" in vitals_df.columns:
            min_spo2 = vitals_df["spo2"].min()
            if min_spo2 < 90:
                insights.append(f"Critical SpO2 dip detected: minimum {min_spo2:.1f}%")
            elif min_spo2 < 92:
                insights.append(f"Low SpO2 detected: minimum {min_spo2:.1f}%")

        # Check for tachycardia episodes
        if "heartRate" in vitals_df.columns:
            tachy_count = (vitals_df["heartRate"] > 120).sum()
            if tachy_count > 0:
                insights.append(f"Tachycardia episodes: {tachy_count} readings with HR > 120 bpm")

        # Check for hypertension
        if "systolicBP" in vitals_df.columns:
            hyper_count = (vitals_df["systolicBP"] > 150).sum()
            if hyper_count > 0:
                insights.append(f"Hypertension episodes: {hyper_count} readings with SBP > 150 mmHg")

        # Check for fever
        if "temperature" in vitals_df.columns:
            fever_count = (vitals_df["temperature"] > 38.5).sum()
            if fever_count > 0:
                insights.append(f"Fever episodes: {fever_count} readings with temp > 38.5°C")

        # Risk escalation
        if mid > 0 and len(df) >= 4:
            first_risk = df.iloc[:mid]["combinedRiskScore"].mean()
            second_risk = df.iloc[mid:]["combinedRiskScore"].mean()
            if second_risk - first_risk > 0.05:
                insights.append("Risk scores are escalating in the recent period")
            elif first_risk - second_risk > 0.05:
                insights.append("Risk scores have improved in the recent period")

        if not insights:
            insights.append("No significant clinical anomalies detected in this window")

        return insights

    @staticmethod
    def _empty_trends(patient_id: str) -> dict:
        return {
            "patientId": patient_id,
            "totalReadings": 0,
            "hrTrend": {"direction": "unknown", "slope": 0.0},
            "spo2Trend": {"direction": "unknown", "slope": 0.0, "downwardTrendDetected": False},
            "bpVolatility": {"level": "unknown"},
            "riskEscalation": {"pattern": "insufficient_data"},
            "halfComparison": {},
            "insights": ["No data available for this patient"],
        }
