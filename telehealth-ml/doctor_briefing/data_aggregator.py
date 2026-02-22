"""
Telehealth ML — Patient Data Aggregator for Doctor Briefings

Aggregates inference results for a specific patient over a time window
(default 24h).  Produces a structured summary object containing:
  • Vital statistics (min/max/mean for HR, SpO2, BP, temp)
  • Risk distribution and peak instability time
  • Anomaly counts and critical event catalogue
  • Alert frequency per hour
  • Fever event tracking

Designed for vectorized pandas processing — target < 50ms for 1440 records.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Aggregates inference results into a structured patient summary.

    All inputs are the JSON dicts produced by InferenceEngine.predict()
    or InferenceEngine.predict_batch().
    """

    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        logger.info("DataAggregator initialized (lookback=%dh)", lookback_hours)

    def aggregate(
        self,
        results: List[dict],
        patient_id: str,
        reference_time: Optional[datetime] = None,
    ) -> dict:
        """
        Filter results to a specific patient and time window, then aggregate.

        Parameters
        ----------
        results : list[dict]
            Full list of inference result dicts.
        patient_id : str
            Patient ID to filter for.
        reference_time : datetime, optional
            End of the lookback window.  Defaults to now (UTC).

        Returns
        -------
        dict : Structured aggregation summary.
        """
        t0 = time.perf_counter()

        if not results:
            return self._empty_summary(patient_id)

        df = pd.DataFrame(results)

        # ── Filter by patient ──
        df = df[df["patientId"] == patient_id].copy()
        if df.empty:
            logger.warning("No records found for patient %s", patient_id)
            return self._empty_summary(patient_id)

        # ── Parse timestamps ──
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # ── Filter by time window ──
        if reference_time is None:
            reference_time = df["ts"].max()  # use latest record as reference
        cutoff = reference_time - timedelta(hours=self.lookback_hours)
        df = df[(df["ts"] >= cutoff) & (df["ts"] <= reference_time)]

        if df.empty:
            logger.warning("No records in %dh window for %s", self.lookback_hours, patient_id)
            return self._empty_summary(patient_id)

        # ── Extract vitals into columns ──
        vitals_df = pd.json_normalize(df["vitals"])
        vitals_df.index = df.index

        # ── Core aggregation ──
        summary = {
            "patientId": patient_id,
            "windowStart": str(cutoff),
            "windowEnd": str(reference_time),
            "totalReadings": len(df),

            # Risk distribution
            "riskDistribution": df["riskLevel"].value_counts().to_dict(),
            "maxRiskScore": round(float(df["combinedRiskScore"].max()), 4),
            "meanRiskScore": round(float(df["combinedRiskScore"].mean()), 4),

            # Anomaly stats
            "anomalyCount": int(df["anomalyDetected"].apply(
                lambda x: x if isinstance(x, bool) else str(x).lower() == "true"
            ).sum()),

            # Vital ranges
            "vitalRanges": {
                "heartRate": self._vital_stats(vitals_df, "heartRate"),
                "spo2": self._vital_stats(vitals_df, "spo2"),
                "systolicBP": self._vital_stats(vitals_df, "systolicBP"),
                "diastolicBP": self._vital_stats(vitals_df, "diastolicBP"),
                "temperature": self._vital_stats(vitals_df, "temperature"),
            },

            # Peak instability
            "peakRiskTime": str(df.loc[df["combinedRiskScore"].idxmax(), "ts"]),
            "peakRiskScore": round(float(df["combinedRiskScore"].max()), 4),

            # Critical events (HIGH / CRITICAL)
            "criticalEvents": self._extract_critical_events(df),

            # Alert frequency per hour
            "alertFrequencyPerHour": self._alert_frequency(df),

            # Fever events
            "feverEvents": self._count_fever_events(vitals_df),

            # Alert type distribution
            "alertTypeDistribution": self._alert_type_distribution(df),
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary["aggregationLatencyMs"] = round(elapsed_ms, 2)
        logger.info(
            "Aggregation for %s: %d records in %.1f ms",
            patient_id, len(df), elapsed_ms,
        )
        return summary

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _vital_stats(vitals_df: pd.DataFrame, col: str) -> dict:
        if col not in vitals_df.columns:
            return {"min": None, "max": None, "mean": None}
        series = pd.to_numeric(vitals_df[col], errors="coerce")
        return {
            "min": round(float(series.min()), 1) if not series.isna().all() else None,
            "max": round(float(series.max()), 1) if not series.isna().all() else None,
            "mean": round(float(series.mean()), 1) if not series.isna().all() else None,
        }

    @staticmethod
    def _extract_critical_events(df: pd.DataFrame) -> list:
        critical = df[df["riskLevel"].isin(["HIGH", "CRITICAL"])]
        events = []
        for _, row in critical.iterrows():
            events.append({
                "timestamp": str(row["ts"]),
                "riskLevel": row["riskLevel"],
                "riskScore": round(float(row["combinedRiskScore"]), 4),
                "reasons": row.get("reasons", []),
            })
        return events

    @staticmethod
    def _alert_frequency(df: pd.DataFrame) -> dict:
        anomaly_mask = df["anomalyDetected"].apply(
            lambda x: x if isinstance(x, bool) else str(x).lower() == "true"
        )
        anomaly_df = df[anomaly_mask]
        if anomaly_df.empty:
            return {}
        anomaly_df = anomaly_df.copy()
        anomaly_df["hour"] = anomaly_df["ts"].dt.strftime("%Y-%m-%dT%H:00")
        return anomaly_df.groupby("hour").size().to_dict()

    @staticmethod
    def _count_fever_events(vitals_df: pd.DataFrame) -> int:
        if "temperature" not in vitals_df.columns:
            return 0
        temps = pd.to_numeric(vitals_df["temperature"], errors="coerce")
        return int((temps > 38.5).sum())

    @staticmethod
    def _alert_type_distribution(df: pd.DataFrame) -> dict:
        if "alertType" not in df.columns:
            # Derive from reasons
            return {}
        anomaly_mask = df["anomalyDetected"].apply(
            lambda x: x if isinstance(x, bool) else str(x).lower() == "true"
        )
        alert_df = df[anomaly_mask]
        if "alertType" in alert_df.columns:
            return alert_df["alertType"].value_counts().to_dict()
        return {}

    @staticmethod
    def _empty_summary(patient_id: str) -> dict:
        return {
            "patientId": patient_id,
            "totalReadings": 0,
            "anomalyCount": 0,
            "riskDistribution": {},
            "maxRiskScore": 0.0,
            "meanRiskScore": 0.0,
            "peakRiskTime": None,
            "peakRiskScore": 0.0,
            "criticalEvents": [],
            "alertFrequencyPerHour": {},
            "feverEvents": 0,
            "vitalRanges": {},
            "alertTypeDistribution": {},
            "aggregationLatencyMs": 0.0,
        }
