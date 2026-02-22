"""
Telehealth ML — Feature Engineering Module

Transforms raw vitals into ML-ready features:
  • Rolling statistics (5-min, 30-min)
  • Heart-rate variability
  • SpO2 minimum over window
  • BP trend slope
  • Z-score normalization
  • Delta (change) between timestamps
  • Composite risk aggregation score
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    BP_SLOPE_WINDOW,
    ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_SHORT,
    VITAL_RANGES,
    ZSCORE_CLIP,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Stateless feature transformer.  All methods operate on a DataFrame
    grouped by patient_id (assumed sorted by timestamp).
    """

    def __init__(self):
        logger.info("FeatureEngineer initialized")

    # ── public API ────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature pipeline.  Expects columns:
            timestamp, patient_id, heart_rate, spo2,
            systolic_bp, diastolic_bp, temperature
        Returns the DataFrame with all engineered columns appended.
        """
        df = df.copy()
        df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        df = self._add_rolling_stats(df)
        df = self._add_hr_variability(df)
        df = self._add_spo2_rolling_min(df)
        df = self._add_bp_slope(df)
        df = self._add_zscores(df)
        df = self._add_deltas(df)
        df = self._add_risk_aggregation(df)

        # forward-fill NaNs produced by rolling windows, then zero-fill remainder
        # Preserve non-numeric columns so groupby/fillna don't corrupt them
        preserve_cols = ["patient_id", "timestamp"]
        preserved = {c: df[c].copy() for c in preserve_cols if c in df.columns}

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df[numeric_cols] = df.groupby("patient_id", group_keys=False)[numeric_cols].apply(
            lambda g: g.ffill()
        )
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        # Restore preserved columns
        for c, vals in preserved.items():
            df[c] = vals

        logger.info("Feature engineering complete — %d features added", self._count_new_cols(df))
        return df

    def transform_single(self, vitals: dict, history: Optional[pd.DataFrame] = None) -> dict:
        """
        Transform a single vitals reading for real-time inference.
        If history is provided, rolling/delta features use it;
        otherwise features default to instantaneous proxies.
        """
        if history is not None and len(history) > 0:
            row_df = pd.DataFrame([vitals])
            combined = pd.concat([history, row_df], ignore_index=True)
            combined = self.transform(combined)
            return combined.iloc[-1].to_dict()

        # Fallback: no history → instant features
        features = dict(vitals)
        features["hr_rolling_mean_5"] = vitals.get("heart_rate", 0)
        features["hr_rolling_mean_30"] = vitals.get("heart_rate", 0)
        features["hr_variability"] = 0.0
        features["spo2_rolling_min_5"] = vitals.get("spo2", 0)
        features["bp_systolic_slope"] = 0.0
        features["bp_diastolic_slope"] = 0.0
        for vital, col in [
            ("heart_rate", "hr_zscore"), ("spo2", "spo2_zscore"),
            ("systolic_bp", "systolic_bp_zscore"),
            ("diastolic_bp", "diastolic_bp_zscore"),
            ("temperature", "temp_zscore"),
        ]:
            vrange = VITAL_RANGES[vital]
            mean = (vrange["high"] + vrange["low"]) / 2
            std = (vrange["high"] - vrange["low"]) / 4
            features[col] = np.clip(
                (vitals.get(vital, mean) - mean) / max(std, 1e-6), -ZSCORE_CLIP, ZSCORE_CLIP
            )
        for vital, col in [
            ("heart_rate", "hr_delta"), ("spo2", "spo2_delta"),
            ("systolic_bp", "systolic_bp_delta"),
            ("diastolic_bp", "diastolic_bp_delta"),
            ("temperature", "temp_delta"),
        ]:
            features[col] = 0.0
        features["risk_aggregation_score"] = self._compute_single_risk(features)
        return features

    # ── rolling statistics ────────────────────────────────────

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        grp = df.groupby("patient_id")["heart_rate"]
        df["hr_rolling_mean_5"] = grp.transform(
            lambda s: s.rolling(ROLLING_WINDOW_SHORT, min_periods=1).mean()
        )
        df["hr_rolling_mean_30"] = grp.transform(
            lambda s: s.rolling(ROLLING_WINDOW_LONG, min_periods=1).mean()
        )
        return df

    def _add_hr_variability(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hr_variability"] = df.groupby("patient_id")["heart_rate"].transform(
            lambda s: s.rolling(ROLLING_WINDOW_SHORT, min_periods=1).std()
        )
        return df

    def _add_spo2_rolling_min(self, df: pd.DataFrame) -> pd.DataFrame:
        df["spo2_rolling_min_5"] = df.groupby("patient_id")["spo2"].transform(
            lambda s: s.rolling(ROLLING_WINDOW_SHORT, min_periods=1).min()
        )
        return df

    # ── BP trend slope ────────────────────────────────────────

    def _add_bp_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, out in [("systolic_bp", "bp_systolic_slope"), ("diastolic_bp", "bp_diastolic_slope")]:
            df[out] = df.groupby("patient_id")[col].transform(
                lambda s: self._rolling_slope(s, BP_SLOPE_WINDOW)
            )
        return df

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope."""
        slopes = np.full(len(series), np.nan)
        values = series.values
        for i in range(len(values)):
            start = max(0, i - window + 1)
            seg = values[start: i + 1]
            if len(seg) < 2:
                slopes[i] = 0.0
                continue
            x = np.arange(len(seg), dtype=float)
            coeffs = np.polyfit(x, seg, 1)
            slopes[i] = coeffs[0]
        return pd.Series(slopes, index=series.index)

    # ── Z-score normalisation ─────────────────────────────────

    def _add_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        mappings = [
            ("heart_rate", "hr_zscore"),
            ("spo2", "spo2_zscore"),
            ("systolic_bp", "systolic_bp_zscore"),
            ("diastolic_bp", "diastolic_bp_zscore"),
            ("temperature", "temp_zscore"),
        ]
        for vital, zcol in mappings:
            vrange = VITAL_RANGES[vital]
            mean = (vrange["high"] + vrange["low"]) / 2.0
            std = (vrange["high"] - vrange["low"]) / 4.0
            df[zcol] = ((df[vital] - mean) / max(std, 1e-6)).clip(-ZSCORE_CLIP, ZSCORE_CLIP)
        return df

    # ── Deltas ────────────────────────────────────────────────

    def _add_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        for vital, dcol in [
            ("heart_rate", "hr_delta"), ("spo2", "spo2_delta"),
            ("systolic_bp", "systolic_bp_delta"),
            ("diastolic_bp", "diastolic_bp_delta"),
            ("temperature", "temp_delta"),
        ]:
            df[dcol] = df.groupby("patient_id")[vital].diff().fillna(0.0)
        return df

    # ── Risk aggregation ──────────────────────────────────────

    def _add_risk_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite risk score ∈ [0, 1].
        Weighted sum of absolute z-scores, clipped and normalised.
        """
        weights = {
            "hr_zscore": 0.25,
            "spo2_zscore": 0.30,
            "systolic_bp_zscore": 0.15,
            "diastolic_bp_zscore": 0.10,
            "temp_zscore": 0.20,
        }
        raw = sum(w * df[col].abs() for col, w in weights.items())
        df["risk_aggregation_score"] = (raw / ZSCORE_CLIP).clip(0.0, 1.0)
        return df

    def _compute_single_risk(self, features: dict) -> float:
        weights = {
            "hr_zscore": 0.25, "spo2_zscore": 0.30,
            "systolic_bp_zscore": 0.15, "diastolic_bp_zscore": 0.10,
            "temp_zscore": 0.20,
        }
        raw = sum(w * abs(features.get(col, 0)) for col, w in weights.items())
        return float(np.clip(raw / ZSCORE_CLIP, 0.0, 1.0))

    @staticmethod
    def _count_new_cols(df: pd.DataFrame) -> int:
        base_cols = {
            "timestamp", "patient_id", "heart_rate", "spo2",
            "systolic_bp", "diastolic_bp", "temperature",
            "is_anomaly", "anomaly_type",
        }
        return len(set(df.columns) - base_cols)
