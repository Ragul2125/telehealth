"""
Telehealth ML — Model Training Module

Trains an IsolationForest anomaly detector on engineered features derived
from normal (non-anomalous) patient vitals.  Also contains the rule-based
risk engine used alongside the ML model during inference.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    FEATURE_COLUMNS,
    IF_CONTAMINATION,
    IF_MAX_SAMPLES,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    RULE_THRESHOLDS,
    SAVED_MODEL_PATH,
    SYNTHETIC_DATASET_PATH,
)
from features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class RuleBasedRiskEngine:
    """
    Deterministic rule engine that maps raw vitals to a risk score ∈ [0, 1]
    and a list of human-readable reasons.

    Each violated rule contributes a weight to the total score.
    """

    RULE_WEIGHTS = {
        "tachycardia":       0.25,
        "bradycardia":       0.20,
        "hypoxia":           0.30,
        "critical_hypoxia":  0.40,
        "hypertension_sys":  0.20,
        "hypertension_dia":  0.15,
        "fever":             0.20,
        "hypothermia":       0.15,
    }

    def evaluate(self, vitals: dict) -> Tuple[float, list]:
        """Return (risk_score, [reasons])."""
        score = 0.0
        reasons = []

        hr = vitals.get("heart_rate", 75)
        spo2 = vitals.get("spo2", 98)
        sys_bp = vitals.get("systolic_bp", 120)
        dia_bp = vitals.get("diastolic_bp", 78)
        temp = vitals.get("temperature", 37.0)

        if hr > RULE_THRESHOLDS["heart_rate_high"]:
            score += self.RULE_WEIGHTS["tachycardia"]
            reasons.append(f"Tachycardia detected: HR={hr:.0f} bpm")
        if hr < RULE_THRESHOLDS["heart_rate_low"]:
            score += self.RULE_WEIGHTS["bradycardia"]
            reasons.append(f"Bradycardia detected: HR={hr:.0f} bpm")

        if spo2 < RULE_THRESHOLDS["spo2_critical"]:
            score += self.RULE_WEIGHTS["critical_hypoxia"]
            reasons.append(f"Critical hypoxia: SpO2={spo2:.1f}%")
        elif spo2 < RULE_THRESHOLDS["spo2_low"]:
            score += self.RULE_WEIGHTS["hypoxia"]
            reasons.append(f"Hypoxia detected: SpO2={spo2:.1f}%")

        if sys_bp > RULE_THRESHOLDS["systolic_bp_high"]:
            score += self.RULE_WEIGHTS["hypertension_sys"]
            reasons.append(f"Systolic hypertension: BP={sys_bp:.0f} mmHg")
        if dia_bp > RULE_THRESHOLDS["diastolic_bp_high"]:
            score += self.RULE_WEIGHTS["hypertension_dia"]
            reasons.append(f"Diastolic hypertension: BP={dia_bp:.0f} mmHg")

        if temp > RULE_THRESHOLDS["temperature_high"]:
            score += self.RULE_WEIGHTS["fever"]
            reasons.append(f"Fever detected: Temp={temp:.1f}°C")
        if temp < RULE_THRESHOLDS["temperature_low"]:
            score += self.RULE_WEIGHTS["hypothermia"]
            reasons.append(f"Hypothermia detected: Temp={temp:.1f}°C")

        return min(score, 1.0), reasons


class ModelTrainer:
    """
    Trains an IsolationForest on engineered features from *normal* data,
    then persists the model + feature scaler via joblib.
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model: IsolationForest | None = None
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None
        logger.info("ModelTrainer initialized")

    def train(self, data_path: Path = SYNTHETIC_DATASET_PATH) -> dict:
        """
        Full training pipeline:
          1. Load CSV
          2. Engineer features
          3. Filter to normal records
          4. Fit IsolationForest
          5. Persist model bundle

        Returns training metadata dict.
        """
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

        logger.info("Running feature engineering on %d records", len(df))
        df = self.feature_engineer.transform(df)

        # Use only normal data for training so anomalies stand out
        normal_mask = df["is_anomaly"] == 0
        train_df = df.loc[normal_mask, FEATURE_COLUMNS].copy()
        logger.info(
            "Training on %d normal records (excluded %d anomalous)",
            len(train_df), (~normal_mask).sum(),
        )

        # Standardise features (store means/stds for inference)
        self.feature_means = train_df.mean().values.copy()
        self.feature_stds = train_df.std().values.copy()
        self.feature_stds[self.feature_stds < 1e-8] = 1.0  # avoid div-by-zero
        X_train = (train_df.values - self.feature_means) / self.feature_stds

        # Train Isolation Forest
        t0 = time.perf_counter()
        self.model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=IF_MAX_SAMPLES,
            contamination=IF_CONTAMINATION,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X_train)
        elapsed = time.perf_counter() - t0
        logger.info("IsolationForest trained in %.2f s", elapsed)

        # Evaluate on full data for logging
        X_all = (df[FEATURE_COLUMNS].values - self.feature_means) / self.feature_stds
        preds = self.model.predict(X_all)
        anomaly_preds = (preds == -1).sum()
        logger.info(
            "Full-data evaluation: %d / %d flagged as anomaly (%.1f%%)",
            anomaly_preds, len(df), 100.0 * anomaly_preds / len(df),
        )

        # Persist
        self._save_model()

        return {
            "training_records": int(len(train_df)),
            "total_records": int(len(df)),
            "training_time_sec": round(elapsed, 3),
            "anomaly_preds_full": int(anomaly_preds),
            "model_path": str(SAVED_MODEL_PATH),
        }

    def _save_model(self):
        """Save model + scaler params as a single bundle."""
        SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_columns": FEATURE_COLUMNS,
        }
        joblib.dump(bundle, SAVED_MODEL_PATH)
        logger.info("Model bundle saved to %s", SAVED_MODEL_PATH)
