"""
Telehealth ML — Synthetic Vitals Data Generator

Generates realistic time-series patient vitals with configurable anomaly
injection. Anomalies are injected as sustained *waves* (not random spikes)
to mimic real clinical deterioration patterns.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ANOMALY_PROFILES,
    ANOMALY_RATE,
    ANOMALY_WAVE_MAX_LEN,
    ANOMALY_WAVE_MIN_LEN,
    NUM_PATIENTS,
    RECORDS_PER_PATIENT,
    SAMPLING_INTERVAL_SEC,
    SYNTHETIC_DATASET_PATH,
    VITAL_RANGES,
)

logger = logging.getLogger(__name__)


class VitalsGenerator:
    """
    Generates synthetic patient vitals as a time-series DataFrame.

    Each patient gets a unique ID and a contiguous time-series of readings.
    A configurable fraction of the data contains anomaly waves — sustained
    deviations that mimic clinical events (tachycardia, hypoxia, etc.).
    """

    def __init__(
        self,
        num_patients: int = NUM_PATIENTS,
        records_per_patient: int = RECORDS_PER_PATIENT,
        anomaly_rate: float = ANOMALY_RATE,
        seed: Optional[int] = 42,
    ):
        self.num_patients = num_patients
        self.records_per_patient = records_per_patient
        self.anomaly_rate = anomaly_rate
        self.rng = np.random.default_rng(seed)
        logger.info(
            "VitalsGenerator initialized: patients=%d, records_each=%d, anomaly_rate=%.2f",
            num_patients, records_per_patient, anomaly_rate,
        )

    # ── public API ────────────────────────────────────────────

    def generate_dataset(self) -> pd.DataFrame:
        """Generate full synthetic dataset for all patients."""
        frames = []
        for i in range(self.num_patients):
            patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
            df = self._generate_patient_series(patient_id, i)
            frames.append(df)
            logger.debug("Generated %d records for %s", len(df), patient_id)

        dataset = pd.concat(frames, ignore_index=True)
        logger.info(
            "Dataset generated: %d total records, %d patients",
            len(dataset), self.num_patients,
        )
        return dataset

    def save_csv(self, df: pd.DataFrame, path=None) -> str:
        """Persist dataset to CSV."""
        path = path or SYNTHETIC_DATASET_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Dataset saved to %s", path)
        return str(path)

    # ── internals ─────────────────────────────────────────────

    def _generate_patient_series(self, patient_id: str, patient_idx: int) -> pd.DataFrame:
        """Build a single patient's time-series with optional anomaly waves."""
        n = self.records_per_patient
        base_time = datetime(2025, 1, 1, 0, 0, 0) + timedelta(hours=patient_idx * 10)
        timestamps = [base_time + timedelta(seconds=SAMPLING_INTERVAL_SEC * i) for i in range(n)]

        # ── baseline vitals with circadian drift + noise ──
        t = np.arange(n, dtype=float)
        circadian = np.sin(2 * np.pi * t / (1440 / SAMPLING_INTERVAL_SEC))  # 24-h cycle scaled

        heart_rate = self._smooth_vital(t, circadian, base=75, amplitude=8, noise_std=3)
        spo2 = self._smooth_vital(t, circadian, base=97.5, amplitude=0.8, noise_std=0.5)
        systolic_bp = self._smooth_vital(t, circadian, base=120, amplitude=5, noise_std=3)
        diastolic_bp = self._smooth_vital(t, circadian, base=77, amplitude=3, noise_std=2)
        temperature = self._smooth_vital(t, circadian, base=36.9, amplitude=0.2, noise_std=0.1)

        # ── clip to physiological bounds ──
        heart_rate = np.clip(heart_rate, 40, 200)
        spo2 = np.clip(spo2, 70, 100)
        systolic_bp = np.clip(systolic_bp, 80, 220)
        diastolic_bp = np.clip(diastolic_bp, 50, 140)
        temperature = np.clip(temperature, 34.0, 42.0)

        # ── anomaly injection ──
        is_anomaly = np.zeros(n, dtype=bool)
        anomaly_type = np.full(n, "none", dtype=object)
        total_anomaly_budget = int(n * self.anomaly_rate)
        injected = 0

        while injected < total_anomaly_budget:
            wave_len = self.rng.integers(ANOMALY_WAVE_MIN_LEN, ANOMALY_WAVE_MAX_LEN + 1)
            wave_len = min(wave_len, total_anomaly_budget - injected)
            start = self.rng.integers(0, max(1, n - wave_len))

            profile_name = self.rng.choice(list(ANOMALY_PROFILES.keys()))
            profile = ANOMALY_PROFILES[profile_name]

            target_low, target_high = profile["target_range"]
            vital_name = profile["vital"]
            target_values = self.rng.uniform(target_low, target_high, size=wave_len)

            local_arrays = {
                "heart_rate": heart_rate,
                "spo2": spo2,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "temperature": temperature,
            }
            local_arrays[vital_name][start: start + wave_len] = target_values

            # handle secondary vitals (e.g., diastolic with systolic in hypertension)
            if "secondary" in profile:
                sec = profile["secondary"]
                sec_low, sec_high = sec["target_range"]
                local_arrays[sec["vital"]][start: start + wave_len] = self.rng.uniform(
                    sec_low, sec_high, size=wave_len
                )

            is_anomaly[start: start + wave_len] = True
            anomaly_type[start: start + wave_len] = profile_name
            injected += wave_len

        df = pd.DataFrame({
            "timestamp": timestamps,
            "patient_id": patient_id,
            "heart_rate": np.round(heart_rate, 1),
            "spo2": np.round(spo2, 1),
            "systolic_bp": np.round(systolic_bp, 1),
            "diastolic_bp": np.round(diastolic_bp, 1),
            "temperature": np.round(temperature, 2),
            "is_anomaly": is_anomaly.astype(int),
            "anomaly_type": anomaly_type,
        })
        return df

    def _smooth_vital(
        self, t: np.ndarray, circadian: np.ndarray,
        base: float, amplitude: float, noise_std: float,
    ) -> np.ndarray:
        """Generate a smooth vital signal = base + circadian modulation + Gaussian noise."""
        return base + amplitude * circadian + self.rng.normal(0, noise_std, size=len(t))
