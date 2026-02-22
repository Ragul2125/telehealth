"""
Telehealth ML Subsystem — Centralized Configuration

All magic numbers, thresholds, model hyperparameters, and feature windows
are defined here. Every module imports from this single source of truth.
"""

import logging
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SAVED_MODEL_PATH = MODEL_DIR / "saved_model.joblib"
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_dataset.csv"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ─────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────
NUM_PATIENTS = 10
RECORDS_PER_PATIENT = 500          # ~8 hours at 1-min intervals
SAMPLING_INTERVAL_SEC = 60         # 1 reading per minute
ANOMALY_RATE = 0.08                # 8 % of records are anomalous
ANOMALY_WAVE_MIN_LEN = 5           # min consecutive anomalous readings
ANOMALY_WAVE_MAX_LEN = 15          # max consecutive anomalous readings

# Normal vital ranges (used for generation + rule engine)
VITAL_RANGES = {
    "heart_rate":   {"low": 60,   "high": 100,  "unit": "bpm"},
    "spo2":         {"low": 95,   "high": 100,  "unit": "%"},
    "systolic_bp":  {"low": 110,  "high": 130,  "unit": "mmHg"},
    "diastolic_bp": {"low": 70,   "high": 85,   "unit": "mmHg"},
    "temperature":  {"low": 36.5, "high": 37.5, "unit": "°C"},
}

# Anomaly definitions (sustained wave targets)
ANOMALY_PROFILES = {
    "tachycardia":  {"vital": "heart_rate",  "target_range": (120, 160)},
    "hypoxia":      {"vital": "spo2",        "target_range": (78, 90)},
    "hypertension": {"vital": "systolic_bp", "target_range": (150, 190),
                     "secondary": {"vital": "diastolic_bp", "target_range": (100, 120)}},
    "fever":        {"vital": "temperature", "target_range": (38.5, 40.5)},
}

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
ROLLING_WINDOW_SHORT = 5           # 5-minute rolling window
ROLLING_WINDOW_LONG = 30           # 30-minute rolling window
BP_SLOPE_WINDOW = 10               # window for BP trend slope
ZSCORE_CLIP = 5.0                  # clip z-scores to [-5, 5]

# Features used for IsolationForest training
FEATURE_COLUMNS = [
    "hr_rolling_mean_5", "hr_rolling_mean_30", "hr_variability",
    "spo2_rolling_min_5", "bp_systolic_slope", "bp_diastolic_slope",
    "hr_zscore", "spo2_zscore", "systolic_bp_zscore",
    "diastolic_bp_zscore", "temp_zscore",
    "hr_delta", "spo2_delta", "systolic_bp_delta",
    "diastolic_bp_delta", "temp_delta",
    "risk_aggregation_score",
]

# ─────────────────────────────────────────────
# Model — Isolation Forest
# ─────────────────────────────────────────────
IF_CONTAMINATION = 0.05            # expected contamination fraction
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = "auto"
IF_RANDOM_STATE = 42

# ─────────────────────────────────────────────
# Rule-Based Risk Engine Thresholds
# ─────────────────────────────────────────────
RULE_THRESHOLDS = {
    "heart_rate_high":   120,
    "heart_rate_low":    50,
    "spo2_low":          92,
    "spo2_critical":     88,
    "systolic_bp_high":  150,
    "diastolic_bp_high": 100,
    "temperature_high":  38.5,
    "temperature_low":   35.0,
}

# ─────────────────────────────────────────────
# Alert / Risk Score Thresholds
# ─────────────────────────────────────────────
# Combined risk score ranges → severity
RISK_LEVEL_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH":     0.60,
    "MODERATE": 0.35,
    # everything below MODERATE → LOW
}

# ─────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────
STREAM_INTERVAL_SEC = 30           # seconds between simulated readings
DEFAULT_SIMULATION_CYCLES = 20     # number of readings per simulation run

# ─────────────────────────────────────────────
# Doctor Briefing Agent
# ─────────────────────────────────────────────
BRIEFING_LOOKBACK_HOURS = 24       # default patient data window
BRIEFING_LLM_MODEL = "gemini-2.0-flash"
BRIEFING_LLM_MAX_RETRIES = 3
BRIEFING_LLM_TIMEOUT_SEC = 10
BRIEFING_MAX_OUTPUT_TOKENS = 500
FULL_RESULTS_PATH = DATA_DIR / "full_results.json"

# ─────────────────────────────────────────────
# Local Vector Store (ChromaDB)
# ─────────────────────────────────────────────
CHROMA_DB_PATH = DATA_DIR / "chroma_db"          # persistent vector index
CHROMA_COLLECTION_NAME = "patient_vitals"         # ChromaDB collection name
CHROMA_EMBED_MODEL = "all-MiniLM-L6-v2"          # local sentence-transformer
CHROMA_TOP_K = 10                                 # results per semantic query

