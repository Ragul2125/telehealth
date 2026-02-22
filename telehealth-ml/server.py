"""
Telehealth ML — FastAPI REST Server

Wraps the existing ML modules (InferenceEngine, AlertEngine, BriefingAgent)
as a production-ready HTTP API for the frontend.

Run:
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Ensure the telehealth-ml root is on sys.path ──────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import FULL_RESULTS_PATH, SAVED_MODEL_PATH
from models.inference import InferenceEngine
from alerts.alert_engine import AlertEngine
from doctor_briefing.briefing_agent import BriefingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telehealth.server")

# ── App-wide singletons (loaded once at startup) ───────────────
_engine: Optional[InferenceEngine] = None
_alert_engine = AlertEngine()
_briefing_agent: Optional[BriefingAgent] = None

# In-memory per-patient vitals store for multi-reading context
_patient_history: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _briefing_agent
    logger.info("Loading ML model…")
    try:
        _engine = InferenceEngine()
        logger.info("✅ InferenceEngine ready")
    except Exception as e:
        logger.error("❌ Could not load model: %s. Run `python main.py demo` first.", e)

    logger.info("Loading BriefingAgent…")
    try:
        _briefing_agent = BriefingAgent()
        logger.info("✅ BriefingAgent ready")
    except Exception as e:
        logger.warning("BriefingAgent init warning: %s", e)

    yield  # app runs here


app = FastAPI(
    title="Telehealth ML API",
    description="Local REST API wrapping the Telehealth ML pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow localhost frontend ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3001",
        "http://localhost:4173",
        "*",  # allow all for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════

class VitalsPayload(BaseModel):
    patientId: str = Field(..., example="PAT-001")
    heartRate: float = Field(..., ge=20, le=300, example=72)
    spo2: float = Field(..., ge=50, le=100, example=98)
    systolicBP: float = Field(..., ge=60, le=250, example=120)
    diastolicBP: float = Field(..., ge=40, le=150, example=75)
    temperature: float = Field(..., ge=34.0, le=42.0, example=36.9)
    timestamp: Optional[str] = None


class InferencePayload(BaseModel):
    patientId: str = Field(..., example="PAT-001")


class TriagePayload(BaseModel):
    patientId: str
    message: str


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════

def _to_internal(v: VitalsPayload) -> dict:
    """Convert camelCase API payload → snake_case internal format."""
    return {
        "patient_id": v.patientId,
        "heart_rate": v.heartRate,
        "spo2": v.spo2,
        "systolic_bp": v.systolicBP,
        "diastolic_bp": v.diastolicBP,
        "temperature": v.temperature,
        "timestamp": v.timestamp or datetime.now(timezone.utc).isoformat(),
    }


def _load_full_results() -> list[dict]:
    """Load batch inference results from disk."""
    if not FULL_RESULTS_PATH.exists():
        return []
    with open(FULL_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("results", [])


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "briefing_ready": _briefing_agent is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── 1. Submit vitals ─────────────────────────────────────────
@app.post("/vitals", tags=["Vitals"])
def submit_vitals(payload: VitalsPayload):
    """Store a vitals reading in the in-memory session history."""
    record = _to_internal(payload)
    pid = payload.patientId
    _patient_history.setdefault(pid, []).append(record)
    # Keep last 500 readings per patient
    _patient_history[pid] = _patient_history[pid][-500:]
    logger.info("Vitals received for %s (history=%d)", pid, len(_patient_history[pid]))
    return {"status": "accepted", "patientId": pid, "readingsInSession": len(_patient_history[pid])}


# ── 2. Run inference ─────────────────────────────────────────
@app.post("/inference", tags=["Inference"])
def run_inference(payload: InferencePayload):
    """Run ML inference using the latest vitals for a patient."""
    pid = payload.patientId
    history_list = _patient_history.get(pid, [])

    # ── Case A: No live session vitals → serve from full_results.json ──
    if not history_list:
        all_results = _load_full_results()
        patient_results = [r for r in all_results if r.get("patientId") == pid]

        if not patient_results:
            # No data at all — return a safe LOW default, never 500
            return {
                "riskLevel": "LOW",
                "anomalyDetected": False,
                "combinedRiskScore": 0.0,
                "reasons": ["No vitals data on record yet. Submit a reading to run inference."],
                "patientId": pid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "default",
            }

        # Return the most recent stored inference result
        latest = sorted(patient_results, key=lambda r: r.get("timestamp", ""), reverse=True)[0]

        # anomalyDetected may be stored as string "True"/"False" from batch output
        anomaly = latest.get("anomalyDetected", False)
        if isinstance(anomaly, str):
            anomaly = anomaly.lower() == "true"

        return {
            "riskLevel": latest.get("riskLevel", "LOW"),
            "anomalyDetected": anomaly,
            "combinedRiskScore": float(latest.get("combinedRiskScore", 0.0)),
            "reasons": latest.get("reasons", []),
            "patientId": pid,
            "timestamp": latest.get("timestamp"),
            "source": "cached",
        }

    # ── Case B: Live session vitals available → run ML inference ──
    if _engine is None:
        return {
            "riskLevel": "LOW",
            "anomalyDetected": False,
            "combinedRiskScore": 0.0,
            "reasons": ["Model not loaded — run `python main.py demo` first."],
            "patientId": pid,
            "source": "unavailable",
        }

    import pandas as pd
    latest_vitals = history_list[-1]
    history_df = pd.DataFrame(history_list[:-1]) if len(history_list) > 1 else None

    try:
        result = _engine.predict(latest_vitals, history=history_df)
        return {
            "riskLevel": result.get("riskLevel", "LOW"),
            "anomalyDetected": bool(result.get("anomalyDetected", False)),
            "combinedRiskScore": float(result.get("combinedRiskScore", 0.0)),
            "reasons": result.get("reasons", []),
            "patientId": pid,
            "timestamp": result.get("timestamp"),
            "source": "live",
        }
    except Exception as e:
        logger.error("Inference error for %s: %s", pid, e)
        raise HTTPException(500, detail=f"Inference failed: {str(e)}")



# ── 3. Get alerts ─────────────────────────────────────────────
@app.get("/alerts/{patient_id}", tags=["Alerts"])
def get_alerts(patient_id: str, limit: int = 50):
    """Return alert history for a patient from full_results.json."""
    all_results = _load_full_results()
    patient_results = [r for r in all_results if r.get("patientId") == patient_id]

    if not patient_results:
        return []

    # Evaluate alerts from stored inference results
    alerts = _alert_engine.evaluate_batch(patient_results)

    # Sort newest first and limit
    alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    return alerts[:limit]


# ── 4. Get doctor briefing ────────────────────────────────────
@app.get("/brief/{patient_id}", tags=["Briefing"])
def get_briefing(patient_id: str, mode: str = "text"):
    """Generate an AI doctor briefing for a patient using local RAG."""
    if _briefing_agent is None:
        raise HTTPException(503, detail="BriefingAgent not ready.")

    all_results = _load_full_results()
    patient_results = [r for r in all_results if r.get("patientId") == patient_id]

    if not patient_results:
        raise HTTPException(404, detail=f"No data found for patient {patient_id}.")

    try:
        briefing = _briefing_agent.generate_briefing(patient_results, patient_id, mode=mode)
        return briefing
    except Exception as e:
        logger.error("Briefing error for %s: %s", patient_id, e)
        raise HTTPException(500, detail=f"Briefing generation failed: {str(e)}")


# ── 5. Triage chat ────────────────────────────────────────────
@app.post("/triage", tags=["Triage"])
def triage_chat(payload: TriagePayload):
    """Answer a patient health question using local context."""
    pid = payload.patientId
    message = payload.message.strip()

    all_results = _load_full_results()
    patient_results = [r for r in all_results if r.get("patientId") == pid]

    # Build a simple context-aware response using RAG / rules
    context_parts = []
    if patient_results:
        recent = sorted(patient_results, key=lambda r: r.get("timestamp", ""), reverse=True)[:5]
        anomalous = [r for r in recent if r.get("anomalyDetected")]
        context_parts.append(
            f"Patient {pid}: {len(patient_results)} readings on record. "
            f"{len(anomalous)} recent anomalies detected."
        )

    # Use BriefingAgent's vector store for semantic search if available
    response_text = _build_triage_response(message, context_parts, patient_results)

    return {
        "patientId": pid,
        "message": message,
        "response": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_triage_response(message: str, context: list, results: list) -> str:
    """Simple keyword + context-aware triage response."""
    msg = message.lower()

    if any(w in msg for w in ["chest pain", "chest pressure", "heart"]):
        recent_anomalies = [r for r in results if r.get("anomalyDetected")]
        if recent_anomalies:
            return (
                f"Based on your monitoring data, {len(recent_anomalies)} anomalous readings "
                "have been recorded. Chest pain combined with abnormal vitals requires immediate "
                "medical evaluation. Please contact your care team or emergency services now."
            )
        return "Chest discomfort should always be evaluated by a medical professional. Please contact your doctor immediately or call emergency services."

    if any(w in msg for w in ["dizzy", "dizziness", "lightheaded"]):
        return (
            "Dizziness can be related to blood pressure changes or oxygen levels. "
            + (f"Your data shows {len(context[0]) if context else 'some'} recent readings. " if context else "")
            + "Please sit down, stay hydrated, and contact your care team if symptoms persist."
        )

    if any(w in msg for w in ["breathe", "breathing", "shortness", "breath", "spo2", "oxygen"]):
        low_spo2 = [r for r in results if r.get("anomalyDetected") and "hypoxia" in str(r.get("reasons", "")).lower()]
        if low_spo2:
            return f"Your monitoring data shows {len(low_spo2)} hypoxia-related readings. Low SpO₂ is serious — please seek immediate medical attention."
        return "Breathing difficulties warrant medical evaluation. Contact your doctor or emergency services if breathing is significantly impaired."

    if any(w in msg for w in ["headache", "head pain", "migraine"]):
        return "Headaches can sometimes be related to blood pressure fluctuations. Based on your monitoring data, please check your recent BP readings and contact your provider if the headache is severe or persistent."

    if any(w in msg for w in ["fever", "temperature", "hot", "cold"]):
        return "Fever requires monitoring. If your temperature exceeds 38.5°C, contact your care team. Stay hydrated and rest."

    if any(w in msg for w in ["reading", "vital", "result", "score", "risk"]):
        if results:
            recent = sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)[:1]
            if recent:
                r = recent[0]
                return (
                    f"Your most recent reading shows risk level: {r.get('riskLevel', 'N/A')}, "
                    f"risk score: {r.get('combinedRiskScore', 0):.1%}. "
                    + (f"Reasons: {', '.join(r.get('reasons', []))}" if r.get('reasons') else "All vitals within normal range.")
                )

    # Default
    return (
        f"I've noted your concern: \"{message}\". "
        + (f"Your monitoring data shows {len(results)} readings on record. " if results else "")
        + "For specific medical advice, please consult your healthcare provider. "
        "If this is an emergency, call emergency services immediately."
    )
