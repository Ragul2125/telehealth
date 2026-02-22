# 🏥 Telehealth ML Subsystem

Production-grade ML pipeline for a cloud-based **Telehealth Patient Monitoring Platform**.

Ingests patient vitals → detects anomalies → generates risk scores → triggers alerts → outputs DynamoDB-compatible JSON → **generates AI doctor briefings**.

---

## 📁 Project Structure

```
telehealth-ml/
│
├── data/
│   ├── __init__.py
│   ├── generator.py              # Synthetic time-series vitals generator
│   └── synthetic_dataset.csv     # Generated dataset (created at runtime)
│
├── features/
│   ├── __init__.py
│   └── feature_engineering.py    # Rolling stats, z-scores, deltas, risk aggregation
│
├── models/
│   ├── __init__.py
│   ├── train.py                  # IsolationForest training + rule-based risk engine
│   ├── inference.py              # Stateless inference engine
│   └── saved_model.joblib        # Trained model bundle (created at runtime)
│
├── alerts/
│   ├── __init__.py
│   └── alert_engine.py           # Structured alert JSON generator
│
├── doctor_briefing/
│   ├── __init__.py
│   ├── data_aggregator.py        # 24h patient data aggregation (vectorized)
│   ├── trend_analyzer.py         # HR/SpO2/BP trends + risk escalation
│   ├── prompt_builder.py         # Deterministic LLM prompt construction
│   └── briefing_agent.py         # Gemini LLM client + template fallback
│
├── simulator/
│   ├── __init__.py
│   └── stream_simulator.py       # Loop-based simulator + CSV batch processor
│
├── config.py                     # All thresholds, hyperparams, paths
├── main.py                       # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd telehealth-ml
pip install -r requirements.txt
```

### 2. Run the Full Demo (Recommended First Run)

This generates data → trains the model → runs batch inference in one command:

```bash
python main.py demo
```

### 3. Generate a Doctor Briefing

After running `demo` or `batch`, generate a clinical briefing for any patient:

```bash
# Text mode (default) — concise 30-second summary
python main.py brief --patient-id PAT-A1B2C3D4

# Structured mode — for dashboard rendering
python main.py brief --patient-id PAT-A1B2C3D4 --mode structured
```

**With Gemini LLM** (optional — set your API key):
```bash
set GEMINI_API_KEY=your-api-key-here
python main.py brief --patient-id PAT-A1B2C3D4
```

Without the API key, a deterministic template-based briefing is generated.

### 4. Run Individual Steps

```bash
# Step 1: Generate synthetic vitals dataset
python main.py generate

# Step 2: Train the IsolationForest anomaly detection model
python main.py train

# Step 3: Run batch inference on the generated dataset
python main.py batch

# Step 4: Run real-time stream simulation (5 cycles, 2s intervals)
python main.py simulate --cycles 5 --interval 2
```

---

## 🧱 Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Generator  │────▶│ Feature Engineer  │────▶│  Model Layer    │
│  (vitals + noise │     │ (rolling stats,   │     │ (IsolationForest│
│   + anomaly      │     │  z-scores, deltas,│     │  + Rule Engine) │
│   waves)         │     │  risk aggregation)│     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Alert Engine    │
                                                 │ (structured JSON │
                                                 │  alerts)         │
                                                 └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Briefing Agent  │
                                                 │ (aggregator →    │
                                                 │  trends → prompt │
                                                 │  → Gemini LLM)   │
                                                 └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Doctor Briefing │
                                                 │  (text / struct) │
                                                 └─────────────────┘
```

### Pipeline Flow

1. **Data Generator** — Creates realistic time-series vitals with sustained anomaly waves (tachycardia, hypoxia, hypertension, fever).
2. **Feature Engineer** — Computes rolling means (5/30 min), HR variability, SpO2 min, BP slope, z-scores, deltas, and a composite risk score.
3. **Model Layer** — IsolationForest (trained on normal data) + rule-based risk engine. Scores are blended 60% ML / 40% rules.
4. **Alert Engine** — Generates structured JSON alerts for `MODERATE`, `HIGH`, and `CRITICAL` risk levels.
5. **Briefing Agent** — Aggregates 24h of patient data → analyzes trends → builds prompt → calls Gemini (or template fallback) → outputs structured briefing.

---

## 🤖 Doctor Briefing Agent

### How It Works

When a doctor opens a virtual consultation, the system:

1. **Fetches** the patient's last 24 hours of vitals and inference results
2. **Aggregates**: risk spikes, anomaly frequency, min/max vitals, alert types, fever events
3. **Analyzes trends**: HR slope, SpO2 downward detection, BP volatility, risk escalation (first 12h vs last 12h)
4. **Generates** a concise 30-second clinical summary (via Gemini LLM or template fallback)

### Text Output Format

```json
{
  "patientId": "PAT-A1B2C3D4",
  "briefingText": "Patient PAT-A1B2C3D4 was monitored over 500 readings...",
  "urgencyLevel": "MODERATE",
  "anomalyCount": 78,
  "totalReadings": 500,
  "disclaimer": "⚕️ DISCLAIMER: This summary is AI-generated...",
  "generatedAt": "2025-01-01T12:00:00+00:00",
  "totalLatencyMs": 82.0
}
```

### Structured Output Format (`--mode structured`)

```json
{
  "patientId": "PAT-A1B2C3D4",
  "summary": "Patient PAT-A1B2C3D4 was monitored over 500 readings...",
  "urgencyLevel": "MODERATE",
  "riskHighlights": [
    "SpO2 dropped to 78.2%",
    "Heart rate peaked at 156.3 bpm",
    "Systolic BP peaked at 189.5 mmHg"
  ],
  "trendFindings": [
    "High anomaly rate: 16% of readings flagged",
    "Critical SpO2 dip detected: minimum 78.2%",
    "Hypertension episodes: 42 readings with SBP > 150 mmHg"
  ],
  "alerts": [ ... ],
  "vitalRanges": { ... },
  "riskDistribution": { "LOW": 422, "MODERATE": 73, "HIGH": 5 },
  "disclaimer": "⚕️ DISCLAIMER: ...",
  "totalLatencyMs": 85.0
}
```

### Safety Guardrails

- **No diagnosis** — The agent never provides diagnoses
- **No treatment plans** — Never recommends medication or treatment
- **Mandatory disclaimer** — Every output includes:
  > ⚕️ This summary is AI-generated and should support, not replace, clinical judgment.

---

## 📊 Vital Ranges & Anomalies

| Vital        | Normal Range     | Anomaly Condition        |
|-------------|------------------|--------------------------|
| Heart Rate  | 60–100 bpm       | Tachycardia >120 bpm     |
| SpO2        | 95–100%          | Hypoxia <90%             |
| Systolic BP | 110–130 mmHg     | Hypertension >150 mmHg   |
| Diastolic BP| 70–85 mmHg       | Hypertension >100 mmHg   |
| Temperature | 36.5–37.5°C      | Fever >38.5°C            |

Anomalies are injected as **sustained waves** (5–15 consecutive readings) to mimic real clinical deterioration.

---

## 🔧 Configuration

All parameters are centralized in `config.py`:

- **Anomaly rate** — `ANOMALY_RATE = 0.08` (8% of records)
- **IsolationForest** — `n_estimators=200`, `contamination=0.05`
- **Risk thresholds** — `CRITICAL ≥ 0.80`, `HIGH ≥ 0.60`, `MODERATE ≥ 0.35`
- **Feature windows** — 5-min and 30-min rolling windows
- **Simulation interval** — 30 seconds between readings
- **Briefing LLM** — `gemini-2.0-flash`, max retries=3, timeout=10s

Modify `config.py` to tune any parameter without changing module code.

---

## 📋 Alert Output Format

```json
{
  "patientId": "PAT-A1B2C3D4",
  "riskLevel": "HIGH",
  "anomalyDetected": true,
  "reasons": [
    "ML anomaly detector triggered (score=-0.142)",
    "Tachycardia detected: HR=138 bpm",
    "Hypoxia detected: SpO2=89.5%"
  ],
  "timestamp": "2025-01-01T03:45:00",
  "combinedRiskScore": 0.7234,
  "vitals": {
    "heartRate": 138.0,
    "spo2": 89.5,
    "systolicBP": 122.0,
    "diastolicBP": 78.0,
    "temperature": 37.1
  },
  "alertType": "CARDIAC",
  "requiresImmediateAttention": true
}
```

---

## 🔁 Retraining the Model

```bash
# Generate new data with different anomaly rate
python main.py generate --anomaly-rate 0.1 --patients 20

# Retrain
python main.py train
```

The old model is overwritten at `models/saved_model.joblib`.

---

## 🧪 Sample API Integration

```python
from models.inference import InferenceEngine
from alerts.alert_engine import AlertEngine

# Initialize once (loads model from disk)
engine = InferenceEngine()
alert_engine = AlertEngine()

# Single-reading inference
vitals = {
    "patient_id": "PAT-001",
    "heart_rate": 135.0,
    "spo2": 88.0,
    "systolic_bp": 155.0,
    "diastolic_bp": 102.0,
    "temperature": 38.8,
}

result = engine.predict(vitals)
alert = alert_engine.evaluate(result)

if alert:
    print(f"🚨 {alert['riskLevel']}: {alert['reasons']}")
```

### Doctor Briefing API Integration

```python
import json
from doctor_briefing.briefing_agent import BriefingAgent

# Load inference results (from batch or real-time)
with open("data/full_results.json") as f:
    results = json.load(f)

# Generate briefing
agent = BriefingAgent()
briefing = agent.generate_briefing(results, "PAT-A1B2C3D4", mode="structured")

print(briefing["summary"])
print(f"Urgency: {briefing['urgencyLevel']}")
```

---

## 📦 Dependencies

| Package       | Purpose                              |
|--------------|--------------------------------------|
| numpy        | Numerical computation                |
| pandas       | Data manipulation and time-series    |
| scikit-learn | IsolationForest anomaly detection    |
| joblib       | Model serialization / persistence    |
| google-genai | Gemini LLM for doctor briefings      |

---

## ⚡ Performance

- **Inference latency**: < 200 ms per reading (single-row, no history)
- **Batch throughput**: ~5,000 records in < 60 seconds
- **Model training**: < 5 seconds on 5,000 records
- **Briefing generation**: < 100 ms (template), < 3s (Gemini LLM)
- **Data aggregation**: < 50 ms for 1,440 records
- **Stateless inference**: No server state between calls
