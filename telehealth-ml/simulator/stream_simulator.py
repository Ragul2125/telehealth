"""
Telehealth ML — Stream Simulator & Batch Processor

Two operational modes:
  1. StreamSimulator  — loop-based real-time simulation (configurable interval).
  2. BatchProcessor   — reads a CSV and runs inference on every row.

Both feed into the inference engine and alert engine, outputting
DynamoDB-compatible JSON results.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from alerts.alert_engine import AlertEngine
from config import (
    DEFAULT_SIMULATION_CYCLES,
    STREAM_INTERVAL_SEC,
    SYNTHETIC_DATASET_PATH,
)
from data.generator import VitalsGenerator
from models.inference import InferenceEngine

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Simulates a real-time vital-sign stream.

    Each cycle:
      1. Generates a fresh vitals reading for a random patient.
      2. Runs inference.
      3. Evaluates alerts.
      4. Logs / collects results.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        alert_engine: AlertEngine,
        interval_sec: float = STREAM_INTERVAL_SEC,
    ):
        self.engine = inference_engine
        self.alert_engine = alert_engine
        self.interval = interval_sec
        self.generator = VitalsGenerator(num_patients=1, records_per_patient=1, seed=None)
        logger.info("StreamSimulator ready (interval=%.1f s)", interval_sec)

    def run(self, cycles: int = DEFAULT_SIMULATION_CYCLES) -> list:
        """
        Execute the simulation loop.

        Returns a list of all inference result dicts (DynamoDB-ready).
        """
        results = []
        logger.info("Starting stream simulation for %d cycles", cycles)

        for i in range(cycles):
            # Generate a single reading
            reading_df = self.generator.generate_dataset()
            vitals = reading_df.iloc[0].to_dict()
            vitals["patient_id"] = f"STREAM-PAT-{(i % 5) + 1:03d}"

            # Inference
            result = self.engine.predict(vitals)
            results.append(result)

            # Alert
            alert = self.alert_engine.evaluate(result)

            # Console output
            status = (
                f"[Cycle {i + 1}/{cycles}] "
                f"Patient={result['patientId']} "
                f"Risk={result['riskLevel']} "
                f"Score={result['combinedRiskScore']:.3f}"
            )
            if alert:
                status += f" ⚠ ALERT: {alert['alertType']}"
                print(f"\n{'=' * 70}")
                print(f"🚨 ALERT — {alert['riskLevel']}")
                print(json.dumps(alert, indent=2, default=str))
                print(f"{'=' * 70}\n")
            else:
                print(status)

            # Wait before next cycle (skip on last)
            if i < cycles - 1:
                time.sleep(self.interval)

        logger.info("Stream simulation complete: %d cycles, %d results", cycles, len(results))
        return results


class BatchProcessor:
    """
    Reads a CSV of vitals and runs batch inference + alerting.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        alert_engine: AlertEngine,
    ):
        self.engine = inference_engine
        self.alert_engine = alert_engine
        logger.info("BatchProcessor ready")

    def process(self, csv_path: Optional[Path] = None) -> dict:
        """
        Process entire CSV through inference + alert pipeline.

        Returns
        -------
        dict with keys:
            results  — list of inference dicts
            alerts   — list of alert dicts
            summary  — aggregated stats
        """
        csv_path = csv_path or SYNTHETIC_DATASET_PATH
        logger.info("Loading batch data from %s", csv_path)

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        logger.info("Loaded %d records for %d patients",
                     len(df), df["patient_id"].nunique())

        # Run batch inference
        results = self.engine.predict_batch(df)

        # Run alert evaluation
        alerts = self.alert_engine.evaluate_batch(results)

        # Summary
        risk_counts = {}
        for r in results:
            lvl = r["riskLevel"]
            risk_counts[lvl] = risk_counts.get(lvl, 0) + 1

        anomaly_count = sum(1 for r in results if r["anomalyDetected"])
        avg_latency = sum(r["inferenceLatencyMs"] for r in results) / max(len(results), 1)

        summary = {
            "totalRecords": len(results),
            "totalAlerts": len(alerts),
            "anomalyCount": anomaly_count,
            "riskDistribution": risk_counts,
            "avgInferenceLatencyMs": round(avg_latency, 2),
        }

        logger.info("Batch processing complete: %s", json.dumps(summary, indent=2))
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total Records:   {summary['totalRecords']}")
        print(f"  Total Alerts:    {summary['totalAlerts']}")
        print(f"  Anomalies Found: {summary['anomalyCount']}")
        print(f"  Avg Latency:     {summary['avgInferenceLatencyMs']:.1f} ms")
        print(f"  Risk Distribution:")
        for level in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
            count = risk_counts.get(level, 0)
            bar = "█" * max(1, count // 10)
            print(f"    {level:10s}: {count:5d}  {bar}")
        print("=" * 60 + "\n")

        # Print sample alerts
        if alerts:
            print(f"📋 Sample Alerts (showing first {min(5, len(alerts))}):\n")
            for alert in alerts[:5]:
                print(json.dumps(alert, indent=2, default=str))
                print()

        return {"results": results, "alerts": alerts, "summary": summary}
