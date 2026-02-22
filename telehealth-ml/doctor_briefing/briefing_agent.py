"""
Telehealth ML — Briefing Agent (Local RAG + Optional Gemini)

Pipeline:
  1. DataAggregator   → 24h vital statistics
  2. TrendAnalyzer    → HR/SpO2/BP slopes, risk escalation
  3. VitalsVectorStore→ semantic search for top anomalies (ChromaDB, LOCAL)
  4. PromptBuilder    → structured prompt with RAG context
  5. LLM / Template  → Gemini (if API key set) or rich local template
  6. Structured output

Local RAG (ChromaDB + all-MiniLM-L6-v2) runs fully offline.
Gemini is an optional enhancement — code works without any API key.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── LLM configuration ────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0     # seconds (handles 429 rate limits)
LLM_TIMEOUT_SEC = 10
MAX_OUTPUT_TOKENS = 600

# ── RAG query templates ───────────────────────────────────────
RAG_QUERIES = [
    "critical anomaly tachycardia hypoxia low oxygen",
    "high blood pressure hypertension spike",
    "fever high temperature critical reading",
    "high risk score anomaly alert triggered",
]


class BriefingAgent:
    """
    Orchestrates the full doctor-briefing pipeline.

    Modes
    -----
    Local RAG (default, fully offline):
      VitalsVectorStore → semantic search → template briefing

    LLM-enhanced (optional, requires GEMINI_API_KEY):
      Same RAG → Gemini generates prose from retrieved context

    Usage
    -----
    agent = BriefingAgent()
    agent.build_index(results)                     # one-time, or after new batch
    briefing = agent.generate_briefing(results, "PAT-001", mode="text")
    """

    def __init__(self, auto_index: bool = False):
        from doctor_briefing.data_aggregator import DataAggregator
        from doctor_briefing.prompt_builder import PromptBuilder
        from doctor_briefing.trend_analyzer import TrendAnalyzer
        from doctor_briefing.vector_store import VitalsVectorStore

        self.aggregator    = DataAggregator()
        self.trend_analyzer = TrendAnalyzer()
        self.prompt_builder = PromptBuilder()
        self.vector_store  = VitalsVectorStore()
        self.llm_client    = self._init_llm_client()

        mode = "LLM-enhanced" if self.llm_client else "local RAG (offline)"
        logger.info("BriefingAgent initialized — mode=%s", mode)

    # ── Public API ────────────────────────────────────────────

    def build_index(self, results: List[dict]) -> int:
        """Index inference results into local ChromaDB. Returns count indexed."""
        logger.info("Building local vector index from %d records…", len(results))
        n = self.vector_store.index_results(results)
        logger.info("Vector index ready — %d documents", n)
        return n

    def generate_briefing(
        self,
        results: List[dict],
        patient_id: str,
        mode: str = "text",
    ) -> dict:
        """
        Generate a doctor briefing for a patient.

        Parameters
        ----------
        results : list[dict]   Full inference result list.
        patient_id : str       Patient to brief on.
        mode : str             "text" or "structured".

        Returns
        -------
        dict : Briefing output.
        """
        t0 = time.perf_counter()

        # Step 1: Statistical aggregation
        aggregation = self.aggregator.aggregate(results, patient_id)
        if aggregation["totalReadings"] == 0:
            return self._empty_briefing(patient_id, mode)

        # Step 2: Trend analysis
        trends = self.trend_analyzer.analyze(results, patient_id)

        # Step 3: RAG — semantic search for most relevant anomalies
        rag_context = self._rag_search(patient_id)

        # Step 4: Build prompt (includes RAG context)
        top_alerts = self._extract_top_alerts(results, patient_id)
        prompt_payload = self.prompt_builder.build(aggregation, trends, top_alerts, rag_context)

        # Step 5: Generate briefing text
        if self.llm_client:
            briefing_text = self._call_llm(prompt_payload, aggregation, trends, rag_context)
        else:
            briefing_text = self._rag_template_briefing(aggregation, trends, rag_context)

        # Step 6: Urgency
        urgency = self._determine_urgency(aggregation, trends)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Step 7: Format output
        from doctor_briefing.prompt_builder import SAFETY_DISCLAIMER

        if mode == "structured":
            output = {
                "patientId":        patient_id,
                "summary":          briefing_text,
                "urgencyLevel":     urgency,
                "riskHighlights":   self._risk_highlights(aggregation),
                "trendFindings":    trends.get("insights", []),
                "ragFindings":      [d["text"] for d in rag_context[:5]],
                "alerts":           top_alerts[:10],
                "vitalRanges":      aggregation.get("vitalRanges", {}),
                "riskDistribution": aggregation.get("riskDistribution", {}),
                "disclaimer":       SAFETY_DISCLAIMER,
                "generatedAt":      datetime.now(timezone.utc).isoformat(),
                "totalLatencyMs":   round(elapsed_ms, 1),
                "briefingMode":     "LLM-enhanced" if self.llm_client else "local-rag",
            }
        else:
            output = {
                "patientId":      patient_id,
                "briefingText":   briefing_text,
                "urgencyLevel":   urgency,
                "anomalyCount":   aggregation.get("anomalyCount", 0),
                "totalReadings":  aggregation.get("totalReadings", 0),
                "disclaimer":     SAFETY_DISCLAIMER,
                "generatedAt":    datetime.now(timezone.utc).isoformat(),
                "totalLatencyMs": round(elapsed_ms, 1),
                "briefingMode":   "LLM-enhanced" if self.llm_client else "local-rag",
            }

        logger.info(
            "Briefing generated for %s: urgency=%s, mode=%s, latency=%.0f ms",
            patient_id, urgency,
            "LLM" if self.llm_client else "local-RAG",
            elapsed_ms,
        )
        return output

    # ── RAG: semantic search ──────────────────────────────────

    def _rag_search(self, patient_id: str) -> List[dict]:
        """
        Run multiple semantic queries against the vector store and
        return deduplicated top anomaly documents for the patient.
        """
        seen_texts = set()
        combined = []

        for query in RAG_QUERIES:
            try:
                hits = self.vector_store.search_anomalies(
                    patient_id=patient_id,
                    query=query,
                    top_k=5,
                )
                for hit in hits:
                    if hit["text"] not in seen_texts:
                        seen_texts.add(hit["text"])
                        combined.append(hit)
            except Exception as e:
                logger.debug("RAG query failed: %s", e)

        # Also get top by metadata risk score
        try:
            risk_hits = self.vector_store.search_by_risk(
                patient_id=patient_id,
                risk_levels=["HIGH", "CRITICAL"],
                top_k=5,
            )
            for hit in risk_hits:
                if hit["text"] not in seen_texts:
                    seen_texts.add(hit["text"])
                    combined.append(hit)
        except Exception as e:
            logger.debug("Risk metadata search failed: %s", e)

        # Sort by similarity then risk score
        combined.sort(
            key=lambda x: (x.get("similarity", 0), x["metadata"].get("riskScore", 0)),
            reverse=True,
        )
        logger.info("RAG retrieved %d unique context docs for %s", len(combined), patient_id)
        return combined[:15]

    # ── LLM call with retry ───────────────────────────────────

    def _call_llm(
        self,
        prompt_payload: dict,
        aggregation: dict,
        trends: dict,
        rag_context: List[dict],
    ) -> str:
        """Call Gemini with retry logic. Falls back to local RAG template on failure."""
        from google.genai import types

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                response = self.llm_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt_payload["user_prompt"],
                    config=types.GenerateContentConfig(
                        system_instruction=prompt_payload["system_prompt"],
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.3,
                    ),
                )
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("LLM call succeeded (attempt %d, %.0f ms)", attempt, elapsed)
                if response and response.text:
                    return response.text.strip()
                logger.warning("Empty LLM response on attempt %d", attempt)

            except Exception as e:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt, MAX_RETRIES, str(e)[:150], wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        logger.error("All LLM attempts failed. Using local RAG template.")
        return self._rag_template_briefing(aggregation, trends, rag_context)

    # ── Local RAG template briefing ───────────────────────────

    def _rag_template_briefing(
        self,
        aggregation: dict,
        trends: dict,
        rag_context: List[dict],
    ) -> str:
        """
        Generate a rich briefing from aggregated stats + RAG-retrieved context.
        Fully offline — no API calls.
        """
        patient  = aggregation.get("patientId", "UNKNOWN")
        readings = aggregation.get("totalReadings", 0)
        anomalies = aggregation.get("anomalyCount", 0)
        max_risk  = aggregation.get("maxRiskScore", 0.0)
        risk_dist = aggregation.get("riskDistribution", {})
        vitals    = aggregation.get("vitalRanges", {})
        insights  = trends.get("insights", [])

        hr   = vitals.get("heartRate",  {})
        spo2 = vitals.get("spo2",       {})
        sbp  = vitals.get("systolicBP", {})

        paragraphs = []

        # ── Para 1: Overall status ──
        if anomalies == 0:
            paragraphs.append(
                f"Patient {patient} was monitored across {readings} readings. "
                "All vital signs remained within normal parameters — no anomalies detected."
            )
        else:
            pct = (anomalies / max(readings, 1)) * 100
            dist_str = ", ".join(f"{k}: {v}" for k, v in risk_dist.items() if v > 0)
            paragraphs.append(
                f"Patient {patient} was monitored across {readings} readings. "
                f"{anomalies} anomalous readings ({pct:.0f}%) were flagged. "
                f"Risk distribution — {dist_str}. Peak risk score: {max_risk:.3f}."
            )

        # ── Para 2: Key abnormalities (stats + trend insights) ──
        key_points = list(insights[:4])  # from TrendAnalyzer

        # Add vital range specifics
        if hr.get("max") and hr["max"] > 120:
            key_points.append(f"Heart rate peaked at {hr['max']} bpm (mean {hr.get('mean', '?')} bpm)")
        if spo2.get("min") and spo2["min"] < 92:
            key_points.append(f"SpO2 dipped to {spo2['min']}% (mean {spo2.get('mean', '?')}%)")
        if sbp.get("max") and sbp["max"] > 150:
            key_points.append(f"Systolic BP peaked at {sbp['max']} mmHg (mean {sbp.get('mean', '?')} mmHg)")

        # Add top RAG-retrieved anomaly reasons (deduplicated)
        seen_reasons = set()
        rag_points = []
        for doc in rag_context[:8]:
            meta = doc.get("metadata", {})
            rl   = meta.get("riskLevel", "")
            rs   = meta.get("riskScore", 0)
            # Pull reason fragment from text
            text = doc.get("text", "")
            if "Reasons:" in text:
                reason_part = text.split("Reasons:")[-1].strip().rstrip(".")
                if reason_part and reason_part not in seen_reasons and reason_part != "no specific triggers":
                    seen_reasons.add(reason_part)
                    if rs > 0.4:
                        rag_points.append(f"{reason_part} (risk={rl}, score={rs:.2f})")

        if rag_points:
            key_points.append("Most significant events: " + "; ".join(rag_points[:3]))

        if key_points:
            paragraphs.append("Key findings: " + "; ".join(key_points) + ".")
        else:
            paragraphs.append(
                f"Heart rate ranged {hr.get('min', 'N/A')}–{hr.get('max', 'N/A')} bpm. "
                f"SpO2 ranged {spo2.get('min', 'N/A')}–{spo2.get('max', 'N/A')}%. "
                f"Systolic BP ranged {sbp.get('min', 'N/A')}–{sbp.get('max', 'N/A')} mmHg."
            )

        # ── Para 3: Urgency & recommendation ──
        urgency = self._determine_urgency(aggregation, trends)
        urgency_text = {
            "CRITICAL": "⚠️ CRITICAL urgency — immediate clinical review is strongly indicated.",
            "HIGH":     "Elevated urgency — close monitoring and prompt review advised.",
            "MODERATE": "Moderate level of concern — continued monitoring warranted.",
            "LOW":      "Low urgency — routine monitoring is appropriate.",
        }
        paragraphs.append(urgency_text.get(urgency, "Routine monitoring continues."))

        return "\n\n".join(paragraphs)

    # ── Urgency determination ─────────────────────────────────

    @staticmethod
    def _determine_urgency(aggregation: dict, trends: dict) -> str:
        max_risk   = aggregation.get("maxRiskScore", 0)
        risk_dist  = aggregation.get("riskDistribution", {})
        escalation = trends.get("riskEscalation", {}).get("pattern", "stable")

        if risk_dist.get("CRITICAL", 0) > 0 or max_risk >= 0.80:
            return "CRITICAL"
        if risk_dist.get("HIGH", 0) > 0 or max_risk >= 0.60 or escalation == "worsening":
            return "HIGH"
        if risk_dist.get("MODERATE", 0) > 5 or max_risk >= 0.40:
            return "MODERATE"
        return "LOW"

    # ── Helpers ────────────────────────────────────────────────

    def _init_llm_client(self):
        """Initialize Gemini client if API key present. Returns None otherwise."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.info("No GEMINI_API_KEY — running in fully local RAG mode.")
            return None
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client ready (model=%s)", GEMINI_MODEL)
            return client
        except ImportError:
            logger.warning("google-genai not installed. Using local RAG.")
            return None
        except Exception as e:
            logger.error("Gemini init failed: %s", e)
            return None

    @staticmethod
    def _extract_top_alerts(results: List[dict], patient_id: str) -> list:
        patient_results = [r for r in results if r.get("patientId") == patient_id]
        anomalous = [
            r for r in patient_results
            if str(r.get("anomalyDetected", "")).lower() == "true"
        ]
        anomalous.sort(key=lambda x: x.get("combinedRiskScore", 0), reverse=True)
        return anomalous[:10]

    @staticmethod
    def _risk_highlights(aggregation: dict) -> list:
        highlights = []
        max_risk = aggregation.get("maxRiskScore", 0)
        if max_risk >= 0.60:
            highlights.append(f"Peak risk score: {max_risk:.3f}")

        vitals = aggregation.get("vitalRanges", {})
        spo2 = vitals.get("spo2", {})
        if spo2.get("min") and spo2["min"] < 90:
            highlights.append(f"SpO2 dropped to {spo2['min']}%")

        hr = vitals.get("heartRate", {})
        if hr.get("max") and hr["max"] > 120:
            highlights.append(f"Heart rate peaked at {hr['max']} bpm")

        bp = vitals.get("systolicBP", {})
        if bp.get("max") and bp["max"] > 150:
            highlights.append(f"Systolic BP peaked at {bp['max']} mmHg")

        fever = aggregation.get("feverEvents", 0)
        if fever > 0:
            highlights.append(f"{fever} fever episodes recorded")

        if not highlights:
            highlights.append("No significant risk indicators")
        return highlights

    @staticmethod
    def _empty_briefing(patient_id: str, mode: str) -> dict:
        from doctor_briefing.prompt_builder import SAFETY_DISCLAIMER
        base = {
            "patientId":    patient_id,
            "urgencyLevel": "LOW",
            "disclaimer":   SAFETY_DISCLAIMER,
            "generatedAt":  datetime.now(timezone.utc).isoformat(),
            "totalLatencyMs": 0.0,
            "briefingMode": "local-rag",
        }
        if mode == "structured":
            base.update({
                "summary":        "No data available for this patient in the monitoring window.",
                "riskHighlights": [],
                "trendFindings":  [],
                "ragFindings":    [],
                "alerts":         [],
            })
        else:
            base["briefingText"]  = "No data available for this patient in the monitoring window."
            base["anomalyCount"]  = 0
            base["totalReadings"] = 0
        return base
