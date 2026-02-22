"""
Telehealth ML — Vitals Vector Store (ChromaDB + Local Embeddings)

Provides a fully offline RAG layer for doctor briefings:
  • Converts inference results into clinical text documents
  • Embeds locally with sentence-transformers (all-MiniLM-L6-v2)
  • Stores in persistent ChromaDB collection
  • Supports semantic search + metadata filtering by patient/risk level

No external API calls. No rate limits. Runs entirely on-device.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────
CHROMA_COLLECTION = "patient_vitals"
EMBED_MODEL = "all-MiniLM-L6-v2"   # ~80 MB, auto-downloaded on first use
DEFAULT_TOP_K = 10


def _make_doc_id(record: dict) -> str:
    """Stable, unique ID for each inference record."""
    key = f"{record.get('patientId','')}-{record.get('timestamp','')}"
    return hashlib.md5(key.encode()).hexdigest()


def _record_to_text(record: dict) -> str:
    """
    Convert one inference result into a clinical narrative sentence.
    This is what gets embedded and searched.
    """
    pid = record.get("patientId", "UNKNOWN")
    ts  = record.get("timestamp", "")
    risk = record.get("riskLevel", "LOW")
    score = record.get("combinedRiskScore", 0.0)
    anomaly = record.get("anomalyDetected", False)
    reasons = record.get("reasons", [])
    vitals  = record.get("vitals", {})

    hr   = vitals.get("heartRate",   "N/A")
    spo2 = vitals.get("spo2",        "N/A")
    sbp  = vitals.get("systolicBP",  "N/A")
    dbp  = vitals.get("diastolicBP", "N/A")
    temp = vitals.get("temperature", "N/A")

    anomaly_str = "ANOMALY DETECTED" if str(anomaly).lower() == "true" else "normal"
    reasons_str = "; ".join(reasons) if reasons else "no specific triggers"

    return (
        f"Patient {pid} at {ts}: {anomaly_str}. "
        f"Risk level: {risk} (score={score:.3f}). "
        f"Heart rate={hr}bpm, SpO2={spo2}%, SystolicBP={sbp}mmHg, "
        f"DiastolicBP={dbp}mmHg, Temperature={temp}C. "
        f"Reasons: {reasons_str}."
    )


class VitalsVectorStore:
    """
    Local ChromaDB-backed vector store for patient vitals inference results.

    Usage
    -----
    store = VitalsVectorStore()
    store.index_results(results)                       # Build / refresh index
    docs = store.search_anomalies("PAT-001", "critical hypoxia tachycardia")
    """

    def __init__(self, persist_dir: Optional[str] = None):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        if persist_dir is None:
            # Default: data/chroma_db/ next to this package
            base = Path(__file__).parent.parent / "data" / "chroma_db"
        else:
            base = Path(persist_dir)

        base.mkdir(parents=True, exist_ok=True)
        self._persist_dir = str(base)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu",
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VitalsVectorStore ready — collection=%s, docs=%d, path=%s",
            CHROMA_COLLECTION, self._collection.count(), self._persist_dir,
        )

    # ── Indexing ──────────────────────────────────────────────

    def index_results(self, results: List[dict], batch_size: int = 200) -> int:
        """
        Index (upsert) a list of inference results into ChromaDB.

        Parameters
        ----------
        results : list[dict]   Full inference result list.
        batch_size : int       Documents per ChromaDB upsert call.

        Returns
        -------
        int : Number of documents indexed.
        """
        if not results:
            logger.warning("No results to index.")
            return 0

        t0 = time.perf_counter()

        # Build parallel lists required by ChromaDB
        ids, docs, metas = [], [], []
        for r in results:
            rid = _make_doc_id(r)
            doc = _record_to_text(r)
            meta = {
                "patientId":   str(r.get("patientId", "")),
                "riskLevel":   str(r.get("riskLevel", "LOW")),
                "anomaly":     str(r.get("anomalyDetected", False)).lower(),
                "timestamp":   str(r.get("timestamp", "")),
                "riskScore":   float(r.get("combinedRiskScore", 0.0)),
            }
            ids.append(rid)
            docs.append(doc)
            metas.append(meta)

        # Upsert in batches
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
            )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Indexed %d records into ChromaDB in %.0f ms (total docs=%d)",
            total, elapsed, self._collection.count(),
        )
        return total

    # ── Search ────────────────────────────────────────────────

    def search_anomalies(
        self,
        patient_id: str,
        query: str = "critical vital sign anomaly tachycardia hypoxia hypertension fever",
        top_k: int = DEFAULT_TOP_K,
    ) -> List[dict]:
        """
        Semantic search for anomalous readings for a specific patient.

        Parameters
        ----------
        patient_id : str   Patient to search within.
        query : str        Natural language query.
        top_k : int        Number of results to return.

        Returns
        -------
        list[dict] : Matched documents with text, metadata, and distance.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            where={"$and": [
                {"patientId": {"$eq": patient_id}},
                {"anomaly":   {"$eq": "true"}},
            ]},
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def search_by_risk(
        self,
        patient_id: str,
        risk_levels: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[dict]:
        """
        Metadata-filtered search — retrieve highest-risk readings for a patient.

        Parameters
        ----------
        patient_id : str        Patient to filter on.
        risk_levels : list[str] e.g. ["HIGH", "CRITICAL"] — None returns all.
        top_k : int             Maximum number of results.

        Returns
        -------
        list[dict] : Matched documents sorted by risk score descending.
        """
        where_clause: dict
        if risk_levels:
            if len(risk_levels) == 1:
                where_clause = {"$and": [
                    {"patientId": {"$eq": patient_id}},
                    {"riskLevel": {"$eq": risk_levels[0]}},
                ]}
            else:
                where_clause = {"$and": [
                    {"patientId": {"$eq": patient_id}},
                    {"riskLevel": {"$in": risk_levels}},
                ]}
        else:
            where_clause = {"patientId": {"$eq": patient_id}}

        results = self._collection.query(
            query_texts=["high risk anomaly critical alert"],
            n_results=min(top_k, self._collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        items = self._format_results(results)
        # Sort by risk score descending
        items.sort(key=lambda x: x["metadata"].get("riskScore", 0), reverse=True)
        return items

    def get_patient_count(self, patient_id: str) -> int:
        """Count total indexed documents for a patient."""
        result = self._collection.get(
            where={"patientId": {"$eq": patient_id}},
            include=[],
        )
        return len(result.get("ids", []))

    def total_docs(self) -> int:
        """Total documents in the collection."""
        return self._collection.count()

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _format_results(raw) -> List[dict]:
        """Flatten ChromaDB query result into a list of dicts."""
        output = []
        docs  = (raw.get("documents")  or [[]])[0]
        metas = (raw.get("metadatas")  or [[]])[0]
        dists = (raw.get("distances")  or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            output.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": round(1.0 - dist, 4),   # cosine dist → similarity
            })
        return output
