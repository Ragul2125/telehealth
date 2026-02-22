"""
Telehealth ML Subsystem — Main Entry Point

CLI orchestrator that wires all modules together.

Usage:
    python main.py generate          Generate synthetic dataset
    python main.py train             Train IsolationForest model
    python main.py batch             Run batch inference on CSV
    python main.py simulate [--cycles N] [--interval S]
                                     Run real-time stream simulation
    python main.py demo              Full pipeline: generate → train → batch → index
    python main.py index             Build local vector index from full_results.json
    python main.py brief --patient-id PAT-XXX [--mode structured]
                                     Generate doctor briefing for a patient
"""

import argparse
import json
import logging
import sys
import time

from config import (
    CHROMA_DB_PATH,
    DEFAULT_SIMULATION_CYCLES,
    FULL_RESULTS_PATH,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    SAVED_MODEL_PATH,
    STREAM_INTERVAL_SEC,
    SYNTHETIC_DATASET_PATH,
)


def setup_logging():
    """Configure production logging."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def cmd_generate(args):
    """Generate synthetic vitals dataset."""
    from data.generator import VitalsGenerator

    logger = logging.getLogger("main.generate")
    gen = VitalsGenerator(
        num_patients=args.patients,
        records_per_patient=args.records,
        anomaly_rate=args.anomaly_rate,
    )
    df = gen.generate_dataset()
    path = gen.save_csv(df)
    logger.info("✅ Dataset generated: %d records → %s", len(df), path)


def cmd_train(args):
    """Train the IsolationForest model."""
    from models.train import ModelTrainer

    logger = logging.getLogger("main.train")
    trainer = ModelTrainer()
    meta = trainer.train()
    logger.info("✅ Training complete")
    print(json.dumps(meta, indent=2))


def cmd_batch(args):
    """Run batch inference on the synthetic dataset."""
    from alerts.alert_engine import AlertEngine
    from models.inference import InferenceEngine
    from simulator.stream_simulator import BatchProcessor

    logger = logging.getLogger("main.batch")

    if not SAVED_MODEL_PATH.exists():
        logger.error("No trained model found at %s. Run 'python main.py train' first.", SAVED_MODEL_PATH)
        sys.exit(1)

    engine = InferenceEngine()
    alert_engine = AlertEngine()
    processor = BatchProcessor(engine, alert_engine)
    output = processor.process()

    # Save full results for briefing agent
    with open(FULL_RESULTS_PATH, "w") as f:
        json.dump(output["results"], f, indent=2, default=str)

    # Save summary
    output_path = SYNTHETIC_DATASET_PATH.parent / "batch_results.json"
    with open(output_path, "w") as f:
        json.dump(output["summary"], f, indent=2, default=str)
    logger.info("✅ Batch results saved to %s", output_path)
    logger.info("✅ Full results saved to %s (for briefing agent)", FULL_RESULTS_PATH)

    # Auto-index into ChromaDB
    print("\n🔍 Auto-indexing results into local vector store...")
    _build_vector_index(output["results"])


def cmd_simulate(args):
    """Run real-time stream simulation."""
    from alerts.alert_engine import AlertEngine
    from models.inference import InferenceEngine
    from simulator.stream_simulator import StreamSimulator

    logger = logging.getLogger("main.simulate")

    if not SAVED_MODEL_PATH.exists():
        logger.error("No trained model found. Run 'python main.py train' first.")
        sys.exit(1)

    engine = InferenceEngine()
    alert_engine = AlertEngine()
    simulator = StreamSimulator(engine, alert_engine, interval_sec=args.interval)

    try:
        results = simulator.run(cycles=args.cycles)
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
        results = []

    logger.info("✅ Simulation complete: %d results collected", len(results))


def cmd_demo(args):
    """Full demo pipeline: generate → train → batch → index."""
    logger = logging.getLogger("main.demo")

    print("\n" + "=" * 60)
    print("🏥 TELEHEALTH ML SUBSYSTEM — FULL DEMO")
    print("=" * 60)

    # Step 1: Generate
    print("\n📊 Step 1/4: Generating synthetic dataset...")
    from data.generator import VitalsGenerator
    gen = VitalsGenerator()
    df = gen.generate_dataset()
    gen.save_csv(df)
    print(f"   ✅ Generated {len(df)} records for {df['patient_id'].nunique()} patients\n")

    # Step 2: Train
    print("🧠 Step 2/4: Training IsolationForest model...")
    from models.train import ModelTrainer
    trainer = ModelTrainer()
    meta = trainer.train()
    print(f"   ✅ Model trained on {meta['training_records']} records in {meta['training_time_sec']}s\n")

    # Step 3: Batch inference
    print("🔍 Step 3/4: Running batch inference...")
    from alerts.alert_engine import AlertEngine
    from models.inference import InferenceEngine
    from simulator.stream_simulator import BatchProcessor

    engine = InferenceEngine()
    alert_engine = AlertEngine()
    processor = BatchProcessor(engine, alert_engine)
    output = processor.process()

    # Save FULL results for briefing agent
    with open(FULL_RESULTS_PATH, "w") as f:
        json.dump(output["results"], f, default=str)
    print(f"   📁 Full results saved to {FULL_RESULTS_PATH}")

    # Save demo summary
    output_path = SYNTHETIC_DATASET_PATH.parent / "demo_results.json"
    serializable = {
        "summary": output["summary"],
        "alerts": output["alerts"][:20],
        "sample_results": output["results"][:10],
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Step 4: Build local vector index
    print("\n🧠 Step 4/4: Building local vector index (ChromaDB)...")
    n_indexed = _build_vector_index(output["results"])
    print(f"   ✅ Indexed {n_indexed} records into local vector store\n")

    # List available patient IDs for briefing
    patient_ids = list({r["patientId"] for r in output["results"] if r.get("patientId")})
    if patient_ids:
        print(f"   📋 Available patient IDs for briefing:")
        for pid in sorted(patient_ids)[:5]:
            print(f"      • {pid}")
        print(f"\n   💡 Try: python main.py brief --patient-id {sorted(patient_ids)[0]}")

    print(f"\n📁 Demo results saved to {output_path}")
    print("\n🏁 Demo complete!\n")


def cmd_index(args):
    """Build / refresh local ChromaDB vector index from full_results.json."""
    if not FULL_RESULTS_PATH.exists():
        print(f"\n❌ No inference results found at {FULL_RESULTS_PATH}")
        print("   Run 'python main.py demo' or 'python main.py batch' first.")
        sys.exit(1)

    print(f"\n📂 Loading results from {FULL_RESULTS_PATH}...")
    with open(FULL_RESULTS_PATH, "r") as f:
        results = json.load(f)
    print(f"   Loaded {len(results)} records")

    print("\n🧠 Building local vector index (ChromaDB)...")
    n = _build_vector_index(results)
    print(f"\n✅ Vector index built — {n} documents indexed at {CHROMA_DB_PATH}\n")


def cmd_brief(args):
    """Generate a doctor briefing for a patient using local RAG."""
    logger = logging.getLogger("main.brief")

    # Load full results
    if not FULL_RESULTS_PATH.exists():
        logger.error(
            "No inference results found at %s.\n"
            "Run 'python main.py demo' or 'python main.py batch' first.",
            FULL_RESULTS_PATH,
        )
        sys.exit(1)

    print(f"\n🏥 Loading inference results from {FULL_RESULTS_PATH}...")
    with open(FULL_RESULTS_PATH, "r") as f:
        results = json.load(f)
    print(f"   Loaded {len(results)} records")

    # Check if patient exists
    patient_ids = {r.get("patientId", "") for r in results}
    if args.patient_id not in patient_ids:
        print(f"\n❌ Patient '{args.patient_id}' not found.")
        available = sorted(pid for pid in patient_ids if pid)
        if available:
            print(f"   Available patients: {', '.join(available[:10])}")
            print(f"\n   💡 Try: python main.py brief --patient-id {available[0]}")
        sys.exit(1)

    # Auto-build index if ChromaDB doesn't exist yet
    if not CHROMA_DB_PATH.exists() or not any(CHROMA_DB_PATH.iterdir()):
        print("\n🧠 Vector index not found — building now (first time only)...")
        _build_vector_index(results)
        print("")

    # Generate briefing
    from doctor_briefing.briefing_agent import BriefingAgent

    mode_label = "structured " if args.mode == "structured" else ""
    print(f"\n🤖 Generating {mode_label}briefing for {args.patient_id}...")
    agent = BriefingAgent()
    briefing = agent.generate_briefing(results, args.patient_id, mode=args.mode)

    # Display
    print("\n" + "=" * 60)
    print(f"📋 DOCTOR BRIEFING — Patient {args.patient_id}")
    print(f"   Urgency: {briefing.get('urgencyLevel', 'N/A')}")
    print(f"   Mode:    {briefing.get('briefingMode', 'N/A')}")
    print(f"   Latency: {briefing.get('totalLatencyMs', 0):.0f} ms")
    print("=" * 60)

    if args.mode == "structured":
        print(f"\n📝 Summary:\n{briefing.get('summary', 'N/A')}")

        print(f"\n🔍 RAG Findings (semantic search):")
        for r in briefing.get("ragFindings", [])[:3]:
            print(f"   • {r}")

        print(f"\n🔴 Risk Highlights:")
        for h in briefing.get("riskHighlights", []):
            print(f"   • {h}")

        print(f"\n📈 Trend Findings:")
        for t in briefing.get("trendFindings", []):
            print(f"   • {t}")

        print(f"\n🚨 Top Alerts ({len(briefing.get('alerts', []))}):")
        for a in briefing.get("alerts", [])[:5]:
            print(f"   [{a.get('riskLevel')}] {', '.join(a.get('reasons', []))}")
    else:
        print(f"\n{briefing.get('briefingText', 'No briefing generated.')}")
        print(f"\n   Anomalies: {briefing.get('anomalyCount', 0)}")
        print(f"   Readings:  {briefing.get('totalReadings', 0)}")

    print(f"\n{briefing.get('disclaimer', '')}")
    print("=" * 60)

    # Save to JSON
    output_path = SYNTHETIC_DATASET_PATH.parent / f"briefing_{args.patient_id}.json"
    with open(output_path, "w") as f:
        json.dump(briefing, f, indent=2, default=str)
    print(f"\n📁 Briefing saved to {output_path}\n")


# ── Shared helper ─────────────────────────────────────────────

def _build_vector_index(results: list) -> int:
    """Build / refresh ChromaDB index. Returns number of records indexed."""
    from doctor_briefing.vector_store import VitalsVectorStore
    store = VitalsVectorStore()
    return store.index_results(results)


# ── Argument parser ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Telehealth ML Subsystem CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate                        Generate synthetic vitals CSV
  python main.py train                           Train anomaly detection model
  python main.py batch                           Batch-process CSV through ML pipeline
  python main.py simulate --cycles 5             Simulate 5 real-time readings
  python main.py demo                            Run full pipeline end-to-end + index
  python main.py index                           Build/refresh local vector index
  python main.py brief --patient-id PAT-A1B2C3D4  Generate doctor briefing (local RAG)
  python main.py brief --patient-id PAT-A1B2C3D4 --mode structured
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--patients", type=int, default=10, help="Number of patients")
    gen_parser.add_argument("--records", type=int, default=500, help="Records per patient")
    gen_parser.add_argument("--anomaly-rate", type=float, default=0.08, help="Anomaly rate (0-1)")

    # train
    subparsers.add_parser("train", help="Train IsolationForest model")

    # batch
    subparsers.add_parser("batch", help="Run batch inference on dataset")

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run real-time simulation")
    sim_parser.add_argument("--cycles", type=int, default=DEFAULT_SIMULATION_CYCLES, help="Number of cycles")
    sim_parser.add_argument("--interval", type=float, default=STREAM_INTERVAL_SEC, help="Seconds between readings")

    # demo
    subparsers.add_parser("demo", help="Run full demo pipeline (generate → train → batch → index)")

    # index
    subparsers.add_parser("index", help="Build/refresh local ChromaDB vector index")

    # brief
    brief_parser = subparsers.add_parser("brief", help="Generate doctor briefing (local RAG, offline)")
    brief_parser.add_argument("--patient-id", required=True, help="Patient ID to brief on")
    brief_parser.add_argument(
        "--mode", choices=["text", "structured"], default="text",
        help="Output mode: 'text' (default) or 'structured'",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()

    commands = {
        "generate": cmd_generate,
        "train":    cmd_train,
        "batch":    cmd_batch,
        "simulate": cmd_simulate,
        "demo":     cmd_demo,
        "index":    cmd_index,
        "brief":    cmd_brief,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
