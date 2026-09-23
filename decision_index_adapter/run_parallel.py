"""Concurrent drop-in for `decision_index run` when the engine is a batching HTTP server (our Modal deployment).
The kit's runner (decision_index/runner.py) sends one request at a time; against a server that batches, that is
about 15x slower than 8 in flight. This writes results.jsonl / environment.json / status.json in exactly the
kit's row format (same fields as runner.py, plus http_wall_ms), resumes like the kit, and `decision_index score`
consumes the output unchanged.

usage: run_parallel.py --rows suite/selected-rows.jsonl.gz --out runs/rev-27b --base-url https://... [--concurrency 8] [--limit N]
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from decision_index.engines import Unsupported, validate
from decision_index.suite.io import atomic_json, dumps, read_jsonl, sha256_file

from decision_index_adapter.rev_engine import RevHttpEngine


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = RevHttpEngine(base_url=a.base_url)
    engine.client = engine.client.__class__(base_url=engine.base_url, timeout=900, limits=__import__("httpx").Limits(max_connections=a.concurrency + 2))
    atomic_json(out / "environment.json", {"engine": "decision_index_adapter.run_parallel:RevHttpEngine", "engine_options": {"base_url": a.base_url, "concurrency": a.concurrency}, "model_source": engine.provenance, **engine.runtime(), "frozen_corpus_sha256": sha256_file(a.rows), "rows_path": str(a.rows), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "latency": engine.latency + f" Concurrency {a.concurrency}: wall time includes queueing behind the other in-flight requests."})
    results_path = out / "results.jsonl"
    done = {}
    if results_path.exists():
        for r in read_jsonl(results_path, complete_lines_only=True):
            done[r["run_id"]] = r["status"]
    rows = [r for r in read_jsonl(a.rows) if done.get(r["_evaluation"]["run_id"], "error") == "error"]
    if a.limit:
        rows = rows[: a.limit]
    lock = threading.Lock()
    counts = collections.Counter()
    start = time.perf_counter()

    def one(row):
        e = row["_evaluation"]
        payload = {"state": row["state"], "questions": row["questions"]}
        result = {**e, "started_utc": stamp(), "engine": engine.name}
        if not a.compact:
            result["payload"] = payload
        t = time.perf_counter()
        try:
            response, raw = engine(**payload)
            validate(payload["questions"], response)
            result.update(status="ok", response=response, http_wall_ms=(time.perf_counter() - t) * 1000)
            if not a.compact:
                result["raw_output"] = raw
        except Unsupported as exc:
            result.update(status="unsupported", error=str(exc))
        except Exception as exc:
            result.update(status="error", error=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc())
        elapsed = (time.perf_counter() - t) * 1000
        result.update(completed_utc=stamp(), total_wall_ms=elapsed, model_request_wall_ms=elapsed)
        with lock:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(dumps(result) + "\n")
            counts[result["status"]] += 1
            n = sum(counts.values())
            if n % 100 == 0:
                atomic_json(out / "status.json", {"time": stamp(), "event": "progress", "completed": len(done) + n, "counts": dict(counts), "elapsed_seconds": round(time.perf_counter() - start, 1)}, indent=None)
        return result["status"]

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(one, rows))
    engine.close()
    final = {"time": stamp(), "event": "complete", "completed": len(done) + sum(counts.values()), "counts": dict(counts), "elapsed_seconds": round(time.perf_counter() - start, 1)}
    atomic_json(out / "status.json", final, indent=None)
    print(json.dumps(final))


if __name__ == "__main__":
    main()
