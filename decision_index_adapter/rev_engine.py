"""Decision Index engines for Rev (github.com/robbalian/rev): a Qwen3.5/3.8 backbone with a merged LoRA and a
pointer head that scores every option of a question in one forward pass and returns a softmax over the options.

Two backends, both speaking Rev's /score contract:

  RevHttpEngine   -> an already-running server (our Modal deployment, or serve_local.py you started yourself)
  RevLocalEngine  -> starts `python serve_local.py --repo <hf repo id>` from the Rev checkout as a subprocess on
                     your own GPU, waits for /ping, then behaves like RevHttpEngine and stops it on close()

Kit interface (decision_index/engines/base.py): __call__(state, questions) -> (response, raw) where
response = {"model": ..., "answers": {key: {"type": "choice", "choice": <option key>, "probabilities": {key: p}}}}
and the runner validates every answer with decision_index.engines.validate. Raise Unsupported(reason) for a
declared capacity limit; anything else propagates as an error and is retried on resume.

Schema translation (mechanical, recorded in provenance): the suite's `questions` map becomes Rev's list of
{id, instructions, criteria}; object-valued instructions or option descriptions are JSON-serialized exactly as the
kit's reference transformers engine renders them (engines/base.py text()); option keys and `state` are untouched.
No prompt tuning, no truncation, no option filtering: a request whose rendered prompt exceeds the server's
16,384-token row limit is refused as Unsupported before it is sent.

Run with the kit:
  python -m decision_index run --engine decision_index_adapter.rev_engine:RevHttpEngine \
      --option base_url=http://127.0.0.1:8000 --out runs/rev-4b
  python -m decision_index run --engine decision_index_adapter.rev_engine:RevLocalEngine \
      --option repo=robbalian/rev-qwen3.5-4b --option rev_dir=/path/to/rev --out runs/rev-4b
(the Rev checkout's parent directory, or this package's parent, must be on PYTHONPATH, or register the class in
decision_index/engines/__init__.py REGISTRY as "rev": "decision_index_adapter.rev_engine:RevHttpEngine").
"""
import json
import os
import subprocess
import sys
import time

from decision_index.engines.base import Engine, Unsupported, text

ROW_TOKEN_LIMIT = 16384  # MAX_ROW_TOKENS in serve_local.py and the assert in server.py
CAPACITY_MARKERS = ("exceeds", "tokens exceeds", "context window", "maximum context length")


def to_request(state, questions):
    qs = []
    for key, q in questions.items():
        if q.get("type") != "choice":
            raise Unsupported("Rev answers choice questions only; got " + str(q.get("type")))
        if len(q["criteria"]) < 2:
            raise Unsupported("a choice needs at least two options")
        criteria = {k: (k if d is None else text(d)) for k, d in q["criteria"].items()}
        qs.append({"id": key, "instructions": text(q.get("instructions", "")), "criteria": criteria})
    return {"state": state, "questions": qs}


class RevHttpEngine(Engine):
    name = "rev-http"
    latency = "HTTP request wall time against the configured Rev /score endpoint (server-side prompt construction, batching queue and inference); excludes server start."

    def __init__(self, base_url=None, timeout=900, retries=2, count_tokens=True, model_label=None, **options):
        super().__init__(**options)
        import httpx

        base_url = base_url or os.environ.get("REV_BASE_URL")
        if not base_url:
            raise ValueError("RevHttpEngine needs base_url (or REV_BASE_URL)")
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.meta = self._wait_ready()
        self.state_format = self.meta.get("state_format", "json")
        self.model_label = model_label or self.meta.get("model")
        self.tok = None
        if count_tokens:
            self.tok = self._tokenizer(self.meta.get("model"), self.meta.get("revision"))
        self.provenance = {
            "kind": "http",
            "base_url": self.base_url,
            "server_metadata": {k: v for k, v in self.meta.items() if k in ("model", "revision", "checkpoint_run", "checkpoint", "checkpoint_sha256", "training_examples", "state_format", "temperature", "precision", "gpu", "device", "dtype", "server")},
            "code": "https://github.com/robbalian/rev",
            "policy": "Unmodified state and questions sent as one /score request per suite row (questions map -> list of {id, instructions, criteria}; object instructions/descriptions JSON-serialized). Rows over the server's 16,384-token row limit are Unsupported, never truncated. Transport errors retried; a valid answer is never retried.",
            "row_token_limit": ROW_TOKEN_LIMIT,
        }

    def _wait_ready(self, budget_s=900):
        t0 = time.time()
        last = None
        while time.time() - t0 < budget_s:
            try:
                r = self.client.post("/ping", json={})
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}"
            except Exception as exc:  # cold start / not up yet
                last = type(exc).__name__
            time.sleep(5)
        raise RuntimeError(f"Rev server at {self.base_url} not ready after {budget_s}s ({last})")

    @staticmethod
    def _tokenizer(model, revision):
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            return Tokenizer.from_file(hf_hub_download(model, "tokenizer.json", revision=revision))
        except Exception as exc:
            print(f"warning: could not load tokenizer for {model}: {exc}; relying on server-side rejection", file=sys.stderr)
            return None

    def _row_tokens(self, req):
        s = req["state"] if (self.state_format == "raw" and isinstance(req["state"], str)) else json.dumps(req["state"], separators=(",", ":"))
        worst = 0
        for q in req["questions"]:
            body = "State:\n" + s + "\nQuestion: " + q["instructions"] + "\nOptions:\n" + "".join(str(k) + ": " + str(d or k) + "\n" for k, d in q["criteria"].items()) + "Decision:"
            worst = max(worst, len(self.tok.encode(body, add_special_tokens=False).ids))
        return worst

    def __call__(self, state, questions):
        req = to_request(state, questions)
        if self.tok is not None:
            n = self._row_tokens(req)
            if n > ROW_TOKEN_LIMIT:
                raise Unsupported(f"prompt longer than the server's {ROW_TOKEN_LIMIT}-token row limit ({n} tokens); no truncation allowed")
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = self.client.post("/score", json=req)
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 400 and any(m in r.text for m in CAPACITY_MARKERS):
                raise Unsupported(r.text[:300])
            if r.status_code in (502, 503, 504) or r.status_code >= 500 and attempt < self.retries:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            raw = r.json()
            answers = {k: {"type": "choice", "choice": a["choice"], "probabilities": a["probabilities"]} for k, a in raw["answers"].items()}
            return {"model": self.model_label, "answers": answers}, {k: raw.get(k) for k in ("server_seconds", "path", "rows")}
        raise RuntimeError(f"Rev request failed after {self.retries + 1} attempts: {last!r}")

    def runtime(self):
        return {"server": self.base_url, "server_metadata": self.meta}

    def close(self):
        self.client.close()


class RevLocalEngine(RevHttpEngine):
    """Starts serve_local.py from the Rev checkout on your GPU, then scores over HTTP on localhost."""

    name = "rev-local"
    latency = "HTTP loopback request wall time against serve_local.py (one request at a time, no cross-request batching); excludes model loading and warmup."

    def __init__(self, repo="robbalian/rev-qwen3.5-4b", rev_dir=None, port=8000, host="127.0.0.1", device="cuda", dtype="bf16", base=None, revision=None, python=None, startup_s=1800, **options):
        rev_dir = rev_dir or os.environ.get("REV_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(rev_dir, "serve_local.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"{script} not found; pass rev_dir=<path to the Rev checkout>")
        cmd = [python or sys.executable, script, "--repo", repo, "--port", str(port), "--host", host, "--device", device, "--dtype", dtype]
        if base:
            cmd += ["--base", base]
        if revision:
            cmd += ["--revision", revision]
        self.proc = subprocess.Popen(cmd, cwd=rev_dir, env={**os.environ, "FLA_USE_COMPILE": "0"})
        try:
            super().__init__(base_url=f"http://{host}:{port}", **options)
        except Exception:
            self.proc.terminate()
            raise
        self.provenance.update(kind="local subprocess", repo=repo, command=cmd, weights=f"https://huggingface.co/{repo}")

    def _wait_ready(self, budget_s=1800):
        t0 = time.time()
        while time.time() - t0 < budget_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"serve_local.py exited with code {self.proc.returncode} before becoming ready")
            try:
                r = self.client.post("/ping", json={})
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError("serve_local.py did not become ready in time")

    def close(self):
        super().close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
