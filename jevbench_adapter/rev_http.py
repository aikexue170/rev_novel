"""Rev adapter: native option probabilities from Rev's `/score` HTTP API (github.com/robbalian/rev).

Rev is a Qwen backbone (Qwen3.5-4B or Qwen3.8-27B) with a rank-16 LoRA and a small pointer head
that scores each offered option in one forward pass; nothing is generated and the softmax over the
option slots is the answer. The same API is served by the repo's Modal endpoint and by
`serve_local.py` (plain Python, one GPU, no Modal), so the maintainer can run the public weights
on his own machine and keep held-out items there.

Request:  POST {endpoint}/score
  {"state": <string or JSON object, sent as-is>,
   "questions": [{"id": "decision", "instructions": <str>, "criteria": {<option key>: <description>}}]}
Response: {"answers": {"decision": {"choice": <key>, "probabilities": {<key>: p, ...}}},
           "server_seconds": <float>, "generated_tokens": 0, ...}
Auth: none by default (`--key-env ''`). If `key_env` names a set variable it is sent as a Bearer
token and never logged.

The server takes one shape of question, a dict of option keys with descriptions, so JevBench's
three primitives map onto it exactly as in the repo's own public-item run
(results/benchmarkheaven/run.py); the mapping was fixed before that run and is unchanged here:
  choice  criteria dict sent as-is; the keys are the benchmark labels; probabilities used as-is
  noul    criteria {"true": ..., "false": ...} sent as the benchmark gives them (if a noul item has no
          criteria, {"true": "Yes", "false": "No"}); answer keys mapped true -> "yes", false -> "no",
          the label set the harness scores over (as the typesafe adapter does with Jev's p_yes)
  score   the ordered level list becomes {"0": level0, "1": level1, ...}; the level indices are
          exactly the benchmark's labels; probabilities used as-is
Every answer is the model's own softmax over the option slots (`native`), never verbalized and
never token logprobs. A distribution whose keys do not cover the exact label set, or a choice
outside it, is a failed answer, not a repaired one.

Input tokens: the server generates nothing, so output_tokens is always 0. It does not report a token
count, so the adapter reports one of its own and says where it came from (usage["input_tokens_source"]):
  * the server's `usage.input_tokens` when the server returns one (serve_local.py may);
  * else, when REV_TOKENIZER names a Hugging Face tokenizer (e.g. Qwen/Qwen3.8-27B), the exact count
    of the prompt the server builds (same template, same tokenizer), loaded once, lazily;
  * else len(prompt) / 4, labelled "estimate_chars_div_4".
The count is for pricing only; it never touches the answer or the timing (the tokenizer runs after
the response is back and its time is excluded from latency_s).

Cost basis: `self_hosted_gpu` by default (override with --cost-basis). The repo's
jevbench_adapter/COST_BASIS.md gives both a self-hosted list-price basis and a hosted-tariff estimate.
"""

from __future__ import annotations

import json
import os
import time

from .base import DecisionResult, http_post_json

USER_AGENT = "JevBench/1.2 (+https://github.com/fstandhartinger/jevbench) rev-adapter"
PING_WAIT_S = 300.0  # Modal cold start of the 27B is up to about two minutes


def rev_criteria(task):
    """(criteria sent to the server, map from the server's option key back to the JevBench label)."""
    qtype, crit = task.question["type"], task.question.get("criteria")
    if qtype == "noul":
        crit = crit if isinstance(crit, dict) else {}
        sent = {"true": crit.get("true") or "Yes", "false": crit.get("false") or "No"}
        return sent, {"true": "yes", "false": "no"}
    if qtype == "score":
        if not isinstance(crit, (list, tuple)) or not crit:
            raise ValueError("score question without a level list")
        sent = {str(i): (str(level) if level is not None else str(i)) for i, level in enumerate(crit)}
        return sent, {k: k for k in sent}
    if not isinstance(crit, dict) or not crit:
        raise ValueError("choice question without an option dict")
    sent = {str(k): v for k, v in crit.items()}
    return sent, {k: k for k in sent}


def rev_prompt(state, instructions, criteria) -> str:
    """The exact text the server tokenizes (server.py `encode`), for token counting only."""
    s = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"))
    text = "State:\n" + s + "\nQuestion: " + instructions + "\nOptions:\n"
    for k, desc in criteria.items():
        text += str(k) + ": " + str(desc or k) + "\n"
    return text + "Decision:"


class RevHttpAdapter:
    name = "rev_http"
    cost_basis = "self_hosted_gpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=120.0,
                 price_input_per_m=None, price_output_per_m=None):
        self.endpoint = (endpoint or os.environ.get("REV_ENDPOINT") or "").rstrip("/")
        if not self.endpoint:
            raise ValueError("rev_http needs --endpoint (or REV_ENDPOINT)")
        self.model = model or "rev"
        self.key_env = key_env or ""
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.tokenizer_name = os.environ.get("REV_TOKENIZER", "")
        self._tokenizer = None
        self.server_metadata = None

    # -- optional warm-up: JEVBENCH_WARM_LOAD=1 makes the CLI call load() before the clock starts
    def load(self):
        """Poll POST /ping until the server answers 200 (a scale-to-zero endpoint may be cold)."""
        deadline = time.time() + PING_WAIT_S
        last = None
        while time.time() < deadline:
            try:
                status, parsed, _ = http_post_json(f"{self.endpoint}/ping", {}, self._headers(), 60.0)
                if status == 200:
                    self.server_metadata = parsed if isinstance(parsed, dict) else None
                    return self.server_metadata
                last = f"HTTP {status}"
            except ConnectionError as e:
                last = str(e)
            time.sleep(5.0)
        raise RuntimeError(f"rev endpoint not ready after {PING_WAIT_S:.0f}s: {last}")

    def _headers(self) -> dict:
        key = os.environ.get(self.key_env, "") if self.key_env else ""
        return {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT,
                **({"Authorization": f"Bearer {key}"} if key else {})}

    def build_request(self, task) -> dict:
        criteria, _ = rev_criteria(task)
        return {"state": task.state,
                "questions": [{"id": "decision", "instructions": task.question["instructions"], "criteria": criteria}]}

    def _count_input_tokens(self, body, parsed):
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), (int, float)) \
                and not isinstance(usage.get("input_tokens"), bool):
            return int(usage["input_tokens"]), "server"
        q = body["questions"][0]
        prompt = rev_prompt(body["state"], q["instructions"], q["criteria"])
        if self.tokenizer_name:
            try:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                return len(self._tokenizer.encode(prompt, add_special_tokens=False)), f"tokenizer:{self.tokenizer_name}"
            except Exception as e:  # noqa: BLE001 - counting must never fail an answer
                return round(len(prompt) / 4), f"estimate_chars_div_4 (tokenizer failed: {type(e).__name__})"
        return round(len(prompt) / 4), "estimate_chars_div_4"

    def run(self, task) -> DecisionResult:
        try:
            criteria, back = rev_criteria(task)
        except ValueError as e:
            return DecisionResult(adapter=self.name, ok=False, error=str(e), probs_source="native", model=self.model)
        body = {"state": task.state,
                "questions": [{"id": "decision", "instructions": task.question["instructions"], "criteria": criteria}]}
        try:
            status, parsed, latency = http_post_json(f"{self.endpoint}/score", body, self._headers(), self.timeout_s)
        except ConnectionError as e:
            return DecisionResult(adapter=self.name, ok=False, error=str(e), probs_source="native",
                                  model=self.model, request_body=body)
        res = DecisionResult(adapter=self.name, ok=False, status=status, latency_s=latency,
                             probs_source="native", model=self.model, raw=parsed, request_body=body)
        if status != 200 or not isinstance(parsed, dict):
            res.error = f"HTTP {status}: {str(parsed)[:300]}"
            return res
        ans = (parsed.get("answers") or {}).get("decision")
        if not isinstance(ans, dict):
            res.error = "missing 'answers.decision'"
            return res
        try:
            probs_raw = ans.get("probabilities")
            if not isinstance(probs_raw, dict):
                raise ValueError("answer missing probabilities")
            if set(probs_raw) != set(back):
                raise ValueError(f"option keys {sorted(probs_raw)} do not match the keys sent {sorted(back)}")
            probs = {}
            for k, v in probs_raw.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"probability for {k!r} is not a number")
                probs[back[k]] = float(v)
            if set(probs) != set(task.labels):
                raise ValueError(f"labels {sorted(probs)} do not match the task's {sorted(task.labels)}")
            choice = ans.get("choice")
            if choice not in back:
                raise ValueError("choice outside the options")
            res.label = back[choice]
        except (KeyError, TypeError, ValueError) as e:
            res.error = f"answer parse failed: {e}"
            return res
        n_in, source = self._count_input_tokens(body, parsed)
        res.usage = {"input_tokens": n_in, "output_tokens": 0, "input_tokens_source": source,
                     "server_seconds": parsed.get("server_seconds"), "generated_tokens": parsed.get("generated_tokens", 0),
                     "path": parsed.get("path")}
        res.probs = probs
        res.ok = True
        return res

    def reserve_estimate(self, task):
        """Self-hosted by default: nothing is billed per request. With explicit prices, a conservative cap."""
        if self.price_input_per_m is None and self.price_output_per_m is None:
            return 0.0
        pin = self.price_input_per_m or 0.0
        pout = self.price_output_per_m or 0.0
        return 100_000 * pin / 1e6 + 4_000 * pout / 1e6
