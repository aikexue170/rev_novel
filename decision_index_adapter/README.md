# Rev on the Decision Index

Engine adapters and run instructions so anyone can score Rev (https://github.com/robbalian/rev) on the
Decision Index (leaderboard: https://huggingface.co/spaces/multimodalart/jev-decision-index; kit:
https://github.com/apolinario/decision-index, commit 52a6989 at the time of writing). Nothing here changes the
prompt, options or scoring: the adapter is a mechanical schema translation between the suite's request shape and
Rev's `/score` API.

## How the kit works (file paths in the kit)

- **Request shape.** One suite row = `state` (string or JSON) + `questions`: a map `key -> {"type": "choice",
  "instructions": str|object, "criteria": {option_key: description}}` (`docs/format.md`). The runner passes exactly
  `state` and `questions` to the engine (`decision_index/runner.py`, `payload = {"state": ..., "questions": ...}`).
- **Engine interface.** Subclass `decision_index.engines.Engine` (`decision_index/engines/base.py`) and implement
  `__call__(state, questions) -> (response, raw)` with
  `response = {"model": ..., "answers": {key: {"type": "choice", "choice": option_key, "probabilities": {option_key: p, ...}}}}`.
  `validate()` in the same file rejects a response unless every question is answered, the choice is one of the
  criteria keys, every option has a finite probability in [0, 1] and they sum to 1 within 0.01. Raise
  `Unsupported(reason)` for a declared capacity limit (recorded as `unsupported`, scored as wrong, never retried);
  any other exception is `error` (retried on resume). Set `provenance` (dict written to `environment.json`) and
  `latency` (one sentence on what the timing covers). Reference engines: `decision_index/engines/http.py`
  (Jev-compatible `POST /v1/systemone`) and `decision_index/engines/transformers_engine.py` (`docs/engines.md`).
- **Registration.** `decision_index/engines/__init__.py` holds `REGISTRY = {"http": ..., "transformers": ...,
  "random": ...}`; `load_engine()` also accepts any dotted `module:Class` path via `--engine`, and `--option k=v`
  (JSON values) become constructor kwargs (`decision_index/cli.py`). No config file is needed.
- **Runner / scorer.** `decision_index run` writes `results.jsonl` (one row per request, lab format, resumable),
  `status.json` and `environment.json` (`decision_index/runner.py`). `decision_index score` writes
  `benchmark-summary.json` (native metric per benchmark, `decision_index/scoring/report.py`), `index.json` (panel
  scores and the Decision Index, `decision_index/scoring/index.py`) and `scores.json` (the submission file,
  `decision_index/pipeline.py`). `decision_index pipeline` chains suite download, run, score and optional upload;
  `decision_index hf-job` runs the same on an RTX PRO 6000 Hugging Face Job (`decision_index/hf_job.py`).
- **Leaderboard bundle.** The Space reads one static file, `data/index.json`, written by
  `evaluation/reproductions/build_leaderboard.py` in the maintainer's private `typesafe-diffusion-lab` checkout from
  each run's `benchmark-summary.json`, `entrant-metadata.json` and the Jev release report (Space `README.md`,
  `data/methodology.json` -> `reproduce`). That script is not in the public kit; entrant metadata (kind, base model,
  parameter count) is derived from what the run loaded and the Hub model card, "nothing is inferred from a name"
  (`data/methodology.json` -> `entrants`).
- **Submissions** go through a **pull request** (kit `README.md`, "Submitting a model to the leaderboard"): run the
  full suite, upload the run directory to a Hub dataset, add a line to `submissions/README.md` (the file does not
  exist yet; create it) with model name, results dataset link (`runs/<name>/scores.json` present and
  `"complete": true`), engine/commit and hardware, and mention any declared capacity limits. Results are re-scored on
  review, so the results file must be untouched.

## Files here

- `rev_engine.py`: `RevHttpEngine` (talks to a running Rev server: our Modal endpoint or `serve_local.py`) and
  `RevLocalEngine` (starts `serve_local.py --repo <hf repo id>` from the Rev checkout on your GPU, waits for
  `/ping`, scores over loopback, stops it on close).
- `run_parallel.py`: a concurrent stand-in for `decision_index run` for batching servers (same output files; the
  kit's runner sends one request at a time).
- `run_full_suite.sh`: the whole sequence for the full 132,422-request suite.
- `ISSUE.md`: draft request to the maintainer.

Mapping details (recorded in `environment.json` -> `model_source.policy`): the `questions` map becomes Rev's list
of `{id, instructions, criteria}`; object-valued `instructions` or option descriptions are JSON-serialized exactly
as the kit's reference engine renders them (`engines/base.py` `text()`); option keys, descriptions and `state` are
otherwise untouched. Rows whose rendered prompt exceeds the server's 16,384-token row limit are refused as
`Unsupported` before sending (counted with the server's own tokenizer, pinned revision from `/ping`); nothing is
truncated, no option is dropped, no prompt differs per benchmark. Rev has no abstention outcome.

## Registration

Option A (no kit changes): put the Rev checkout's parent on `PYTHONPATH` and pass the dotted path:

```sh
git clone https://github.com/robbalian/rev && export PYTHONPATH=$PWD/rev
python -m decision_index run --engine decision_index_adapter.rev_engine:RevHttpEngine --option base_url=http://127.0.0.1:8000 --out runs/rev-4b
```

Option B (registry entry, one line in `decision_index/engines/__init__.py`):

```python
REGISTRY = {
    "http": "decision_index.engines.http:HttpSystemOne",
    "transformers": "decision_index.engines.transformers_engine:TransformersEngine",
    "random": "decision_index.engines.base:RandomEngine",
    "rev": "decision_index_adapter.rev_engine:RevHttpEngine",        # running Rev server
    "rev-local": "decision_index_adapter.rev_engine:RevLocalEngine", # starts serve_local.py on this GPU
}
```

then `--engine rev --option base_url=...` or `--engine rev-local --option repo=robbalian/rev-qwen3.5-4b --option rev_dir=/path/to/rev`.

Engine options: `RevHttpEngine(base_url, timeout=900, retries=2, count_tokens=True, model_label=None)`;
`RevLocalEngine(repo, rev_dir, port=8000, host=127.0.0.1, device=cuda, dtype=bf16, base=None, revision=None,
python=None)` plus the HTTP options. Dependencies for the HTTP backend: `httpx`, `huggingface_hub`, `tokenizers`
(all pip). `serve_local.py` needs `torch transformers accelerate huggingface_hub fastapi uvicorn`; the 4B fits a
12 GB GPU, the 27B needs room for the base model in bf16 (about 60 GB).

Weights: https://huggingface.co/robbalian/rev-qwen3.5-4b and https://huggingface.co/robbalian/rev-qwen3.8-27b
(each holds `checkpoint.pt`: LoRA adapters, pointer head and metadata; `serve_local.py` merges the LoRA into the
pinned Qwen base at load).

## Running the suite

```sh
pip install -e ".[transformers]"   # in the kit; or plain `pip install -e .` plus httpx tokenizers for HTTP only
export HF_HUB_DISABLE_XET=1
hf auth login                      # REQUIRED: multimodalart/decision-index-suite returns 401 without a token

python -m decision_index suite download --dir suite      # 132,422 rows, verified against the pinned sha256
# alternative if the dataset is unreachable: rebuild from the pinned public sources (about 4 GB, HLE is gated)
python -m decision_index suite rebuild --work work        # writes work/artifacts/benchmark-suite/release-v1-rebuilt/
python -m decision_index suite import --dir suite --rows work/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl.gz \
    --exclusions hub/excluded-questions.json --manifest hub/manifest.json

# local weights on your GPU (starts serve_local.py itself)
python -m decision_index run --engine decision_index_adapter.rev_engine:RevLocalEngine \
    --option repo=robbalian/rev-qwen3.5-4b --option rev_dir=/path/to/rev --out runs/rev-4b
# or an already running server (ours: https://reprompt--jev-beat-serve-dense27jb.us-west.modal.direct)
python decision_index_adapter/run_parallel.py --rows suite/selected-rows.jsonl.gz --out runs/rev-27b \
    --base-url http://127.0.0.1:8000 --concurrency 8

python -m decision_index score --results runs/rev-4b/results.jsonl --suite-dir suite --engine rev-4b
```

`score` needs the **full** suite: `index_entry()` also builds the 25-benchmark frozen panel and indexes every
panel benchmark, so on a partial rows file it raises `KeyError` (we hit `KeyError: 36` on a slice without BRIGHT).
For a quick check on a slice, call `decision_index.scoring.report.benchmark_summary` and
`decision_index.scoring.index.score_panel` directly (that is what our sample scoring did).

Wall time: the kit's serial runner at ~0.7 s per request is roughly 25-30 h for 132k requests against our Modal
27B; `run_parallel.py` at concurrency 8 ran our sample at ~10 requests/s, i.e. about 3.7 h, so budget 4-6 h
(ToolRet, BRIGHT and ACOS rows carry 30+ questions each). `serve_local.py` serves one request at a time, so a local
run is bounded by your GPU: on the sample, requests took 0.05-0.5 s of GPU time each on a B200.

## What we verified

- HTTP backend end to end through the kit's own `decision_index run` (serial) on a 101-request stratified slice of
  the frozen suite against the 27B endpoint: 101/101 `ok`, 0 errors, 45 s; `environment.json` records the checkpoint
  (`jb_20260922-211844_27b`); the kit's `benchmark_summary` scored every benchmark in the slice and `score_panel`
  produced an index (59.28 on 101 requests, illustrative only).
- Our own sample run (1,046 requests, 823 cases, 18 of the 19 panel benchmarks; rows verified byte-identical to the
  frozen release per benchmark via the kit's `hub/manifest.json` hashes): Decision Index **61.07** (95% case
  bootstrap 56.7-65.0) for the 27B; Jev recomputed on the same 18 benchmarks from the Space's published values:
  **63.15**. BRIGHT was not run. Details: `results/decision_index/summary.json` in the Rev repo.
- Not verified: `RevLocalEngine` (no local GPU here; `serve_local.py` is the same code path as `server.py`),
  the 4B on the index, the Hub weight repos (ids assumed), and the maintainer's private leaderboard build.
