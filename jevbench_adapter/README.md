# JevBench adapter for Rev

Files for running Rev under the JevBench harness (https://github.com/fstandhartinger/jevbench, protocol
`jevbench::v1.2`, scoring v1.3.0), written against harness commit `f79a1ca` (tag `v1.3.0` + 4).

| File | What |
|---|---|
| `rev_http.py` | The adapter: drops into `jevbench/adapters/`. Talks to Rev's `POST /score` over HTTP (Modal endpoint or `serve_local.py`), maps the three JevBench primitives onto Rev's option dict, returns the model's own softmax over the exact label set (`native`). |
| `jevbench-registration.patch` | Three-line registration: import in `jevbench/adapters/__init__.py`, the `kinds` entry and the `--adapter` choice in `jevbench/cli.py`. Applies cleanly to `f79a1ca` with `git apply`. |
| `test_rev_http_adapter.py` | Unit tests in the harness's own style (pytest, `http_post_json` monkeypatched): request shape, the three mappings, failure on wrong keys, token-count source. Drops into `tests/`. |
| `COST_BASIS.md` | Self-hosted list-price and hosted-tariff cost bases, token counts, speed notes. |
| `ISSUE.md` | Draft of the bench-request issue. |

## Registering the adapter

```sh
git clone https://github.com/fstandhartinger/jevbench && cd jevbench
cp <rev>/jevbench_adapter/rev_http.py jevbench/adapters/
cp <rev>/jevbench_adapter/test_rev_http_adapter.py tests/
git apply <rev>/jevbench_adapter/jevbench-registration.patch
python -m pytest -q tests            # 105 existing + 8 new tests pass on f79a1ca
```

Or by hand: add `from .rev_http import RevHttpAdapter` to `jevbench/adapters/__init__.py`, add
`"rev_http": RevHttpAdapter` to the `kinds` dict in `cmd_run` and `"rev_http"` to the `--adapter` choices in `cli.py`.
The adapter needs only the standard library; `transformers` is optional (exact token counts, see below).

## Running

Against a server (the repo's `serve_local.py` on any CUDA GPU with the public weights, or the Modal endpoint):

```sh
# server: see serve_local.py in the repo root; it serves the same /score and /ping API on :8000
export REV_TOKENIZER=Qwen/Qwen3.8-27B          # optional: exact input-token counts (Qwen/Qwen3.5-4B for the 4B)
JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter rev_http --endpoint http://127.0.0.1:8000 --key-env '' --model rev-qwen3.8-27b \
  --cost-basis self_hosted_gpu --reserve-usd 0 --cap-usd 1 \
  --results RUN/results.jsonl --raw-dir RUN/raw --ledger RUN/ledger.jsonl --manifest RUN/manifest.json
python -m jevbench.cli summarize --tasks datasets/public/hard.jsonl --results RUN/results.jsonl
```

- `--key-env ''` sends no Authorization header; the server has no auth. If a key is ever needed, name its env var.
- `JEVBENCH_WARM_LOAD=1` makes the CLI call `adapter.load()`, which polls `POST /ping` until the server answers
  (the Modal endpoint scales to zero; cold start is up to about two minutes). Without it the first request pays the cold
  start, as the harness notes for scale-to-zero endpoints.
- One request at a time, as the harness does. The Modal endpoint should not be driven at more than 4 in flight.
- `--price-in-per-m` / `--price-out-per-m` are left unset for a self-hosted run; the run then reports cost `null` with
  basis `self_hosted_gpu` and the maintainer prices the row from `usage.input_tokens` (see `COST_BASIS.md`).

## What the adapter sends and reads

Rev's server takes one question shape: `{"instructions": str, "criteria": {key: description}}` and returns
`{"choice": key, "probabilities": {key: p}}` from a pointer head over the option slots (nothing is generated,
`generated_tokens` is always 0). The mapping is the one used for the repo's own public-item run
(`results/benchmarkheaven/run.py`) and was fixed before that run:

| JevBench type | Sent | Read back |
|---|---|---|
| `choice` | criteria dict as-is (keys are the labels) | probabilities as-is |
| `noul` | criteria `{"true": ..., "false": ...}` as the benchmark gives them (`{"true": "Yes", "false": "No"}` if absent) | `true -> "yes"`, `false -> "no"` |
| `score` | the ordered level list as `{"0": level0, "1": level1, ...}` | probabilities as-is; keys are the level indices |

State is sent as-is (string or JSON object; the server serializes objects compactly). A distribution whose keys do
not match the labels sent, or a choice outside them, is a failed answer (`ok=false`), never repaired.

`usage` per decision: `input_tokens` (from the server's `usage.input_tokens` if it returns one; else the exact count
under the server's prompt template when `REV_TOKENIZER` is set; else `len(prompt)/4`), `output_tokens: 0`,
`input_tokens_source` naming which, `server_seconds`, `path` (`rows` or `shared`; single-question requests always take
`rows`).

## Verification run (public items, this adapter, the harness's own runner and scorer)

`runs/dense27jb_T1.70_public231/` holds the harness's own `results.jsonl`, `summary.json` (its `summarize --public-export`),
`manifest.json` and a per-tier `tiers.json`. Run 23 Sep 2026, 04:48-04:51 UTC, from a laptop over the internet to the
Modal endpoint (B200, us-west), one request at a time, `JEVBENCH_WARM_LOAD=1`, `REV_TOKENIZER=Qwen/Qwen3.8-27B`;
checkpoint `jb_20260922-211844_27b`, sha256 `6a4667b7…`, temperature 1.70. Raw responses were kept outside the repo, as
the harness requires.

| | Harness run (this adapter) | Our own run.py scoring (`results/benchmarkheaven/`) |
|---|---:|---:|
| Easy | 48/48 | 48/48 |
| Standard (original) | 71/72 | 71/72 |
| Hard (public half) | 89/111 | 88/111 |
| Pooled | **208/231 (0.900)** | 207/231 (0.896) |
| Strict-valid distributions | 231/231 | 231/231 |
| Failed requests | 0 | 0 |
| Hard-tier top-label ECE | 0.078 | 0.074 |
| Mean TVD, 10 public probability items | 0.249 | 0.239 |
| Calibration axis on the public hard items | 79.8 | 80.6 |

Choices agree item for item except one: `hard-sol-b-judge_hard-02`, a near-even yes/no (yes 0.61 in the recorded
run before the temperature, yes 0.34 live at T = 1.70; a temperature alone cannot cross 0.5), which the live run gets
right. Batching and padding differ between runs and the BF16 forward is not bit-reproducible: in an earlier live pass at
T = 1 against the same recorded run, 7 of 219 distributions moved by more than 0.05 (max 0.28, on this item) and this
was the only flip. The token count the adapter
reported (mean 603.9 per decision, exact tokenizer) matches the offline count in `COST_BASIS.md`.

The same harness command against a local replay of the recorded answers (`results/benchmarkheaven/dense27jb.jsonl` served
through Rev's `/score` shape) reproduced 207/231 with zero per-item differences, which checks the mapping and the
scoring path independently of GPU nondeterminism.

Latency as the harness measured it from this laptop (network included, one request in flight): p50 0.377 s, p95
0.563 s, first request 4.5 s (the server's canonical batch shapes warm on first use even after `/ping`); the server's
own `server_seconds` p50 was 51 ms. The site measures from Germany, so its raw figures will be higher again.
