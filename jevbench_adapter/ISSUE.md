DRAFT, not posted. Items marked TODO need a real value before posting (weights revisions, repo commit, overlap-check
path). Everything else is from the repo's measured results.

---

**Title:** [bench request]: Add Rev (Qwen3.5-4B and Qwen3.8-27B + LoRA and pointer head, native option probabilities; HTTP adapter and local server included)

Request to add **Rev** (https://github.com/robbalian/rev) in two sizes: **Rev Qwen3.5-4B** and **Rev Qwen3.8-27B**.
Both are meant to be run on your own GPU from the public weights; the adapter and a plain local server are in the repo.

**What it is.** A Qwen backbone (Qwen3.5-4B or Qwen3.8-27B, BF16) with a rank-16 LoRA on the attention and DeltaNet
projections and a small pointer head: the state, the question and the options are encoded in one forward pass and the
head scores each option's slot; the softmax over the slots, divided by one calibration temperature (T = 1.70, fitted on
the synthetic dev split, stored in the checkpoint, changes no choices), is the answer. Nothing is generated (0 output tokens), no
token logprobs, no verbalized probabilities. One question per request in the benchmark setting; the server also takes
several questions about one state in one request. Trained for one epoch on 19,792 public decisions (the 27B additionally on
6,285 synthetic decisions with rule-derived labels), each seen in two option orders with a consistency term so the
answer does not depend on where the right option sits.

**Links.**
- Repo (MIT code): https://github.com/robbalian/rev — commit `TODO`
- Weights (MIT for the LoRA + head checkpoint; the base Qwen weights it merges into are Apache-2.0):
  `robbalian/rev-qwen3.5-4b` revision `TODO`, `robbalian/rev-qwen3.8-27b` revision `TODO`
  (base `Qwen/Qwen3.5-4B`; `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`)
- Adapter and registration patch: `jevbench_adapter/` (`rev_http.py`, `jevbench-registration.patch`,
  `test_rev_http_adapter.py`; written against `f79a1ca`)
- Local server, no Modal: `serve_local.py` — one CUDA GPU, FastAPI + uvicorn, serves the same `POST /score` and `POST /ping` on :8000; `--selftest` loads the weights and answers one built-in request
- Cost basis and token counts: `jevbench_adapter/COST_BASIS.md`

**How to run.**

```sh
git clone https://github.com/robbalian/rev && cd rev && git checkout TODO
pip install torch transformers accelerate huggingface_hub fastapi uvicorn   # optional: flash-linear-attention, causal-conv1d
python serve_local.py --repo robbalian/rev-qwen3.8-27b --port 8000        # or robbalian/rev-qwen3.5-4b (fits in 12 GB); BF16
# the base Qwen weights and their pinned revision come from the checkpoint's metadata

cd <jevbench>
cp <rev>/jevbench_adapter/rev_http.py jevbench/adapters/ && git apply <rev>/jevbench_adapter/jevbench-registration.patch
REV_TOKENIZER=Qwen/Qwen3.8-27B JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run \
  --tasks datasets/public/original.jsonl --adapter rev_http --endpoint http://127.0.0.1:8000 --key-env '' \
  --model rev-qwen3.8-27b --cost-basis self_hosted_gpu --reserve-usd 0 ...
```

The mapping is the obvious one and is fixed in the adapter: `choice` criteria as-is, `noul` `true/false` criteria
sent as given and read back as `yes/no`, `score` levels as `{"0": ..., "1": ...}`. Inputs over 16,384 tokens are refused
(HTTP 500 from the server's assertion), not truncated; at most 256 options.

**Our own numbers on the public items** (self-measured, 27B only, the 231 public items through our server with your
scoring rules, `results/benchmarkheaven/` in the repo; the run of your own runner through this adapter is in
`jevbench_adapter/runs/`):

| tier | items | Rev Qwen3.8-27B |
|---|---|---|
| easy | 48 | 1.000 |
| standard (original) | 72 | 0.986 |
| hard (public half) | 111 | 0.793 |
| pooled | 231 | 0.896 (207/231), 231/231 strict-valid |

Hard-tier top-label ECE 0.074; mean total variation to the gold distributions on the 10 public probability items 0.239.
Your runner with our adapter against the same endpoint gives 208/231 (hard 89/111; ECE 0.078, TVD 0.249): one
near-even item flips between runs, the rest agree item for item.
Weak families on the public hard tier: temporal_numeric 5/15, long_policy 9/19. We have not run the 4B on the
public items (it scores 87.6 % to the 27B's 92.0 % on our own 975-question holdout), so its numbers here are yours to
find. These are public items only, half the hard tier and none of the judge tier; we are not claiming a rank from
them, and speed and cost are measured from your server.

**What we ask.** An evaluation of both sizes on the full 534-decision suite on your own GPU from the public weights,
under your normal held-out, speed and cost policies. Priority if only one: the 27B.

**Cost basis** (details and every assumption in `jevbench_adapter/COST_BASIS.md`). Zero output tokens; 604 input tokens
per decision on the public items under the server's own template (exact tokenizer count; the adapter reports the count per
decision in `usage.input_tokens`).
- Self-hosted at Modal list price (B200 + 8 CPU + 96 GiB, $0.00205/s), divided by measured answers per second at full
  load: **4B $0.0220 per 1k decisions** (93.2/s), **27B $0.0812 per 1k** (25.3/s); at one request in flight $0.148 and $0.242.
- Hosted-tariff estimate for the size class, as you priced reflex/LitJev/SemIf: 4B at DeepInfra `Qwen/Qwen3.5-4B`
  $0.03/M input = **$0.0181 per 1k**; 27B at the OpenRouter DeepInfra BF16 route for `qwen/qwen3.8-27b` $0.15/M input =
  **$0.0906 per 1k** ($0.129 at the $0.214/M you used for the existing 27B rows). Read 23 Sep 2026.
- Serial latency we measured end to end from a cloud client in the same region: 4B p50 65 ms / p95 102 ms, 27B
  p50 88 ms / p95 240 ms; your x2 + 0.15 s rule applies and your client in Germany adds the round trip.

**Disclosure.** No JevBench item, public or held out, and no Jev output was used for training, tuning or selection.
An exact normalized-text overlap check of the training set against the 231 public items found zero matches
(`TODO: path to the overlap-check output in the repo`). The public 231 were used as a development gate: after the
first 27B scored 198/231 we read its misses, generated synthetic data of the failing shapes (long policies, date and
quantity arithmetic, planted wrong notes) with rule-derived labels, and retrained once; that retrain is the model
submitted, so expect the held-out hard items to score somewhat below the public ones. The 4B was not changed after
seeing any JevBench number.

Happy to answer questions here or rerun anything.

---

Verification note for the reviewer (not part of the issue): `jevbench_adapter/README.md` records the run of this
adapter through the harness's own runner and scorer against the live 27B endpoint; the per-tier numbers there must
match the table above within a couple of items before posting.
