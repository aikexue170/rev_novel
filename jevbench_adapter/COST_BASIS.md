# Cost basis for Rev on JevBench

JevBench's Cost axis is US dollars per 1,000 **decisions** (one decision = one whole question: state, rubric,
options), scored `100 - 30 log10($/1k / $0.001)`. This file gives two bases per model and states every input so
the maintainer can substitute his own. Nothing here is a claim of a score; the score comes from his run.

Models: `robbalian/rev-qwen3.5-4b` (Qwen3.5-4B backbone) and `robbalian/rev-qwen3.8-27b` (Qwen3.8-27B backbone).
Both answer in one forward pass and generate nothing, so output tokens are 0 per decision everywhere below.
The 9B (`Qwen3.5-9B`) is listed for completeness; it is not part of the request.

## (a) Self-hosted at Modal list price

Rate for one B200 container as we run it (Modal list prices, https://modal.com/pricing, read 23 Sep 2026):

| Component | List price | Per second |
|---|---|---:|
| NVIDIA B200 | $0.001736 / s | $0.001736 |
| 8 CPU cores | $0.0000131 / core / s | $0.0001048 |
| 96 GiB RAM | $0.00000222 / GiB / s | $0.0002131 |
| **Total** | | **$0.0020539 / s ($7.39 / h)** |

Cost per 1,000 decisions = rate per second / measured answers per second x 1,000. The throughput numbers are from
`results/holdout/summary.json` (975-question public holdout, one question per request, a cloud client in the same
region as the server, `results/holdout/REPORT.md`): "full load" is the highest concurrency level we ran (256 in
flight for the 4B and 9B, 128 for the 27B; GPU busy 99-100 %); "one in flight" is concurrency 1, i.e. what a serial
benchmark client sees, with the GPU idle roughly half the time.

| Model | Full load: answers/s | $ per 1k decisions | Cost axis | One in flight: answers/s | $ per 1k decisions | Cost axis |
|---|---:|---:|---:|---:|---:|---:|
| Rev Qwen3.5-4B | 93.2 | **$0.0220** | 59.7 | 13.9 | $0.1478 | 34.9 |
| Rev Qwen3.8-27B | 25.3 | **$0.0812** | 42.7 | 8.5 | $0.2416 | 28.5 |
| Rev Qwen3.5-9B (not requested) | 73.6 | $0.0279 | 56.5 | 15.0 | $0.1369 | 35.9 |

Assumptions: (1) the GPU is billed only while it serves, i.e. a busy server with no idle time at the full-load figure
and a half-idle one at the one-in-flight figure; (2) Modal list price, no committed-use discount; (3) the holdout
questions average about 700 input tokens, a little more than JevBench's public items (604, below), so per-decision
GPU time on JevBench items is if anything slightly lower; (4) the 4B and 9B ran with `max_padded_tokens` 49,152 and
the 27B with 32,768; nothing was quantized (BF16 merged LoRA, FP32 head).

For comparison, JevBench's own self-host sensitivity for a comparable row (SemIf, Qwen3.5-4B) assumes a rented GPU at
$/h, concurrency 4 and 30 % utilisation; on that assumption a B200 at $7.39/h serving the 4B at our concurrency-8
throughput (47.4/s) would be $0.0433 per 1k at full utilisation and $0.144 per 1k at 30 % utilisation.

## (b) Estimate at hosted-provider tariffs for the same size class

JevBench prices open self-hosted rows at the public per-token tariff of the exact base weights (or the nearest listed
size class) times the system's own measured input tokens per decision (e.g. reflex-27b and LitJev: OpenRouter
Qwen3.8-27B at $0.214/M input; reflex 4B and SemIf: DeepInfra Qwen/Qwen3.5-4B at $0.03/M input). The same method:

**Input tokens per decision.** Counted on the 231 public items (`results/benchmarkheaven/jevbench_public_231.jsonl`)
with the exact prompt the server builds (`State:\n{state}\nQuestion: {instructions}\nOptions:\n{key}: {description}\n...Decision:`,
JSON states serialized compactly) and the base models' own tokenizers (the Qwen3.5 and Qwen3.8 tokenizers give
identical counts on these items):

| | Mean | Median | Min | Max | Easy mean | Standard mean | Hard mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact tokenizer count | **604** | 102 | 48 | 3,899 | 76 | 83 | 1,170 |
| chars / 4 estimate | 609 | 129 | | | | | |

The 534-item suite has a different tier mix (146 judge items, 109 held-out hard items), so the maintainer's own mean
will differ; Jev 1.13.0 reads 950 input tokens per decision on the full suite under its own tokenizer. The adapter
reports a per-decision count in `usage.input_tokens` (exact when `REV_TOKENIZER` is set, else chars/4, and it says
which in `usage.input_tokens_source`).

**Tariffs** (read 23 Sep 2026 from the OpenRouter public models API, https://openrouter.ai/api/v1/models and
`/models/<id>/endpoints`, and the DeepInfra public model list, https://api.deepinfra.com/models/list):

| Size class | Listing | $/M input | $/M output | $ per 1k decisions at 604 tokens | Cost axis |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B | OpenRouter `qwen/qwen3.8-27b`, headline (default route, Novita) | 0.42 | 3.00 | $0.2537 | 27.9 |
| Qwen3.8-27B | OpenRouter, DeepInfra route, BF16 (the exact unquantized weights) | 0.15 | 1.875 | **$0.0906** | 41.3 |
| Qwen3.8-27B | OpenRouter, cheapest route (Darkbloom, FP4) | 0.10 | 1.80 | $0.0604 | 46.6 |
| Qwen3.8-27B | tariff used for the existing reflex-27b and LitJev rows | 0.214 | 0 | $0.1292 | 36.7 |
| Qwen3.5-4B | DeepInfra `Qwen/Qwen3.5-4B` (not on OpenRouter; the reference used for the reflex 4B and SemIf rows) | 0.03 | 0.15 | **$0.0181** | 62.3 |
| Qwen3.5-4B | nearest OpenRouter Qwen listing, `qwen/qwen3.5-9b` (DeepInfra BF16 route) | 0.10 | 0.15 | $0.0604 | 46.6 |

OpenRouter lists 16 providers for `qwen/qwen3.8-27b` from $0.10 to $0.45 per M input; the bold rows are the ones we
would propose (exact base weights, unquantized, from the same provider the maintainer already used for the 4B class).
Output price is irrelevant: 0 output tokens.

Assumptions: (1) a hosted provider would charge the base model's tariff for a LoRA-plus-head variant, which no
provider actually offers, so this is a size-class proxy, not a price anyone can pay; (2) input tokens are the public-item
mean under the server's template; (3) tariffs are the ones listed on 23 Sep 2026 and change; (4) no volume discount,
no cached-prefix discount (every decision is a fresh state).

## Speed rule and what we measured

JevBench measures latency serially from its own server in Germany and, for self-hosted and demo endpoints, publishes
`latency x 2 + 0.15 s` (the +0.15 s only on the maintainer's own servers) as an assumed production-load adjustment;
score = mean of `100 - 20 log10(s / 0.1 s)` at p50 and p95.

Our measured serial (one request in flight) latency, end to end from a cloud client in the same region (Modal
container in us-west, HTTPS, network + tokenization + forward included; `results/holdout/summary.json`, 975 questions):

| Model | p50 | p95 | Under the x2 + 0.15 s rule (illustrative only) |
|---|---:|---:|---:|
| Rev Qwen3.5-4B | 65 ms | 102 ms | 0.28 s / 0.35 s -> Speed 90.0 |
| Rev Qwen3.8-27B | 88 ms | 240 ms | 0.33 s / 0.63 s -> Speed 86.9 |

Those are not what the maintainer will see: his client is in Germany, so a transatlantic round trip (about 150 ms) is
added to every request against the Modal endpoint, and if he runs `serve_local.py` on his own GPU the numbers depend
on that GPU. On JevBench's public items from a laptop over the internet (this adapter, one request at a time, 23 Sep 2026) the
harness measured p50 0.377 s and p95 0.563 s against the Modal endpoint (server-side p50 51 ms; `README.md` in this
directory), i.e. the network dominates. The 27B's latency grows with input
length (87 ms under 256 tokens, 216 ms over 2k tokens); the hard tier averages 1,170 tokens.

Cold start: the Modal endpoint scales to zero and takes up to about two minutes to come back; the adapter's `load()`
(`JEVBENCH_WARM_LOAD=1`) polls `POST /ping` before the clock starts, as the harness does for in-process entrants.
