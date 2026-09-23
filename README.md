# Rev: Beating Jev on accuracy, speed, and cost
Rev is a set of 3 decision models built on Qwen that beat Jev on accuracy and speed on a 975-question public benchmark. The 4B and 9B beat it on cost too.

The models, training data, and train scripts are available here.

![holdout](post/hero_holdout.png)

## Base model, training, serving, and eval
| Model | Accuracy | Speed (server-to-server) | Cost per 1k |
|---|---:|---:|---:|
| Hosted Jev | 85.0% | 142 ms | $0.0366 |
| Qwen3.5-4B | 87.7% | 37 ms | $0.0220 |
| Qwen3.5-9B | 88.7% | 40 ms | $0.0297 |
| Qwen3.8-27B | 92.0% | 75 ms | $0.0819 |

Rev is a LoRA tune plus a small pointer head on top of Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B. The head picks one of the offered options directly, so nothing is generated.

![how it works](post/how_it_works.svg)

The training data is 19,792 public decisions: 4,000 each from the train splits of RACE, CosmosQA, MultiRC and Social IQa, 2,000 from ContractNLI, and 1,792 short classification and policy rows from 13 other public sets. The 4B and 27B also see 6,285 synthetic decisions we generated with rule-based labels. Every eval question, and every document it came from, was held out.

The eval set is 975 held-out questions from the validation splits of those same 5 datasets. It's public and frozen in [`eval_sets/`](eval_sets/).

Speed and cost use one B200 per model on Modal. Cost is Modal's list price for the GPU plus its CPU and RAM, divided by the answers per second we measured with the GPU fully busy. Speed is the median time for a single request, sent from a separate Modal server in us-west to ours and to Jev's hosted API, network included.

Serving puts every question from every request in flight into shared GPU passes, which is what makes it cheap. A long document with many questions is read once and reused.

Training took 33 minutes for the 9B, 63 for the 4B and 150 for the 27B, on one B200 each: about $30 for all three at Modal's price. The whole project, every experiment and benchmark run included, cost $438 on Modal.

## What didn't work and what it teaches us about Jev
Jev is an impressive piece of engineering and it's clearly flipped how developers think about coding and LLM decisions.

Here's a few of the ideas that didn't work:
- **JSON on hosted models** Accuracy was fine but answers generate serially: 20 questions took 5.10 s against Jev's 0.27 s, at 15x the cost per decision.
- **Single-token outputs per question** Accuracy fell and latency barely improved. It's clear a lot of the models we use today are leaning hard on reasoning
- **Train a head only** Frozen Qwen with a pointer head trains in ~10 s per configuration. Tried adding a head to a single layer and a learned best of multiple layers. Best accuracy was 80% on dev but much poorer on external test sets
- **LoRA plus head** - The architecture today. Takes < 1h to tune on the full 20k tuning set

### Reverse engineering Jev
Question and token scaling- This architecture scaled well from 1, 5, 20 questions but hosted Jev is impressively flat. 
![multiq](post/multiq.png)

Same with input length. Our method gets slower
![latency by length](post/latency_by_length.png)


## Honestly

This is all Claude Code and GPT plus minor intuition. One person, two coding agents, four days and $438 of GPU rebuilt a well-funded startup's flagship model. We didn't use their weights, their data or their team: just their public API, public datasets and a credit card.

The model was never the moat. The idea, the launch, the developer mindshare and the API people build against are.

Jev is pretty amazing as a concept and idea. It moved things forward and they had an amazing launch. It clearly caught the attention of developers, which is exactly what it should do. They're going to get cloned, copied, and beaten, but maybe it doesn't matter because they're top of mind.

## Snake

For fun, and because a decision model should be able to play a game one multiple-choice question at a time. Each turn the player gets the board and is asked which way to move. Same seeds for everyone, 10 games, a 1,000-move cap. Moves survived per game, on average:

| Player | Moves per game |
|---|---:|
| Qwen3.8-27B | 187 |
| Qwen3.5-4B | 172 |
| Qwen3.5-9B | 144 |
| Hosted Jev | 113 |

This is game 5, where Jev runs into itself at move 38 and the 27B keeps going to 185.

![Snake head-to-head](results/snake/headtohead.gif)

All ten games: [`results/snake/replay.html`](results/snake/replay.html). The head-to-head above: [`results/snake/headtohead.html`](results/snake/headtohead.html).

**It takes images too.** The backbone is a vision-language model, so the same text-trained LoRA and head can take the board as a PNG, with no coordinates anywhere in the prompt. Nothing was retrained for this. Under a 200-move cap the 27B survived 64 moves per game from the image against 176 from text, so it plays worse from pixels, but it plays. Jev is text only, so there's no Jev number here. Details in [`results/snake/vision/`](results/snake/vision/).

## Run it

Weights are on Hugging Face: [`rev-qwen3.5-4b`](https://huggingface.co/robbalian/rev-qwen3.5-4b), [`rev-qwen3.5-9b`](https://huggingface.co/robbalian/rev-qwen3.5-9b), [`rev-qwen3.8-27b`](https://huggingface.co/robbalian/rev-qwen3.8-27b). Each is the LoRA and head only, 60 to 200 MB. `serve_local.py` pulls the Qwen base from the Hub, merges them and serves the same API the benchmarks used, on any CUDA GPU.

```bash
pip install torch transformers fastapi uvicorn huggingface_hub
python serve_local.py --repo robbalian/rev-qwen3.5-4b --port 8000
```

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "state": {"invoice_total": 1250, "po_total": 1000, "vendor": "Acme"},
  "questions": [{"id": "approve", "instructions": "Should this invoice be approved without review?",
                 "criteria": {"yes": "Approve as is", "no": "Send to review"}}]}'
```

You get back a choice and a probability per option. Any number of options and questions per request works.

## Benchmarks

### How we measured

Both sides are called over HTTPS from the same client, a Modal container in us-west. Ours are Modal web endpoints on one B200 each. Jev is `typesafe/jev-1.13` through OpenRouter; where its servers sit isn't public. Latency is the p50 the client sees with one request in flight, network included. Cost is per 1,000 answers: Jev as billed by OpenRouter, ours at Modal's list price for the B200 plus its CPU and RAM, divided by the answers per second we measured at full load. That's the cost of a busy GPU with no idle time.

All four systems in each table were measured in one run from one client. Latency shifts by tens of milliseconds between runs depending on where the client container lands, so compare within a table, not across tables. Jev's accuracy also moves by a few questions between runs, 84.7% to 85.2% across ours.

### The test set

975 questions from the validation splits of five public datasets, frozen in [`eval_sets/public_holdout_975.jsonl`](eval_sets/public_holdout_975.jsonl) with a checksum. Every question has the shape Jev's API takes: a state (the document), instructions (the question) and criteria (the options). Labels are the datasets' own. Training excluded every one of these questions and every document they come from.

| Source | Questions | Task | Options | Input length |
|---|---:|---|---:|---|
| ContractNLI | 175 | Does this NDA support, contradict, or not mention a claim? 53 real NDAs, claims drawn from the 17 standard hypotheses. | 3 | 500 to 3,600 tokens, median 2,000 |
| RACE | 200 | Reading comprehension from English exams: a passage and a question about it. | 4 | 300 to 550 tokens |
| CosmosQA | 200 | Commonsense reading of short first-person narratives: why something happened, what happens next. | 4 | 100 to 250 tokens |
| MultiRC | 200 | Multi-sentence reading comprehension: is this candidate answer correct? 83 passages. | 2 | 200 to 650 tokens |
| Social IQa | 200 | Social commonsense about a one- or two-sentence situation: motivations, reactions, what comes before. | 3 | 50 to 100 tokens |

The set runs from two-line situations to 3,600-token contracts, and from yes/no to four-way choices. 95% intervals are about plus or minus 2 points, so the 4B's lead over Jev is at the edge of a tie; the 9B and 27B are clearly ahead. Full report with every concurrency level: [`results/holdout/`](results/holdout/).

### Speed by input length

Same run, bucketed by input tokens. Jev is flat at every length and ours grow with the document. The 4B and 9B stay under Jev all the way up to the longest contracts. The 27B crosses Jev above 2,000 tokens.

![latency by length](post/latency_by_length.png)

| Model | under 256 tokens | 256 to 512 | 512 to 1k | 1k to 2k | over 2k |
|---|---:|---:|---:|---:|---:|
| Hosted Jev | 140 ms | 141 ms | 144 ms | 144 ms | 150 ms |
| Qwen3.5-4B | 37 ms | 37 ms | 38 ms | 44 ms | 63 ms |
| Qwen3.5-9B | 39 ms | 39 ms | 40 ms | 52 ms | 76 ms |
| Qwen3.8-27B | 74 ms | 74 ms | 75 ms | 113 ms | 196 ms |

### Speed by questions per request

Jev's API takes several questions about one state in a single request, so we tested that too: four synthetic invoices with 1, 5 or 20 yes/no questions each. Every request carries a fresh reference id so neither side can answer it from a cache. Jev is flat in the number of questions. Ours grows, but the 4B and 9B answer 20 questions in half Jev's time and the 27B about matches it. This run's client sat farther from every server than the load test's, which is why Jev's one-question time is 191 ms here and 142 ms above.

![multiq](post/multiq.png)

| Model | 1 question | 5 questions | 20 questions |
|---|---:|---:|---:|
| Hosted Jev | 191 ms | 203 ms | 210 ms |
| Qwen3.5-4B | 64 ms | 65 ms | 97 ms |
| Qwen3.5-9B | 64 ms | 69 ms | 111 ms |
| Qwen3.8-27B | 100 ms | 107 ms | 218 ms |

### Cost under load

![cost](post/cost_concurrency.png)

Self-hosting is only cheap when the GPU is busy. At one request in flight our cost per answer is several times Jev's; it drops below Jev's for the 4B and 9B once enough requests are in flight to keep the GPU full.

Modal is close to the cheapest B200 we could find. At $5.98 an hour on RunPod, the lowest on-demand price, the 4B costs $0.0178 per 1,000 answers, half of Jev's price, and the 27B $0.0662. Every provider we checked is in [`results/holdout/GPU_PRICES.md`](results/holdout/GPU_PRICES.md).

### Outside leaderboards

From our own testing, the 27B is the top open model on both public Jev leaderboards: ahead of Jev on JevBench, still behind it on the Decision Index.

| Our run of | Qwen3.8-27B | Hosted Jev | Best other open model |
|---|---:|---:|---:|
| [JevBench](https://benchmarkheaven.com/jev-models), 231 public items | 89.6% | 86.6% | 87.0% |
| [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index), 1,046-request sample | 61.1 | 63.2 | 59.2 |

These are our measurements, not the boards'; we've asked JevBench to run the full suite ([#52](https://github.com/fstandhartinger/jevbench/issues/52)). No JevBench item was trained on, but we read our first model's misses on its public items to decide what synthetic data to make. Details in [`results/`](results/).

### Credit: Kev

We borrowed the recipe, a LoRA and a pointer head on Qwen, from Jared Palmer's [Kev](https://github.com/jaredpalmer/kev). Thanks Jared.

## Reproduce

Everything runs on [Modal](https://modal.com). The only secret is an OpenRouter key in a Modal secret named `openrouter`, for the Jev arm. Run outputs go to `runs/`, which is git-ignored.

| File | What it does |
|---|---|
| `train/train.py` | Trains the LoRA and pointer head on one B200 per model, from `data/`, holding out `eval_sets/public_holdout_975.jsonl`. |
| `train/synth_jevbench.py` | Generates the synthetic decisions with rule-derived labels, and checks for overlap with every eval set. |
| `train/fit_temperature.py`, `train/set_temperature.py` | Fit a calibration temperature on held-out data and store it in a checkpoint. |
| `train/publish_weights.py` | Packages a training run into a Hugging Face repo with a model card. |
| `serve/server.py` | The Modal server: merged weights, cross-request batching, the shared-document path and its cost-model routing. `cd serve && modal deploy server.py`. |
| `serve_local.py` | The same API without Modal, on any CUDA GPU. |
| `bench/loadtest.py` | Every model over the 975-question benchmark at each concurrency level, from a cloud client. `bench/report.py` summarizes it. |
| `bench/multiq.py` | The 1, 5 and 20 questions per request test, with cache-busting. `bench/report_multiq.py` summarizes it. Cases are in `cases/`. |
| `bench/refresh_numbers.py`, `bench/post_charts.py` | Build `post/numbers.json` from one load-test run and one multi-question run, then draw this README's charts. |
| `bench/external_eval.py` | Scores Jev's 102 published workflow questions against Jev's saved answers. |
| `bench/head_only.py`, `bench/kev_eval.py` | The frozen-backbone ablation and layer sweep; Kev's checkpoints through their own serving path. |
| `snake/` | The Snake arena, text and image boards, and its replay pages. |
| `jevbench_adapter/`, `decision_index_adapter/` | Adapters, cost basis and submission notes for the two outside leaderboards. |
| `results/` | Every report behind the numbers here. |
