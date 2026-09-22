# Rev: Beating Jev on accuracy, speed, and cost

Rev is a set of open-weight decision models: a Qwen backbone with a rank-16 LoRA and a small pointer head that picks one of the offered options without generating a token. On a 975-question public benchmark, Qwen3.5-4B and Qwen3.5-9B are more accurate than Hosted Jev, answer in a third of the time end to end, and cost 1.3x to 1.8x less per answer at load. One person and two coding agents built this in about three days.

## The benchmark

**Hosted against hosted, measured the same way.** Both systems are called over HTTPS from the same cloud client (a Modal container in us-west). Our models run as Modal web endpoints on one B200 each; Jev is `typesafe/jev-1.13` through OpenRouter. Every latency in this post is end to end: what that client sees from sending the request to receiving the answer, including network, tokenization and the model forward, with one request in flight. Cost is per 1,000 answers: Jev as billed by OpenRouter, ours at Modal's B200 list price (GPU plus provisioned CPU and RAM) divided by the answers per second we measured at full load, so it is the cost of a busy GPU with no idle time.

**The test set** is 975 questions drawn from five public reading and reasoning datasets, frozen in [`eval_sets/public_holdout_975.jsonl`](eval_sets/public_holdout_975.jsonl) with its checksum. Every question has the same shape Jev's API uses: a state (the document), instructions (the question) and criteria (the options), and the label is the dataset's own answer. None of it was used for training: the training set excludes these questions by id and by source document.

| Source | Questions | Task | Options | Input length |
|---|---:|---|---:|---|
| ContractNLI | 175 | Does this NDA support, contradict, or not mention a claim? 53 real non-disclosure agreements, the 17 standard claims each. | 3 | 500 to 3,600 tokens, median 2,000 |
| RACE | 200 | Reading comprehension from English exams: a passage and a question about it. | 4 | 300 to 550 tokens |
| CosmosQA | 200 | Commonsense reading of short first-person narratives: why something happened, what happens next. | 4 | 100 to 250 tokens |
| MultiRC | 200 | Multi-sentence reading comprehension: is this candidate answer to a question about the passage correct? 83 passages. | 2 | 200 to 650 tokens |
| Social IQa | 200 | Social commonsense about a one- or two-sentence situation: motivations, reactions, what comes before. | 3 | 50 to 100 tokens |

So the set spans two-line situations to 3,600-token contracts, yes/no to four-way choices, and three kinds of reasoning. The multi-question test further down uses a separate set of four synthetic invoices with 20 yes/no questions each.

### Accuracy, speed, cost

![holdout](post/hero_holdout.png)

| Model | Accuracy | Speed (end to end) | Cost per 1k |
|---|---:|---:|---:|
| Hosted Jev | 84.7% | 187 ms | $0.0366 |
| Qwen3.5-4B | 87.6% | 61 ms | $0.0209 |
| Qwen3.5-9B | 88.7% | 65 ms | $0.0286 |
| Qwen3.8-27B | 91.1% | 97 ms | $0.0821 |

95% intervals are about plus or minus 2 points, so the three models are ahead of Jev on accuracy, not tied. For the record, on Jev's own published set of 102 workflow questions Jev still leads, 91 to our best 89 of 102; the full numbers are in [`results/`](results/).

### Speed by input length

Same run, bucketed by input tokens. Jev is flat at every length; ours grow with the document, and the 4B and 9B stay under Jev all the way up to the longest contracts.

![latency by length](post/latency_by_length.png)

| Model | under 256 tokens | 256 to 512 | 512 to 1k | 1k to 2k | over 2k |
|---|---:|---:|---:|---:|---:|
| Hosted Jev | 188 ms | 185 ms | 180 ms | 195 ms | 187 ms |
| Qwen3.5-4B | 61 ms | 61 ms | 61 ms | 66 ms | 83 ms |
| Qwen3.5-9B | 64 ms | 64 ms | 65 ms | 77 ms | 103 ms |
| Qwen3.8-27B | 95 ms | 96 ms | 103 ms | 147 ms | 224 ms |

### Speed by questions per request

Jev's API takes several questions about one state in a single request, so we tested that too: four synthetic invoices, 1, 5 or 20 yes/no questions each, every request carrying a fresh reference id so neither side can serve it from a cache. Jev's latency is flat in the number of questions. Ours grows, but the 4B and 9B answer 20 questions in about half Jev's time.

![multiq](post/multiq.png)

| Model | 1 question | 5 questions | 20 questions |
|---|---:|---:|---:|
| Hosted Jev | 152 ms | 154 ms | 160 ms |
| Qwen3.5-4B | 57 ms | 58 ms | 86 ms |
| Qwen3.5-9B | 57 ms | 59 ms | 103 ms |
| Qwen3.8-27B | 82 ms | 99 ms | 198 ms |

### Cost under load

![cost](post/cost_concurrency.png)

## The journey

<!-- ROB: this section is yours. It should read as written by hand, and the first line should say so. The notes below are the facts in order, with the numbers, to write from or to delete. -->

![journey](post/journey.png)

Notes to write from:

- **JSON on hosted models.** Asked hosted chat models for named JSON answers. Accuracy was fine (on 160 matched invoice decisions, hosted Qwen got 154 right to Jev's 147), but every answer is generated serially: a 20-question request took 5.10 s against Jev's 0.27 s, and cost 15x more per decision.
- **Compact outputs.** Generate less: integer arrays, spaced Y/N letters, bitstrings. Accuracy fell (97.5% for JSON, about 92% for arrays, 85 to 87% for letters, 56 to 66% for bitstrings) and latency did not improve (3.25 s for JSON, 4.00 s for bitstrings). Priority routing on hosted providers did not get near Jev either.
- **First-token letter logits.** Stop generating: put the options in the prompt and read the logits of the answer letters at one position. Zero training, Qwen3.8-27B at about 85% on the benchmark, Qwen3.5-9B at 80%. Fast, but sensitive to option order, and packing several questions into one sequence let later questions see earlier ones.
- **Head only.** Freeze Qwen3.5-4B, cache hidden states, train a pointer head in about 10 seconds per configuration. Best single layer (18 to 20 of 32) reached 84.3%; the final layer only 80.6%; learned mixes over layers did not beat the best single layer. Out of distribution (Jev's own workflow questions) it collapsed to 58 to 74 of 102.
- **LoRA plus head.** Adding a rank-16 adapter on the attention and DeltaNet projections was the win: 87.6% on the benchmark and 87 of 102 on Jev's questions. The first trained version already matched Jev's accuracy at lower latency, but at one request in flight the 9B cost 4.1x more than Jev per answer. The model was fine; the serving was the problem.
- **Speeding it up.** torch.compile with CUDA graphs cut 20-question compute from 116 ms to 67 ms in an early test but needs fixed shapes and flipped a few near-tied answers, so the final server runs eager. What actually moved cost below Jev: one row per question, batched across requests, which takes a GPU from 40% busy to near 100%. Then a shared-context path for many questions on one long document (prefill once, fork the cache per question), chosen per request by a cost model the server calibrates at startup.

## Honestly

This is all Claude Code and GPT plus minor intuition. An ordinary guy reverse engineered a well-funded startup's flagship model in about three days, for less than a few hundred dollars of GPU time, without their weights, their data or their team. What was needed was their public API contract, public datasets, two coding agents and a credit card.

That is the bombshell, and it is worth being precise about what it means. We did not steal anything. We rebuilt the product from its description, which is worse news for anyone whose pitch is the model. The model was never the moat. What is left as a moat is the idea, the developer mindshare from a great launch, the eval set, and the API contract people build against. Every "we trained a small specialized model" company should read this the same way: the model is a weekend, the distribution is the company.

Jev is pretty amazing as a concept and idea. It moved things forward and they had an amazing launch. It clearly caught the attention of developers, which is exactly what it should do. They're going to get cloned, copied, and beaten, but maybe it doesn't matter because they're top of mind.

## How it works

![how it works](post/how_it_works.svg)

**Model.** Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B, frozen, with a rank-16 LoRA on every layer's attention and DeltaNet projections and a 256-d pointer head. The head scores each option's last token against the decision token and takes a softmax over the offered options only, so any number of options works and nothing is decoded.

**Training.** One epoch over 19,792 public decisions ([`data/train.jsonl.gz`](data/train.jsonl.gz)), with the benchmark's questions and documents excluded. Each example is seen in two option orders with a KL consistency term, so the answer does not depend on where the right option sits. One B200 per model: 28 minutes for the 4B, 33 for the 9B, 94 for the 27B.

**Serving.** One Modal server per model with the LoRA merged into the weights and cross-request dynamic batching: every question is one row, rows from all in-flight requests are sorted by length and run together, which is what takes the GPU from 40% busy to nearly 100% and puts our cost below Jev's. A request with several questions on one long document is served by prefilling the document once and forking the cache per question, and the server picks that path or plain rows per request from a cost model it calibrates at startup.

**Kev.** Credit where it is due: [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer was the first open reconstruction of Jev, and it is where the recipe comes from. Kev put a rank-16 LoRA and a pointer head on Qwen3.5-Base, published the weights, the training code and the serving path, and showed that a small open model can play Jev's game at all. We started from that recipe and changed the parts that limited it: Kev trains on about 12.6k short decisions and rejects states over 384 tokens, so it cannot read a contract; ours trains on states up to 4,096 tokens in two option orders, goes up to 27B, and gets its speed and cost from the batched serving stack rather than the model. With Kev's length guard lifted, Kev-9B scores 76.5% on our benchmark, which is what you would expect from a model that never saw a long document in training, and it beats us on the short classification tasks it was built for.

## Reproduce

Everything runs on [Modal](https://modal.com); the only secret is an OpenRouter key in a Modal secret named `openrouter` for the Jev arm.

| File | What it does |
|---|---|
| `train.py` | Trains the LoRA and pointer head for the 4B, 9B and 27B on one B200 each, from `data/train.jsonl.gz`, holding out `eval_sets/public_holdout_975.jsonl`. |
| `server.py` | The Modal server: merged weights, dynamic batching, the shared-context path with calibrated routing, and a slow-host guard. `modal deploy server.py`. |
| `loadtest.py` | Every model over the 975-question benchmark at each concurrency level, from a cloud client; `report.py` and `charts.py` turn it into the tables and charts. |
| `multiq.py` | The 1, 5 and 20 questions per request test, with cache-busting; `report_multiq.py` summarizes it. Cases in `cases/`. |
| `external_eval.py` | Scores Jev's own published 102 workflow questions (plus a 160-question public slice) against Jev's saved answers in `results/jev_official_answers.jsonl`. |
| `head_only.py` | The frozen-backbone ablation and layer sweep. |
| `synth_workflows.py`, `train_synth.py` | Official-style synthetic workflow packets and the warm start on them. |
| `kev_eval.py` | Kev's released checkpoints through their own serving path. |
| `post_charts.py` | Draws the charts in this README into `post/`. |
| `snake_arena.py`, `snake_vision.py`, `snake_replay.py` | The Snake arena (text and image boards) and its replay pages; see below. |
| `eval_sets/` | The frozen evaluation sets with checksums. Nothing in them was used for training. |
| `results/` | The full load-test report with every concurrency level, the cache-busted multi-question runs, Jev's official-set results, and the Snake arena summary. |

## Snake

For fun, and because a decision model should be able to play a game one multiple-choice question at a time: each turn the player gets the board and is asked which direction to move. Same seeds for everyone, 10 games, a 1,000-move cap. Qwen3.8-27B ate 201 food, Hosted Jev 144, Qwen3.5-4B 114, Qwen3.5-9B 109. This is game 5, where Jev runs into itself at move 38 and the 27B keeps going to 185.

![Snake head-to-head](results/snake/headtohead.gif)

Live replay with all ten games: [`results/snake/replay.html`](results/snake/replay.html). The head-to-head above: [`results/snake/headtohead.html`](results/snake/headtohead.html).

**It takes images too.** The backbone is a vision-language model, so the same text-trained LoRA and head can be handed the board as a PNG instead of text, with no coordinates anywhere in the prompt. Nothing was retrained for this. Under a 200-move cap the 27B ate 66 food from the image board over 10 games against 173 to 189 from the text board, so it plays worse from pixels but it plays, with a head that never saw an image during training. Jev is text only, so there is no Jev number here. Details in [`results/snake/vision/`](results/snake/vision/).
