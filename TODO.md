# Launch checklist (target: 8am)

Owner key: **C** = Claude does it, **R** = Rob does it, **C→R** = Claude prepares, Rob runs or approves.

## Models
- [ ] **C** Retrain Qwen3.5-4B on the public + synthetic mix (`jb_` run; launched).
- [ ] **C** Fit a calibration temperature per checkpoint on the synthetic dev split; serve it (`server.py`), record it in the checkpoint metadata.
- [ ] **C** Deploy the retrained 4B, score it on JevBench public items, Jev's official 102, and the holdout.
- [ ] **C** One final load test with Jev + 4B + 9B + 27B (all retrained where applicable) and the 1/5/20 test; refresh README numbers and charts (`refresh_numbers.py`, `post_charts.py`).

## Weights on Hugging Face
- [x] **C** `publish_weights.py`: packages LoRA adapters + pointer head + temperature + metadata from the Modal volume into a Hugging Face repo per model, with a model card.
- [x] **C** `serve_local.py`: plain FastAPI server (no Modal) that loads the base model from the Hub and our checkpoint, exposes `/score` with the same contract; this is what the JevBench maintainer runs.
- [ ] **R** Create the Hugging Face repos (or org) and run `publish_weights.py` with your token (`HF_TOKEN`); Claude cannot log in for you.

## Repo public
- [ ] **C** `LICENSE` (MIT, your name), README pass for public readers, remove anything internal.
- [ ] **R** Write "The journey" section (marked in README). Keep or cut the "Honestly" section wording.
- [ ] **R** Flip `robbalian/rev` to public.

## JevBench submission
- [ ] **C** Adapter for their harness (`jevbench/adapters/`) mapping our `/score` to protocol `jevbench::v1.2`, tested against the public items.
- [ ] **C** Cost basis document: Modal list-price cost at measured throughput, plus a hosted-provider estimate for the size class.
- [ ] **C→R** Issue text for `fstandhartinger/jevbench` requesting evaluation (links to repo, weights, adapter, cost basis). Claude posts it once the repo and weights are public.
- [ ] **C→R** Adapter PR from a fork (after the issue).

## Decision Index (the Hugging Face Space leaderboard)
- [ ] **C** Engine adapter for the maintainer's kit (HTTP backend + local `serve_local.py` backend), tested on a slice; registration steps and full-suite run commands.
- [ ] **C→R** Issue text for `apolinario/decision-index` requesting inclusion of the 4B and 27B. Claude posts it once repo and weights are public.
- [ ] **R** Hugging Face login (`HF_TOKEN`) so we can pull the frozen suite `multimodalart/decision-index-suite` and score all 132k requests ourselves (about 4 hours at 10 requests/s); optional but it replaces the sample number.
