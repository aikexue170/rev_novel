# Launch checklist (target: 8am)

Owner key: **C** = Claude does it, **R** = Rob does it, **C→R** = Claude prepares, Rob runs or approves.

## Models
- [x] **C** Retrain Qwen3.5-4B on the public + synthetic mix (`jb_20260923-002700_4b`; holdout 87.7%).
- [x] **C** Calibration temperatures fitted and stored: 27B T=1.70 (JevBench calibration axis 71.9 to 80.6), 4B T=1.96 (dev ECE 0.121 to 0.045).
- [x] **C** Retrained 4B deployed and scored: JevBench 181/231 (hard 59.5%, I=70.6), Jev's official 84/102, holdout 87.7%.
- [x] **C** Final load test `final9_20260923-014354` and multiq `multiq_invoice_bust_final9_20260923-014923`; README tables and charts refreshed from them.

## Weights on Hugging Face
- [x] **C** `publish_weights.py`: packages LoRA adapters + pointer head + temperature + metadata from the Modal volume into a Hugging Face repo per model, with a model card.
- [x] **C** `serve_local.py`: plain FastAPI server (no Modal) that loads the base model from the Hub and our checkpoint, exposes `/score` with the same contract; this is what the JevBench maintainer runs.
- [x] **C** Weights uploaded as private repos: `robbalian/rev-qwen3.5-4b` (T=1.96), `robbalian/rev-qwen3.5-9b`, `robbalian/rev-qwen3.8-27b` (T=1.70). All three public (2026-09-23).

## Repo public
- [x] **C** `LICENSE` (MIT, your name), README pass for public readers (Run it section added).
- [ ] **R** Write "The journey" section (marked in README). Keep or cut the "Honestly" section wording.
- [x] Repo public (2026-09-23).

## JevBench submission
- [x] **C** Adapter for their harness (`jevbench_adapter/rev_http.py` + registration patch + tests), verified through their own runner: 208/231 live (one near-even item flips run to run).
- [x] **C** Cost basis: `jevbench_adapter/COST_BASIS.md` (Modal list price at measured throughput; hosted-tariff estimate at 604 tokens/decision).
- [x] **C** Posted: https://github.com/fstandhartinger/jevbench/issues/52 (repo commit c4b25a2, weight revisions 6c07755 / 3622d32).
- [ ] **C→R** Adapter PR from a fork (after the issue).

## Decision Index (the Hugging Face Space leaderboard)
- [x] **C** Engine adapter for the maintainer's kit (`decision_index_adapter/`), tested through their runner on a 101-request slice; registration steps and full-suite commands in its README. Note: their path is a PR to `submissions/README.md` plus a results dataset on the Hub, not just an issue.
- [~] **C→R** Issue text drafted at `decision_index_adapter/ISSUE.md`. Claude posts it once repo and weights are public.
- [ ] **R** Hugging Face login (`HF_TOKEN`) so we can pull the frozen suite `multimodalart/decision-index-suite` and score all 132k requests ourselves (about 4 hours at 10 requests/s); optional but it replaces the sample number.
