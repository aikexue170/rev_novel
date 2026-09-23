# Launch checklist (target: 8am)

Owner key: **C** = Claude does it, **R** = Rob does it, **C→R** = Claude prepares, Rob runs or approves.

## Models
- [ ] **C** Retrain Qwen3.5-4B on the public + synthetic mix (`jb_` run; launched).
- [~] **C** Fit a calibration temperature per checkpoint on the synthetic dev split; serve it, record it in the checkpoint metadata. 27B done (T=1.70, dev ECE 0.059 to 0.038); 4B after its retrain.
- [ ] **C** Deploy the retrained 4B, score it on JevBench public items, Jev's official 102, and the holdout.
- [ ] **C** One final load test with Jev + 4B + 9B + 27B (all retrained where applicable) and the 1/5/20 test; refresh README numbers and charts (`refresh_numbers.py`, `post_charts.py`).

## Weights on Hugging Face
- [x] **C** `publish_weights.py`: packages LoRA adapters + pointer head + temperature + metadata from the Modal volume into a Hugging Face repo per model, with a model card.
- [x] **C** `serve_local.py`: plain FastAPI server (no Modal) that loads the base model from the Hub and our checkpoint, exposes `/score` with the same contract; this is what the JevBench maintainer runs.
- [~] **C** Weights uploaded as private repos: `robbalian/rev-qwen3.8-27b` (with T=1.70), `robbalian/rev-qwen3.5-9b`; the 4B follows its retrain. **R** flips them public with the repo.

## Repo public
- [x] **C** `LICENSE` (MIT, your name), README pass for public readers (Run it section added).
- [ ] **R** Write "The journey" section (marked in README). Keep or cut the "Honestly" section wording.
- [ ] **R** Flip `robbalian/rev` to public.

## JevBench submission
- [x] **C** Adapter for their harness (`jevbench_adapter/rev_http.py` + registration patch + tests), verified through their own runner: 208/231 live (one near-even item flips run to run).
- [x] **C** Cost basis: `jevbench_adapter/COST_BASIS.md` (Modal list price at measured throughput; hosted-tariff estimate at 604 tokens/decision).
- [~] **C→R** Issue text drafted at `jevbench_adapter/ISSUE.md` (TODOs: commit hash, weight revisions, 4B numbers). Claude posts it once the repo and weights are public.
- [ ] **C→R** Adapter PR from a fork (after the issue).

## Decision Index (the Hugging Face Space leaderboard)
- [x] **C** Engine adapter for the maintainer's kit (`decision_index_adapter/`), tested through their runner on a 101-request slice; registration steps and full-suite commands in its README. Note: their path is a PR to `submissions/README.md` plus a results dataset on the Hub, not just an issue.
- [~] **C→R** Issue text drafted at `decision_index_adapter/ISSUE.md`. Claude posts it once repo and weights are public.
- [ ] **R** Hugging Face login (`HF_TOKEN`) so we can pull the frozen suite `multimodalart/decision-index-suite` and score all 132k requests ourselves (about 4 hours at 10 requests/s); optional but it replaces the sample number.
