# Request: add Rev (Qwen3.5-4B and Qwen3.8-27B LoRA + pointer head) to the Decision Index

Draft for a GitHub issue / PR description on https://github.com/apolinario/decision-index (not posted).

## What it is

**Rev** is an open Jev-style decision model: a Qwen backbone with a rank-16 LoRA merged into the weights and a small
FP32 pointer head that scores every supplied option in a single forward pass and returns a softmax over the option
keys, so every question comes back with a choice and a full probability distribution. Two sizes:

| Model | Base | Weights | Kind (per the index's rules) |
|---|---|---|---|
| Rev 4B | Qwen/Qwen3.5-4B @851bf6e | https://huggingface.co/robbalian/rev-qwen3.5-4b | LoRA + head |
| Rev 27B | Qwen/Qwen3.8-27B @1d4bf0f | https://huggingface.co/robbalian/rev-qwen3.8-27b | LoRA + head |

Code, training scripts, serving and the adapter: https://github.com/robbalian/rev. Each Hub repo holds one
`checkpoint.pt` (adapters, head, metadata incl. the pinned base revision); `serve_local.py` merges the LoRA at load
and serves the same `/score` contract we use in production (state + list of `{id, instructions, criteria}` ->
per-question choice and probabilities). Not affiliated with TypeSafe.

## Adapter for the kit

`decision_index_adapter/rev_engine.py` in the Rev repo follows `decision_index/engines/base.py`:

- `RevHttpEngine` (`--engine decision_index_adapter.rev_engine:RevHttpEngine --option base_url=...`) for a running
  server.
- `RevLocalEngine` (`--option repo=robbalian/rev-qwen3.5-4b --option rev_dir=/path/to/rev`) starts
  `serve_local.py` on the local GPU (4B fits 12 GB; 27B needs bf16 room for the base) and scores over loopback, so
  the run can be reproduced on the index's RTX PRO 6000 or via `hf-job`.
- One fixed prompt for every benchmark (`State / Question / Options: key: description / Decision:`), no truncation
  (rows over the 16,384-token row limit are `Unsupported`, counted by the server's own tokenizer), no option
  filtering, no abstention outcome, no correctness-based retry. Object-valued instructions/descriptions are
  JSON-serialized like the kit's reference engine. Provenance and the policy are written to `environment.json`.
- Optional registry lines: `"rev": "decision_index_adapter.rev_engine:RevHttpEngine"`,
  `"rev-local": "decision_index_adapter.rev_engine:RevLocalEngine"`.

Verified: the kit's own `decision_index run` with `RevHttpEngine` on a 101-request stratified slice of the frozen
suite (101/101 answered, 0 errors), scored with the kit's `benchmark_summary` / `score_panel`.

## Self-measured numbers (sample, not a full run)

We could not download `multimodalart/decision-index-suite` (401 without credentials), so we rebuilt the panel
benchmarks with `suite rebuild` and checked each against `hub/manifest.json` (normalized-file sha256 and the
selected group ids all match for 18 of the 19 panel benchmarks; BRIGHT's rebuild did not finish). On a
`stratified_sample` draw (seed 20260919, whole linked cases, exclusions applied) of **1,046 requests / 823 cases /
3,487 fields** over those 18 benchmarks, scored with the kit's scorers and the leaderboard formula:

| | Decision Index (18 benchmarks) | balanced_skill |
|---|---:|---:|
| Rev 27B (`jb_20260922-211844_27b`) | **61.07** (95% case-bootstrap CI 56.7-65.0) | 46.2 |
| Jev jev-1.13.0, same 18 benchmarks (recomputed from `data/index.json`) | **63.15** | 49.1 |
| Jev, full 19-panel headline in the Space | 59.51 | 46.3 |

Area means (Rev 27B / Jev, same 18): Knowledge & Reasoning 0.638 / 0.689; Language Understanding 0.637 / 0.623;
Retrieval & Classification (ESCI only) 0.456 / 0.552; Tools & Automation 0.764 / 0.734; Arts & Human Judgment
0.559 / 0.559. Per benchmark and the per-field results are in `results/decision_index/` of the Rev repo.

Caveats, stated plainly: this is ~46 cases per benchmark, not the 132,422-request suite, so about +/-4 index
points; BRIGHT is missing, so 61.07 is not comparable to the Space's 19-benchmark headline numbers (Jev's is 63.15
on the same 18); iSarcasmEval's headline track had only 14 cases in the sample; latency was measured over the
internet against a shared batching server and is not comparable to the index's in-process timings. The 4B has not
been run on the index yet. A full run with `run_full_suite.sh` (about 4-6 h at concurrency 8 against our server)
and an uploaded `scores.json` will follow, or the maintainer can run `RevLocalEngine` directly.

## Training data statement

No Decision Index item was used for training. Rev's training data are our own synthetic workflow sets plus public
task data; the sampled index rows live only under `results/decision_index/` (excluded from the training paths, which
assert no id/group/state overlap with our frozen evaluation sets). We did not touch the 442 excluded rows
differently from anyone else. If it helps review, we can list the public source datasets that overlap with the
index's sources (MMLU, ContractNLI and others appear in both) so the entry can be flagged like other entrants
trained on public benchmarks.

## What we ask

1. Accept `decision_index_adapter/rev_engine.py` as the entrant adapter (either as a `module:Class` path or the two
   registry lines above).
2. Run, or let us submit, the full frozen suite for Rev 4B and Rev 27B under the standard rules (no truncation, no
   option filtering, unanswered = wrong) and add both to `data/index.json` with kind "LoRA + head", base and
   parameter count read from the Hub repos as the methodology prescribes.
3. Tell us if the results-dataset layout in the README (`runs/<name>/scores.json`, `results.jsonl.gz`,
   `environment.json`) is still what `build_leaderboard.py` expects, since that script is not in the public kit.
