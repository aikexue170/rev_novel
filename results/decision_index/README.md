# Decision Index 0.1 sample (external benchmark, evaluation only)

Source: Hugging Face Space https://huggingface.co/spaces/multimodalart/jev-decision-index (leaderboard bundle
`data/index.json` generated 2026-09-22T11:12Z) and its reproduction kit https://github.com/apolinario/decision-index
(commit 52a6989, 2026-09-21). The Space's frozen suite is a Hub dataset (`multimodalart/decision-index-suite`,
132,422 requests, uncompressed sha256 `288d372...`) that is **not readable without credentials** (HTTP 401 for
anonymous access on 2026-09-22). The rows here were rebuilt from the pinned public sources with the kit's own
`suite rebuild` and verified byte-identical per benchmark: every normalized file's sha256 and every selected group
id match the kit's `hub/manifest.json` for the frozen release.

**Never train on this directory. Do not copy it into `data/` or `eval_sets/`.** GPQA Diamond's authors ask that its
questions not be posted in plain text; keep this directory local.

## Files

- `items.jsonl`: the sampled benchmark requests in our request format (`state`, `questions: [{id, instructions,
  criteria}]`, plus `expected`, `scoring`, `metadata`, `source`, `catalog_id`, `group_id`, `payload_sha256`).
  Sampling: the kit's `stratified_sample` (seed 20260919, round-robin over benchmarks, whole linked cases,
  442-row exclusion list applied), ~46 cases per panel benchmark: 1,000 requests over 17 panel benchmarks plus 46
  ToolRet queries drawn with the same rule (ToolRet's build finished during scoring). BRIGHT (catalog 36) was
  **not run**: its rebuild was still normalizing when the report was due, so the index here is over 18 of the 19
  panel benchmarks and Jev's index is recomputed on the same 18 for comparison. Counts and the file's sha256 are in
  `summary.json` (`requests`, `cases`, `question_fields`, `items_sha256`).
- `dense27b.jsonl`: one line per question field: `id` (suite run_id), `question`, `source`, `expected`,
  `accepted` (tie set for ChessBench/Habermas), `choice`, `probabilities`, `correct`, `latency_ms` (HTTP wall time
  of the whole request), `prompt_tokens_max`.
- `summary.json`: Decision Index on the sample (kit formula, `balanced_raw`), 300-replicate case bootstrap CI,
  per-area and per-benchmark raw/skill with Jev's published values, Jev and leaderboard models recomputed on the
  same benchmark set, field accuracy by source, HTTP latency p50/p95.

Mapping to our API: the suite's `questions` map became a list of `{id, instructions, criteria}`; object-valued
`instructions`/descriptions were JSON-serialized (the kit's reference engine renders them the same way); option
keys and descriptions are otherwise unchanged; `state` passed through as given. Requests over the server's
16,384-token limit would have been recorded as `unsupported` (counted wrong), never truncated; none occurred in
this sample.
