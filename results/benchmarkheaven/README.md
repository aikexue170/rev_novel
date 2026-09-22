# JevBench public items in our request format

- Source: https://benchmarkheaven.com/jev-models (JevBench v1.3.0 scoring, protocol jevbench::v1.2; harness and public tasks: https://github.com/fstandhartinger/jevbench, MIT)
- Items: 231 (all public items; the benchmark has 534, the other 303 are held out or imported and not redistributed)
- File: `jevbench_public_231.jsonl`, sha256 `2ae89901417ae2b636612758601181b774f0a735ea4bcb7efd86907117e7da05`

| Upstream file | Items | Upstream sha256 (matches datasets/manifest.json) | Tier |
|---|---:|---|---|
| datasets/public/easy.jsonl | 48 | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` | easy |
| datasets/public/original.jsonl | 72 | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` | standard |
| datasets/public/hard.jsonl | 111 | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` | hard |

Public items by tier and question type: easy/choice: 36, easy/noul: 12, hard/choice: 67, hard/noul: 38, hard/score: 6, standard/choice: 36, standard/noul: 24, standard/score: 12

Per-tier public families: easy/extraction: 12, easy/fact: 12, easy/intent: 12, easy/tool_selection: 12, hard/adversarial: 6, hard/ambiguous: 7, hard/judge_hard: 17, hard/long_policy: 19, hard/multi_hop: 18, hard/probability: 10, hard/routing_hard: 5, hard/temporal_numeric: 15, hard/tradeoff: 6, hard/trap: 8, standard/adequacy: 12, standard/extraction: 12, standard/intent: 12, standard/ordinal: 12, standard/policy: 12, standard/routing: 12

## Mapping

- `choice`: criteria dict sent as-is; keys are the benchmark labels.
- `noul`: criteria keyed `true`/`false` sent as-is (as the benchmark gives them); the answer key is mapped `true->yes`, `false->no`, the labels the harness scores over (its TypeSafe adapter does the same with Jev's p_yes).
- `score`: the benchmark's ordered level list becomes `{"0": level0, "1": level1, ...}`; our server accepts only dict criteria, and the level indices are exactly the benchmark's labels.
- One question per request, as in the harness (each item has its own state).
- `gold_probs` (10 public probability items) is kept for the Calibration axis's fidelity term.

Rules: not used for training; not copied into data/ or eval_sets/.
