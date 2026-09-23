# Load test final9_20260923-014354

Exact 975-question public holdout, one question per request, cloud client in us-west. Self-hosted cost = (GPU + provisioned CPU/RAM list price per second) / measured answers per second, i.e. cost at the utilization actually achieved at that concurrency. Hosted Jev cost = billed usage returned by the API.

| Arm | GPU | Conc. | Accuracy | p50 ms | p95 ms | Answers/s | GPU busy | Cost / 1k answers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small4jb | NVIDIA B200 | 1 | 87.69% | 37 | 64 | 24.5 | 59% | $0.0837 |
| small4jb | NVIDIA B200 | 8 | 87.69% | 110 | 206 | 66.5 | 97% | $0.0309 |
| small4jb | NVIDIA B200 | 32 | 87.69% | 338 | 632 | 84.2 | 97% | $0.0244 |
| small4jb | NVIDIA B200 | 128 | 87.79% | 913 | 4094 | 93.3 | 97% | $0.0220 |
| small4jb | NVIDIA B200 | 256 | 87.90% | 2329 | 6158 | 75.5 | 89% | $0.0272 |
| small9b | NVIDIA B200 | 1 | 88.72% | 40 | 82 | 17.8 | 49% | $0.1156 |
| dense27jb | NVIDIA B200 | 1 | 92.00% | 75 | 202 | 10.0 | 71% | $0.2055 |
| jev | hosted | 1 | 85.03% | 142 | 237 | 6.3 |  | $0.0366 |
| small9b | NVIDIA B200 | 8 | 88.62% | 134 | 263 | 55.4 | 97% | $0.0371 |
| small9b | NVIDIA B200 | 32 | 88.72% | 432 | 791 | 69.2 | 99% | $0.0297 |
| small9b | NVIDIA B200 | 128 | 88.51% | 1612 | 3783 | 60.8 | 88% | $0.0338 |
| dense27jb | NVIDIA B200 | 8 | 92.00% | 377 | 949 | 17.6 | 86% | $0.1170 |
| jev | hosted | 4 | 85.13% | 140 | 441 | 20.4 |  | $0.0366 |
| small9b | NVIDIA B200 | 256 | 88.62% | 2374 | 6390 | 69.2 | 95% | $0.0297 |
| jev | hosted | 16 | 85.23% | 138 | 231 | 102.7 |  | $0.0366 |
| dense27jb | NVIDIA B200 | 32 | 92.00% | 1256 | 2309 | 23.8 | 99% | $0.0865 |
| dense27jb | NVIDIA B200 | 128 | 92.00% | 4729 | 7722 | 25.1 | 100% | $0.0819 |

Errors per level: none

- **small9b**: Qwen/Qwen3.5-9B · checkpoint `full_20260922-091011_9b` (19601 training examples) · NVIDIA B200 in seattle · batching {'max_rows': 128, 'max_padded_tokens': 49152, 'window_ms': 2.0, 'overhead_tokens': 256} · cold start 65s
- **small4jb**: Qwen/Qwen3.5-4B · checkpoint `jb_20260923-002700_4b` (25886 training examples) · NVIDIA B200 in seattle · batching {'max_rows': 128, 'max_padded_tokens': 49152, 'window_ms': 2.0, 'overhead_tokens': 256} · cold start 0s
- **dense27jb**: Qwen/Qwen3.8-27B · checkpoint `jb_20260922-211844_27b` (25886 training examples) · NVIDIA B200 in seattle · batching {'max_rows': 128, 'max_padded_tokens': 32768, 'window_ms': 2.0, 'overhead_tokens': 256} · cold start 0s
