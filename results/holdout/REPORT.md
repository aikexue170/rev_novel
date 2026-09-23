# Load test final8_20260923-001846

Exact 975-question public holdout, one question per request, cloud client in us-west. Self-hosted cost = (GPU + provisioned CPU/RAM list price per second) / measured answers per second, i.e. cost at the utilization actually achieved at that concurrency. Hosted Jev cost = billed usage returned by the API.

| Arm | GPU | Conc. | Accuracy | p50 ms | p95 ms | Answers/s | GPU busy | Cost / 1k answers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small9b | NVIDIA B200 | 1 | 88.72% | 62 | 99 | 15.0 | 40% | $0.1370 |
| small4b | NVIDIA B200 | 1 | 87.59% | 65 | 102 | 13.9 | 48% | $0.1477 |
| small9b | NVIDIA B200 | 8 | 88.62% | 139 | 266 | 53.1 | 97% | $0.0387 |
| small4b | NVIDIA B200 | 8 | 87.59% | 142 | 321 | 47.4 | 96% | $0.0433 |
| small9b | NVIDIA B200 | 32 | 88.41% | 428 | 811 | 68.8 | 98% | $0.0299 |
| small4b | NVIDIA B200 | 32 | 87.59% | 395 | 749 | 74.4 | 98% | $0.0276 |
| small9b | NVIDIA B200 | 128 | 88.51% | 1474 | 2559 | 72.6 | 99% | $0.0283 |
| small4b | NVIDIA B200 | 128 | 87.59% | 1262 | 2121 | 80.9 | 92% | $0.0254 |
| small9b | NVIDIA B200 | 256 | 88.51% | 2754 | 4566 | 73.6 | 98% | $0.0279 |
| small4b | NVIDIA B200 | 256 | 87.59% | 2037 | 3272 | 93.2 | 99% | $0.0220 |
| dense27jb | NVIDIA B200 | 1 | 92.00% | 88 | 240 | 8.5 | 53% | $0.2416 |
| jev | hosted | 1 | 85.23% | 159 | 278 | 5.7 |  | $0.0366 |
| dense27jb | NVIDIA B200 | 8 | 92.00% | 335 | 713 | 21.5 | 99% | $0.0955 |
| jev | hosted | 4 | 85.13% | 157 | 279 | 23.3 |  | $0.0366 |
| jev | hosted | 16 | 84.92% | 148 | 237 | 99.1 |  | $0.0366 |
| dense27jb | NVIDIA B200 | 32 | 92.00% | 1260 | 2241 | 24.0 | 99% | $0.0857 |
| dense27jb | NVIDIA B200 | 128 | 92.00% | 4658 | 7930 | 25.3 | 100% | $0.0812 |

Errors per level: none

- **small4b**: Qwen/Qwen3.5-4B · checkpoint `full_20260922-091011_4b` (19601 training examples) · NVIDIA B200 in us-west · batching {'max_rows': 128, 'max_padded_tokens': 49152, 'window_ms': 2.0, 'overhead_tokens': 256} · cold start 0s
- **small9b**: Qwen/Qwen3.5-9B · checkpoint `full_20260922-091011_9b` (19601 training examples) · NVIDIA B200 in us-west · batching {'max_rows': 128, 'max_padded_tokens': 49152, 'window_ms': 2.0, 'overhead_tokens': 256} · cold start 0s
- **dense27jb**: Qwen/Qwen3.8-27B · checkpoint `jb_20260922-211844_27b` (25886 training examples) · NVIDIA B200 in us-west · batching {'max_rows': 128, 'max_padded_tokens': 32768, 'window_ms': 2.0, 'overhead_tokens': 256} · cold start 0s
