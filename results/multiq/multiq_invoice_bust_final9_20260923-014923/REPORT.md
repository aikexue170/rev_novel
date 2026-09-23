# Multi-question latency multiq_invoice_bust_final9_20260923-014923

Questions per request on one shared context, 5 repetitions each, one request in flight, cloud client in us-west. GPU = server-side forward (prefill + fork + branches on the shared path); network = e2e minus server time.

| Arm | Q/req | Accuracy | e2e p50 | e2e p95 | GPU | prefill | fork | branches | network | path | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| dense27jb | 1 | 100.0% | 100 ms | 107 ms | 56 ms |  |  |  | 41 ms | rows |  |
| dense27jb | 5 | 100.0% | 107 ms | 112 ms | 62 ms |  |  |  | 40 ms | rows |  |
| dense27jb | 20 | 97.8% | 218 ms | 224 ms | 171 ms | 60 | 1 | 110 | 43 ms | shared |  |
| jev | 1 | 100.0% | 191 ms | 312 ms |  |  |  |  |  |  | $0.0185 |
| jev | 5 | 95.0% | 203 ms | 317 ms |  |  |  |  |  |  | $0.0059 |
| jev | 20 | 91.0% | 210 ms | 318 ms |  |  |  |  |  |  | $0.0035 |
| small4jb | 1 | 75.0% | 64 ms | 67 ms | 21 ms |  |  |  | 40 ms | rows |  |
| small4jb | 5 | 90.0% | 65 ms | 74 ms | 22 ms |  |  |  | 40 ms | rows |  |
| small4jb | 20 | 83.0% | 97 ms | 102 ms | 49 ms |  |  |  | 42 ms | rows |  |
| small9b | 1 | 100.0% | 64 ms | 70 ms | 21 ms |  |  |  | 40 ms | rows |  |
| small9b | 5 | 86.0% | 69 ms | 75 ms | 25 ms |  |  |  | 41 ms | rows |  |
| small9b | 20 | 83.2% | 111 ms | 114 ms | 67 ms | 23 | 1 | 43 | 41 ms | shared |  |

- **dense27jb**: Qwen/Qwen3.8-27B · `jb_20260922-211844_27b` · NVIDIA B200 in seattle

- **small4jb**: Qwen/Qwen3.5-4B · `jb_20260923-002700_4b` · NVIDIA B200 in seattle

- **small9b**: Qwen/Qwen3.5-9B · `full_20260922-091011_9b` · NVIDIA B200 in seattle
