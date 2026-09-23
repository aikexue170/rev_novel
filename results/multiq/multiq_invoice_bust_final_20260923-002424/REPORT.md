# Multi-question latency multiq_invoice_bust_final_20260923-002424

Questions per request on one shared context, 5 repetitions each, one request in flight, cloud client in us-west. GPU = server-side forward (prefill + fork + branches on the shared path); network = e2e minus server time.

| Arm | Q/req | Accuracy | e2e p50 | e2e p95 | GPU | prefill | fork | branches | network | path | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| dense27jb | 1 | 100.0% | 80 ms | 147 ms | 45 ms |  |  |  | 32 ms | rows |  |
| dense27jb | 5 | 100.0% | 100 ms | 335 ms | 63 ms |  |  |  | 32 ms | rows |  |
| dense27jb | 20 | 97.0% | 200 ms | 467 ms | 159 ms | 49 | 1 | 109 | 36 ms | shared |  |
| jev | 1 | 100.0% | 142 ms | 252 ms |  |  |  |  |  |  | $0.0185 |
| jev | 5 | 95.0% | 166 ms | 236 ms |  |  |  |  |  |  | $0.0059 |
| jev | 20 | 90.8% | 170 ms | 333 ms |  |  |  |  |  |  | $0.0035 |
| small4b | 1 | 100.0% | 60 ms | 101 ms | 23 ms |  |  |  | 33 ms | rows |  |
| small4b | 5 | 85.0% | 61 ms | 95 ms | 24 ms |  |  |  | 32 ms | rows |  |
| small4b | 20 | 79.2% | 98 ms | 142 ms | 60 ms | 26 | 1 | 36 | 33 ms | shared |  |
| small9b | 1 | 100.0% | 54 ms | 56 ms | 21 ms |  |  |  | 31 ms | rows |  |
| small9b | 5 | 85.0% | 59 ms | 62 ms | 24 ms |  |  |  | 30 ms | rows |  |
| small9b | 20 | 82.8% | 100 ms | 113 ms | 66 ms | 22 | 1 | 43 | 31 ms | shared |  |

- **dense27jb**: Qwen/Qwen3.8-27B · `jb_20260922-211844_27b` · NVIDIA B200 in us-west

- **small4b**: Qwen/Qwen3.5-4B · `full_20260922-091011_4b` · NVIDIA B200 in us-west

- **small9b**: Qwen/Qwen3.5-9B · `full_20260922-091011_9b` · NVIDIA B200 in us-west
