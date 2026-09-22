# Multi-question latency multiq_invoice_bust_20260922-174831

Questions per request on one shared context, 5 repetitions each, one request in flight, cloud client in us-west. GPU = server-side forward (prefill + fork + branches on the shared path); network = e2e minus server time.

| Arm | Q/req | Accuracy | e2e p50 | e2e p95 | GPU | prefill | fork | branches | network | path | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| dense27b | 1 | 75.0% | 82 ms | 91 ms | 47 ms |  |  |  | 31 ms | rows |  |
| dense27b | 5 | 90.0% | 99 ms | 105 ms | 64 ms |  |  |  | 30 ms | rows |  |
| dense27b | 20 | 95.0% | 198 ms | 204 ms | 163 ms | 52 | 1 | 110 | 31 ms | shared |  |
| jev | 1 | 100.0% | 152 ms | 284 ms |  |  |  |  |  |  | $0.0185 |
| jev | 5 | 95.0% | 154 ms | 288 ms |  |  |  |  |  |  | $0.0059 |
| jev | 20 | 91.0% | 160 ms | 292 ms |  |  |  |  |  |  | $0.0035 |
| small4b | 1 | 100.0% | 57 ms | 59 ms | 23 ms |  |  |  | 31 ms | rows |  |
| small4b | 5 | 85.0% | 58 ms | 64 ms | 24 ms |  |  |  | 30 ms | rows |  |
| small4b | 20 | 79.8% | 86 ms | 91 ms | 49 ms |  |  |  | 31 ms | rows |  |
| small9b | 1 | 100.0% | 57 ms | 64 ms | 22 ms |  |  |  | 31 ms | rows |  |
| small9b | 5 | 85.0% | 59 ms | 60 ms | 25 ms |  |  |  | 30 ms | rows |  |
| small9b | 20 | 82.8% | 103 ms | 115 ms | 68 ms | 25 | 1 | 43 | 31 ms | shared |  |

- **dense27b**: Qwen/Qwen3.8-27B · `full_20260922-091011_27b` · NVIDIA B200 in seattle

- **small4b**: Qwen/Qwen3.5-4B · `full_20260922-091011_4b` · NVIDIA B200 in seattle

- **small9b**: Qwen/Qwen3.5-9B · `full_20260922-091011_9b` · NVIDIA B200 in seattle
