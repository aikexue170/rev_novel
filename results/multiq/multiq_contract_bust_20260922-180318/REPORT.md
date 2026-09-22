# Multi-question latency multiq_contract_bust_20260922-180318

Questions per request on one shared context, 5 repetitions each, one request in flight, cloud client in us-west. GPU = server-side forward (prefill + fork + branches on the shared path); network = e2e minus server time.

| Arm | Q/req | Accuracy | e2e p50 | e2e p95 | GPU | prefill | fork | branches | network | path | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| dense27b | 1 | 100.0% | 152 ms | 197 ms | 127 ms |  |  |  | 19 ms | rows |  |
| dense27b | 5 | 85.0% | 220 ms | 273 ms | 188 ms | 125 | 2 | 61 | 20 ms | shared |  |
| dense27b | 17 | 86.8% | 319 ms | 365 ms | 269 ms | 127 | 2 | 140 | 19 ms | shared |  |
| jev | 1 | 100.0% | 114 ms | 252 ms |  |  |  |  |  |  | $0.0966 |
| jev | 5 | 65.0% | 130 ms | 307 ms |  |  |  |  |  |  | $0.0226 |
| jev | 17 | 75.0% | 137 ms | 205 ms |  |  |  |  |  |  | $0.0097 |
| small4b | 1 | 100.0% | 55 ms | 75 ms | 33 ms |  |  |  | 16 ms | rows |  |
| small4b | 5 | 80.0% | 85 ms | 95 ms | 59 ms | 33 | 1 | 25 | 15 ms | shared |  |
| small4b | 17 | 85.3% | 132 ms | 155 ms | 80 ms | 34 | 1 | 45 | 22 ms | shared |  |
| small9b | 1 | 100.0% | 68 ms | 78 ms | 42 ms |  |  |  | 21 ms | rows |  |
| small9b | 5 | 85.0% | 96 ms | 113 ms | 68 ms | 42 | 1 | 25 | 17 ms | shared |  |
| small9b | 17 | 85.0% | 143 ms | 167 ms | 94 ms | 42 | 1 | 51 | 19 ms | shared |  |

- **dense27b**: Qwen/Qwen3.8-27B · `full_20260922-091011_27b` · NVIDIA B200 in seattle

- **small4b**: Qwen/Qwen3.5-4B · `full_20260922-091011_4b` · NVIDIA B200 in seattle

- **small9b**: Qwen/Qwen3.5-9B · `full_20260922-091011_9b` · NVIDIA B200 in sea
