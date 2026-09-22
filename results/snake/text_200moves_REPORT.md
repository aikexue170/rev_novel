# Snake arena snake_20260922-161608

10 games per player, 10x10 grid, up to 200 steps, identical seeds. One `choice` question per tick (up/down/left/right) from the same board text. Ranked by total food eaten; steps survived as tie-break.

| Player | Food total | Food / game | Best game | Steps / game | Deaths | Median decision | Cost |
|---|---:|---:|---:|---:|---|---:|---:|
| Qwen 27B + head | **189** | 18.9 | 23 | 176 | self 5, timeout 5 | 96 ms | — |
| Hosted Jev | **113** | 11.3 | 21 | 88 | starved 2, self 8 | 168 ms | $0.0275 |
| Qwen 9B + head | **107** | 10.7 | 15 | 138 | self 8, timeout 2 | 68 ms | — |
| Qwen 4B + head | **101** | 10.1 | 15 | 148 | self 5, starved 3, timeout 2 | 127 ms | — |
