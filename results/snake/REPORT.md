# Snake arena snake_20260922-162345

10 games per player, 10x10 grid, up to 1000 steps, identical seeds. One `choice` question per tick (up/down/left/right) from the same board text. Ranked by total food eaten; steps survived as tie-break.

| Player | Food total | Food / game | Best game | Steps / game | Deaths | Median decision | Cost |
|---|---:|---:|---:|---:|---|---:|---:|
| Qwen 27B + head | **201** | 20.1 | 28 | 187 | self 10 | 101 ms | — |
| Hosted Jev | **144** | 14.4 | 19 | 113 | self 10 | 180 ms | $0.0351 |
| Qwen 4B + head | **114** | 11.4 | 19 | 172 | self 7, starved 3 | 127 ms | — |
| Qwen 9B + head | **109** | 10.9 | 16 | 144 | self 10 | 70 ms | — |
