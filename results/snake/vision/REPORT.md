# Snake arena snakevis_20260922-155551

10 games per player, 10x10 grid, up to 200 steps, identical seeds. One `choice` question per tick (up/down/left/right) from the same board text. Ranked by total food eaten; steps survived as tie-break.

| Player | Food total | Food / game | Best game | Steps / game | Deaths | Median decision | Cost |
|---|---:|---:|---:|---:|---|---:|---:|
| Qwen 27B + head (image board) | **66** | 6.6 | 14 | 64 | self 8, wall 1, starved 1 | 158 ms | — |
| Qwen 9B + head (image board) | **19** | 1.9 | 4 | 14 | self 7, wall 3 | 102 ms | — |
| Qwen 4B + head (image board) | **17** | 1.7 | 4 | 11 | self 7, wall 3 | 73 ms | — |
