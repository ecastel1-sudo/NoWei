# DS-R1 Cadence Sweep Aggregate

Averages are unweighted across the 16 DS-R1 dataset/EP cases. Lower mean PAR is better; higher transmit means more expert-slot movement.

| variant | mean PAR | total transmit | mean transmit ratio vs 1024 | mean PAR ratio vs 1024 | mean PAR improvement vs Default |
|---|---:|---:|---:|---:|---:|
| Default | 3.246764 | 0 | 0.000 | 0.619 | 0.000 |
| DS-EPLB@256:layers=all | 1.512024 | 6151924 | 3.596 | 1.202 | 0.474 |
| DS-EPLB@512:layers=all | 1.694188 | 3449713 | 1.999 | 1.079 | 0.422 |
| DS-EPLB@1024:layers=all | 1.855533 | 1725811 | 1.000 | 1.000 | 0.381 |
| DS-EPLB@2048:layers=all | 2.228625 | 684565 | 0.397 | 0.868 | 0.288 |
| DS-EPLB@4096:layers=all | 2.798583 | 347543 | 0.201 | 0.709 | 0.131 |
