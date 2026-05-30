# DS-EPLB Transfer/PAR Sweep

PAR is lower-is-better. `transmit_amount` counts changed expert slots.

| dataset | EP | variant | mean PAR | transmit | PAR ratio vs DS-EPLB@1024 | transmit ratio |
|---|---:|---|---:|---:|---:|---:|
| ShareGPT | 32 | Default | 1.643631 | 0 | 0.764785 | 0.000000 |
| ShareGPT | 32 | DS-EPLB@512:layers=all | 1.239803 | 164969 | 1.013889 | 2.003267 |
| ShareGPT | 32 | DS-EPLB@1024:layers=all | 1.257024 | 82350 | 1.000000 | 1.000000 |
| ShareGPT | 32 | DS-EPLB@2048:layers=all | 1.315802 | 32681 | 0.955329 | 0.396855 |
