# DS-R1 Sweep Expected Composite Scores

Scores use `DS-EPLB@1024:layers=all` as the per-case baseline and the hardware-modeled runtime formula.

Constants: `balanced_compute_seconds = 60.0`, `expert_bytes = 88_080_384`, `bandwidth_bytes_per_second = 900_000_000_000`.

| variant | expected composite score | mean modeled runtime (s) | min case score | max case score |
|---|---:|---:|---:|---:|
| Default | 68.017465 | 194.805831 | 48.111164 | 86.376476 |
| DS-EPLB@256:layers=all | 93.127214 | 128.350883 | 82.057151 | 118.403732 |
| DS-EPLB@512:layers=all | 97.986051 | 122.752087 | 91.819755 | 114.225252 |
| DS-EPLB@1024:layers=all | 100.000000 | 121.888220 | 100.000000 | 100.000000 |
| DS-EPLB@2048:layers=all | 92.264696 | 137.904780 | 70.737172 | 102.224241 |
| DS-EPLB@4096:layers=all | 76.819013 | 170.040784 | 56.763571 | 93.584571 |
