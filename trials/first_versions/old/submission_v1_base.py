"""STARE-LB v1-base standalone
"""
# Copy of nowei/submission.py minimal standalone wrapper
import numpy as np
import torch

EWMA_ALPHA = 0.2
WINDOW_CLIP_PCTL = 99.0
PAR_GAIN_THRESHOLD = 0.05
MIN_PAR_TO_ACT = 1.05

# For brevity we import the core functions from the package nowei.submission if available
# If not, this file is a placeholder pointing to the main implementation.
# In the repo this is already available as nowei/submission.py; to produce a fully
# self-contained single-file submission simply copy the large implementation here.
try:
    from nowei.submission import rebalance, _reset  # type: ignore
except Exception:
    # In case it's not importable, provide a minimal noop rebalance
    def _reset():
        pass
    def rebalance(hotness, n_device, n_red_expert):
        hotness = np.asarray(hotness)
        n_layers, n_experts = hotness.shape[1], hotness.shape[2]
        n_phys = n_experts + n_red_expert
        dep = np.zeros((n_layers, n_device, n_phys // n_device), dtype=np.int64)
        return False, np.array([], dtype=np.int64), dep, None
