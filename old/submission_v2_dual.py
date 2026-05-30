"""STARE-LB v2-dual standalone (thin wrapper)

This file imports the implementation from `nowei/submission_v2_dual.py` when
available. For a single-file submission, copy the full source of that module
into this file.
"""
import numpy as np
try:
    from nowei.submission_v2_dual import rebalance, _reset  # type: ignore
except Exception:
    def _reset():
        pass
    def rebalance(hotness, n_device, n_red_expert):
        hotness = np.asarray(hotness)
        n_layers, n_experts = hotness.shape[1], hotness.shape[2]
        dep = np.zeros((n_layers, n_device, (n_experts + n_red_expert)//n_device), dtype=np.int64)
        return False, np.array([], dtype=np.int64), dep, None
