from enum import Enum, auto
from typing import Optional
import os
import torch
import numpy as np
from . import deepseek


class EplbAlgorithm(Enum):
    deepseek = auto()
    deepseek_hierarchical = auto()
    proposed = auto()


def rebalance_experts(
        tokens_per_expert: torch.Tensor,
        num_physical_experts: int = 320,
        num_local_physical_experts: int = 64,
        num_groups: Optional[int] = 1,
        num_nodes: int = 1,
        algorithm: EplbAlgorithm = EplbAlgorithm.deepseek,
):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        physical_to_logical_map, logical_to_all_physical_map, expert_count = \
            deepseek.rebalance_experts(
                weight=tokens_per_expert.sum(dim=0),
                num_replicas=num_physical_experts,
                num_groups=num_groups,
                num_nodes=num_nodes,
                num_gpus=num_physical_experts // num_local_physical_experts,
                enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
            )
        return physical_to_logical_map, logical_to_all_physical_map, expert_count, list(
            range(physical_to_logical_map.shape[0]))
    if algorithm in [EplbAlgorithm.proposed]:
        # TODO: add algorithm details
        pass
    raise NotImplementedError


def rebalance(n_device=64, n_red_expert=64, algorithm='deepseek'):
    """Build a per-cadence rebalance closure for the simulator loop.

    - 'deepseek' / 'deepseek_hierarchical': go through the DeepSeek reference
      implementation and mark every layer as a redeploy candidate.
    - 'proposed' / 'submission': load submission.py at the repo root and
      delegate to its `rebalance(hotness, n_device, n_red_expert)`. This is
      how quickstart.py scores our STARE-LB policy.
    """
    # Map algorithm alias -> source filename at the repo root.
    sub_aliases = {
        'proposed': 'submission.py',
        'submission': 'submission.py',
        'submission_v2': 'submission_v2.py',
        'submission_v3': 'submission_v3.py',
        'submission_v4': 'submission_v4.py',
        'submission_v5': 'submission_v5.py',
        'submission_v6': 'submission_v6.py',
    }
    if algorithm in sub_aliases:
        # Lazy import so the DeepSeek path is free of any submission cost.
        import sys
        import importlib.util
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        sub_path = repo_root / sub_aliases[algorithm]
        # Always load the submission fresh from disk so:
        #   (1) edits between interactive runs are picked up automatically;
        #   (2) per-run module state starts clean even when run_case() is
        #       invoked multiple times in one process.
        mod_name = sub_path.stem  # 'submission' or 'submission_v2'
        spec = importlib.util.spec_from_file_location(mod_name, str(sub_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load submission from {sub_path}")
        _sub = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = _sub
        spec.loader.exec_module(_sub)
        if hasattr(_sub, '_reset'):
            _sub._reset()

        def rebalance_(hotness, _sub=_sub, n_device=n_device, n_red_expert=n_red_expert):
            return _sub.rebalance(hotness, n_device, n_red_expert)

        return rebalance_

    def rebalance_(hotness, n_device=n_device, n_red_expert=n_red_expert, algorithm=algorithm):
        n_expert = hotness.shape[-1]
        physical_to_logical_map, logical_to_all_physical_map, expert_count, priority = \
            rebalance_experts(torch.from_numpy(hotness), n_expert + n_red_expert, (n_expert + n_red_expert) // n_device,
                              algorithm=EplbAlgorithm[algorithm])

        return len(priority) > 0, np.array(priority), physical_to_logical_map.numpy().reshape((-1, n_device, (n_expert + n_red_expert) // n_device)), None

    return rebalance_
