"""Minimal placeholder for a transmission-aware rebalancer.

This module provides a working `rebalance_experts` entrypoint compatible with
the DeepSeek implementation used elsewhere in the repo. It currently implements
a simple round-robin replication scheme so the simulator can call the
`proposed` algorithm without errors. You can iterate on this file to add the
transmission-aware greedy heuristic later.
"""
from typing import Tuple
import torch
import numpy as np

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple, safe fallback rebalancer.

    It assigns physical experts in a round-robin fashion over logical experts
    and produces the same return-shape as the DeepSeek implementation. This
    makes it easy to swap in a more advanced, transmission-aware algorithm
    later while keeping the simulator working.

    Returns:
        phy2log: [num_layers, num_replicas]
        log2phy: [num_layers, num_logical_experts, X]
        logcnt: [num_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape

    device = weight.device

    # Assign physical experts to logical experts in a round-robin manner.
    phy2log = (
        torch.arange(num_replicas, dtype=torch.int64, device=device)
        .unsqueeze(0)
        .expand(num_layers, -1)
        % num_logical_experts
    )

    # Replica rank (0..)
    phyrank = torch.zeros_like(phy2log, dtype=torch.int64)

    # Count replicas per logical expert
    logcnt = torch.zeros((num_layers, num_logical_experts), dtype=torch.int64, device=device)
    for i in range(num_layers):
        counts = torch.bincount(phy2log[i], minlength=num_logical_experts)
        logcnt[i, : counts.numel()] = counts

    # Build log2phy mapping with shape [num_layers, num_logical_experts, maxlogcnt]
    maxlogcnt = int(logcnt.max().item()) if logcnt.numel() > 0 else 0
    if maxlogcnt == 0:
        # no replicas
        log2phy = torch.full((num_layers, num_logical_experts, 0), -1, dtype=torch.int64, device=device)
        return phy2log, log2phy, logcnt

    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64, device=device
    )

    for layer in range(num_layers):
        # iterate phy indices and place into log2phy according to phy2log assignment
        fill_ptr = {int(i): 0 for i in range(num_logical_experts)}
        for phy_idx in range(num_replicas):
            log = int(phy2log[layer, phy_idx].item())
            pos = fill_ptr[log]
            log2phy[layer, log, pos] = phy_idx
            fill_ptr[log] += 1

    return phy2log, log2phy, logcnt

def rebalance(hotness, n_device, n_red_expert):
    """
    :param hotness: A numpy.ndarray of shape [n_expert] or [n_layer, n_expert]
    :param n_device: int, number of devices to partition over
    :param n_red_expert: int, number of replicas per expert (total number of 'physical' experts per logical expert)
    :return: assign, a numpy.ndarray of shape [n_layer, n_red_expert * n_device] or [n_red_expert * n_device]
             Each value is the logical expert index that should be mapped to the i-th (layer, physical expert) for the system.
    """
    hotness = np.asarray(hotness)
    if hotness.ndim == 1:
        # No layers, add dummy layer axis for uniform handling
        hotness = hotness[None, :]
        added_layer_axis = True
    else:
        added_layer_axis = False

    n_layer, n_expert = hotness.shape
    # Arrange assignments in round-robin by default (as a valid placeholder)
    assign = np.empty((n_layer, n_device * n_red_expert), dtype=np.int32)
    for layer in range(n_layer):
        for p in range(n_device * n_red_expert):
            assign[layer, p] = p % n_expert

    if added_layer_axis:
        assign = assign[0]  # Remove dummy layer axis for 1D hotness input

    return assign

__all__ = ["rebalance_experts", "rebalance"]
