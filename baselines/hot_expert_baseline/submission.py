import numpy as np


def rebalance(hotness, n_device, n_red_expert):
    """Assign redundant slots to the hottest experts in each layer."""
    load = hotness.sum(axis=0)
    n_layers, n_experts = load.shape
    n_exp_per_dev = (n_experts + n_red_expert) // n_device

    deployment = np.zeros((n_layers, n_device, n_exp_per_dev), dtype=np.int64)
    base_slots = n_exp_per_dev - 1

    for layer in range(n_layers):
        for device in range(n_device):
            for slot in range(base_slots):
                deployment[layer, device, slot] = (
                    device * base_slots + slot
                ) % n_experts

        hottest = np.argsort(load[layer])[::-1]
        for device in range(n_device):
            deployment[layer, device, -1] = hottest[device % len(hottest)]

    layers_priority = np.arange(n_layers, dtype=np.int64)
    return True, layers_priority, deployment, None
