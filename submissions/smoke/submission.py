import numpy as np


def rebalance(hotness, n_device, n_red_expert):
    """Minimal API smoke-test submission that does not request redeployment."""
    n_layers = hotness.shape[1]
    n_experts = hotness.shape[2]
    n_exp_per_dev = (n_experts + n_red_expert) // n_device

    deployment = np.zeros((n_layers, n_device, n_exp_per_dev), dtype=np.int64)

    for layer in range(n_layers):
        for device in range(n_device):
            for slot in range(n_exp_per_dev - 1):
                deployment[layer, device, slot] = (
                    device * (n_exp_per_dev - 1) + slot
                ) % n_experts
            deployment[layer, device, -1] = deployment[layer, device, -2]

    return False, [], deployment, None
