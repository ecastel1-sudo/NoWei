import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import os

mp.set_start_method("spawn", force=True)

BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
TRANSFER_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000


def transmission_time_seconds(transmit_amount):
    return transmit_amount * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND


def modeled_runtime_seconds(mean_par, transmit_amount):
    return BALANCED_COMPUTE_SECONDS * mean_par + transmission_time_seconds(transmit_amount)


class Configure(object):
    def __init__(self, dataset, model, n_red_experts, n_devices, n_nodes, n_layers, n_experts, collection_interval,
                 exp_adjust_iters, n_batch, figure_flag, trace_dir, output_dir, algo_set,
                 cpu_per_process, n_processes):
        self.dataset = dataset
        self.model = model
        self.n_red_experts = n_red_experts
        self.n_devices = n_devices
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.collection_interval = collection_interval
        self.exp_adjust_iters = exp_adjust_iters
        self.n_batch = n_batch
        self.figure_flag = figure_flag
        self.trace_dir = trace_dir
        self.output_dir = output_dir
        self.algo_set = algo_set
        self.cpu_per_process = cpu_per_process
        self.n_processes = n_processes
        self.n_exp_per_dev = (n_experts + n_red_experts) // n_devices
        self.algo_type = self.algo_set[0]
        self.top_k = 8

    def copy(self):
        return Configure(
            dataset=self.dataset,
            model=self.model,
            n_red_experts=self.n_red_experts,
            n_devices=self.n_devices,
            n_nodes=self.n_nodes,
            n_layers=self.n_layers,
            n_experts=self.n_experts,
            collection_interval=self.collection_interval,
            exp_adjust_iters=self.exp_adjust_iters,
            n_batch=self.n_batch,
            figure_flag=self.figure_flag,
            trace_dir=self.trace_dir,
            output_dir=self.output_dir,
            algo_set=self.algo_set.copy() if isinstance(self.algo_set, (list, tuple)) else self.algo_set,
            cpu_per_process=self.cpu_per_process,
            n_processes=self.n_processes
        )


class DynamicAlg:
    def __init__(self, cfg: Configure):
        self.cfg = cfg
        self.expert_hotness = None
        self.pars_per_iter = None
        self.transmit_amount_per_iter = None

        from eplb_algorithms import rebalance
        self.method = {
            'DS-EPLB': partial(rebalance(self.cfg.n_devices, self.cfg.n_red_experts, 'deepseek')),
            # TODO: add proposed algorithm
            # 'Proposed': partial(rebalance(self.cfg.n_devices, self.cfg.n_red_experts, 'proposed')),
        }

    def compute_hotness(self):
        trace_path = os.path.join(self.cfg.trace_dir, self.cfg.dataset + '.npy')
        self.expert_hotness = np.load(trace_path)
        mask = self.expert_hotness == 0
        self.expert_hotness[mask] += 1  # ensure hotness is not zero
        print(self.expert_hotness.shape)

    @staticmethod
    def calculate_par(hotness, deployment):
        n_experts = hotness.shape[0]
        n_devices, exp_per_dev = deployment.shape
        cut = np.bincount(deployment.reshape(-1), minlength=n_experts)
        weights = hotness / cut
        loads = weights[deployment.reshape(-1)].reshape((n_devices, exp_per_dev)).sum(-1)
        par = loads.max() / loads.mean()
        return par

    @staticmethod
    def cal_par_per_iter(cur_hotness, cur_deploy_table):
        pars = np.zeros((cur_hotness.shape[0],))
        for layer_idx in range(cur_hotness.shape[0]):
            pars[layer_idx] = DynamicAlg.calculate_par(cur_hotness[layer_idx], cur_deploy_table[layer_idx])
        return pars

    def init_deploy_table(self):
        deploy_tables = np.zeros((self.cfg.n_devices, self.cfg.n_exp_per_dev), dtype=int)
        for i in range(self.cfg.n_devices):
            if 'Default' in self.cfg.algo_type:
                for j in range(self.cfg.n_exp_per_dev):
                    deploy_tables[i, j] = (i * self.cfg.n_exp_per_dev + j) % self.cfg.n_experts
            else:
                for j in range(self.cfg.n_exp_per_dev - 1):
                    deploy_tables[i, j] = (i * (self.cfg.n_exp_per_dev - 1) + j) % self.cfg.n_experts
                deploy_tables[i, -1] = deploy_tables[i, -2]
        return deploy_tables

    def store_raw_data(self):
        raw_data_dir = os.path.join(self.cfg.output_dir, 'raw_data',
                                f'{self.cfg.dataset}_{self.cfg.model}_EP{self.cfg.n_devices}')
        if not os.path.exists(raw_data_dir):
            os.makedirs(raw_data_dir)
        np.save(os.path.join(raw_data_dir, f'{self.cfg.algo_type}_par.npy'), self.pars_per_iter)
        np.save(os.path.join(raw_data_dir, f'{self.cfg.algo_type}_transmit_amount.npy'), self.transmit_amount_per_iter)

    @staticmethod
    def compute_redeploy_cost(old, new):
        diff = old != new
        row_costs = np.sum(diff, axis=1)
        return np.sum(row_costs)

    def forward(self):
        n_iterations = len(self.expert_hotness)
        self.pars_per_iter = np.zeros((n_iterations, self.cfg.n_layers), dtype=float)
        self.transmit_amount_per_iter = np.zeros((n_iterations,), dtype=int)
        cur_deploy_table = list()
        for i in range(self.cfg.n_layers):
            cur_deploy_table.append(self.init_deploy_table())
        cur_deploy_table = np.array(cur_deploy_table)
        next_deploy_table = np.zeros_like(cur_deploy_table)
        redeploy_finish_iter = 0
        algo_finish_iter = 0
        expert_ready = True
        cur_layers_priority = list()
        for i in tqdm(range(1, n_iterations + 1)):
            cur_hotness = self.expert_hotness[i - 1]  # n_layers, n_experts
            self.pars_per_iter[i - 1] = self.cal_par_per_iter(cur_hotness, cur_deploy_table)

            if 'Default' in self.cfg.algo_type:
                pass
            else:
                if not expert_ready and i > algo_finish_iter and len(cur_layers_priority) > 0:
                    adjust_layer_idx = cur_layers_priority.pop(0)
                    self.transmit_amount_per_iter[i - 1] += self.compute_redeploy_cost(
                        cur_deploy_table[adjust_layer_idx], next_deploy_table[adjust_layer_idx])
                    cur_deploy_table[adjust_layer_idx] = next_deploy_table[adjust_layer_idx]

                # reset
                if len(cur_layers_priority) == 0 and not expert_ready:
                    expert_ready = True
                    redeploy_finish_iter = i

                # execute load balancing algorithm
                if i == redeploy_finish_iter + self.cfg.collection_interval + 1 and expert_ready:
                    st = time.time()
                    train_window = self.expert_hotness[i - self.cfg.collection_interval: i]
                    change, layers_priority, deployment_table, _ = self.method[self.cfg.algo_type](
                        train_window)

                    if change:
                        cur_layers_priority = layers_priority.tolist()
                        expert_ready = False
                        next_deploy_table[layers_priority] = deployment_table[layers_priority]
                    et = time.time()
                    algo_execute_iters = math.ceil((et - st) / 0.08)
                    algo_finish_iter = i + algo_execute_iters - 1


def single_process(cfg: Configure):
    algo = DynamicAlg(cfg)
    # algo.compute_hotness()
    algo.compute_hotness()
    algo.forward()
    if cfg.figure_flag:
        algo.store_raw_data()
    return [np.mean(algo.pars_per_iter), np.sum(algo.transmit_amount_per_iter)]


def multi_process(worker_args: list, process, n_processes: int):
    task_remain = len(worker_args)
    results = []
    start_idx = 0
    while task_remain > 0:
        task_num = min(n_processes, task_remain)
        with mp.Pool(task_num) as pool:
            cur_results = pool.starmap(process, worker_args[start_idx: start_idx + task_num])
        for result in cur_results:
            if result is None:
                print('ERROR: can not return the result')
                return None
        task_remain -= task_num
        start_idx += task_num
        results.extend(cur_results)
    return results


def run(cfg: Configure):
    worker_args = []
    cfgs = list()
    for algo_type in cfg.algo_set:
        new_cfg = cfg.copy()
        new_cfg.algo_type = algo_type
        if 'Default' in algo_type:
            new_cfg.n_red_experts = 0
            new_cfg.n_exp_per_dev = (new_cfg.n_experts + new_cfg.n_red_experts) // new_cfg.n_devices
            new_cfg.collection_interval = 0
        elif 'DS-EPLB' in algo_type:
            new_cfg.collection_interval = 1024

        cfgs.append(new_cfg)
        worker_args.append((new_cfg,))

    results = multi_process(worker_args, single_process, n_processes)
    par_per_method = dict()
    transmit_amount_per_method = dict()

    for i, algo_type in enumerate(cfg.algo_set):
        # print(f'{np.array(results[i].values())=}')
        par_per_method[algo_type] = results[i][0]
        transmit_amount_per_method[algo_type] = results[i][1]
    baseline_algo = 'DS-EPLB' if 'DS-EPLB' in par_per_method else cfg.algo_set[0]
    baseline_time = modeled_runtime_seconds(
        par_per_method[baseline_algo],
        transmit_amount_per_method[baseline_algo],
    )

    for algo_type in cfg.algo_set:
        mean_par = par_per_method[algo_type]
        transmit_amount = transmit_amount_per_method[algo_type]
        transmission_time = transmission_time_seconds(transmit_amount)
        total_time = modeled_runtime_seconds(mean_par, transmit_amount)
        score = 100.0 * baseline_time / total_time
        print(
            f'{cfg.dataset} {cfg.model} EP{cfg.n_devices} {algo_type} '
            f'PAR: {mean_par} '
            f'total_transit: {transmit_amount} '
            f'transmission_time_seconds: {transmission_time} '
            f'total_time_seconds: {total_time} '
            f'score: {score}')
    if cfg.figure_flag:
        case_name = f'{cfg.dataset}_{cfg.model}_EP{cfg.n_devices}'
        figure_dir = os.path.join(cfg.output_dir, 'figure')
        data_dir = os.path.join(cfg.output_dir, 'raw_data', case_name)
        plot_fig(data_dir, figure_dir)


def load_raw_data(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'Error: {data_dir} does not exist')
    par_data = dict()
    transmit_amount_data = dict()
    for file_name in os.listdir(data_dir):
        algo_type = file_name.split('.')[0].split('_')[0]
        file_path = os.path.join(data_dir, file_name)
        if file_name.endswith(f'par.npy'):
            par_data[algo_type] = np.load(file_path)
        if file_name.endswith(f'transmit_amount.npy'):
            transmit_amount_data[algo_type] = np.load(file_path)
    return par_data, transmit_amount_data


def _plot_single_fig(data_dict, title, xlabel, ylabel, save_path, case_name):
    fig = plt.figure(figsize=(12, 6))

    for i, method_name in enumerate(data_dict.keys()):
        data = data_dict[method_name]
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        n_iterations = len(data)

        plt.plot(
            np.arange(n_iterations),
            data,
            label=method_name,
            linewidth=1.5,
            alpha=0.7,
            markevery=10,
        )

    all_data = np.concatenate([v.mean(axis=1) if v.ndim == 2 else v for v in data_dict.values()])
    if len(all_data) == 0:
        plt.ylim([0, 1])
    else:
        data_min, data_max = all_data.min(), all_data.max()
        margin = (data_max - data_min) * 0.1 if (data_max != data_min) else 0.1
        plt.ylim([data_min - margin, data_max + margin])

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title}: {case_name}', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()

    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_fig(data_dir, figure_dir):
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    case_name = os.path.basename(data_dir)
    par_data, transmit_amount_data = load_raw_data(data_dir)

    par_save_path = os.path.join(figure_dir, f'PAR_{case_name}.png')
    _plot_single_fig(
        data_dict=par_data,
        title='Average PAR along iterations',
        xlabel='Iteration',
        ylabel='Average PAR',
        save_path=par_save_path,
        case_name=case_name
    )

    transmit_save_path = os.path.join(figure_dir, f'Transmit_Amount_{case_name}.png')
    _plot_single_fig(
        data_dict=transmit_amount_data,
        title='Transmit amount along iterations',
        xlabel='Iteration',
        ylabel='Transmit Amount',
        save_path=transmit_save_path,
        case_name=case_name
    )


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    params = list()
    for ds in ['Mix', 'ShareGPT', 'WildChat', 'LmSys']:
        for md in ['DS-R1', 'Qwen3']:
            for ep in [32, 64, 128, 256]:
                if not (md == 'Qwen3' and ep == 256) and not (ds == 'Mix' and md == 'Qwen3'):
                    dataset = ds
                    model = md
                    n_red_experts = ep
                    n_devices = ep
                    n_nodes = ep // 16
                    n_layers = 58 if md == 'DS-R1' else 94
                    n_experts = 256 if md == 'DS-R1' else 128
                    collection_interval = 128 #TODO: hyperparameter to further determined for proposed algorithm
                    exp_adjust_iters = 1
                    n_batch = 32
                    figure_flag = True
                    trace_dir = os.path.join(cur_dir, 'trace', md)
                    output_dir = os.path.join(cur_dir, 'output')
                    algo_set = ['Default', 'DS-EPLB']
                    #algo_set = ['Default', 'DS-EPLB', 'Proposed'] #TODO: add proposed algorithm
                    cpu_per_process = 1
                    n_processes = (mp.cpu_count() - 3) // cpu_per_process
                    params.append(Configure(dataset, model, n_red_experts, n_devices, n_nodes, n_layers, n_experts,
                                            collection_interval, exp_adjust_iters, n_batch, figure_flag, trace_dir,
                                            output_dir, algo_set, cpu_per_process, n_processes))

    for param in params:
        run(param)
