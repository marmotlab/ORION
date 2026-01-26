import ray
import numpy as np
import torch
import csv

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *


def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    checkpoint = torch.load(f'{model_path}/checkpoint_50000.pth', map_location=device)

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    max_dist_history = []
    min_dist_history = []
    mean_dist_history = []
    explored_rate_history = []
    success_rate_history = []
    fail_history = []
    total_step_history = []
    per_map_records = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(success_rate_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job

                per_map_records.append({
                    "map_path": metrics.get("map_path"),                           
                    "episode": metrics.get("global_step"),
                    "max_dist": metrics.get("max_dist"),
                    "min_dist": metrics.get("min_dist"),
                    "mean_dist": metrics.get("mean_dist"),
                    "explored_rate": metrics.get("explored_rate"),
                    "total_step": metrics.get("total_step"),        
                    "success": metrics.get("success_rate"),
                })

                if metrics['success_rate']:
                    max_dist_history.append(metrics['max_dist'])
                    min_dist_history.append(metrics['min_dist'])
                    mean_dist_history.append(metrics['mean_dist'])
                    total_step_history.append(metrics['total_step'])
                else:
                    fail_history.append(metrics['global_step'])
                explored_rate_history.append(metrics['explored_rate'])
                success_rate_history.append(metrics['success_rate'])

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        print('=====================================')
        print('| Test:', FOLDER_NAME)
        print('| Total test:', NUM_TEST)
        print('| Number of agents:', TEST_N_AGENTS)
        print('| Average max length:', np.array(max_dist_history).mean())
        print('| Std max length:', np.array(max_dist_history).std())
        print('| Average min length:', np.array(min_dist_history).mean())
        print('| Std min length:', np.array(min_dist_history).std())
        print('| Average mean distance:', np.array(mean_dist_history).mean())
        print('| Std mean distance:', np.array(mean_dist_history).std())
        print('| Average explored rate:', np.array(explored_rate_history).mean())
        print('| Average success rate:', np.array(success_rate_history).mean())
        print('| Fail episode number:', np.array(fail_history))
        print('| Average total step:', np.array(total_step_history).mean())

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, device=self.device,
                            save_image=SAVE_GIFS, greedy=True, test=True)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()
