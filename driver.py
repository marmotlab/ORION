import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to Multi-agent navigation!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)
if not os.path.exists(fail_examples_path):
    os.makedirs(fail_examples_path)
map_data_folder = "maps_GT_data"
if not os.path.exists(f'{map_data_folder}'):
            os.makedirs(f'{map_data_folder}')


def main():
    
    seed = 42
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha'] 
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs are available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['max_dist','min_dist', 'mean_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(37):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(4):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        assert len(experience_buffer[i]) == len(experience_buffer[0]), \
                            f"Experience buffer {i} has length {len(experience_buffer[i])}, expected {len(experience_buffer[0])}"
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    navi_edge_mask = torch.stack(rollouts[6]).to(device)
                    navi_node_inputs = torch.stack(rollouts[7]).to(device)
                    prev_option_inputs = torch.stack(rollouts[8]).to(device)
                    prev_option_inputs = prev_option_inputs.to(torch.int64)

                    action = torch.stack(rollouts[9]).to(device)
                    option = torch.stack(rollouts[10]).to(device)
                    is_termination = torch.stack(rollouts[11]).to(device)
                    reward = torch.stack(rollouts[12]).to(device)
                    done = torch.stack(rollouts[13]).to(device)

                    all_agent_indices = torch.stack(rollouts[14]).to(device)
                    all_agent_next_indices = torch.stack(rollouts[15]).to(device)

                    next_node_inputs = torch.stack(rollouts[16]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[17]).to(device)
                    next_edge_mask = torch.stack(rollouts[18]).to(device)
                    next_current_index = torch.stack(rollouts[19]).to(device)
                    next_current_edge = torch.stack(rollouts[20]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[21]).to(device)
                    next_navi_edge_mask = torch.stack(rollouts[22]).to(device)
                    next_navi_node_inputs = torch.stack(rollouts[23]).to(device)
                    next_pre_option_inputs = torch.stack(rollouts[24]).to(device)
                    next_pre_option_inputs = next_pre_option_inputs.to(torch.int64)

                    next_all_agent_indices = torch.stack(rollouts[25]).to(device)
                    next_all_agent_next_indices = torch.stack(rollouts[26]).to(device)
                    
                    state_node_inputs = torch.stack(rollouts[27]).to(device)
                    state_node_padding_mask = torch.stack(rollouts[28]).to(device)
                    state_edge_mask = torch.stack(rollouts[29]).to(device)
                    state_current_index = torch.stack(rollouts[30]).to(device)
                    state_current_edge = torch.stack(rollouts[31]).to(device)
                    next_state_node_inputs = torch.stack(rollouts[32]).to(device)
                    next_state_node_padding_mask = torch.stack(rollouts[33]).to(device)
                    next_state_edge_mask = torch.stack(rollouts[34]).to(device)
                    next_state_current_index = torch.stack(rollouts[35]).to(device)
                    next_state_current_edge = torch.stack(rollouts[36]).to(device)

                    observation = [node_inputs, node_padding_mask, edge_mask, current_index,
                                   current_edge, edge_padding_mask, navi_edge_mask, navi_node_inputs,
                                   prev_option_inputs]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask,
                                        next_current_index, next_current_edge, next_edge_padding_mask,
                                        next_navi_edge_mask, next_navi_node_inputs, 
                                        next_pre_option_inputs]
                    q_state = [state_node_inputs, state_node_padding_mask, state_edge_mask, state_current_index,
                               state_current_edge, all_agent_indices, all_agent_next_indices,
                               prev_option_inputs]
                    q_state_termination = [state_node_inputs, state_node_padding_mask, state_edge_mask,
                                           state_current_index, state_current_edge, all_agent_indices,
                                           all_agent_next_indices, 1-prev_option_inputs]
                    next_q_state = [next_state_node_inputs, next_state_node_padding_mask, next_state_edge_mask,
                                    next_state_current_index, next_state_current_edge, 
                                    next_all_agent_indices, next_all_agent_next_indices,
                                    next_pre_option_inputs]

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*q_state)
                        q_values2 = dp_q_net2(*q_state)
                        q_values = torch.min(q_values1, q_values2)
                        q_values_termination1 = dp_q_net1(*q_state_termination)
                        q_values_termination2 = dp_q_net2(*q_state_termination)
                        q_values_termination = torch.min(q_values_termination1, q_values_termination2)
                    logp, termination_logits, _ = dp_policy(*observation, check=True)
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()) ),
                        dim=1).mean()
                    # print(f"policy_loss shape: {policy_loss.shape}")
                    # termination head loss
                    q_values_max = q_values.max(dim=1, keepdim=True)[0]
                    q_values_termination_max = q_values_termination.max(dim=1, keepdim=True)[0]
                    advantage = (q_values_termination_max - q_values_max).detach()
                    # print(f"advantage shape: {advantage.shape}, termination_logits shape: {termination_logits.shape}")
                    log_prob_termination = F.logsigmoid(termination_logits)
                    log_prob_continue = F.logsigmoid(-termination_logits)
                    is_termination = is_termination.int().squeeze(-1)
                    # print(f"is_termination shape: {is_termination.shape}, log_prob_termination shape: {log_prob_termination.shape}, log_prob_continue shape: {log_prob_continue.shape}")
                    log_termination = log_prob_termination * is_termination + log_prob_continue * (1 - is_termination)
                    termination_head_loss = -(log_termination * advantage).mean()
                    # print(f"termination_head_loss shape: {termination_head_loss.shape}")
                    overall_policy_loss = policy_loss + 0.05 * termination_head_loss
                    global_policy_optimizer.zero_grad()
                    overall_policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(),max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp, _, _ = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*next_q_state)
                        next_q_values2 = dp_target_q_net2(*next_q_state)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*q_state)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*q_state)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1
                    # print("target q update counter", target_q_update_counter % 1024)

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), overall_policy_loss.item(), policy_loss.item(), termination_head_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 128:  # every 32 episode update
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 64 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                
            # every 5000 episodes save a checkpoint
            if curr_episode >= 15000 and curr_episode % 5000 == 0:
                new_path = f"./{model_path}/checkpoint_{curr_episode}.pth"
                torch.save(checkpoint, new_path)
                print(f"Also saved checkpoint at episode {curr_episode} to {new_path}")

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, overall_policy_loss, policy_loss, termination_head_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, \
        alpha_loss, max_dist, min_dist, mean_dist, success_rate, explored_rate = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Overall Policy Loss', scalar_value=overall_policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Termination Head Loss', scalar_value=termination_head_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Max Distance', scalar_value=max_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Min Dist', scalar_value=min_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Mean Dist', scalar_value=mean_dist, global_step=curr_episode)


if __name__ == "__main__":
    main()
