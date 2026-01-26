import torch
import numpy as np
import matplotlib.colors as mcolors
from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from node_manager import NodeManager
import matplotlib.pyplot as plt
from copy import deepcopy
import random

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.reach_target = [0] * N_TARGET
        self.store_flag = [0] * N_TARGET
        self.explict_flag = [0] * N_TARGET
        self.explore_over = [0] * N_TARGET
        # self.stop = True

        self.env = Env(global_step, plot=self.save_image)
        self.target_list = self.env.approximate_target_list
        self.target_cell_list = self.env.target_list
        self.Ground_Truth_Node_Manager = self.env.Ground_Truth_Node_Manager
        self.node_manager = self.env.node_manager
        self.robot_list = [
            Agent(i, self.global_step, self.target_list, self.target_cell_list, policy_net, self.node_manager,
                  self.device, self.save_image, test=False, Ground_Truth_Node_Manager=deepcopy(self.Ground_Truth_Node_Manager), 
                  prior_nodes_coords=deepcopy(self.env.priori_node_coords)) for i in range(N_AGENTS)]

        self.episode_buffer = []
        self.episode_buffer_length = [0] * N_AGENTS
        self.perf_metrics = dict()
        for i in range(37):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False

        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]), self.env.update_belief_info)
        for robot in self.robot_list:
            robot.update_planning_state(deepcopy(self.env.robot_locations), self.env.finding_target, self.reach_target, self.explore_over)
            robot.update_underlying_state(deepcopy(self.env.robot_locations), self.env.finding_target)

        if self.save_image:
            self.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            index_list = []
            next_node_index_list = []
            next_location_list = []
            new_index_list = []

            for robot in self.robot_list:
                if self.explore_over[robot.id]:  # finish the exploration, no need to move
                    selected_locations.append(self.env.target_position_list[robot.id])
                    dist_list.append(0)
                    next_location_list.append(robot.location)

                    is_termination = torch.tensor([robot.prev_option != 0], dtype=torch.bool).to(self.device)
                    robot.prev_option = 0
                    robot.current_stage = 1

                    continue
                elif not self.reach_target[robot.id] and robot.rapid_convergence_flag:
                    action_index = 0
                    observation = robot.get_observation()
                    state = robot.get_state()
                    robot.save_observation(observation)
                    robot.save_state(state)
                    next_location = robot.path.pop(0)
                    next_location = np.array(next_location).reshape(-1)
                    for index, value in enumerate(robot.neighbor_indices):
                        node = robot.node_coords[value]
                        if node[0] == next_location[0] and node[1] == next_location[1]:
                            action_index = index
                            break
                    action_index = torch.tensor([action_index], dtype=torch.long).to(self.device)
                    robot.save_action(action_index)

                    is_termination = torch.tensor([robot.prev_option != 0], dtype=torch.bool).to(self.device)
                    robot.save_is_termination(is_termination)
                    robot.prev_option = 0
                    robot.current_stage = 1
                    option = torch.tensor([robot.prev_option], dtype=torch.bool).to(self.device)
                    robot.save_option(option)
                    if next_location[0] == robot.target[0] and next_location[1] == robot.target[1]:
                        selected_locations.append(next_location)
                        next_location_list.append(next_location)
                        dist_list.append(0)
                    else:
                        selected_locations.append(next_location)
                        next_location_list.append(next_location)
                        dist_list.append(np.linalg.norm(next_location - robot.location))
                elif not self.reach_target[robot.id] and not robot.rapid_convergence_flag:
                    observation = robot.get_observation()
                    state = robot.get_state()
                    robot.save_observation(observation)
                    robot.save_state(state)

                    # astar_trick
                    if robot.astar_flag:
                        action_index = 0
                        next_location = robot.astar_path.pop(0)
                        next_location = np.array(next_location).reshape(-1)
                        for index, value in enumerate(robot.neighbor_indices):
                            node = robot.node_coords[value]
                            if node[0] == next_location[0] and node[1] == next_location[1]:
                                action_index = index
                                break
                        action_index = torch.tensor([action_index], dtype=torch.long).to(self.device)
                        robot.save_action(action_index)

                        is_termination = torch.tensor([robot.prev_option != 0], dtype=torch.bool).to(self.device)
                        robot.save_is_termination(is_termination)
                        robot.prev_option = 0
                        robot.current_stage = 0
                        option = torch.tensor([robot.prev_option], dtype=torch.bool).to(self.device)
                        robot.save_option(option)
                    else:
                        next_location, action_index, option = robot.select_next_waypoint(observation)
                        robot.save_action(action_index)

                        is_termination = torch.tensor([robot.prev_option != option], dtype=torch.bool).to(self.device)
                        robot.save_is_termination(is_termination)
                        robot.prev_option = option
                        robot.current_stage = 0
                        option = torch.tensor([robot.prev_option], dtype=torch.bool).to(self.device)
                        robot.save_option(option)

                    # astar_trick check conditions
                    if not robot.astar_flag and self.global_step >= 3000:
                        if robot.nodes_in_radius_range:
                            exist = tuple(next_location) in robot.nodes_in_radius_range
                            if exist:
                                robot.repetition += 1
                            else:
                                robot.repetition = 0
                                robot.nodes_in_radius_range = set()
                                if robot.astar_flag:
                                    robot.astar_flag = False
                        if robot.repetition >= 6: # robot.repetition should be 10
                            robot.astar_flag = True
                            # robot.repetition = 0
                    if robot.astar_flag and len(robot.astar_path) == 0:
                        robot.astar_path = self.out_of_dead_space(next_location, robot.target)

                    selected_locations.append(next_location)
                    next_location_list.append(next_location)
                    dist_list.append(np.linalg.norm(next_location - robot.location))
                elif self.reach_target[robot.id] and not self.explore_over[robot.id]:
                    observation = robot.get_observation()
                    state = robot.get_state()
                    robot.save_observation(observation)
                    robot.save_state(state)
                    
                    if robot.back_trick:
                        action_index = 0
                        next_location = robot.back_path.pop(0)
                        next_location = np.array(next_location).reshape(-1)
                        for index, value in enumerate(robot.neighbor_indices):
                            node = robot.node_coords[value]
                            if node[0] == next_location[0] and node[1] == next_location[1]:
                                action_index = index
                                break
                        action_index = torch.tensor([action_index], dtype=torch.long).to(self.device)
                        robot.save_action(action_index) 

                        is_termination = torch.tensor([robot.prev_option != 0], dtype=torch.bool).to(self.device)
                        robot.save_is_termination(is_termination)
                        robot.prev_option = 0
                        robot.current_stage = 1
                        option = torch.tensor([robot.prev_option], dtype=torch.bool).to(self.device)
                        robot.save_option(option)
                    else:                      
                        next_location, action_index, option = robot.select_next_waypoint(observation)
                        robot.save_action(action_index)

                        is_termination = torch.tensor([robot.prev_option != option], dtype=torch.bool).to(self.device)
                        robot.save_is_termination(is_termination)
                        robot.prev_option = option
                        robot.current_stage = 1
                        option = torch.tensor([robot.prev_option], dtype=torch.bool).to(self.device)
                        robot.save_option(option)
                        
                    if next_location[0] == robot.target[0] and next_location[1] == robot.target[1]:
                        selected_locations.append(next_location)
                        next_location_list.append(next_location)
                        dist_list.append(0)
                    else:
                        selected_locations.append(next_location)
                        next_location_list.append(next_location)
                        dist_list.append(np.linalg.norm(next_location - robot.location))
                    
                    if all(x == 1 for x in self.env.finding_target):
                        robot.back_trick = True
                    if robot.back_trick and len(robot.back_path) == 0:
                        robot.back_path = self.out_of_dead_space(next_location, robot.target)

            selected_locations, next_location_list = self.solve_path_conflict(selected_locations, dist_list, next_location_list)
            cur_node_indices = np.array([robot.gt_current_index for robot in self.robot_list])
            for id, location in enumerate(next_location_list):
                gt_nodes_to_check = self.robot_list[id].gt_node_coords[:, 0] + self.robot_list[id].gt_node_coords[:, 1] * 1j
                location_index = np.argwhere(gt_nodes_to_check == location[0] + location[1] * 1j)[0][0]
                next_node_index_list.append(location_index)

            reward_list = [0] * N_TARGET
            arrive_list = []
            for index, flag in enumerate(self.reach_target):
                if not flag:
                    index_list.append(index)
                else:
                    new_index_list.append(index)
            if new_index_list:
                if len(new_index_list) == 2:
                    random.shuffle(new_index_list)
                for new_index in new_index_list:
                    index_list.append(new_index)
            else:
                pass
            for index in index_list:
                next_location = selected_locations[index]
                robot = self.robot_list[index]
                if self.explore_over[robot.id]:
                    continue
                elif not self.reach_target[robot.id]:
                    _, astar_cur_dist2target = self.env.Ground_Truth_Node_Manager.a_star(robot.location, robot.target)
                    _, astar_next_dist2target = self.env.Ground_Truth_Node_Manager.a_star(next_location, robot.target)
                    robot.travel_dist += np.linalg.norm(robot.location - next_location)
                    dist_to_target = np.linalg.norm(next_location - robot.target)
                    individual_reward = self.env.step(robot.id, next_location, astar_cur_dist2target,
                                                      astar_next_dist2target, selected_locations)
                    collision = self.sensor_coverage(robot.id, next_location)

                    if not collision and not robot.rapid_convergence_flag:
                        robot.path = self.effective_path_finding(robot.id)
                        if len(robot.path) > 0:
                            robot.rapid_convergence_flag = True
                            subtask_reward = 20
                            if robot.astar_path:
                                robot.astar_path = []
                        else:
                            subtask_reward = 0
                    else:
                        subtask_reward = 0

                    if dist_to_target == 0:
                        self.reach_target[robot.id] = 1
                        arrive_list.append(index)

                    total_reward = individual_reward + subtask_reward
                    reward_list[index] = total_reward

                elif not self.explore_over[robot.id] and self.reach_target[robot.id]:
                    robot.travel_dist += np.linalg.norm(robot.location - next_location)
                    dist_to_target = np.linalg.norm(next_location - robot.target)
                    indiv_discover_reward = self.env.explore_step(robot.id, next_location, selected_locations)
                    useful_explore_count = self.env.check_useful_explore(robot.id, self.reach_target)
                    if not robot.explore_flag and useful_explore_count:
                        robot.explore_flag = True
                    if dist_to_target == 0:
                        self.explore_over[robot.id] = 1
                        if robot.explore_flag:
                            subtask_explore_reward = 20
                        else:
                            subtask_explore_reward = 0
                    else:
                        subtask_explore_reward = 0

                    total_reward = subtask_explore_reward - 0.4 + indiv_discover_reward
                    reward_list[index] = total_reward

            count = self.reach_target.count(1)
            if count == 3 and len(arrive_list):
                for index in arrive_list:
                    self.explore_over[index] = 1
                    self.robot_list[index].last_stop = True
            else:
                pass

            self.env.update_robot_belief(deepcopy(self.env.robot_cells))
            self.env.delete_wrong_belief()
            self.env.update_old_belief()
            self.env.evaluate_exploration_rate()

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]), self.env.update_belief_info)

            for robot in self.robot_list:
                robot.update_planning_state(deepcopy(self.env.robot_locations), self.env.finding_target, self.reach_target, self.explore_over)
                robot.update_underlying_state(deepcopy(self.env.robot_locations), self.env.finding_target)

            team_explore_reward = self.env.calculate_team_explore_reward(self.reach_target)
            for flag_index, flag in enumerate(self.store_flag):
                if flag == 1:
                    continue
                else:
                    reward_list[flag_index] += team_explore_reward

            if all(x == 1 for x in self.explore_over):
                done = True
                final_reward = 25
            else:
                final_reward = 0

            for robot, reward in zip(self.robot_list, reward_list):
                if self.explore_over[robot.id] and not self.store_flag[robot.id]:
                    robot.save_all_indices(np.array(cur_node_indices), np.array(next_node_index_list))
                    robot.save_reward_done(reward + final_reward, done)
                    observation = robot.get_observation()
                    state = robot.get_state()
                    robot.save_next_observations(observation)
                    robot.save_next_state(state)
                    self.episode_buffer_length[robot.id] = len(robot.episode_buffer[0])
                    self.store_flag[robot.id] = 1
                    if self.save_image or self.global_step >= 7000:
                        robot.trajectory_x.append(self.env.target_position_list[robot.id][0])
                        robot.trajectory_y.append(self.env.target_position_list[robot.id][1])
                elif self.explore_over[robot.id] and self.store_flag[robot.id]:
                    continue
                elif not self.explore_over[robot.id]:
                    robot.save_all_indices(np.array(cur_node_indices), np.array(next_node_index_list))
                    robot.save_reward_done(reward + final_reward, done)
                    observation = robot.get_observation()
                    state = robot.get_state()
                    robot.save_next_observations(observation)
                    robot.save_next_state(state)
                    
                if i == MAX_EPISODE_STEP - 1 and self.episode_buffer_length[robot.id] == 0:
                    self.episode_buffer_length[robot.id] = len(robot.episode_buffer[0])

            if self.save_image:
                self.belief_refine()
                self.plot_env(i + 1)

            if done or i == MAX_EPISODE_STEP - 1:
                self.save_action_index_pairs(next_location_list, i)
            if done:
                break
            else:
                self.env.reset_Dijkstra_path()
                self.env.update_Dijkstra_path()

        self.perf_metrics['max_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['min_dist'] = min([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['mean_dist'] = sum(robot.travel_dist for robot in self.robot_list) / len(self.robot_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        for robot in self.robot_list:
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

        if self.global_step >= 7000 and not done:
            self.plot_fail_trajectory()
            
    def save_action_index_pairs(self, next_location_list, current_step):
        terminate_step = current_step + 1
        robot_index = self.episode_buffer_length.index(terminate_step)
        for robot in self.robot_list:
            robot.episode_buffer[25] = deepcopy(robot.episode_buffer[14])[1:]
            robot.episode_buffer[26] = deepcopy(robot.episode_buffer[15])[1:]
        for robot in self.robot_list:
            if self.episode_buffer_length[robot.id] == terminate_step:
                gt_node_coords_to_check = robot.gt_node_coords[:, 0] + robot.gt_node_coords[:, 1] * 1j
                next_next_index_list = []
                for location in next_location_list:
                    index = np.argwhere(gt_node_coords_to_check == location[0] + location[1] * 1j)
                    index = index[0][0]
                    next_next_index_list.append(index)
                robot.episode_buffer[25] += torch.tensor(np.array(next_next_index_list)).reshape(1, -1, 1).to(self.device)
                robot.episode_buffer[26] += torch.tensor(np.array(next_next_index_list)).reshape(1, -1, 1).to(self.device)
            elif self.episode_buffer_length[robot.id] < terminate_step:
                new_index = self.episode_buffer_length[robot.id]
                current_index = new_index - 1
                robot.episode_buffer[25].append(deepcopy(self.robot_list[robot_index].episode_buffer[25][current_index]))
                robot.episode_buffer[26].append(deepcopy(self.robot_list[robot_index].episode_buffer[26][current_index]))

    def update_target_state(self):
        for target_index, target_cell in enumerate(self.target_cell_list):
            if self.env.update_belief[target_cell[1], target_cell[0]] == 255:
                self.explict_flag[target_index] = 1
            else:
                continue

    def belief_refine(self):
        enclose_coords, enclose_mask = find_enclosed_region(self.env.update_belief)
        if enclose_coords.size > 0:
            self.env.robot_belief[enclose_mask] = 1

    def out_of_dead_space(self, location, target):
        path, astar_dist = self.Ground_Truth_Node_Manager.a_star(location, target)
        if astar_dist == 1e8:
            dist_dict, prev_dict = self.Ground_Truth_Node_Manager.Dijkstra(self.env.robot_locations[id])
            path, _ = self.Ground_Truth_Node_Manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict, target)
        return path

    def effective_path_finding(self, id):
        robot = self.robot_list[id]
        path, astar_dist = self.Ground_Truth_Node_Manager.a_star(self.env.robot_locations[id], robot.target)
        if astar_dist == 1e8:
            dist_dict, prev_dict = self.Ground_Truth_Node_Manager.Dijkstra(self.env.robot_locations[id])
            path, _ = self.Ground_Truth_Node_Manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict, robot.target)
        return path

    def solve_path_conflict(self, selected_locations, dist_list, next_location_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list)) 
        selected_locations_in_arriving_sequence = np.array(selected_locations)[
            arriving_sequence]  

        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                id = arriving_sequence[j]
                current_location = self.robot_list[id].location
                nearby_nodes = self.robot_list[id].node_manager.priori_nodes_dict.nearest_neighbors(selected_location.tolist(), 25)  
                current_node = self.robot_list[id].node_manager.priori_nodes_dict.find((current_location[0], current_location[1]))
                neighbor_nodes = np.array(list(current_node.data.neighbor_set)).reshape(-1, 2)
                for node in nearby_nodes:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                        continue
                    else:
                        if coords[0] + coords[1] * 1j in neighbor_nodes[:, 0] + neighbor_nodes[:, 1] * 1j:
                            selected_location = coords
                            break
                selected_locations_in_arriving_sequence[j] = selected_location
                selected_locations[id] = selected_location
                next_location_list[id] = selected_location

        return selected_locations, next_location_list

    def sensor_coverage(self, robot_id, next_location):
        collision_flag = None
        target = self.target_list[robot_id]
        dis = np.linalg.norm(next_location - target)
        if dis < SENSOR_RANGE:
            collision = check_collision(next_location, target, self.env.update_belief_info)
            if not collision:
                reverse_collision = check_collision(target, next_location, self.env.update_belief_info)
                if not reverse_collision:
                    collision_flag = False
                else:
                    collision_flag = True
            else:
                collision_flag = True
        else:
            collision_flag = True
        return collision_flag
        
    def plot_env(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(10, 5))

        priori_belief = np.zeros_like(self.env.belief_mask)
        priori_belief[self.env.belief_mask == 127] = 255
        color_list = ['r', 'b', 'g']

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 1:
                alpha_mask = priori_belief / 255 / 3
                plt.imshow(priori_belief, cmap='Greens', alpha=alpha_mask)
                plt.axis('off')

            if self.explore_over[robot.id]:
                robot_cell = self.env.target_list[robot.id]
            else:
                robot_cell = self.env.robot_cells[robot.id]
            target_cell = self.env.target_list[robot.id]
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=8, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - self.env.belief_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - self.env.belief_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)
            plt.plot(target_cell[0], target_cell[1], c + 'o', markersize=8)

        plt.subplot(1, 2, 2)
        plt.imshow(self.env.ground_truth, cmap='gray')
        plt.axis('off')
        for robot in self.robot_list:
            c = color_list[robot.id]
            if self.explore_over[robot.id]:
                robot_cell = self.env.target_list[robot.id]
            else:
                robot_cell = self.env.robot_cells[robot.id]
            target_cell = self.env.target_list[robot.id]
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=8, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - self.env.belief_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - self.env.belief_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)
            plt.plot(target_cell[0], target_cell[1], c + 'o', markersize=8)

        plt.suptitle('Robot_0: {:.4g} | Robot_1: {:.4g} | robot_2: {:.4g}'.format 
                     (self.robot_list[0].travel_dist, self.robot_list[1].travel_dist, self.robot_list[2].travel_dist))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        plt.close()
        self.env.frame_files.append(frame)

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    checkpoint = torch.load(model_path + '/checkpoint_50000.pth', map_location='cpu')
    model.load_state_dict(checkpoint['policy_model'])
    worker = Multi_agent_worker(0, model, 0, save_image=True)
    worker.run_episode() 

