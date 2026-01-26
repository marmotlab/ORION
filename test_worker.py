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
from test_parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)
    
if not os.path.exists(fail_tests_path):
    os.makedirs(fail_tests_path)

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, greedy=True, test=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.reach_target = [0] * TEST_N_AGENTS
        self.explore_over = [0] * TEST_N_AGENTS
        self.greedy = greedy
        self.test = test
        self.total_step = 0

        self.env = Env(global_step, n_agent=TEST_N_AGENTS, plot=self.save_image, test=test)
        self.map_path = self.env.gt_map_path
        self.target_list = self.env.approximate_target_list
        self.target_cell_list = self.env.target_list
        self.Ground_Truth_Node_Manager = self.env.Ground_Truth_Node_Manager
        self.node_manager = self.env.node_manager
        self.robot_list = [
            Agent(i, self.global_step, self.target_list, self.target_cell_list, policy_net, self.node_manager,
                  self.device, self.save_image, self.test) for i in range(N_AGENTS)]

        self.perf_metrics = dict()

    def run_episode(self):
        done = False

        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]), self.env.update_belief_info)
        for robot in self.robot_list:
            robot.update_planning_state(deepcopy(self.env.robot_locations), self.env.finding_target, self.reach_target, self.explore_over)

        if self.save_image:
            self.plot_env_test(0)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            index_list = []
            new_index_list = []
            target_found_signal = [self.env.finding_target[i] for i in robot.collaboration_list]

            for robot in self.robot_list:
                if self.explore_over[robot.id]:  # finish the exploration, no need to move
                    selected_locations.append(self.env.target_position_list[robot.id])
                    dist_list.append(0)
                    robot.prev_option = 0
                    continue
                elif not self.reach_target[robot.id] and robot.rapid_convergence_flag:
                    next_location = robot.path.pop(0)
                    next_location = np.array(next_location).reshape(-1)
                    robot.prev_option = 0
                    if next_location[0] == robot.target[0] and next_location[1] == robot.target[1]:
                        selected_locations.append(next_location)
                        dist_list.append(0)
                    else:
                        selected_locations.append(next_location)
                        dist_list.append(np.linalg.norm(next_location - robot.location))
                elif not self.reach_target[robot.id] and not robot.rapid_convergence_flag:
                    observation = robot.get_observation(pad=False)
                    next_location, _, option = robot.select_next_waypoint(observation,greedy=self.greedy)
                    robot.prev_option = option
                    selected_locations.append(next_location)
                    dist_list.append(np.linalg.norm(next_location - robot.location))
                elif self.reach_target[robot.id] and not self.explore_over[robot.id]:
                    observation = robot.get_observation(pad=False)

                    if robot.back_trick:
                        next_location = robot.back_path.pop(0)
                        next_location = np.array(next_location).reshape(-1)
                        robot.prev_option = 0
                    else:
                        next_location, _, option = robot.select_next_waypoint(observation, greedy=True)
                        robot.prev_option = option

                    if next_location[0] == robot.target[0] and next_location[1] == robot.target[1]:
                        selected_locations.append(next_location)
                        dist_list.append(0)
                    else:
                        selected_locations.append(next_location)
                        dist_list.append(np.linalg.norm(next_location - robot.location))

                    if all(x == 1 for x in self.env.finding_target):
                        robot.back_trick = True
                    if robot.back_trick and len(robot.back_path) == 0:
                        robot.back_path = self.out_of_dead_space(next_location, robot.target)

            selected_locations = self.solve_path_conflict(selected_locations, dist_list)

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
                    robot.travel_dist += np.linalg.norm(robot.location - next_location)
                    dist_to_target = np.linalg.norm(next_location - robot.target)
                    self.env.test_step(robot.id, next_location)

                    collision = self.sensor_coverage(robot.id, next_location)

                    if not collision and not robot.rapid_convergence_flag:
                        robot.path = self.effective_path_finding(robot.id)
                        if len(robot.path) > 0:
                            robot.rapid_convergence_flag = True

                    if dist_to_target == 0:
                        self.reach_target[robot.id] = 1
                        arrive_list.append(index)
                        # count = self.reach_target.count(1)
                        # if count == 3:
                        #     self.explore_over[robot.id] = 1
                elif not self.explore_over[robot.id] and self.reach_target[robot.id]:
                    robot.travel_dist += np.linalg.norm(robot.location - next_location)
                    dist_to_target = np.linalg.norm(next_location - robot.target)
                    self.env.test_step(robot.id, next_location)
                    if dist_to_target == 0:
                        self.explore_over[robot.id] = 1

            if all(x == 1 for x in target_found_signal) and len(arrive_list):
                for index in arrive_list:
                    self.explore_over[index] = 1
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

            if all(x == 1 for x in self.explore_over):
                done = True
                self.total_step = i

            if self.save_image:
                self.belief_refine()
                self.plot_env_test(i + 1)

            if done:
                break

        self.perf_metrics['max_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['min_dist'] = min([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['mean_dist'] = sum(robot.travel_dist for robot in self.robot_list) / len(self.robot_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['global_step'] = self.global_step
        self.perf_metrics['total_step'] = self.total_step
        self.perf_metrics['map_path'] = self.map_path

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)
            
        if not done and self.test:
            self.plot_fail_trajectory()

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

    def solve_path_conflict(self, selected_locations, dist_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))  
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]  

        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:,
                                                                                               1] * 1j:
                id = arriving_sequence[j]
                current_location = self.robot_list[id].location
                nearby_nodes = self.robot_list[id].node_manager.priori_nodes_dict.nearest_neighbors(
                    selected_location.tolist(), 25)  
                current_node = self.robot_list[id].node_manager.priori_nodes_dict.find((current_location[0],
                                                                                        current_location[1]))
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

        return selected_locations

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

    def plot_env_test(self, step):
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
                nodes = get_cell_position_from_coords(robot.node_coords, robot.global_map_info)
                # plt.imshow(robot.global_map_info.map, cmap='gray')
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

if __name__ == '__main__':
    from model import PolicyNet
    net = PolicyNet(9, 128)
    checkpoint = torch.load(f'{model_path}/checkpoint_50000.pth', map_location='cpu')
    net.load_state_dict(checkpoint['policy_model'])
    test_worker = TestWorker(0, net, 36, save_image=True, greedy=True, test=True)
    test_worker.run_episode()
