import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
from node_manager import NodeManager
from node_manager_GroundTruth import NodeManager_GroundTruth
from node_manager_GT_for_reward import NodeManager_GroundTruth4reward
from sensor import update_map_sensor, make_belief_mask
from utils import *
import random
from parameter import *
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import pickle


class Env:
    def __init__(self, episode_index, n_agent=N_AGENTS, plot=False, test=False):
        self.episode_index = episode_index
        self.n_agent = n_agent
        self.plot = plot
        self.test = test
        self.prior_map_path = None
        self.gt_map_path = None
        self.priori_truth, start_cell, self.target_list, self.potential_start_cell = self.import_priori_map(
            episode_index)
        self.ground_truth = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.area_discovered = 0

        self.cell_size = CELL_SIZE  # meter
        self.sensor_range = SENSOR_RANGE  # meter
        # self.travel_dist = 0  # meter
        self.explored_rate = 0
        self.area_discovered_rate = 0
        self.target_position_list = []
        self.approximate_target_list = []
        # approximate target cell, will be deleted in the final version
        self.approximate_target_cell_list = []
        self.increase_mask = None
        self.flag = True
        self.finding_target = [0] * N_TARGET

        initial_coords = np.array([0.0, 0.0])  # meter
        self.belief_origin_x = -np.round(start_cell[0] * self.cell_size, 1)  # meter
        self.belief_origin_y = -np.round(start_cell[1] * self.cell_size, 1)  # meter

        self.robot_belief = copy.deepcopy(self.priori_truth)
        self.update_belief = np.ones(self.ground_truth_size) * 127
        self.belief_mask = np.ones(self.ground_truth_size) * 127
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.update_belief_info = MapInfo(self.update_belief, self.belief_origin_x, self.belief_origin_y,
                                          self.cell_size)

        self.priori_truth_info = MapInfo(self.priori_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.node_manager = NodeManager([], plot=False)
        self.node_manager.update_priori_truth_graph(initial_coords, self.priori_truth_info)
        self.node_manager.update_all_current_graph(initial_coords, self.update_belief_info)
        
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.store_load_nodemanager(self.gt_map_path, initial_coords)

        self.ground_truth_node_coords, self.priori_node_coords = self.get_uniform_nodes()

        # get the start coords
        self.robot_locations = self.set_initial_location()
        # self.robot_locations = np.array([[0., 0.], [-4., 0.]])
        self.robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info).reshape(-1, 2)
        for robot_cell in self.robot_cells:
            self.update_update_belief(robot_cell)
        self.update_robot_belief(self.robot_cells)

        # get the target coords
        for i in range(N_TARGET):
            target_cell = self.target_list[i]
            target_position = get_coords_from_cell_position(target_cell, self.priori_truth_info)
            self.target_position_list.append(target_position)
            approximate_target, approximate_cell = self.nearest_target_position(target_position)
            self.approximate_target_list.append(approximate_target)
            self.approximate_target_cell_list.append(approximate_cell)

        self.node_manager.update_target_list(np.array(self.approximate_target_list))

        self.delete_wrong_belief()
        self.old_belief_mask = deepcopy(self.belief_mask)
        self.old_update_belief = deepcopy(self.update_belief)

        self.Dijkstra_path_list_1 = []
        self.Dijkstra_path_list_2 = []
        self.curr_coverage_list = []
        self.update_Dijkstra_path()

        if self.plot:
            self.frame_files = []
            
    def store_load_nodemanager(self, map_path, initial_coords):
        map_name = map_path.split('/')[-1].split('.')[0]
        folder_name = map_path.split('/')[-2] + '_data'
        data_path = f'{folder_name}/{map_name}'
        
        try:
            with open(f'{data_path}.pkl', 'rb') as f:
                cache = pickle.load(f)
                self.Ground_Truth_Node_Manager = cache['Ground_Truth_Node_Manager']
                self.GT_Node_Manager_reward = cache['GT_Node_Manager_reward']
        except FileNotFoundError:
            # print("File not found, creating new ground truth node managers")
            self.Ground_Truth_Node_Manager = NodeManager_GroundTruth(plot=False)
            self.Ground_Truth_Node_Manager.update_ground_truth_graph(initial_coords, self.ground_truth_info)
            self.GT_Node_Manager_reward = NodeManager_GroundTruth4reward(plot=False)
            self.GT_Node_Manager_reward.update_GT_graph_reward(initial_coords, self.ground_truth_info)
            os.makedirs(os.path.dirname(f'{data_path}.pkl'), exist_ok=True)
            cache = {
            'Ground_Truth_Node_Manager': self.Ground_Truth_Node_Manager,
            'GT_Node_Manager_reward': self.GT_Node_Manager_reward,
                }
            with open(f'{data_path}.pkl', 'wb') as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            # print("Error loading cache:", e)
            self.Ground_Truth_Node_Manager = NodeManager_GroundTruth(plot=False)
            self.Ground_Truth_Node_Manager.update_ground_truth_graph(initial_coords, self.ground_truth_info)
            self.GT_Node_Manager_reward = NodeManager_GroundTruth4reward(plot=False)
            self.GT_Node_Manager_reward.update_GT_graph_reward(initial_coords, self.ground_truth_info)
            cache = {
            'Ground_Truth_Node_Manager': self.Ground_Truth_Node_Manager,
            'GT_Node_Manager_reward': self.GT_Node_Manager_reward,
                }
            with open(f'{data_path}.pkl', 'wb') as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
                
    def import_priori_map(self, episode_index):
        if not self.test:
            map_dir = f'maps_priori'
        else:
            map_dir = f'maps_priori_test_new_3'
        map_list = os.listdir(map_dir)
        map_list.sort()
        map_index = episode_index % np.size(map_list)
        self.prior_map_path = map_dir + '/' + map_list[map_index]
        priori_info = (io.imread(self.prior_map_path, 1)).astype(int)

        priori_info = block_reduce(priori_info, 2, np.min)

        potential_start_cell = []
        for i in range(N_AGENTS):
            if i == 0:
                robot_cell = np.array((np.nonzero(priori_info == 208 + i)))
                robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])
            else:
                potential_cell = np.nonzero(priori_info == 208 + i)
                potential_cell = np.array(list(zip(potential_cell[1], potential_cell[0])))
                potential_start_cell.append(potential_cell)

        final_target = []
        for i in range(N_TARGET):
            target_cell = np.array((np.nonzero(priori_info == 50 + i)))
            target_cell = np.array([target_cell[1, 10], target_cell[0, 10]])
            final_target.append(target_cell)

        priori_info = (priori_info > 150) | ((priori_info <= 80) & (priori_info >= 50))
        priori_info = priori_info * 254 + 1

        return priori_info, robot_cell, final_target, potential_start_cell

    def import_ground_truth(self, episode_index):
        if self.test:
            map_dir = f'maps_GT_test_new_3'
        else:
            map_dir = f'maps_GT'
        map_list = os.listdir(map_dir)
        map_list.sort()
        map_index = episode_index % np.size(map_list)
        self.gt_map_path = map_dir + '/' + map_list[map_index]

        ground_truth = (io.imread(self.gt_map_path, 1)).astype(int)
        ground_truth = block_reduce(ground_truth, 2, np.min)

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth

    def set_initial_location(self):
        robot_locations = [np.array([0, 0])]
        for i in range(self.n_agent - 1):
            potential_start_coords = get_coords_from_cell_position(self.potential_start_cell[i], self.priori_truth_info)
            for coord in potential_start_coords:
                if np.any(np.all(self.priori_node_coords == coord, axis=1)) and \
                        np.any(np.all(self.ground_truth_node_coords == coord, axis=1)):
                    robot_locations.append(coord)
                    break
        robot_locations = np.array(robot_locations)
        return robot_locations

    def get_uniform_nodes(self):
        all_ground_truth_node_coords = []
        all_priori_node_coords = []
        for node in self.Ground_Truth_Node_Manager.nodes_dict.__iter__():
            all_ground_truth_node_coords.append(node.data.coords)
        for node in self.node_manager.priori_nodes_dict.__iter__():
            all_priori_node_coords.append(node.data.coords)
        all_ground_truth_node_coords = np.array(all_ground_truth_node_coords).reshape(-1, 2)
        all_priori_node_coords = np.array(all_priori_node_coords).reshape(-1, 2)
        return all_ground_truth_node_coords, all_priori_node_coords

    def get_current_nodes(self):
        all_current_node_coords = []
        for node in self.node_manager.current_nodes_dict.__iter__():
            if node.data.status == FREE:
                all_current_node_coords.append(node.data.coords)
        all_current_node_coords = np.array(all_current_node_coords).reshape(-1, 2)
        return all_current_node_coords

    def nearest_target_position(self, target_position):
        nearest_node = None
        distances = np.linalg.norm(self.priori_node_coords - target_position, axis=1)
        sorted_indices = np.argsort(distances)
        for index in sorted_indices:
            nearest_node = self.priori_node_coords[index]
            if np.any(np.all(self.ground_truth_node_coords == nearest_node, axis=1)):
                break
        nearest_cell = get_cell_position_from_coords(nearest_node, self.priori_truth_info)
        return nearest_node, nearest_cell

    def calculate_indiv_navi_reward(self, astar_cur_dist2target, astar_next_dist2target):
        reward = 0
        
        reward -= 0.4

        reward += (astar_cur_dist2target - astar_next_dist2target) / 32

        return reward

    def delete_wrong_belief_1(self):
        self.increase_mask = deepcopy(self.belief_mask)
        if self.flag:
            self.flag = False
        else:
            self.increase_mask[self.increase_mask == self.old_belief_mask] = 127
        self.node_manager.delete_mask_node(self.increase_mask, self.belief_info)
        
    def delete_wrong_belief(self):
        self.increase_mask = deepcopy(self.update_belief)
        if self.flag:
            self.flag = False
        else:
            self.increase_mask[self.increase_mask == self.old_update_belief] = 127
        self.node_manager.delete_mask_node(self.increase_mask, self.belief_info)
        self.node_manager.delete_current_node(self.increase_mask, self.belief_info)

    def update_update_belief(self, robot_cell):
        self.update_belief = update_map_sensor(robot_cell, round(self.sensor_range / self.cell_size),
                                               self.update_belief, self.ground_truth)

    def update_robot_belief(self, robot_cell):
        if robot_cell.ndim == 2:
            for cell in robot_cell:
                self.belief_mask = make_belief_mask(cell, round(self.sensor_range / self.cell_size), self.belief_mask,
                                                    self.update_belief)
            mask = self.update_belief != 127
            self.robot_belief[mask] = self.update_belief[mask]
            mask_indice = np.argwhere(self.update_belief != 127)
            self.update_false_information(mask_indice)
        else:
            self.belief_mask = make_belief_mask(robot_cell, round(self.sensor_range / self.cell_size), self.belief_mask,
                                                self.update_belief)
            mask = self.update_belief != 127
            self.robot_belief[mask] = self.update_belief[mask]
            mask_indice = np.argwhere(self.update_belief != 127)
            self.update_false_information(mask_indice)

    def update_false_information(self, mask_indice):
        indices = np.argwhere(self.belief_mask == 1)
        mask_indice = np.ascontiguousarray(mask_indice)
        indices = np.ascontiguousarray(indices)
        mask_view = mask_indice.view([('x', mask_indice.dtype), ('y', mask_indice.dtype)])
        indices_view = indices.view([('x', indices.dtype), ('y', indices.dtype)])
        index_list = np.setdiff1d(indices_view, mask_view).view(indices.dtype).reshape(-1, 2)
        for index in index_list:
            if self.robot_belief[(index[0], index[1])] == 1 or self.robot_belief[(index[0], index[1])] == FALSE_NEGATIVE:
                self.robot_belief[(index[0], index[1])] = FALSE_NEGATIVE
            else:
                self.robot_belief[(index[0], index[1])] = FALSE_POSITIVE

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.update_belief == 255) / np.sum(self.ground_truth == 255)

    def update_robot_location(self, robot_location, robot_id, next_cell):
        self.robot_locations[robot_id] = robot_location
        self.robot_cells[robot_id] = next_cell

    def reset_Dijkstra_path(self):
        self.Dijkstra_path_list_1 = []
        self.Dijkstra_path_list_2 = []
        self.curr_coverage_list = []

    def update_Dijkstra_path(self):
        for location, target in zip(self.robot_locations, self.approximate_target_list):
            dist_dict, prev_dict = self.GT_Node_Manager_reward.Enhanced_Dijkstra(location)
            all_paths, _ = self.GT_Node_Manager_reward.find_all_shortest_paths(dist_dict, prev_dict, target)
            all_paths_coords = merge_and_deduplicate_coordinates(all_paths)
            if len(all_paths_coords):
                all_paths_cells = get_cell_position_from_coords(all_paths_coords, self.ground_truth_info).reshape(-1, 2)
                self.Dijkstra_path_list_1.append(all_paths_coords)
                self.Dijkstra_path_list_2.append(all_paths_cells)
            else:
                self.Dijkstra_path_list_1.append(all_paths_coords)
                self.Dijkstra_path_list_2.append(all_paths_coords)
        self.initiate_Dijkstra_path()

    def initiate_Dijkstra_path(self):
        for i in range(3):
            optimal_path_1 = self.Dijkstra_path_list_1[i]
            if len(optimal_path_1):
                location = self.robot_locations[i]
                dis = np.linalg.norm(optimal_path_1 - location, axis=1)
                curr_in_range = optimal_path_1[dis < self.sensor_range]
                for curr_cell in curr_in_range:
                    if check_collision(curr_cell, location, self.update_belief_info) and \
                        check_collision(location, curr_cell, self.update_belief_info):
                        curr_in_range = remove_coordinate(curr_in_range, curr_cell)
                self.curr_coverage_list.append(curr_in_range)
            else:
                self.curr_coverage_list.append(optimal_path_1)

            optimal_path_2 = self.Dijkstra_path_list_2[i]
            if len(optimal_path_2):
                for cell in optimal_path_2:
                    if self.update_belief[cell[1], cell[0]] == 255:
                        self.Dijkstra_path_list_2[i] = remove_coordinate(self.Dijkstra_path_list_2[i], cell)
            else:
                continue

    def calculate_indiv_explore_reward(self, robot_id):
        location = self.robot_locations[robot_id]
        Dijkstra_path_cell_1 = self.Dijkstra_path_list_1[robot_id]
        dis = np.linalg.norm(Dijkstra_path_cell_1 - location, axis=1)
        next_in_range = Dijkstra_path_cell_1[dis < self.sensor_range]
        for next_cell in next_in_range:
            if check_collision(next_cell, location, self.update_belief_info) and \
                check_collision(location, next_cell, self.update_belief_info):
                next_in_range = remove_coordinate(next_in_range, next_cell)
        curr_in_range = self.curr_coverage_list[robot_id]
        new_update = np.array([row for row in next_in_range if not any(np.all(row == A_row) for A_row in curr_in_range)])
        indiv_update_count = len(new_update)

        Dijkstra_path_cell_2 = self.Dijkstra_path_list_2[robot_id]
        if len(Dijkstra_path_cell_2):
            location_cell = self.robot_cells[robot_id]
            distances = np.linalg.norm(Dijkstra_path_cell_2 - location_cell, axis=1)
            cells_in_range = Dijkstra_path_cell_2[distances < self.sensor_range / self.cell_size]
            if len(cells_in_range):
                coords_in_range = get_coords_from_cell_position(cells_in_range, self.update_belief_info).reshape(-1, 2)
            else:
                coords_in_range = cells_in_range
            for index, coord in enumerate(coords_in_range):
                if not check_collision(coord, location, self.update_belief_info) or \
                    not check_collision(location, coord, self.update_belief_info):
                    cell = cells_in_range[index]
                    self.Dijkstra_path_list_2[robot_id] = remove_coordinate(self.Dijkstra_path_list_2[robot_id], cell)

        indiv_explore_reward = indiv_update_count / 20
        return indiv_explore_reward

    def calculate_team_explore_reward(self, reach_target):
        team_explore_count = 0
        for i in range(3):
            if reach_target[i]:
                continue
            else:
                Dijkstra_path_cell_2 = self.Dijkstra_path_list_2[i]
                for cell_2 in Dijkstra_path_cell_2:
                    if self.update_belief[cell_2[1], cell_2[0]] == 255:
                        team_explore_count += 1
        team_explore_reward = team_explore_count / 20
        return team_explore_reward
    
    def discover_target_reward(self, robot_id, selected_locations):
        discover_reward = 0
        for index, target_flag in enumerate(self.finding_target):
            target_cell = self.approximate_target_cell_list[index]
            if index == robot_id:
                if self.update_belief[target_cell[1], target_cell[0]] == 255:
                    self.finding_target[index] = 1
            else:
                if target_flag:
                    continue
                else:
                    if self.update_belief[target_cell[1], target_cell[0]] == 255:
                        self.finding_target[index] = 1
                        correspond_location = selected_locations[index]
                        target_location = self.approximate_target_list[index]
                        valid_signal = self.judge_target_reward(correspond_location, target_location)
                        if valid_signal:
                            discover_reward += 10
        return discover_reward
    
    def judge_target_reward(self, correspond_location, target_location):
        valid_signal = False
        if np.linalg.norm(correspond_location - target_location) <= SENSOR_RANGE:
            collision = check_collision(correspond_location, target_location, self.update_belief_info)
            if collision:
                valid_signal = True
        else:
            valid_signal = True
        return valid_signal

    def check_useful_explore(self, robot_id, reach_target):
        explore_count = False
        for i in range(3):
            if i == robot_id or reach_target[i]:
                continue
            else:
                Dijkstra_path_cell_3 = self.Dijkstra_path_list_2[i]
                if len(Dijkstra_path_cell_3):
                    location_cell = self.robot_cells[robot_id]
                    location = self.robot_locations[robot_id]
                    dis_array = np.linalg.norm(Dijkstra_path_cell_3 - location_cell, axis=1)
                    filtered_cells = Dijkstra_path_cell_3[dis_array < self.sensor_range / self.cell_size]
                    if len(filtered_cells):
                        filtered_coords = get_coords_from_cell_position(filtered_cells, self.update_belief_info).reshape(-1, 2)
                    else:
                        filtered_coords = filtered_cells
                    for coord in filtered_coords:
                        if not check_collision(coord, location, self.update_belief_info) or not \
                            check_collision(location, coord, self.update_belief_info):
                            explore_count = True
                            break
                else:
                    continue
        return explore_count

    def step(self, robot_id, next_waypoint, astar_cur_dist2target, astar_next_dist2target, selected_locations):
        intermediate_cells, next_cell = self.sensor_smooth(next_waypoint, robot_id)
        self.update_robot_location(next_waypoint, robot_id, next_cell)
        for q in range(NUM_SIM_STEPS):
            self.update_update_belief(intermediate_cells[q])
        indiv_navi_reward = self.calculate_indiv_navi_reward(astar_cur_dist2target, astar_next_dist2target)
        indiv_explore_reward = self.calculate_indiv_explore_reward(robot_id)
        indiv_discover_reward = self.discover_target_reward(robot_id, selected_locations)
        reward = indiv_navi_reward + indiv_explore_reward + indiv_discover_reward

        return reward

    def sensor_smooth(self, next_waypoint, robot_id):
        cur_cell = self.robot_cells[robot_id]
        next_cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        intermediate_cells = np.linspace(cur_cell, next_cell, NUM_SIM_STEPS + 1)[1:]
        intermediate_cells = np.round(intermediate_cells).astype(int)
        return intermediate_cells, next_cell

    def test_step(self, robot_id, next_waypoint):
        intermediate_cells, next_cell = self.sensor_smooth(next_waypoint, robot_id)
        self.update_robot_location(next_waypoint, robot_id, next_cell)
        for q in range(NUM_SIM_STEPS):
            self.update_update_belief(intermediate_cells[q])
        self.test_update_target_state()
            
    def test_update_target_state(self):
        for index, target_flag in enumerate(self.finding_target):
            if target_flag:
                continue
            else:
                target_cell = self.approximate_target_cell_list[index]
                if self.update_belief[target_cell[1], target_cell[0]] == 255:
                    self.finding_target[index] = 1
            
    def explore_step(self, robot_id, next_waypoint, selected_locations):
        intermediate_cells, next_cell = self.sensor_smooth(next_waypoint, robot_id)
        self.update_robot_location(next_waypoint, robot_id, next_cell)
        for q in range(NUM_SIM_STEPS):
            self.update_update_belief(intermediate_cells[q])        
        indiv_discover_reward = self.discover_target_reward(robot_id, selected_locations)
        return indiv_discover_reward

    def update_old_belief(self):
        self.old_belief_mask = deepcopy(self.belief_mask)
        self.old_update_belief = deepcopy(self.update_belief)

    def plot_env_test(self, step):
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        for i in range(N_AGENTS):
            robot_location = self.robot_locations[i]
            plt.plot((robot_location[0] - self.belief_origin_x) / self.cell_size,
                     (robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        for i in range(N_TARGET):
            target_cell = self.target_list[i]
            plt.plot(target_cell[0], target_cell[1], 'o', markersize=12)

        plt.subplot(1, 3, 2)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_info)
        for i in range(N_AGENTS):
            robot_location = self.robot_locations[i]
            plt.plot((robot_location[0] - self.belief_origin_x) / self.cell_size,
                     (robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        plt.imshow(self.ground_truth_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c='r', s=5, zorder=2)
        for i in range(N_TARGET):
            target_cell = self.target_list[i]
            plt.plot(target_cell[0], target_cell[1], 'o', markersize=4)
            approximate_target = self.approximate_target_list[i]
            approximate_target_cell = get_cell_position_from_coords(approximate_target, self.ground_truth_info)
            plt.plot(approximate_target_cell[0], approximate_target_cell[1], 'g', marker='*', markersize=8)
        for coords in self.ground_truth_node_coords:
            node = self.Ground_Truth_Node_Manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.ground_truth_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.ground_truth_info.map_origin_y) / self.cell_size, 'tan',
                         zorder=1)

        plt.subplot(1, 3, 3)
        plt.imshow(self.update_belief, cmap='gray')
        plt.axis('off')
        for i in range(N_AGENTS):
            robot_location = self.robot_locations[i]
            plt.plot((robot_location[0] - self.belief_origin_x) / self.cell_size,
                     (robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        nodes = get_cell_position_from_coords(self.current_node_coords, self.ground_truth_info)
        plt.scatter(nodes[:, 0], nodes[:, 1], color=(0.988, 0.557, 0.675), s=5, zorder=2)
        for coords in self.current_node_coords:
            node = self.node_manager.current_nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set_free:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.belief_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.belief_origin_y) / self.cell_size, c=(0.988, 0.557, 0.675), zorder=1)
        for i in range(N_TARGET):
            target_cell = self.target_list[i]
            plt.plot(target_cell[0], target_cell[1], c=(0.988, 0.557, 0.675), markersize=12)

        plt.suptitle('Explored ratio: {:.4g} | Travel distance: {:.4g}'.format
                     (self.explored_rate, 0))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)


if __name__ == '__main__':
    env = Env(episode_index=18, plot=True, test=True)
    env.plot_env_test(1)
