import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from sklearn.neighbors import NearestNeighbors

from utils import *
from parameter import *
from node_manager import NodeManager


class Agent:
    def __init__(self, id, global_step, target_list, target_cell_list, policy_net, node_manager, device='cpu',
                 plot=False, test=False, Ground_Truth_Node_Manager=None, prior_nodes_coords=None):
        self.id = id
        self.global_step = global_step
        self.device = device
        self.policy_net = policy_net
        self.plot = plot
        self.target = target_list[id]
        self.target_list = target_list
        self.target_cell_list = target_cell_list
        self.node_manager = node_manager
        self.x_center = None
        self.y_center = None
        self.test = test
        if not self.test:
            self.Ground_Truth_Node_Manager = Ground_Truth_Node_Manager
            self.prior_nodes_coords = prior_nodes_coords
        self.collaboration_list = []
        self.cluster_targets()
        # location and map
        self.location = None
        # self.flag = False
        
        self.last_stop = False

        # flag to perform rapid coverage
        self.rapid_convergence_flag = False
        # flag to use Astar trick
        self.astar_flag = False
        self.last_frontiers = None
        # flag to represent whether exploration is useful
        self.explore_flag = False
        # flag to use Astar beck to start postion and corresponding path
        self.back_trick = False
        self.back_path = []

        # map related parameters
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.global_map_info = None
        self.current_map_info = None
        # self.update_belief_info = None
        self.updating_map_info = None
        # self.updating_map_belief_info = None
        # self.increase_belief_mask = None
        self.current_local_map_info = None

        # combined_frontiers
        self.frontier = set()
        # cur_frontiers
        self.cur_frontier = set()

        # A_star path
        self.path = []

        # Astar_trick
        self.astar_path = []
        self.nodes_in_radius_range = set()
        self.utility_in_radius_range = set()
        self.repetition = 0

        self.travel_dist = 0

        # graph
        self.node_coords, self.prior_utility, self.cur_utility, self.guidepost, self.occupancy = None, None, None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None
        self.target_signal, self.navigation_matrix, self.confidence_signal, self.update_node_coords = None, None, None, None
        self.combined_navi_signal, self.combined_explore_signal, self.update_belief_node, self.update_utility = None, None, None, None
        self.cur_navi_signal, self.cur_explore_signal, self.node_visited, self.stage_signal = None, None, None, None
        self.stage_list = []
        
        # ground truth graph (only for critic)
        self.gt_node_coords, self.gt_adjacent_matrix, self.gt_occupancy, self.gt_target = None, None, None, None
        self.gt_valid_signal, self.gt_navi_signal, self.gt_explore_signal, self.gt_current_index, self.gt_neighbor_indice = None, None, None, None, None
        self.gt_stage = None
        
        # test parameter can be deleted
        self.explore_utility = None
        self.navi_utility = None
        self.centers = None
        self.cur_centers = None

        # option-critic initialization
        self.prev_option = 0
        self.current_stage = 0
        self.is_termination = False
        
        self.episode_buffer = []
        for i in range(37):
            self.episode_buffer.append([])

        self.past_trajectory = []

        if self.plot or self.global_step >= 7000 or self.test:
        # if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []

    def update_global_map(self, global_map_info):
        # no need in training because of shallow copy
        self.global_map_info = global_map_info  # robot_belief_info
        
    def update_current_map(self, current_map_info):
        self.current_map_info = current_map_info

    def update_updating_map(self, location, map_info):
        self.updating_map_info = self.get_updating_map(location, map_info)
        
    def update_current_local_map(self, location, map_info):
        self.current_local_map_info = self.get_updating_map(location, map_info)

    def update_location(self, location):
        self.location = location

    def update_visited(self, location):
        node = self.node_manager.priori_nodes_dict.find(location.tolist())
        if self.node_manager.priori_nodes_dict.__len__() == 0:
            pass
        else:
            if node is None:
                print(self.location, location, self.id, self.global_step)
            node.data.set_visited(self.id)
        if self.plot or self.global_step >= 7000 or self.test:
        # if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])
        self.past_trajectory.append(location)

    def update_frontiers(self):
        self.frontier, self.cur_frontier = get_frontier_in_map(self.updating_map_info, self.current_local_map_info)

    def get_updating_map(self, location, map_info):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[
                                     0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                     1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = (map_info.map_origin_x + self.cell_size * (map_info.map.shape[1] - 1))
        max_y = (map_info.map_origin_y + self.cell_size * (map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, map_info)

        updating_map = map_info.map[
                       updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
                       updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info
    
    def update_collaboration_list(self):
        _, chosen_index = sorted_dis_in_array(np.array(self.target_list), self.location, 2)
        self.collaboration_list = list(chosen_index)
        if self.id not in self.collaboration_list:
            self.collaboration_list.append(self.id)
        else:
            _, chosen_index = sorted_dis_in_array(np.array(self.target_list), self.location, 3)
            self.collaboration_list = list(chosen_index)

    def cluster_targets(self):
        _, chosen_index = sorted_dis_in_array(np.array(self.target_list), self.target, 3)
        self.collaboration_list = list(chosen_index)

    def update_graph(self, global_map_info, location, current_belief_info):
        self.update_global_map(global_map_info)  # update the global_map
        self.update_current_map(current_belief_info)
        self.update_location(location)  # update agent_location
        self.update_updating_map(self.location, self.global_map_info)
        self.update_current_local_map(self.location, self.current_map_info)
        self.update_frontiers()  # update the extended_map_frontier
        self.node_manager.update_posteriori_graph(self.id, self.location, self.frontier, self.cur_frontier, self.updating_map_info, 
                                                  self.global_map_info, self.current_map_info, self.current_local_map_info)
        self.node_manager.update_current_graph(self.location, self.current_local_map_info)
        self.update_visited(self.location)

    def update_planning_state(self, robot_locations, finding_target, reach_target, explore_over):
        self.node_coords, self.prior_utility, self.cur_utility, self.guidepost, self.occupancy, self.adjacent_matrix, \
            self.current_index, self.neighbor_indices, self.target_signal, self.confidence_signal, self.navigation_matrix, \
            self.update_node_coords, self.cur_navi_signal, self.update_belief_node, self.cur_explore_signal, \
            self.update_utility, self.node_visited \
                = self.update_observation(robot_locations, finding_target, reach_target, explore_over)

        # judge whether to stop using the astar_trick
        if self.global_step >= 3000:
            if not self.nodes_in_radius_range:
                self.get_radius_circle(self.location)
                
        if self.global_step >= 3000:
            if self.last_frontiers is None:
                self.last_frontiers = self.frontier
            else:
                if not reach_target[self.id] and reach_target.count(1) == 2 and self.astar_flag:
                    pass
                else:
                    if self.repetition >= 8:
                        pass
                    else:
                        if self.last_frontiers != self.frontier and self.astar_flag:
                            self.astar_flag = False
                            self.astar_path = []
                self.last_frontiers = self.frontier
                
    def update_underlying_state(self, robot_locations, finding_target, test=False):
        self.gt_node_coords, self.gt_adjacent_matrix, self.gt_valid_signal, self.gt_occupancy, \
            self.gt_target, self.gt_navi_signal, self.gt_explore_signal, self.gt_current_index, self.gt_neighbor_indice, \
             = self.get_underlying_node_graph(self.update_belief_node, robot_locations, finding_target)
            
        # if not test:
        #     self.last_coords_to_check_list.append(self.gt_node_coords)

    def get_radius_circle(self, center):
        distances = np.linalg.norm(self.update_belief_node - center, axis=1)
        mask = distances < 20
        self.nodes_in_radius_range = self.update_belief_node[mask]
        self.nodes_in_radius_range = set(map(tuple, self.nodes_in_radius_range))

    def update_observation(self, robot_locations, finding_target, reach_target, explore_over):
        # self.x_center = []
        # self.y_center = []
        all_node_coords = []
        update_node_coords = []
        update_belief_node = []
        update_utility = []
        node_visited = []
        for node in self.node_manager.priori_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        guidepost = []
        prior_utility = []
        cur_utility = []
        confidence_signal = []
        potential_target = [self.target_list[i] for i in self.collaboration_list]
        potential_target_finding = [finding_target[i] for i in self.collaboration_list]

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        navigation_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.priori_nodes_dict.find((coords[0], coords[1])).data
            cur_node = self.node_manager.current_nodes_dict.find((coords[0], coords[1])).data
            guidepost.append(node.visited[self.id])
            prior_utility.append(node.prior_utility)
            cur_utility.append(node.cur_utility)
            confidence_signal.append(node.confidence)
            
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                assert index is not None
                index = index[0][0]
                adjacent_matrix[i, index] = 0
                
            if cur_node.status == FREE:
                update_node_coords.append(coords)
                update_belief_node.append(coords)
                update_utility.append(node.cur_utility)
                node_visited.append(node.visited[self.id])
                for neighbor in cur_node.neighbor_set:
                    neighbor_node = self.node_manager.current_nodes_dict.find((neighbor[0], neighbor[1])).data
                    index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                    assert index is not None
                    index = index[0][0]
                    if neighbor_node.status == FREE:
                        navigation_matrix[i, index] = 0
                    else:
                        pass                  
            else:
                if np.any(np.all(coords == potential_target, axis=1)):
                    update_node_coords.append(coords)
                    update_belief_node.append(coords)
                    update_utility.append(node.cur_utility)
                    node_visited.append(node.visited[self.id])
                else:
                    update_node_coords.append(self.location)

        prior_utility = np.array(prior_utility)
        cur_utility = np.array(cur_utility)
        guidepost = np.array(guidepost)
        confidence_signal = np.array(confidence_signal)
        update_node_coords = np.array(update_node_coords)
        update_belief_node = np.array(update_belief_node)
        update_utility = np.array(update_utility)
        node_visited = np.array(node_visited)

        node = self.node_manager.priori_nodes_dict.find((self.location[0], self.location[1])).data
        neighbor_set = node.neighbor_set
        cur_node = self.node_manager.current_nodes_dict.find((self.location[0], self.location[1])).data
        cur_neighbot_set = cur_node.neighbor_set
        assert neighbor_set == cur_neighbot_set, print(self.location, neighbor_set, cur_neighbot_set)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        occupancy = np.zeros((n_nodes, 1))
        potential_locations = [robot_locations[i] for i in self.collaboration_list]
        for location in potential_locations:
            location_node = self.node_manager.priori_nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(node_coords_to_check == location_node[0] + location_node[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1
                
        target_signal = np.zeros((n_nodes, 1))
        cur_navi_prompt = np.argwhere(cur_utility > 0).reshape(-1)
        
        for flag, coord in enumerate(potential_target):
            target_node = self.node_manager.priori_nodes_dict.find((coord[0], coord[1])).data.coords
            target_index = np.argwhere(node_coords_to_check == target_node[0] + target_node[1] * 1j)[0][0]
            if coord[0] == self.target[0] and coord[1] == self.target[1]:
                target_signal[target_index] = -1
                navigation_matrix[target_index, current_index] = 0
                navigation_matrix[current_index, target_index] = 0
                for i in cur_navi_prompt:
                    navigation_matrix[i, target_index] = 0
                    navigation_matrix[target_index, i] = 0
            else:
                if not potential_target_finding[flag]:
                    target_signal[target_index] = 1
                    for i in cur_navi_prompt:
                        navigation_matrix[i, target_index] = 0
                        navigation_matrix[target_index, i] = 0
                        
        # get all non-zero utility nodes
        cur_non_zero_utility_nodes = all_node_coords[cur_navi_prompt]
        cur_dist_dict, cur_prev_dict = self.node_manager.current_Dijkstra(self.location)
        cur_reachable_coords, cur_reachable_dis = get_reachable_nodes(cur_non_zero_utility_nodes, self.location, cur_dist_dict)
        current_beacons, _ = get_beacons(cur_reachable_coords, all_node_coords, self.current_map_info)

        cur_navi_signal = np.zeros_like(cur_utility)
        path = self.choose_navi_guidepost(cur_navi_prompt, finding_target, cur_reachable_coords, cur_non_zero_utility_nodes,
                                          cur_dist_dict, cur_prev_dict, current_beacons)
        for coords in path:
            if coords[0] != self.location[0] or coords[1] != self.location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                cur_navi_signal[index] = 1

        if path:
            self.navi_utility = np.array(path[-1])
        else:
            self.navi_utility = self.target

        explict_of_others = 0
        explict_index_list = []
        for index, explict in enumerate(finding_target):
            if index == self.id:
                continue
            else:
                if index in self.collaboration_list and not explict:
                    explict_of_others += 1
                    explict_index_list.append(index)

        cur_explore_signal = np.zeros_like(cur_utility)
        traj = self.choose_explore_guidepost(cur_navi_prompt, finding_target, cur_reachable_coords, cur_dist_dict, cur_prev_dict, 
                                             current_beacons, explict_index_list, cur_non_zero_utility_nodes)
        for coords in traj:
            if coords[0] != self.location[0] or coords[1] != self.location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                cur_explore_signal[index] = 1

        if traj:
            self.explore_utility = np.array(traj[-1])
        else:
            self.explore_utility = self.target
            
        return all_node_coords, prior_utility, cur_utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices, \
            target_signal, confidence_signal, navigation_matrix, update_node_coords, cur_navi_signal, update_belief_node, cur_explore_signal, \
            update_utility, node_visited
            
    def get_underlying_node_graph(self, update_belief_node, robot_locations, finding_target):
        gt_node_coords = copy.deepcopy(update_belief_node).tolist()
        
        for node in self.Ground_Truth_Node_Manager.nodes_dict.__iter__():
            coords = node.data.coords
            if not (coords == update_belief_node).all(1).any(0):
                gt_node_coords.append(coords)
        
        gt_node_coords = np.array(gt_node_coords).reshape(-1, 2)
        n_nodes = gt_node_coords.shape[0]

        gt_valid_signal = np.zeros(n_nodes)
        mask = np.isin(self.to_structured(gt_node_coords), self.to_structured(self.prior_nodes_coords)).reshape(-1)
        gt_valid_signal[mask] = 1

        gt_adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        gt_nodes_to_check = gt_node_coords[:, 0] + gt_node_coords[:, 1] * 1j
        
        for i, gt_coords in enumerate(gt_node_coords):
            node = self.Ground_Truth_Node_Manager.nodes_dict.find((gt_coords[0], gt_coords[1])).data
            for neighbor in node.neighbor_set:
                index = np.argwhere(gt_nodes_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    gt_adjacent_matrix[i, index] = 0
                    
        gt_current_index = np.argwhere(gt_nodes_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        gt_neighbor_indice = np.argwhere(gt_adjacent_matrix[gt_current_index] == 0).reshape(-1)

        update_utility_prompt = np.argwhere(self.update_utility > 0).reshape(-1)
        gt_target = np.zeros((n_nodes, 1))
        for flag, coord in enumerate(self.target_list):
            target_node = self.Ground_Truth_Node_Manager.nodes_dict.find((coord[0], coord[1])).data.coords
            target_index = np.argwhere(gt_nodes_to_check == target_node[0] + target_node[1] * 1j)[0][0]
            if coord[0] == self.target[0] and coord[1] == self.target[1]:
                gt_adjacent_matrix[target_index, gt_current_index] = 0
                gt_adjacent_matrix[gt_current_index, target_index] = 0
                gt_target[target_index] = -1
                for i in update_utility_prompt:
                    gt_adjacent_matrix[i, target_index] = 0
                    gt_adjacent_matrix[target_index, i] = 0
            else:
                gt_target[target_index] = 1
                if not finding_target[flag]:
                    for i in update_utility_prompt:
                        gt_adjacent_matrix[i, target_index] = 0
                        gt_adjacent_matrix[target_index, i] = 0

        gt_occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_node = self.Ground_Truth_Node_Manager.nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(gt_nodes_to_check == location_node[0] + location_node[1] * 1j)[0][0]
            if index == gt_current_index:
                gt_occupancy[index] = -1
            else:
                gt_occupancy[index] = 1

        gt_navi_signal = np.zeros_like(gt_valid_signal)
        gt_path, _ = self.Ground_Truth_Node_Manager.a_star(self.location, self.navi_utility)
        for coords in gt_path:
            if coords[0] != self.location[0] or coords[1] != self.location[1]:
                index = np.argwhere(gt_nodes_to_check == coords[0] + coords[1] * 1j)[0][0]
                gt_navi_signal[index] = 1

        gt_explore_signal = np.zeros_like(gt_valid_signal)
        gt_traj, _ = self.Ground_Truth_Node_Manager.a_star(self.location, self.explore_utility)
        for coords in gt_traj:
            if coords[0] != self.location[0] or coords[1] != self.location[1]:
                index = np.argwhere(gt_nodes_to_check == coords[0] + coords[1] * 1j)[0][0]
                gt_explore_signal[index] = 1
            
        return gt_node_coords, gt_adjacent_matrix, gt_valid_signal, gt_target, gt_occupancy, gt_navi_signal, gt_explore_signal, \
                gt_current_index, gt_neighbor_indice

    def to_structured(self, arr):
        return arr.view([('', arr.dtype)] * arr.shape[1])

    def get_observation(self, pad=True):
        node_coords = self.node_coords
        update_node_coords = self.update_node_coords
        node_prior_utility = self.prior_utility.reshape(-1, 1)  # shape(n_node,1)
        node_cur_utility = self.cur_utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)  # shape(n_node,1)
        node_occupancy = self.occupancy.reshape(-1, 1)
        target_signal = self.target_signal.reshape(-1, 1)
        node_confidence_signal = self.confidence_signal.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices  # the maximum of current_edge is 25
        navi_edge_mask = self.navigation_matrix
        cur_navi_signal = self.cur_navi_signal.reshape(-1, 1)
        cur_explore_signal = self.cur_explore_signal.reshape(-1, 1)

        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                     axis=-1) / UPDATING_MAP_SIZE  # scale the nodes coordinates in [0,1], decrease the computation
        update_node_coords = np.concatenate((update_node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                            update_node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                            axis=-1) / UPDATING_MAP_SIZE
        node_prior_utility = node_prior_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_cur_utility = node_cur_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_prior_utility, node_guidepost, node_occupancy, target_signal,
                                      node_confidence_signal, cur_navi_signal, cur_explore_signal), axis=1)  # shape(n_node, 9)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # shape(1, n_node, 9)

        navi_node_inputs = np.concatenate((update_node_coords, node_cur_utility, node_guidepost, node_occupancy,
                                           target_signal, node_confidence_signal, cur_navi_signal, cur_explore_signal), axis=1)
        navi_node_inputs = torch.FloatTensor(navi_node_inputs).unsqueeze(0).to(self.device)
        
        if pad:          
            assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
            padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
            node_inputs = padding(node_inputs)  # shape(1, NODE_PADDING_SIZE, 10)
            navi_node_inputs = padding(navi_node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)

        if pad:
            node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)  # shape(1, 1, NODE_PADDING_SIZE)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)  # shape(1, 1, 1)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)  # shape(1, n_node, n_node)
        navi_edge_mask = torch.tensor(navi_edge_mask).unsqueeze(0).to(self.device)

        if pad:
            padding = torch.nn.ConstantPad2d(
                (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
            edge_mask = padding(edge_mask)  # edge_mask final_shape=(1, NODE_PADDING_SIZE, NODE_PADDING_SIZE)
            navi_edge_mask = padding(navi_edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]  # current_position_index in the neighbor_set()
        current_edge = torch.tensor(current_edge).unsqueeze(0)  # shape:(1, len(neighbor_set())
        k_size = current_edge.size()[-1]  # k_size = len(neighbor_set()), max of k_size = K_SIZE = 25
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
            current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)  # shape(1, K_SIZE, 1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1  # means current position is 1,other positions are 0
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
            edge_padding_mask = padding(edge_padding_mask)  # shape:(1, 1, K_SIZE)

        # prev optin and current stage
        prev_option = torch.tensor([self.prev_option]).reshape(1, 1, 1).to(self.device)  # shape(1, 1, 1)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask,
                navi_edge_mask, navi_node_inputs, 
                prev_option]
        
    def get_state(self):
        gt_node_coords = self.gt_node_coords
        gt_node_cur_utility = self.update_utility.reshape(-1, 1)
        gt_guidepost = self.node_visited.reshape(-1, 1)
        gt_valid_signal = self.gt_valid_signal.reshape(-1, 1)
        gt_target = self.gt_target.reshape(-1, 1)
        gt_occupancy = self.gt_occupancy.reshape(-1, 1)
        gt_navi_signal = self.gt_navi_signal.reshape(-1, 1)
        gt_current_index = self.gt_current_index
        gt_explore_signal = self.gt_explore_signal.reshape(-1, 1)
        gt_current_edge = self.gt_neighbor_indice
        state_edge_mask = self.gt_adjacent_matrix
        n_gt_node = gt_node_coords.shape[0]
        n_padding = n_gt_node - self.update_belief_node.shape[0]
        
        gt_node_guidepost = np.pad(gt_guidepost, ((0, n_padding), (0, 0)), mode='constant', constant_values=0)
        utility_padding_value = SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE
        gt_node_cur_utility = np.pad(gt_node_cur_utility, ((0, n_padding), (0, 0)), mode='constant', constant_values=-utility_padding_value)
        
        current_node_coords = gt_node_coords[self.gt_current_index]
        gt_node_coords = np.concatenate((gt_node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                         gt_node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                        axis=-1) / UPDATING_MAP_SIZE
        gt_node_cur_utility = gt_node_cur_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        state_node_inputs = np.concatenate((gt_node_coords, gt_node_cur_utility, gt_node_guidepost, gt_occupancy, gt_target,
                                            gt_valid_signal, gt_navi_signal, gt_explore_signal), axis=1)
        state_node_inputs = torch.FloatTensor(state_node_inputs).unsqueeze(0).to(self.device)
        
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_gt_node))
        state_node_inputs = padding(state_node_inputs)
        
        state_node_padding_mask = torch.zeros((1, 1, n_gt_node), dtype=torch.int16).to(self.device)
        global_node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_gt_node), dtype=torch.int16).to(self.device)
        state_node_padding_mask = torch.cat((state_node_padding_mask, global_node_padding), dim=-1)
        
        gt_current_index = torch.tensor([gt_current_index]).reshape(1, 1, 1).to(self.device)
        
        state_edge_mask = torch.tensor(state_edge_mask).unsqueeze(0).to(self.device)
        
        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_gt_node, 0, NODE_PADDING_SIZE - n_gt_node), 1)
        state_edge_mask = padding(state_edge_mask)
        
        gt_current_edge = torch.tensor(gt_current_edge).unsqueeze(0).to(self.device)
        k_size = gt_current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        gt_current_edge = padding(gt_current_edge)
        gt_current_edge = gt_current_edge.unsqueeze(-1)
        
        return [state_node_inputs, state_node_padding_mask, state_edge_mask, gt_current_index, gt_current_edge]
        
    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _, _, _, prev_option = observation
        with torch.no_grad():
            logp, _, option = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        # pick the value from tensor(current_edge) as next_node_index
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]

        return next_position, action_index, option

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, navi_edge_mask, navi_node_inputs, pre_option = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()
        self.episode_buffer[6] += navi_edge_mask.bool()
        self.episode_buffer[7] += navi_node_inputs
        self.episode_buffer[8] += pre_option.bool()

    def save_action(self, action_index):
        self.episode_buffer[9] += action_index.reshape(1, 1, 1)
    
    def save_option(self, option_index):
        self.episode_buffer[10] += option_index.reshape(1, 1, 1)

    def save_is_termination(self, is_termination):
        self.episode_buffer[11] += is_termination.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[12] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[13] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)
        
    def save_all_indices(self, all_agent_curr_indices, next_node_index_list):
        self.episode_buffer[14] += torch.tensor(all_agent_curr_indices).reshape(1, -1, 1).to(self.device)
        self.episode_buffer[15] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, navi_edge_mask, navi_node_inputs, pre_option = observation
        self.episode_buffer[16] += node_inputs
        self.episode_buffer[17] += node_padding_mask.bool()
        self.episode_buffer[18] += edge_mask.bool()
        self.episode_buffer[19] += current_index
        self.episode_buffer[20] += current_edge
        self.episode_buffer[21] += edge_padding_mask.bool()
        self.episode_buffer[22] += navi_edge_mask.bool()
        self.episode_buffer[23] += navi_node_inputs
        self.episode_buffer[24] += pre_option.bool()

    def save_state(self, state):
        global_node_inputs, global_node_padding_mask, global_edge_mask, global_current_index, \
            global_current_edge = state
        self.episode_buffer[27] += global_node_inputs
        self.episode_buffer[28] += global_node_padding_mask.bool()
        self.episode_buffer[29] += global_edge_mask.bool()
        self.episode_buffer[30] += global_current_index
        self.episode_buffer[31] += global_current_edge

    def save_next_state(self, state):       
        global_node_inputs, global_node_padding_mask, global_edge_mask, global_current_index, \
            global_current_edge = state
        self.episode_buffer[32] += global_node_inputs
        self.episode_buffer[33] += global_node_padding_mask.bool()
        self.episode_buffer[34] += global_edge_mask.bool()
        self.episode_buffer[35] += global_current_index
        self.episode_buffer[36] += global_current_edge

    def get_nearest_target_explicit(self, non_zero_utility_nodes, reachable_coords, reachable_dis):
        total_dis = []
        target_dist_dict, _ = self.node_manager.current_Dijkstra(self.target)
        target_reachable_coords, _ = get_reachable_nodes(non_zero_utility_nodes, self.target,
                                                         target_dist_dict)
        if target_reachable_coords.size == 0 and reachable_coords.size:
            relative = reachable_coords - self.target
            dis_list = np.linalg.norm(relative, axis=-1)
            for index, _ in enumerate(reachable_coords):
                total_dist = reachable_dis[index] + dis_list[index]
                total_dis.append(total_dist)
        elif target_reachable_coords.size and reachable_coords.size:
            for index, coord in enumerate(reachable_coords):
                min_dist, target_point = sorted_non_utility(target_reachable_coords, coord)
                target_dist = target_dist_dict[(target_point[0], target_point[1])]
                total_dist = min_dist + reachable_dis[index] + target_dist
                total_dis.append(total_dist)
        total_dis = np.array(total_dis).reshape(-1)
        if total_dis.size > 0:
            chosed_index = np.argsort(total_dis)[0]
            chosed_node = reachable_coords[chosed_index]
        else:
            chosed_node = np.array([])
        return chosed_node
        
    def get_nearest_non_utility_target(self, beacons, target, cur_dist_dict, cur_prev_dict, reachable_coords, non_zero_utility_nodes):
        _, _, chosen_beacon = self.node_manager.Dijkstra_for_guidepost(target, beacons)
        if chosen_beacon is not None:
            path, dist = self.node_manager.current_astar(self.location, chosen_beacon)
            if dist == 1e8:
                path, _ = self.node_manager.get_Dijkstra_path_and_dist(cur_dist_dict, cur_prev_dict, chosen_beacon)
        else:
            path = []
        #     path = self.get_nearest_non_utility(self.location, beacons, cur_dist_dict)
        return path
        
    def get_nearest_non_utility(self, robot_location, beacons, dist_dict):
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in beacons:
            if coords[0] != robot_location[0] or coords[1] != robot_location[1]:
                dist = dist_dict[(coords[0], coords[1])]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_utility_coords = coords
        path_coords, _ = self.node_manager.current_astar(robot_location, nearest_utility_coords)
        return path_coords
    
    def choose_navi_guidepost(self, navi_prompt, finding_target, reachable_coords, non_zero_utility_nodes, dist_dict, prev_dict, beacons):
        if navi_prompt.size > 0:
            if finding_target[self.id]:
                path, dist = self.node_manager.current_astar(self.location, self.target)
                if dist == 1e8:
                    path = self.get_nearest_non_utility_target(beacons, self.target, dist_dict, prev_dict, reachable_coords, non_zero_utility_nodes)
            else:
                path = self.get_nearest_non_utility_target(beacons, self.target, dist_dict, prev_dict, reachable_coords, non_zero_utility_nodes)
        else:
            path, dist = self.node_manager.current_astar(self.location, self.target)
            if dist == 1e8:
                path, _ = self.node_manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict, self.target)
        return path

    def choose_explore_guidepost(self, navi_prompt, finding_target, reachable_coords, dist_dict, prev_dict, beacons, explict_index_list, non_zero_utility_nodes):
        if navi_prompt.size > 0:
            if explict_index_list:
                if len(explict_index_list) > 2:
                    potential_targets = []
                    for index in explict_index_list:
                        potential_targets.append(self.target_list[index])
                    potential_targets = np.array(potential_targets).reshape(-1, 2)
                    chosen_target_list, _ = sorted_dis_in_array(potential_targets, self.location, 2)
                    potential_target_1 = chosen_target_list[0]
                    dist_dict_1, prev_dict_1, chosen_beacon_1 = self.node_manager.Dijkstra_for_guidepost(
                        potential_target_1, beacons)
                    potential_target_2 = chosen_target_list[1]
                    dist_dict_2, prev_dict_2, chosen_beacon_2 = self.node_manager.Dijkstra_for_guidepost(
                        potential_target_2, beacons)
                    if chosen_beacon_1 is not None and chosen_beacon_2 is not None:
                        _, min_dis_1 = self.node_manager.get_Dijkstra_path_and_dist(dist_dict_1, prev_dict_1, chosen_beacon_1)
                        _, min_dis_2 = self.node_manager.get_Dijkstra_path_and_dist(dist_dict_2, prev_dict_2, chosen_beacon_2)
                        if min_dis_1 <= min_dis_2:
                            traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_1)
                        else:
                            traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_2)
                    elif chosen_beacon_1 is not None and chosen_beacon_2 is None:
                        traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_1)
                    elif chosen_beacon_1 is None and chosen_beacon_2 is not None:
                        traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_2)
                    else:
                        traj = []
                elif len(explict_index_list) == 2:
                    index_1 = explict_index_list[0]
                    dist_dict_1, prev_dict_1, chosen_beacon_1 = self.node_manager.Dijkstra_for_guidepost(
                        self.target_list[index_1], beacons)
                    index_2 = explict_index_list[1]
                    dist_dict_2, prev_dict_2, chosen_beacon_2 = self.node_manager.Dijkstra_for_guidepost(
                        self.target_list[index_2], beacons)
                    if chosen_beacon_1 is not None and chosen_beacon_2 is not None:
                        _, min_dis_1 = self.node_manager.get_Dijkstra_path_and_dist(dist_dict_1, prev_dict_1, chosen_beacon_1)
                        _, min_dis_2 = self.node_manager.get_Dijkstra_path_and_dist(dist_dict_2, prev_dict_2, chosen_beacon_2)
                        if min_dis_1 <= min_dis_2:
                            traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_1)
                        else:
                            traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_2)
                    elif chosen_beacon_1 is not None and chosen_beacon_2 is None:
                        traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_1)
                    elif chosen_beacon_1 is None and chosen_beacon_2 is not None:
                        traj, _ = self.node_manager.current_astar(self.location, chosen_beacon_2)
                    else:
                        traj = []
                else:
                    index = explict_index_list[0]
                    traj = self.get_nearest_non_utility_target(beacons, self.target_list[index], dist_dict, prev_dict, reachable_coords, non_zero_utility_nodes)
            else:
                if finding_target[self.id]:
                    traj, dist = self.node_manager.current_astar(self.location, self.target)
                    if dist == 1e8:
                        traj = self.get_nearest_non_utility_target(beacons, self.target, dist_dict, prev_dict, reachable_coords, non_zero_utility_nodes)
                else:
                    traj = self.get_nearest_non_utility_target(beacons, self.target, dist_dict, prev_dict, reachable_coords, non_zero_utility_nodes)
        else:
            traj, dist = self.node_manager.current_astar(self.location, self.target)
            if dist == 1e8:
                traj, _ = self.node_manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict, self.target)
        return traj
