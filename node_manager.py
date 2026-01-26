import time
import heapq
import numpy as np
from utils import *
from parameter import *
import quads


class NodeManager:
    def __init__(self, target_list, plot=False):
        self.priori_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.current_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.target_list = target_list if target_list else None
        self.y_min, self.y_max, self.x_min, self.x_max = None, None, None, None

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.priori_nodes_dict.find(key)
        return exist
    
    def check_node_exist_in_current_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.current_nodes_dict.find(key)
        return exist

    def check_node_status_in_current_dict(self, coords):
        key = (coords[0], coords[1])
        status = self.current_nodes_dict.find(key).data.status
        return status

    def add_node_to_dict(self, coords, frontiers, cur_frontiers, updating_map_info, current_local_map_info, confidence, state):
        key = (coords[0], coords[1])
        node = LocalNode(coords, frontiers, cur_frontiers, updating_map_info, current_local_map_info, confidence, state)
        self.priori_nodes_dict.insert(point=key, data=node)
        return node

    def add_node_to_current_dict(self, coords, status):
        key = (coords[0], coords[1])
        node = Current_node(coords, status)
        self.current_nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.priori_nodes_dict.find(neighbor_coords)
                neighbor_node.data.neighbor_set.remove((node.coords[0], node.coords[1]))
        self.priori_nodes_dict.remove(node.coords.tolist())

    def remove_node_from_current_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.current_nodes_dict.find(neighbor_coords)
                neighbor_node.data.neighbor_set.remove((node.coords[0], node.coords[1]))
        for neighbor_coords in node.neighbor_set_free:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.current_nodes_dict.find(neighbor_coords)
                neighbor_node.data.neighbor_set_free.remove((node.coords[0], node.coords[1]))
        self.current_nodes_dict.remove(node.coords.tolist())

    def update_all_current_graph(self, robot_location, current_map_info):
        current_node_coords, _ = get_updating_current_node_coords(robot_location,current_map_info)
        
        self.y_min, self.y_max, self.x_min, self.x_max = get_min_max_in_nodes(current_node_coords)
        current_node_coords = filter_boundary_nodes(current_node_coords, self.x_min, self.x_max, self.y_min, self.y_max)

        for coords in current_node_coords:
            node = self.check_node_exist_in_current_dict(coords)
            if node is None:
                self.add_node_to_current_dict(coords, UNKNOWN)
            else:
                pass
        for coords in current_node_coords:
            node = self.current_nodes_dict.find((coords[0], coords[1])).data
            node.update_current_edges(current_map_info, self.current_nodes_dict)

    def update_current_graph(self, robot_location, current_local_map_info):
        node_coords, _ = get_updating_current_node_coords(robot_location, current_local_map_info)
        
        node_coords = filter_boundary_nodes(node_coords, self.x_min, self.x_max, self.y_min, self.y_max)

        all_node_list = []
        for coords in node_coords:
            node = self.check_node_exist_in_current_dict(coords)
            if node is None:
                node = self.add_node_to_current_dict(coords, FREE)
            else:
                node = node.data
            all_node_list.append(node)

        for node in all_node_list:
            if np.linalg.norm(node.coords - robot_location) < (SENSOR_RANGE + NODE_RESOLUTION):
                node.update_free_neighbor_edges(current_local_map_info, self.current_nodes_dict)
                if node.status == FREE:
                    node.update_neighbor_edges(current_local_map_info, self.current_nodes_dict)

    def update_priori_truth_graph(self, robot_location, priori_truth_map_info):
        priori_truth_node_coords, _ = get_updating_node_coords(robot_location, priori_truth_map_info)

        for coords in priori_truth_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                self.add_node_to_dict(coords, None, None, None, None, UNCERTAIN_CONFIDENCE, UNCERTAIN)
            else:
                pass

        for coords in priori_truth_node_coords:
            node = self.priori_nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_nodes(priori_truth_map_info, self.priori_nodes_dict)

    def update_posteriori_graph(self, robot_id, robot_location, frontiers, cur_frontiers, updating_map_info, map_info, current_map_info, current_local_map_info):
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)
        
        all_node_list = []
        global_frontiers, global_cur_frontiers = get_frontier_in_map(map_info, current_map_info)
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, cur_frontiers, updating_map_info, current_local_map_info, CERTAIN_CONFIDENCE, CERTAIN)
            else:
                node = node.data
                if not node.state or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_prior_frontiers(frontiers, global_frontiers, updating_map_info)
                    node.update_node_observable_cur_frontiers(cur_frontiers, global_cur_frontiers, current_local_map_info)
            all_node_list.append(node)

        for node in all_node_list:
            if np.linalg.norm(node.coords - robot_location) < (SENSOR_RANGE + NODE_RESOLUTION):
                node.update_posteriori_neighbor_nodes(map_info, self.priori_nodes_dict)

    def update_target_list(self, target_list):
        self.target_list = target_list
        for target in target_list:
            target_node = self.priori_nodes_dict.find((target[0], target[1])).data
            target_node.confidence = CERTAIN_CONFIDENCE
                    
    def delete_mask_node(self, mask_belief, global_map_info):
        for node in self.priori_nodes_dict.__iter__():
            coords = node.data.coords
            cell_coords = get_cell_position_from_coords(coords, global_map_info)
            if mask_belief[cell_coords[1], cell_coords[0]] == 127:
                continue
            else:
                node = node.data
                self.remove_node_from_dict(node)

    def delete_current_node(self, mask_belief, global_map_info):
        for node in self.current_nodes_dict.__iter__():
            coords = node.data.coords
            cell_coords = get_cell_position_from_coords(coords, global_map_info)
            if mask_belief[cell_coords[1], cell_coords[0]] == 127:
                continue
            else:
                node = node.data
                self.remove_node_from_current_dict(node)

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.priori_nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)
            
            node = self.priori_nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict
    
    def Dijkstra_for_guidepost(self, start, interest_points, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        interest_set = set(tuple(p) for p in interest_points)

        for node in self.current_nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            if u in interest_set:
                return dist_dict, prev_dict, u

            node = self.current_nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set_free:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** 0.5
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict, None

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            print("destination is not in Dijkstra graph")
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        h = np.linalg.norm(np.array([coords_1[0] - coords_2[0], coords_1[1] - coords_2[1]]))
        return h

    def a_star(self, start, destination, max_dist=None):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        open_heap = []
        heapq.heappush(open_heap, (0, (start[0], start[1])))

        while len(open_list) > 0:
            _, n = heapq.heappop(open_heap)
            n_coords = n
            node = self.priori_nodes_dict.find(n).data

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()

                return path, np.round(length, 2)

            costs = np.linalg.norm(np.array(list(node.neighbor_set)).reshape(-1, 2) - [n_coords[0], n_coords[1]],
                                   axis=1)
            for cost, neighbor_node_coords in zip(costs, node.neighbor_set):
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m], m))
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

            open_list.remove(n)
            closed_list.add(n)

        return [], 1e8

    def current_astar(self, start, destination, max_dist=None):
        # the path does not include the start
        if self.check_node_status_in_current_dict(start) != FREE:
            print(start)
            Warning("start node status is not FREE")
            return [], 1e8
        if self.check_node_status_in_current_dict(destination) != FREE:
            Warning("end node status is not FREE")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        open_heap = []
        heapq.heappush(open_heap, (0, (start[0], start[1])))

        while len(open_list) > 0:
            _, n = heapq.heappop(open_heap)
            n_coords = n
            node = self.current_nodes_dict.find(n).data

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()

                return path, np.round(length, 2)

            costs = np.linalg.norm(np.array(list(node.neighbor_set)).reshape(-1, 2) - [n_coords[0], n_coords[1]],
                                   axis=1)
            for cost, neighbor_node_coords in zip(costs, node.neighbor_set):
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m], m))
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

            open_list.remove(n)
            closed_list.add(n)

        return [], 1e8

    def current_Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.current_nodes_dict.__iter__():
            node = node.data
            if node.status == UNKNOWN:
                continue
            else:
                coords = node.coords
                key = (coords[0], coords[1])
                dist_dict[key] = 1e8
                prev_dict[key] = None
                q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            node = self.current_nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict


class LocalNode:
    def __init__(self, coords, prior_frontiers, cur_frontiers,updating_map_info, current_local_map_info, confidence, state):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.prior_utility = 0
        self.cur_utility = 0
        self.visited = [0] * N_AGENTS
        self.confidence = confidence
        self.state = state
        if self.state:
            self.prior_observable_frontiers = self.initialize_observable_prior_frontiers(prior_frontiers, updating_map_info)
            self.cur_observable_frontiers = self.initialize_observable_cur_frontiers(cur_frontiers, current_local_map_info)
            self.need_update_neighbor = True
        else:
            self.prior_observable_frontiers = set()
            self.cur_observable_frontiers = set()
            self.need_update_neighbor = False

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    # update prior_utility and prior_observable_frontiers
    def initialize_observable_prior_frontiers(self, frontiers, updating_map_info):   
        if len(frontiers) == 0:
            self.prior_utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_frontiers_collision(self.coords, point, updating_map_info)
                if not collision:
                    reverse_collision = check_frontiers_collision(point, self.coords, updating_map_info)
                    if not reverse_collision:
                        observable_frontiers.add((point[0], point[1]))
                    else:
                        continue
                else:
                    continue
            self.prior_utility = len(observable_frontiers)
            if self.prior_utility <= MIN_UTILITY:
                self.prior_utility = 0
                observable_frontiers = set()
            return observable_frontiers

    # update cur_utility and cur_observable_frontiers
    def initialize_observable_cur_frontiers(self, frontiers, current_local_map_info):   
        if len(frontiers) == 0:
            self.cur_utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, current_local_map_info)
                if not collision:
                    reverse_collision = check_collision(point, self.coords, current_local_map_info)
                    if not reverse_collision:
                        observable_frontiers.add((point[0], point[1]))
                    else:
                        continue
                else:
                    continue
            self.cur_utility = len(observable_frontiers)
            if self.cur_utility <= MIN_UTILITY:
                self.cur_utility = 0
                observable_frontiers = set()
            return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, priori_nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        # self.neighbor_list.append(self.coords)
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = priori_nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_node_collision(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))

    def update_posteriori_neighbor_nodes(self, updating_map_info, priori_nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                center_index = self.neighbor_matrix.shape[0] // 2
                if i == center_index and j == center_index:
                    self.neighbor_matrix[i, j] = 1
                    continue

                neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                      self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                neighbor_node = priori_nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                if neighbor_node is None:
                    continue
                else:
                    neighbor_node = neighbor_node.data
                    collision = check_node_collision(self.coords, neighbor_coords, updating_map_info)
                    neighbor_matrix_x = center_index + (center_index - i)
                    neighbor_matrix_y = center_index + (center_index - j)
                    if not collision:
                        self.neighbor_matrix[i, j] = 1
                        self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                        neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                        neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
                    else:
                        reverse_collision = check_node_collision(neighbor_coords, self.coords, updating_map_info)
                        if not reverse_collision:
                            continue
                        else:
                            if (neighbor_coords[0], neighbor_coords[1]) in self.neighbor_set:
                                self.neighbor_matrix[i, j] = -1
                                self.neighbor_set.remove((neighbor_coords[0], neighbor_coords[1]))

                            if (self.coords[0], self.coords[1]) in neighbor_node.neighbor_set:
                                neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = -1
                                neighbor_node.neighbor_set.remove((self.coords[0], self.coords[1]))

    # update_node_observable_prior_frontiers
    def update_node_observable_prior_frontiers(self, frontiers, global_frontiers, updating_map_info):
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.prior_observable_frontiers:
            if frontier not in global_frontiers:  # frontiers are updated by Lidar
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.prior_observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        new_frontiers = frontiers - self.prior_observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_frontiers_collision(self.coords, point, updating_map_info)
            if not collision:
                reverse_collision = check_frontiers_collision(point, self.coords, updating_map_info)
                if not reverse_collision:
                    self.prior_observable_frontiers.add((point[0], point[1]))
                else:
                    continue
            else:
                continue

        self.prior_utility = len(self.prior_observable_frontiers)
        if self.prior_utility <= MIN_UTILITY:
            self.prior_utility = 0
            self.prior_observable_frontiers = set()

    # update_node_observable_cur_frontiers
    def update_node_observable_cur_frontiers(self, cur_frontiers, global_frontiers, cur_local_map_info):
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.cur_observable_frontiers:
            if frontier not in global_frontiers:  # frontiers are updated by Lidar
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.cur_observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        new_frontiers = cur_frontiers - self.cur_observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, cur_local_map_info)
            if not collision:
                reverse_collision = check_collision(point, self.coords, cur_local_map_info)
                if not reverse_collision:
                    self.cur_observable_frontiers.add((point[0], point[1]))
                else:
                    continue
            else:
                continue

        self.cur_utility = len(self.cur_observable_frontiers)
        if self.cur_utility <= MIN_UTILITY:
            self.cur_utility = 0
            self.cur_observable_frontiers = set()

    def set_visited(self, id):
        self.visited[id] = 1
        self.prior_observable_frontiers = set()
        self.cur_observable_frontiers = set()
        self.prior_utility = 0
        self.cur_utility = 0
        
class Current_node:
    def __init__(self, coords, status):
        self.coords = coords

        self.neighbor_matrix_free = -np.ones((5, 5))
        self.neighbor_set_free = set()
        self.neighbor_matrix_free[2, 2] = 1
        self.neighbor_set_free.add((self.coords[0], self.coords[1]))
        self.status = status
        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def update_free_neighbor_edges(self, updating_map_info, current_nodes_dict):
        for i in range(self.neighbor_matrix_free.shape[0]):
            for j in range(self.neighbor_matrix_free.shape[1]):
                center_index = self.neighbor_matrix_free.shape[0] // 2
                if i == center_index and j == center_index:
                    self.neighbor_matrix_free[i, j] = 1
                    continue

                neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                      self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                neighbor_node = current_nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                if neighbor_node is None:
                    continue
                else:
                    neighbor_node = neighbor_node.data
                    collision = check_unknown_collision(self.coords, neighbor_coords, updating_map_info)
                    neighbor_matrix_x = center_index + (center_index - i)
                    neighbor_matrix_y = center_index + (center_index - j)
                    if not collision:
                        self.neighbor_matrix_free[i, j] = 1
                        self.neighbor_set_free.add((neighbor_coords[0], neighbor_coords[1]))

                        neighbor_node.neighbor_matrix_free[neighbor_matrix_x, neighbor_matrix_y] = 1
                        neighbor_node.neighbor_set_free.add((self.coords[0], self.coords[1]))
                    else:
                        reverse_collision = check_unknown_collision(neighbor_coords, self.coords, updating_map_info)
                        if not reverse_collision:
                            continue
                        else:
                            if (neighbor_coords[0], neighbor_coords[1]) in self.neighbor_set_free:
                                self.neighbor_matrix_free[i, j] = -1
                                self.neighbor_set_free.remove((neighbor_coords[0], neighbor_coords[1]))

                            if (self.coords[0], self.coords[1]) in neighbor_node.neighbor_set_free:
                                neighbor_node.neighbor_matrix_free[neighbor_matrix_x, neighbor_matrix_y] = -1
                                neighbor_node.neighbor_set_free.remove((self.coords[0], self.coords[1]))

    def update_current_edges(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix_free.shape[0]):
            for j in range(self.neighbor_matrix_free.shape[1]):
                if self.neighbor_matrix_free[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix_free.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix_free[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]),1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_unknown_collision(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix_free[i, j] = 1
                            self.neighbor_set_free.add((neighbor_coords[0], neighbor_coords[1]))

                            neighbor_node.neighbor_matrix_free[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_set_free.add((self.coords[0], self.coords[1]))

    def update_neighbor_edges(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]),1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        if neighbor_node.status == UNKNOWN:
                            continue
                        else:
                            collision = check_collision(self.coords, neighbor_coords, updating_map_info)
                            neighbor_matrix_x = center_index + (center_index - i)
                            neighbor_matrix_y = center_index + (center_index - j)
                            if not collision:
                                self.neighbor_matrix[i, j] = 1
                                self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                                neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                                neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
