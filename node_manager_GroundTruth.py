import time
import heapq
import numpy as np
from utils import *
from parameter import *
import quads
from collections import defaultdict


class NodeManager_GroundTruth:
    def __init__(self, plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.frontier = None

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = LocalNode(coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.nodes_dict.find(neighbor_coords)
                neighbor_node.data.neighbor_set.remove(node.coords.tolist())
        self.nodes_dict.remove(node.coords.tolist())

    def update_ground_truth_graph(self, robot_location, ground_truth_map_info):
        ground_truth_node_coords, _ = get_updating_node_coords(robot_location, ground_truth_map_info)
        for coords in ground_truth_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                self.add_node_to_dict(coords)
            else:
                pass

        for coords in ground_truth_node_coords:
            node = self.nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_nodes(ground_truth_map_info, self.nodes_dict)

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
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

            # assert self.nodes_dict.find(u) is not None

            node = self.nodes_dict.find(u).data
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
            node = self.nodes_dict.find(n).data

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

        print('Path does not exist!')

        return [], 1e8


class LocalNode:
    def __init__(self, coords):
        self.coords = coords

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
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
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
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
