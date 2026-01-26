import numpy as np
import imageio
import os
from skimage.morphology import label
from scipy.ndimage import convolve
from collections import deque
from parameter import *
from sklearn.neighbors import NearestNeighbors


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords
    

def sorted_dis_in_array(array, point, k):
    relative = array - point
    dis_list = np.linalg.norm(relative, axis=1)
    sorted_index = np.argsort(dis_list)
    k_eff = min(k, len(dis_list)) 
    chosen_index = sorted_index[:k_eff]
    target = array[chosen_index]
    return target, chosen_index


def get_beacons(non_zero_utility_node_coords, all_node_coords, global_map_info):
    local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
    center_indices = []
    centers = non_zero_utility_node_coords
    if centers.shape[0] >= MIN_CENTERS_BEFORE_SPARSIFY:
        knn = NearestNeighbors(radius=SPARSIFICATION_CENTERS_KNN_RAD)
        knn.fit(centers)
        key_center_indices = []
        coverd_center_indices = []
        for i, center in enumerate(centers):
            if i in coverd_center_indices:
                pass
            else:
                _, indices = knn.radius_neighbors(center.reshape(1,2))
                key_center_indices.append(i)
                for index in indices[0]:
                    node = centers[index]
                    if not check_collision(center, node, global_map_info):
                        coverd_center_indices.append(index)
        for i in key_center_indices:
            tmp = centers[i]
            center_indices.append(np.argwhere(local_node_coords_to_check == tmp[0] + tmp[1] * 1j)[0][0])
    else:
        for center in centers:
            center_indices.append(np.argwhere(local_node_coords_to_check == center[0] + center[1] * 1j)[0][0])
    center_indices = list(set(center_indices))
    beacons = all_node_coords[center_indices]
    return beacons, center_indices


def remove_coordinate(coords, target_coord):
    return coords[~np.all(coords == target_coord, axis=1)]


def merge_and_deduplicate_coordinates(list_of_lists):
    merged_set = {coord for sublist in list_of_lists for coord in sublist}
    return np.array(list(merged_set))


def find_enclosed_region(belief, target_value=127):
    h, w = belief.shape
    not_occupied = (belief != 1)

    reachable = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)

    q = deque()

    for i in range(h):
        for j in range(w):
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                if not_occupied[i, j]:
                    q.append((i, j))
                    reachable[i, j] = True
                    visited[i, j] = True

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if not visited[nx, ny] and not_occupied[nx, ny]:
                    visited[nx, ny] = True
                    reachable[nx, ny] = True
                    q.append((nx, ny))

    enclosed_mask = (belief == target_value) & (~reachable)
    enclosed_coords = np.argwhere(enclosed_mask)
    return enclosed_coords, enclosed_mask


def count_vibration_pair(trajectory, current_pos, next_pos, window=None):
    if window is not None:
        traj = trajectory[-(window + 1): -1]
    else:
        traj = trajectory

    count = 0
    for i in range(0, len(traj), 2):
        if np.array_equal(traj[i], current_pos) and np.array_equal(traj[i + 1], next_pos):
            count += 1
    return count


def sorted_non_utility(non_utility_coords, target):
    relative = non_utility_coords - target
    dis_list = np.linalg.norm(relative, axis=-1)
    sorted_index = np.argsort(dis_list)
    sorted_dis_list = dis_list[sorted_index]
    min_dist = sorted_dis_list[0]
    min_index = sorted_index[0]
    target_point = non_utility_coords[min_index]
    return min_dist, target_point


def sorted_dis_utility(non_utility_coords, target):
    relative = non_utility_coords - target
    dis_list = np.linalg.norm(relative, axis=-1)
    sorted_index = np.argsort(dis_list)
    sorted_coords = non_utility_coords[sorted_index]
    sorted_dis_list = dis_list[sorted_index]
    return sorted_dis_list, sorted_coords


def get_reachable_nodes(non_zero_utility_nodes, location, dist_dict):
    reachable_list = []
    dist_list = []
    for coords in non_zero_utility_nodes:
        if coords[0] != location[0] or coords[1] != location[1]:
            dist = dist_dict[(coords[0], coords[1])]
            if dist == 1e8:
                continue
            else:
                reachable_list.append(coords)
                dist_list.append(dist)
    reachable_list = np.array(reachable_list).reshape(-1, 2)
    dist_list = np.array(dist_list).reshape(-1)
    return reachable_list, dist_list


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    # a binary map for free and connected areas
    free = ((map_info.map == FREE) | (map_info.map == FALSE_POSITIVE)) .astype(float)
    # free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)  
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map  # using binary image to represent the connectivity of current location


def get_current_all_node(location, map_info):
    # a binary map for free and connected areas
    free = ((map_info.map == FREE) | (map_info.map == UNKNOWN)) .astype(float)
    # free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)  
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map  # using binary image to represent the connectivity of current location


def get_updating_node_coords(location, updating_map_info, check_connectivity=False):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE or updating_map_info.map[cell[1], cell[0]] == FALSE_POSITIVE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map

def get_updating_current_node_coords(location, current_map_info, check_connectivity=True):
    x_min = current_map_info.map_origin_x
    y_min = current_map_info.map_origin_y
    x_max = current_map_info.map_origin_x + (current_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = current_map_info.map_origin_y + (current_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, current_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < current_map_info.map.shape[0] and 0 <= cell[0] < current_map_info.map.shape[1]
            if current_map_info.map[cell[1], cell[0]] == UNKNOWN:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_current_all_node(location, current_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, current_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map


def get_min_max_in_nodes(nodes):
    y_min = np.min(nodes[:, 1])  
    y_max = np.max(nodes[:, 1])  
    x_min = np.min(nodes[:, 0])  
    x_max = np.max(nodes[:, 0])  
    return y_min, y_max, x_min, x_max


def filter_boundary_nodes(nodes, x_min, x_max, y_min, y_max):
    """
    Remove all nodes on the boundary from nodes, including:
    Nodes whose row coordinate is equal to y_min or y_max
    Nodes whose column coordinate is equal to x_min or x_max
    """
    mask = ~(
        (nodes[:, 0] == x_min) | (nodes[:, 0] == x_max) |
        (nodes[:, 1] == y_min) | (nodes[:, 1] == y_max)
    )
    new_nodes = nodes[mask]
    return new_nodes


def get_frontier_in_map(map_info_1, map_info_2):  # get the intersect frontiers in map_1 and map_2
    frontier_coords_1 = initiate_frontier_in_prior_map(map_info_1)
    frontier_coords_2 = initiate_frontier_in_map(map_info_2)
    frontier_coords_1 = np.ascontiguousarray(frontier_coords_1)
    frontier_coords_2 = np.ascontiguousarray(frontier_coords_2)
    frontier_coords_1_view = frontier_coords_1.view([('x', frontier_coords_1.dtype), ('y', frontier_coords_1.dtype)])
    frontier_coords_2_view = frontier_coords_2.view([('x', frontier_coords_2.dtype), ('y', frontier_coords_2.dtype)])
    prior_frontier_coords = np.intersect1d(frontier_coords_1_view, frontier_coords_2_view).view(frontier_coords_1.dtype).reshape(-1, 2)
    cur_frontier_coords = frontier_coords_2
    if prior_frontier_coords.size > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        prior_frontier_coords = prior_frontier_coords.reshape(-1 ,2)
        prior_frontier_coords = frontier_down_sample(prior_frontier_coords)
    else:
        prior_frontier_coords = set(map(tuple, prior_frontier_coords))
        
    if cur_frontier_coords.size > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        cur_frontier_coords = cur_frontier_coords.reshape(-1 ,2)
        cur_frontier_coords = frontier_down_sample(cur_frontier_coords)
    else:
        cur_frontier_coords = set(map(tuple, cur_frontier_coords))
    return prior_frontier_coords, cur_frontier_coords


def initiate_frontier_in_prior_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = ((map_info.map == FALSE_POSITIVE) | (map_info.map == FALSE_NEGATIVE)) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)  
    return frontier_coords


def initiate_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    return frontier_coords


def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point
    # use 'map' function to transfer every voxel_values into tuple, and then stored in set
    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data


def eliminate_belief_error_1(belief_map, ground_truth):
    kernel_4 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    is_free = (belief_map == FREE).astype(np.uint8)
    is_occ = (belief_map == OCCUPIED).astype(np.uint8)
    is_unknown = (belief_map == UNKNOWN)

    free_neighbors = convolve(is_free, kernel_4, mode='constant', cval=0)
    occ_neighbors = convolve(is_occ, kernel_4, mode='constant', cval=0)

    needs_correction = is_unknown & (free_neighbors >= 1) & (occ_neighbors >= 1)
    belief_map[needs_correction] = ground_truth[needs_correction]

    return belief_map


def eliminate_belief_error(belief_map, ground_truth, robot_cell):
    sensor_range = SENSOR_RANGE / CELL_SIZE
    H, W = belief_map.shape
    x, y = int(robot_cell[0]), int(robot_cell[1])
    r = int(np.ceil(sensor_range))
    
    xmin = max(0, x - r)
    xmax = min(W, x + r + 1)
    ymin = max(0, y - r)
    ymax = min(H, y + r + 1)
    
    belief_roi = belief_map[ymin:ymax, xmin:xmax]
    ground_truth_roi = ground_truth[ymin:ymax, xmin:xmax]
    
    is_free = (belief_roi == FREE).astype(np.uint8)
    is_occ = (belief_roi == OCCUPIED).astype(np.uint8)
    is_unknown = (belief_roi == UNKNOWN)

    kernel_4 = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]])
    
    free_neighbors = convolve(is_free, kernel_4, mode='constant', cval=0)
    occ_neighbors = convolve(is_occ, kernel_4, mode='constant', cval=0)
    
    h, w = belief_roi.shape
    grid_x, grid_y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    dist_sq = (grid_x - x) ** 2 + (grid_y - y) ** 2
    within_range = dist_sq <= sensor_range ** 2
    
    needs_correction_roi = is_unknown & (free_neighbors >= 1) & (occ_neighbors >= 1) & within_range
    belief_map[ymin:ymax, xmin:xmax][needs_correction_roi] = ground_truth_roi[needs_correction_roi]

    return belief_map


def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_unknown_collision(start, end, map_info):
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_frontiers_collision(start, end, map_info):
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == FALSE_NEGATIVE or k == FALSE_POSITIVE or k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_node_collision(start, end, map_info):
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == FALSE_NEGATIVE:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_priori_edge(start, end, map_info):
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1.0) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gifs complete\n')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y

