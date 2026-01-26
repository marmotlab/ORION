import numpy as np


def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 2

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def update_mask(x0, y0, x1, y1, mask, belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    old_item = None

    while 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        k = belief.item(y, x)
        j = mask.item(y, x)

        if old_item is not None and old_item == 255 and k == 127:
            break

        if x == x1 and y == y1:
            break

        if k == 1:
            mask.itemset((y, x), 255)
        else:
            if k == 127:
                mask.itemset((y, x), 1)
            elif k != 127 and j != 255:
                mask.itemset((y, x), 255)

        old_item = k

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return mask


def posteriori_check(x0, y0, x1, y1, priori_info, robot_belief, posteriori_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < priori_info.shape[1] and 0 <= y < priori_info.shape[0]:
        k1 = priori_info.item(y, x)
        k2 = robot_belief.item(y, x)

        if k2 == 127:
            break

        if x == x1 and y == y1:
            break

        if k1 != k2:
            posteriori_belief.itemset((y, x), 255)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return posteriori_belief


def update_map_sensor(robot_position, sensor_range, robot_belief, ground_truth):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief


def make_belief_mask(robot_position, sensor_range, mask, belief):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        mask = update_mask(x0, y0, x1, y1, mask, belief)
        sensor_angle += sensor_angle_inc
    return mask


def posteriori_sensor(robot_position, sensor_range, posteriori_belief, priori_info, robot_belief):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        posteriori_belief = posteriori_check(x0, y0, x1, y1, priori_info, robot_belief, posteriori_belief)
        sensor_angle += sensor_angle_inc
    return posteriori_belief
