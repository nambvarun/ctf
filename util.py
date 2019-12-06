import numpy as np
from typing import List


def get_points_between(point_a: np.array, point_b: np.array) -> np.array:
    x_data = np.linspace(point_a[0], point_b[0], 100)
    y_data = np.linspace(point_a[1], point_b[1], 100)
    return np.vstack((x_data, y_data))


def get_intersection(line_a: List[np.array], line_b: List[np.array]) -> (bool, np.array):
    # print(line_a)
    # print(line_b)
    A = np.vstack((-(line_a[1] - line_a[0]), line_b[1] - line_b[0])).T
    b = line_a[0] - line_b[0]

    t = np.linalg.inv(A) @ b

    # print(t)
    # print(t <= 1)
    # print(0 <= t)

    return np.all(t <= 1) and np.all(0 <= t), (line_a[1] - line_a[0]) * t[0] + line_a[0]


# a = [np.array([1, 1]), np.array([0, 0])]
# b = [np.array([0, 1]), np.array([1, 0])]
#
# print(get_intersection(a, b))
