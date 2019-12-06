import copy
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from scipy.spatial import KDTree
import sys
from typing import List, Tuple
import util


class Agent:
    def __init__(self, name: str = 'agent', center: np.array = np.array([5, 5]), dim_size: float = 0.2, theta: float = 1e-6):
        self._name = name
        self._center = center
        self._theta = theta
        self._dim_size = dim_size
        self._transform = np.array([[np.cos(self._theta), -np.sin(self._theta), self._center[0]],
                                    [np.sin(self._theta), np.cos(self._theta), self._center[1]],
                                    [0, 0, 1]])
        self._lidar_beams = 50

    def get_bounds(self) -> List[List[np.array]]:
        bounds_list = []
        previous_item = self._center
        for i in range(30):
            item_x = self._dim_size * np.cos(np.pi/2 + i/30 * 2 * np.pi)
            item_y = self._dim_size * np.sin(np.pi/2 + i/30 * 2 * np.pi)
            out = (self._transform @ np.array([item_x, item_y, 1]))[:-1].flatten()

            bounds_list.append([previous_item, out])
            previous_item = (self._transform @ np.array([item_x, item_y, 1]))[:-1].flatten()

        return bounds_list

    def draw(self) -> List[np.array]:
        points_to_return = []
        for edge in self.get_bounds():
            points_to_return.append(util.get_points_between(*edge))

        return points_to_return

    def move(self, drho, dtheta):
        self._theta += dtheta
        self._center = self._center + (self._transform @ np.array([0, drho, 0]))[:-1]
        self._transform = np.array([[np.cos(self._theta), -np.sin(self._theta), self._center[0]],
                                    [np.sin(self._theta), np.cos(self._theta), self._center[1]],
                                    [0, 0, 1]])

    def observation(self, objs, sample_points, flag_points):
        pc = self.lidar_observation(objs)
        flag_pos = self.find_flag(objs, flag_points)

        fov = None
        if flag_pos is None:
            fov = self.fov_observation(objs, sample_points).T
        else:
            flag_pos = flag_pos.T

        return pc.T, fov, flag_pos

    def is_ray_blocked(self, objs, line):
        for key in objs.keys():
            if key != self._name:
                collide, where, t = Simulation.collide(line, objs[key].get_bounds())
                if collide and key != 'bounds':
                    return True

        return False

    def fov_observation(self, objs, sample_points):
        dists, sample_idx = sample_points.query(self._center, 30, distance_upper_bound=2.0)
        sample_idx = sample_idx[sample_idx < sample_points.data.shape[0]]
        fov_points = sample_points.data[sample_idx]

        can_i_see_this_fov_point = []
        for point in fov_points:
            if np.all(self._center == point):
                can_i_see_this_fov_point.append(True)
            else:
                fov_ray = [[self._center + np.random.random(2) * 1e-6, point]]
                # print(fov_ray)
                if self.is_ray_blocked(objs, fov_ray):
                    can_i_see_this_fov_point.append(True)
                else:
                    can_i_see_this_fov_point.append(False)

        return fov_points[can_i_see_this_fov_point]

    def find_flag(self, objs, flag_points):
        for point in flag_points:
            flag_ray = [[self._center, point]]
            if not self.is_ray_blocked(objs, flag_ray):
                return point

        return None

    def lidar_observation(self, objs):
        o = []
        for i in range(self._lidar_beams):
            item_x = 20 * np.cos(np.pi / 2 + i / self._lidar_beams * 2 * np.pi)
            item_y = 20 * np.sin(np.pi / 2 + i / self._lidar_beams * 2 * np.pi)
            out = (self._transform @ np.array([item_x, item_y, 1]))[:-1].flatten()

            lidar_ray = [[self._center, out]]

            min_t = np.finfo(np.float).max
            point = None
            for key in objs.keys():
                if key != self._name:
                    collide, where, t = Simulation.collide(lidar_ray, objs[key].get_bounds())
                    if collide:
                        if t[0] < min_t:
                            point = where
                            min_t = t[0]

            o.append(point)
        pc = np.array(o)

        return pc


class Wall:
    def __init__(self, bounds: List[List[np.array]]):
        self._bounds = bounds

    def get_bounds(self) -> List[List[np.array]]:
        return self._bounds

    def draw(self) -> List[np.array]:
        points_to_return = []
        for edge in self.get_bounds():
            points_to_return.append(util.get_points_between(*edge))

        return points_to_return


class Simulation:
    def __init__(self, min_bound: float = 0, max_bound: float = 10, flag_center: np.array = np.array([5, 5])):
        plt.ion()
        self._fig = plt.figure(figsize=(8, 8))
        self._fig.canvas.mpl_connect('key_press_event', self._press)
        self._ax = self._fig.add_subplot(111)

        self._name2obj = {}
        self._name2axitem = {}

        self._lidar_list = []

        # setting bounds of the viz
        plt.xlim(min_bound - 1, max_bound + 1)
        plt.ylim(min_bound - 1, max_bound + 1)

        self._min_bound = min_bound
        self._max_bound = max_bound

        # plotting the wall boundary
        bounds = [[np.array([min_bound, min_bound]), np.array([min_bound, max_bound])],
                  [np.array([min_bound, min_bound]), np.array([max_bound, min_bound])],
                  [np.array([max_bound, max_bound]), np.array([max_bound, min_bound])],
                  [np.array([max_bound, max_bound]), np.array([min_bound, max_bound])]]
        wall = Wall(bounds)

        self._flag_center = flag_center
        flag_bounds = [[np.array([flag_center[0] + 0.03, flag_center[1] + 0.15]), np.array([flag_center[0] - 0.07, flag_center[1] - 0.08])],
                       [np.array([flag_center[0] - 0.08, flag_center[1] - 0.08]), np.array([flag_center[0] + 0.16, flag_center[1] - 0.08])],
                       [np.array([flag_center[0] + 0.16, flag_center[1] - 0.08]), np.array([flag_center[0] + 0.03, flag_center[1] + 0.15])]]

        lc = mc.LineCollection(flag_bounds, linewidths=2)
        self._ax.add_collection(lc)

        bound = {"bounds": wall}

        self._found_flag = False
        self.add_elements(bound)

    def has_flag(self, name: str) -> bool:
        if np.linalg.norm(self._name2obj[name]._center - self._flag_center) < 0.5 or self._found_flag:
            self._found_flag = True
            return True

        return False

    def get_state(self, name: str) -> Tuple:
        center = self._name2obj[name]._center.tolist()
        has_flag = self.has_flag(name)

        return center[0], center[1], has_flag

    def add_elements(self, items):
        for name in items.keys():
            item = items[name]
            if name not in self._name2axitem.keys():
                if name == 'agent':
                    color = 'r'
                else:
                    color = 'b'

                lines_for_item = []
                for points in item.draw():
                    line, = self._ax.plot(points[0], points[1], c=color)
                    lines_for_item.append(line)

                self._name2axitem[name] = lines_for_item
            if name not in self._name2obj.keys():
                self._name2obj[name] = item

    def update(self, items):
        for name in items.keys():
            self._update_element(name, items[name])
        plt.pause(0.001)

    def update_abs(self, items):
        for name in items.keys():
            dtheta = self._name2obj[name]._theta - items[name][1]
            self._update_element(name, [items[name][0], dtheta])
        plt.pause(0.001)


    def get_observation(self, name: str, points_to_sample: KDTree, plot_lidar: bool = False) -> np.array:
        dists, sample_idx = points_to_sample.query(self._flag_center, 30, distance_upper_bound=1.0)
        sample_idx = sample_idx[sample_idx < points_to_sample.data.shape[0]]
        flag_points = points_to_sample.data[sample_idx]
        pc, pobs, pf = self._name2obj[name].observation(self._name2obj, points_to_sample, flag_points)

        if plot_lidar:
            list_of_lidar_points = []
            agent_center = self._name2obj[name]._center
            for idx, obs in enumerate(pc.T):
                if obs is not None:
                    list_of_lidar_points.append(util.get_points_between(agent_center, obs))

            if len(self._lidar_list) == 0:
                for point_pair in list_of_lidar_points:
                    self._lidar_list.append(self._ax.plot(point_pair[0], point_pair[1], c='r', ls='--', linewidth=0.4))
            else:
                for point_pair, lidar_ax in zip(list_of_lidar_points, self._lidar_list):
                    lidar_ax[0].set_data(point_pair)

        return pc, pobs, pf

    def _press(self, event):
        sys.stdout.flush()
        if event.key == 'w':
            self._update_element("agent", [0.1, 0])
        if event.key == 'a':
            self._update_element("agent", [0, np.deg2rad(30)])
        if event.key == 'd':
            self._update_element("agent", [0, np.deg2rad(-30)])

    def _update_element(self, name: str, args: List[float]):
        self._name2obj[name].move(args[0], args[1])

        for key in self._name2obj.keys():
            if key != name:
                if self.collide(self._name2obj[key].get_bounds(), self._name2obj[name].get_bounds())[0]:
                    self._name2obj[name].move(-args[0], -args[1])
                    break

        for idx, points in enumerate(self._name2obj[name].draw()):
            self._name2axitem[name][idx].set_data(points)

    @staticmethod
    def collide(bounds1: List[List[np.array]], bounds2: List[List[np.array]]) -> [bool, np.array, np.array]:
        to_return = [False, None, None]
        for side_from_obj1 in bounds1:
            for side_from_obj2 in bounds2:
                intersects, where, t = util.get_intersection(side_from_obj1, side_from_obj2)
                if intersects and (to_return[2] is None or to_return[2][0] > t[0]):
                    to_return = [True, where, t]

        return to_return


# sim = Simulation(flag_center=np.array([8.75, 1.5]))
# agent = Agent(center=np.array([1.5, 8.5]))
#
# enemy_base = Wall([[np.array([7.5, 0]), np.array([7.5, 1.0])],
#                    [np.array([7.5, 1.9]), np.array([7.5, 3.0])],
#                    [np.array([7.5, 3.0]), np.array([9.0, 3.0])]])
#
# home_base = Wall([[np.array([0.0, 7.5]), np.array([1.5, 7.5])],
#                   [np.array([2.25, 7.5]), np.array([2.5, 7.5])],
#                   [np.array([2.5, 7.5]), np.array([2.5, 8.0])],
#                   [np.array([2.5, 9.0]), np.array([2.5, 10.0])]])
#
# barrier_1 = Wall([[np.array([6.0, 6.0]), np.array([7.0, 8.0])]])
# barrier_2 = Wall([[np.array([2.0, 3.0]), np.array([5.0, 3.0])],
#                   [np.array([5.0, 3.0]), np.array([5.0, 1.5])]])
#
# objs = {"agent": agent, "enemy base": enemy_base, "home base": home_base, "barrier 1": barrier_1, "barrier 2": barrier_2}
# sim.add_elements(objs)
#
# while True:
#     commands = {}
#     sim.update(commands)
#     sim.get_observation('agent', plot_lidar=True)
