import copy
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import sys
from typing import List, Tuple
import util


class Agent:
    def __init__(self, center_x: float = 5, center_y: float = 5, dim_size: float = 0.3, theta: float = 1e-6):
        self._center = np.array([center_x, center_y])
        # self._dim = dim_size
        self._theta = theta
        self._dim_size = dim_size
        self._transform = np.array([[np.cos(self._theta), -np.sin(self._theta), self._center[0]],
                                    [np.sin(self._theta), np.cos(self._theta), self._center[1]],
                                    [0, 0, 1]])

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

    def observation(self):
        return None


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
    def __init__(self, min_bound: float = 0, max_bound: float = 10):
        plt.ion()
        self._fig = plt.figure(figsize=(8, 8))
        self._fig.canvas.mpl_connect('key_press_event', self._press)
        self._ax = self._fig.add_subplot(111)

        self._name2obj = {}
        self._name2axitem = {}

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

        bound = {"bounds": wall}

        self.add_elements(bound)

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

    def _press(self, event):
        sys.stdout.flush()
        if event.key == 'w':
            self._update_element("agent", [0.1, 0])
        if event.key == 'a':
            self._update_element("agent", [0, 0.1])
        if event.key == 'd':
            self._update_element("agent", [0, -0.1])

    def _update_element(self, name: str, args: List[float]):
        self._name2obj[name].move(args[0], args[1])

        for key in self._name2obj.keys():
            if key != name:
                if self.collide(self._name2obj[key].get_bounds(), self._name2obj[name].get_bounds()):
                    self._name2obj[name].move(-args[0], -args[1])
                    break

        for idx, points in enumerate(self._name2obj[name].draw()):
            self._name2axitem[name][idx].set_data(points)

    def collide(self, bounds1: List[List[np.array]], bounds2: List[List[np.array]]) -> bool:
        for side_from_obj1 in bounds1:
            for side_from_obj2 in bounds2:
                intersects, _ = util.get_intersection(side_from_obj1, side_from_obj2)
                if intersects:
                    return True

        return False

sim = Simulation()
agent = Agent()
# sim.update_element("agent", agent)
objs = {"agent": agent}
sim.add_elements(objs)

# enemy_base = Wall([[np.array([]), []]
#                    [[], []]])

while True:
    commands = {}
    sim.update(commands)
