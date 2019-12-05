import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import sys
from typing import List


class Agent:
    def __init__(self, center_x: float = 5, center_y: float = 5, dim_size: float = 0.3, theta: float = 0):
        self._center = np.array([center_x, center_y])
        self._theta = theta
        self._dim_size = dim_size
        self._transform = np.array([[np.cos(self._theta), -np.sin(self._theta), self._center[0]],
                              [np.sin(self._theta), np.cos(self._theta), self._center[1]],
                              [0, 0, 1]])

    def draw(self) -> List[np.array]:
        def get_points_between(point_a: List[float], point_b: List[float]) -> np.array:
            x_data = np.linspace(point_a[0], point_b[0], 100)
            y_data = np.linspace(point_a[1], point_b[1], 100)
            return np.vstack((x_data, y_data))

        top_corner = (self._transform @ np.array([0, self._dim_size/2, 1]))[:-1].flatten()
        bottom_left_corner = (self._transform @ np.array([-self._dim_size/3, -self._dim_size/2, 1]))[:-1].flatten()
        bottom_right_corner = (self._transform @ np.array([self._dim_size/3, -self._dim_size/2, 1]))[:-1].flatten()

        return [get_points_between(top_corner, bottom_left_corner),
                get_points_between(bottom_left_corner, bottom_right_corner),
                get_points_between(bottom_right_corner, top_corner)]

    def move(self, drho, dtheta):
        self._theta += dtheta
        self._center = self._center + (self._transform @ np.array([0, drho, 0]))[:-1]
        self._transform = np.array([[np.cos(self._theta), -np.sin(self._theta), self._center[0]],
                                    [np.sin(self._theta), np.cos(self._theta), self._center[1]],
                                    [0, 0, 1]])


class Simulation:
    def __init__(self, min_bound: float = 0, max_bound: float = 10):
        plt.ion()
        self._fig = plt.figure(figsize=(8, 8))
        self._fig.canvas.mpl_connect('key_press_event', self.press)
        self._ax = self._fig.add_subplot(111)

        # setting bounds of the viz
        plt.xlim(min_bound - 1, max_bound + 1)
        plt.ylim(min_bound - 1, max_bound + 1)

        self._min_bound = min_bound
        self._max_bound = max_bound

        # plotting the wall boundary
        bounds = [[(min_bound, min_bound), (min_bound, max_bound)],
                  [(min_bound, min_bound), (max_bound, min_bound)],
                  [(max_bound, max_bound), (max_bound, min_bound)],
                  [(max_bound, max_bound), (min_bound, max_bound)]]

        lc = mc.LineCollection(bounds, linewidths=1)
        self._ax.add_collection(lc)

        self._name2obj = {}
        self._name2axitem = {}

    def press(self, event):
        sys.stdout.flush()
        if event.key == 'w':
            self._update_element("agent", [0.1, 0])
        if event.key == 'a':
            self._update_element("agent", [0, 0.1])
        if event.key == 'd':
            self._update_element("agent", [0, -0.1])

    def add_elements(self, items):
        for name in items.keys():
            item = items[name]
            if name not in self._name2axitem.keys():
                lines_for_item = []
                for points in item.draw():
                    line, = self._ax.plot(points[0], points[1], c='r')
                    lines_for_item.append(line)

                self._name2axitem[name] = lines_for_item
            if name not in self._name2obj.keys():
                self._name2obj[name] = item

    def update(self, items):
        for name in items.keys():
            self._update_element(name, items[name])
        plt.pause(0.001)

    def _update_element(self, name: str, args: List[float]):
        self._name2obj[name].move(args[0], args[1])
        for idx, points in enumerate(self._name2obj[name].draw()):
            self._name2axitem[name][idx].set_data(points)


sim = Simulation()
agent = Agent(theta=0.1)
# sim.update_element("agent", agent)
objs = {"agent": agent}
sim.add_elements(objs)

while True:
    commands = {}
    sim.update(commands)
