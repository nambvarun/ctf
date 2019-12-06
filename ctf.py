import numpy as np
import scipy as sp
from sim_simple import Simulation, Agent, Wall
import qlearn as q
import util


sim = Simulation(flag_center=np.array([8.75, 1.5]))
agent = Agent(center=np.array([1.5, 8.5]))

enemy_base = Wall([[np.array([7.5, 0]), np.array([7.5, 1.0])],
                   [np.array([7.5, 1.9]), np.array([7.5, 3.0])],
                   [np.array([7.5, 3.0]), np.array([9.0, 3.0])]])

home_base = Wall([[np.array([0.0, 7.5]), np.array([1.5, 7.5])],
                  [np.array([2.25, 7.5]), np.array([2.5, 7.5])],
                  [np.array([2.5, 7.5]), np.array([2.5, 8.0])],
                  [np.array([2.5, 9.0]), np.array([2.5, 10.0])]])

barrier_1 = Wall([[np.array([6.0, 6.0]), np.array([7.0, 8.0])]])
barrier_2 = Wall([[np.array([2.0, 3.0]), np.array([5.0, 3.0])],
                  [np.array([5.0, 3.0]), np.array([5.0, 1.5])]])

objs = {"agent": agent, "enemy base": enemy_base, "home base": home_base, "barrier 1": barrier_1, "barrier 2": barrier_2}
sim.add_elements(objs)

g = q.SAGrid(xdef=[0.5, 9.5, 0.5], ydef=[0.5, 9.5, 0.5])
alg = q.QLN(q.RewardModel(g))

# g.P: list of points for sampling the simulation.
print(g.P.shape)
# points_to_sample = sp.kd
# while True:
#     commands = {}
#     sim.update(commands)
#     sim.get_observation('agent', plot_lidar=True)
