import numpy as np
from scipy.spatial import kdtree
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

g = q.SAGrid(xdef=[0.5, 9.5, 0.5], ydef=[0.5, 9.5, 0.5], rdef=[0.1,1.1,0.5])
alg = q.QLN(q.RewardModel(g))

# g.P: list of points for sampling the simulation.
# print(g.P.shape)
points_to_sample = kdtree.KDTree(g.P.T)
while True:
    commands = {}
    pc, pobs, pf = sim.get_observation('agent', points_to_sample, plot_lidar=True)
    if pobs is not None:
        pobs = pobs.reshape(g.nx, g.ny)
    # print(pc.shape)
    # print(pobs.shape)
    # print(pf)
    alg.updateRewardModel(pc, pobs, pf)

    i = 0
    dQ = np.finfo(np.float).max

    while i < 1000 and dQ > 0.1:
        dQ = alg.iterateQUpdate()
        i += 1

    ap = alg.policy(sim.get_state('agent'))
    # print(ap)
    # break
    commands['agent'] = ap
    print(ap)
    sim.update_abs(commands)
