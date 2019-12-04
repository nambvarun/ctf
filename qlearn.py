import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sps
import scipy.interpolate as spi
import itertools
from functools import reduce
from math import sin, cos, radians

# The state action grid: all matrices are defined on this grid
class SAGrid:

    # resolution and extent of the state-action space:
    # start, stop(exclusive), step
    # xdef =  [0, 10, 1]    # pos/state - x
    # ydef =  [0, 10, 1]    # pos/state - y
    # flag =  [False, True] # state - flag?
    # rdef =  [0, 1, 0.25]  # action - range
    # thdef = [0, 360, 30]  # action - angle (deg)

    def __init__(self, xdef=[0,10,1], ydef=[0,10,1], rdef=[0,1,0.25], thdef=[0,360,30]):
        self.xdef = xdef
        self.ydef = ydef
        self.rdef = rdef
        self.thdef = thdef

        self.x = np.arange(*self.xdef)
        self.y = np.arange(*self.ydef)
        self.f = [False, True]
        self.r = np.arange(*self.rdef)
        self.th = np.arange(*self.thdef)
    
# this class keeps track of the estimated reward model based on the rewards/penalties for hitting the wall, retrieving the flag, etc.
class RewardModel:
    pos_thres = 0.9
    goal_reward = 100
    flag_reward = 100
    wall_penalty = -10

    def __init__(self, sagrid, ind_goal=[0,0]):
        # state-action grid object
        self.sagrid = sagrid
        g = self.sagrid

        # goal position
        self.ind_goal = ind_goal

        # possible flag positions (x,y) (an unnormalized distribution)
        self.flag_dist = np.ones((len(g.x), len(g.y)), dtype=np.bool)
        
        # the Map: a list of lines
        self.Map = []

        # the position-action collision matrix (x,y,r,th)
        self.C = np.zeros(tuple(map(len, (g.x, g.y, g.r, g.th))), dtype=np.uint8)


    # TODO: # complete this functions
    def updateFlagDist(self, curr_pos, flag_pos):
        # update self.flag_dist
        if flag_pos == None: # no flag found
            # self.flag_dist[things near curr_pos] = False
            pass
        else:
            # the flag is at flag_pos
            # self.flag_dist = False
            # self.flag_dist[indices near flag pos] = True
            pass

    def updateMap(self, line):
        self.Map.append(line)
        self.updateCollisionMatrix(line)

    def updateCollisionMatrix(self, line):
        f = lambda sa: self.collision(sa[0:2], sa[3:5], line)
        g = self.sagrid
        C = list(map(f, itertools.product(g.x, g.y, g.r, g.th)))
        C = np.reshape(C, tuple(map(len,(g.x, g.y, g.r, g.th))))
        self.C += C

    ##############################COLLISION SECTION############################
    # THIS SECTION determines whether some (s,a) pair collides with a line in the map
    # ccw and intersection are efficient fxns to detect intersection
    @staticmethod
    def intersection(A,B,C,D):
        def ccw(A,B,C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    # pol2cart transfroms rho and theta action values to delta x and delta y
    @staticmethod
    def pol2cart(rho, theta):
        x = rho * cos(radians(theta))
        y = rho * sin(radians(theta))
        return x, y

    # computes the next position given the current position and action
    def nextPos(self, pos, action):
        delx, dely = self.pol2cart(action[0], action[1])
        p1 = [pos[0], pos[1]]
        p2 = [p1[0] + delx, p1[1] + dely]
        return p2
    
    # collision returns a boolean if we collide with a line or not
    # TODO: "line" input should be x and y coordinates of endpoints (linep1 and linep2) of line from SLAM 
    def collision(self, pos, next_pos, line):
        return self.intersection(list(pos), next_pos, line[0], line[1])
    ###############################END COLLISION SECTION#######################
    
    def ind2pos(self, ind):
        Pos = self.sagrid.pos
        pos = (Pos[0][ind[0]], Pos[1][ind[1]])
        return pos

    # define the reward function for any arbitrary state/action given the flag/goal pos / Map (i.e. reward @ s, a)
    def reward(self, state, action):
        r = 0
        # add the goal reward
        if state[3]: # we have the flag; add if within distance to the goal
            r += self.goal_reward * \
                (la.norm(np.array(state[0:1]) - np.array(self.ind2pos(self.ind_goal))) \
                    < self.pos_thres)

        # add the flag position reward  
        else: # we don't have the flag: get the interpolated flag distribution
            flag_dist_interp = spi.RegularGridInterpolator(self.sagrid.pos, self.flag_dist)
            r += self.flag_reward * (flag_dist_interp(state[0:1]) / np.sum(self.flag_dist))

        # add collision penalty
        pos, next_pos = (state[0:2], self.nextPos(state[0:2], action))
        if any(self.collision(pos, next_pos, line) for line in self.Map):
            r -= self.wall_penalty

        return r

    def getRewardMatrix(self):
        g = self.sagrid
        sz = tuple(map(len, (g.x, g.y, g.f, g.r, g.th)))
        R = np.zeros(sz, dtype=np.single)
        
        # add the goal reward (pos = ind_goal, flag = True)
        R[self.ind_goal][1] += self.goal_reward

        # add the flag position reward
        R[:,:,0,:,:] += self.flag_reward * np.reshape((self.flag_dist / np.sum(self.flag_dist)), tuple(map(len, (g.x, g.y,[1],[1]))))

        # add the collision penalty
        R[:,:,0,:,:] -= self.wall_penalty * self.C
        R[:,:,1,:,:] -= self.wall_penalty * self.C

        # return the matrix 
        return R


# Q-learning object
class QLN:

    def __init__(self, reward_model=RewardModel(SAGrid())):
        self.reward_model = reward_model
        self.sagrid = reward_model.sagrid
        self.R = self.reward_model.getRewardMatrix()
        self.Q = np.zeros_like(self.R)

    def updateRewardModel(self, newlines, pos, flag_pos=None):
        self.reward_model.updateFlagDist(pos, flag_pos)
        [self.reward_model.updateMap(line) for line in newlines]
        self.R = self.reward_model.getRewardMatrix()
    
    # TODO: perform an update iteration on the Q matrix with the R matrix
    def iterateQUpdate(self):
        """
        Using self.R, self.Q, and self.reward_model.reward(s-prime, action), 
        update self.Q
        """

        """
        # continously update the Qmatrix
        gamma = 0.95             #discount factor
        alpha = 0.01             #learning rate
        r = Rmat[sa]             #reward from stateaction pair
        qp = 2 #interpolation from max_a of Q(s+a,a')
        for sa in sagrid.getStateActions():
            Q[sa] += alpha*(r + gamma*qp - Q[sa])
        """
    
        pass

    
    def policy(self, state):
        # the grid
        g = self.sagrid

        # create an interpolator
        Qinterp = spi.RegularGridInterpolator(g.StateActionRng(), self.Q)
        
        # for this set of actions ...
        A = (g.r, g.th)
        actions = itertools.product(A)

        # interpolate Q from the state
        Qs = np.reshape(np.array(list(map(lambda act: Qinterp(*state, *act), actions))), tuple(map(len, A)))

        # choose the best action from the set
        a_star = np.unravel_index(np.argmax(Qs), tuple(map(len, A)))
        
        return a_star




# define the action-utility functon for the Qmat (i.e. value of Q @ s, a)

# define the utility function (i.e. max of Q over a @ s)

# compute a test version of the problem
def main():
    # define the grid to operate on
    sagrid = SAGrid()
    
    # define the reward model (on the grid)
    reward_model = RewardModel(sagrid)

    # define the Q-learning object
    alg = QLN(reward_model)
    
    """ 
    Here is where we start computing stuff:
        at the same time we want to
        - call alg.updateRewardModel(newlines, pos[, flagpos]) whenever we observe something
        - call alg.policy(state=(x,y,f)) whenever we need to take a step
        - call alg.iterateQUpdate() always, in order to continuously converge

    """
    
    print("I'm Mr. Meseeks, look at me!")



if __name__ == "__main__":
    main()
