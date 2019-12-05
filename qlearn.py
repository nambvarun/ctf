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

    # defines/holds a grid that the matrices are defined on (for convenience)
    def __init__(self, xdef=[0,10,1], ydef=[0,10,1], rdef=[0.1,1,0.5], thdef=[0,360,30]):
        self.xdef = xdef
        self.ydef = ydef
        self.rdef = rdef
        self.thdef = thdef

        self.x = np.arange(*self.xdef)
        self.y = np.arange(*self.ydef)
        self.f = [False, True]
        self.r = np.arange(*self.rdef)
        self.th = np.arange(*self.thdef)

        self.nx, self.ny, self.nf, self.nr, self.nth = tuple(map(len, (self.x, self.y, self.f, self.r, self.th)))

    def getBorderLines(self):
        # border positions: half a step past the grid
        xl, xu = min(self.x) - self.xdef[-1]/2, max(self.x) + self.xdef[-1]/2
        yl, yu = min(self.y) - self.ydef[-1]/2, max(self.y) + self.ydef[-1]/2

        # the 4 corners
        p1 = [xl, yl]; p2 = [xu, yl]; p3 = [xu, yu]; p4 = [xl, yu]

        # the 4 lines
        l1 = [p1, p2]; l2 = [p2, p3]; l3 = [p3, p4]; l4 = [p4, p1]

        # return a list of lines
        return [l1, l2, l3, l4]


    
# this class keeps track of the estimated reward model based on the rewards/penalties for hitting the wall, retrieving the flag, etc.
class RewardModel:
    pos_thres = 0.9 # closeness threshold
    flag_pos = None # flag position
    goal_reward = 10 # reward for being at the goal
    flag_reward = 10 # reward for being at the flag
    wall_penalty = -100 # reward for colliding with the wall

    # constructor for a reward model
    def __init__(self, g=SAGrid(), goal_ind=[0,0]):
        # state-action grid object
        self.sagrid = g

        # goal position (by index: use ind2pos for the position)
        self.goal_ind = goal_ind
        self.goal_pos = np.array(self.ind2pos(self.goal_ind))

        # possible flag positions (x,y) (an unnormalized distribution)
        self.flag_dist = np.ones((g.nx, g.ny), dtype=np.bool)
        self.flag_dist_interp = spi.RegularGridInterpolator((g.x, g.y), self.flag_dist)
        
        # the position-action collision matrix (x,y,r,th) binary to save memory
        self.C = np.zeros((g.nx, g.ny, g.nr, g.nth), dtype=np.bool)
        
        # the Map: a list of lines
        self.Map = g.getBorderLines()
        [self.updateCollisionMatrix(line) for line in self.Map]

    # computes the position of the position index
    def ind2pos(self, ind):
        return self.sagrid.x[ind[0]], self.sagrid.y[ind[1]]

    def updateFlagDist(self, curr_pos, flag_pos, horizon):
        if flag_pos == None: # no flag found: rule out positions within distance "horizon"       
            self.flag_dist &= np.reshape(np.array(list( \
                la.norm(curr_pos - np.array([x,y])) >= horizon \
                for (x,y) in itertools.product(self.sagrid.x, self.sagrid.y) \
                ), dtype=np.bool), (self.sagrid.nx, self.sagrid.ny))
        else: # the flag is at flag_pos
            self.flag_dist.flat = False
            self.flag_pos = flag_pos

        # reset the interpolator
        self.flag_dist_interp.values = self.flag_dist

    ############################# COLLISION SECTION ############################
    # THIS SECTION determines whether some (s,a) pair collides with a line in the map
    # add more lines to the map
    def updateMap(self, line):
        self.Map.append(line) # add a line
        self.updateCollisionMatrix(line) # update the collision matrix

    # code to update the collision matrix; it is stored in binary to save memory
    def updateCollisionMatrix(self, line):
        # collision function per state-action pair
        f = lambda posact: self.collision(posact[0:2], self.nextPos(posact[0:2], posact[2:4]), line)

        # compute whether the line collides for each state-action pair
        g = self.sagrid
        self.C |= np.reshape(np.array(list(map(f, itertools.product(g.x, g.y, g.r, g.th))), dtype=np.bool), self.C.shape)

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
        delp = np.array(self.pol2cart(action[0], action[1]))
        return pos + delp
    
    # collision returns a boolean if we collide with a line or not
    def collision(self, pos, del_pos, line):
        return self.intersection(pos, pos + del_pos, line[0], line[1])
    ############################## END COLLISION SECTION ######################
    
    ##############################  REWARDS ######################################################
    # define the reward function for any arbitrary state/action given the flag/goal pos / Map (i.e. reward @ s, a)
    def reward(self, state, action):
        # add the goal reward or flag position reward
        r = self.collisionReward(state[0:2], state[3:5]) \
        + (self.goalReward(np.array(state[0:2])) if state[3] else self.flagReward(np.array(state[0:2])))
        return r
    
    def goalReward(self, pos):
        return self.goal_reward * (la.norm(pos - self.goal_pos) < self.pos_thres)

    def flagReward(self, pos):
        return self.flag_reward * ( \
                self.flag_dist_interp(pos) / np.sum(self.flag_dist) \
                if self.flag_pos == None \
                else la.norm(pos - self.flag_pos) < self.pos_thres \
                    )

    def collisionReward(self, pos, action):
        return self.wall_penalty * any(self.collision(pos, np.array(self.pol2cart(*action)), line) for line in self.Map)
         

    # computes the sampled reward matrix defined on the grid
    def getRewardMatrix(self):
        # initialize the matrix
        g = self.sagrid
        sz = (g.nx, g.ny, g.nf, g.nr, g.nth)
        R = np.zeros(sz, dtype=np.single)
        
        # add the goal reward (for all actions) (flag = True)
        R[:,:,1,:,:] += self.goal_reward * np.reshape( \
            list(map(self.goalReward, itertools.product(g.x,g.y))), \
                (g.nx, g.ny, 1, 1))

        # add the flag position reward (for all actions) (flag = False)
        R[:,:,0,:,:] += self.flag_reward * np.reshape( \
            list(map(self.flagReward, itertools.product(g.x, g.y))), \
                (g.nx, g.ny, 1, 1))

        # add the collision penalty
        R += self.wall_penalty * np.reshape(self.C,  (g.nx, g.ny, 1, g.nr, g.nth))

        # return the matrix 
        return R


# Q-learning object
class QLN:

    # static constants
    alpha = 0.01
    gamma = 0.95

    # constructor
    def __init__(self, reward_model=RewardModel(SAGrid())):
        self.reward_model = reward_model
        self.sagrid = reward_model.sagrid
        self.R = self.reward_model.getRewardMatrix()
        self.Q = np.zeros_like(self.R)
        g = self.sagrid
        self.Qinterp = spi.RegularGridInterpolator((g.x, g.y, tuple(map(int, g.f)), g.r, g.th), self.Q)

    # update the reward model based on a new observation
    def updateRewardModel(self, newlines, pos, flag_pos=None):
        self.reward_model.updateFlagDist(pos, flag_pos) # update flag position
        [self.reward_model.updateMap(line) for line in newlines] # update Map
        self.R = self.reward_model.getRewardMatrix() # update R matrix
    
    # define the max utility kernel over all actions at the next state
    def QsPrimeMax(self, sa):
        p = np.array(sa[0:2]) + np.array(self.reward_model.pol2cart(*sa[3:5])) # next state, s-prime; ignores flag handover
        q = np.max(list(self.Qinterp((*p, sa[3], *a)) for a in itertools.product((g.r, g.th)))) # max over all subsequent actions
        return q
    
    # perform an update iteration on the Q matrix 
    def iterateQUpdate(self):
        # max utility over all subsequent states    
        g = self.sagrid       
        Qp = np.empty_like(self.Q)
        Qp.flat = list(self.QsPrimeMax(sa) for sa in itertools.product((g.x, g.y, g.f, g.r, g.th)))
        
        # update Q
        self.Q += self.alpha * (self.R + self.gamma * Qp - self.Q)
        self.Qinterp.values = self.Q
   

    # return an optimal action based at the given state 
    def policy(self, state):
        # for this set of actions ...
        g = self.sagrid
        A = (g.r, g.th)
        actions = itertools.product(A)

        # interpolate Q from the state
        Qs = np.empty((g.nr, g.nth))
        Qs.flat = list(map(lambda action: self.Qinterp((*state, *action)), actions))

        # choose the best action from the set
        a_star = np.unravel_index(np.argmax(Qs), tuple(map(len, A)))
        
        return a_star



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
