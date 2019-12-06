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
    def __init__(self, xdef=[0,10,1], ydef=[0,10,1], rdef=[1e-3,1,0.5], thdef=[0,360,30]):
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

    # pol2cart transfroms rho and theta action values to delta x and delta y
    @staticmethod
    def pol2cart(rho, theta):
        x = rho * cos(radians(theta))
        y = rho * sin(radians(theta))
        return x, y

    # ccw and intersection are efficient fxns to detect intersection
    @staticmethod
    def intersection(A,B,C,D):
        def ccw(A,B,C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    

    # computes the position of the position index
    def ind2pos(self, ind):
        return self.x[ind[0]], self.y[ind[1]]


    
# this class keeps track of the estimated reward model based on the rewards/penalties for hitting the wall, retrieving the flag, etc.
class RewardModel:
    pos_thres = 0.9 # closeness threshold
    flag_pos = None # flag position
    goal_reward = 10 # reward for being at the goal
    flag_reward = 10 # reward for being at the flag
    wall_penalty = -5 # reward for colliding with a point in a region, per point

    # constructor for a reward model
    def __init__(self, g=SAGrid(), goal_ind=[0,0]):
        # state-action grid object
        self.sagrid = g

        # goal position (by index: use ind2pos for the position)
        self.goal_ind = goal_ind
        self.goal_pos = np.array(g.ind2pos(self.goal_ind))

        # possible flag positions (x,y) (an unnormalized distribution)
        self.flag_dist = np.ones((g.nx, g.ny), dtype=np.bool)
        self.flag_dist_interp = spi.RegularGridInterpolator((g.x, g.y), self.flag_dist,\
            method='linear', bounds_error=False, fill_value=0)
        
        # the position-action collision matrix (x,y,r,th) binary to save memory
        self.C = np.zeros((g.nx, g.ny, g.nr, g.nth), dtype=np.uint8)
        self.Cinterp = spi.RegularGridInterpolator((g.x, g.y, g.r, g.th), self.C,\
            method='linear', bounds_error=False, fill_value=np.Inf)
        
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
    # THIS SECTION determines whether some (s,a) pair collides with the map
    
    # TODO: collision returns a boolean if we collide with a line or not
    def collision(self, x, y, r, th, point_cloud):
        pass

    def getCollisionCount(self, point_cloud):
        # compute the collision function for all state-action pairs
        g = self.sagrid
        return np.reshape(np.array(list(map(\
            lambda sa: self.collision(*sa, point_cloud),\
                itertools.product(g.x, g.y, g.r, g.th))),dtype=np.uint8), self.C.shape)
    
    # code to update the collision matrix; it is stored in binary to save memory
    def updateCollisionMatrix(self, point_cloud):
        # compute whether the line collides for each state-action pair 
        self.C = np.maximum(self.C, self.getCollisionCount(point_cloud))
        self.Cinterp.values = self.C
    ############################## END COLLISION SECTION ######################
    
    ##############################  REWARDS ######################################################
    # define the reward function for any arbitrary state/action given the flag/goal pos / Map (i.e. reward @ s, a)
    def reward(self, state, action):
        # add the goal reward or flag position reward
        r = self.collisionReward(state[0:2], action) \
        + (self.goalReward(np.array(state[0:2])) if state[2] else self.flagReward(np.array(state[0:2])))
        return r
    
    # reward for distance to the goal
    def goalReward(self, pos):
        return self.goal_reward * (la.norm(pos - self.goal_pos) < self.pos_thres)

    # reward for distance to flag (known or unknown position)
    def flagReward(self, pos):
        return self.flag_reward * ( \
                self.flag_dist_interp(pos) / np.sum(self.flag_dist) \
                if self.flag_pos == None \
                else la.norm(pos - self.flag_pos) < self.pos_thres \
                    )

    # interpolated collision likelihood
    def collisionReward(self, pos, action):
        return self.wall_penalty * self.Cinterp((*pos, *action))
         

    # computes the sampled reward matrix defined on the grid
    def getRewardMatrix(self):
        # initialize the matrix
        g = self.sagrid
        R = np.empty((g.nx, g.ny, g.nf, g.nr, g.nth), dtype=np.single)

        # get R per state action
        R.flat = list(self.reward(sa[0:3], sa[3:5]) for sa in itertools.product(g.x, g.y, g.f, g.r, g.th))
        
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
        self.Qinterp = spi.RegularGridInterpolator((g.x, g.y, tuple(map(int, g.f)), g.r, g.th), self.Q,\
             method='linear', bounds_error=False, fill_value=-np.Inf)

    # update the reward model based on a new observation
    def updateRewardModel(self, point_cloud, pos, flag_pos=None):
        self.reward_model.updateFlagDist(pos, flag_pos) # update flag position
        self.reward_model.updateCollisionMatrix(point_cloud) # update Map
        self.R = self.reward_model.getRewardMatrix() # update R matrix
    
    # define the max utility kernel over all actions at the next state
    def QsPrimeMax(self, sa):
        g = self.sagrid
        p = np.array(sa[0:2]) + np.array(self.sagrid.pol2cart(*sa[3:5])) # next state, s-prime; ignores flag switching
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
