import numpy as np
import itertools
from scipy.interpolate import RegularGridInterpolator
from numpy.linalg import norm
from functools import reduce
from math import sin, cos, radians, degrees
from scipy.spatial import distance

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
    def __init__(self, xdef=[0,10,1], ydef=[0,10,1], rdef=[1e-3,1.1,0.5], thdef=[0,360,30]):
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

        # get 2 x P position array
        self.P = np.array(list((x,y) for (x,y) in itertools.product(self.x, self.y))).T

        # get 2 x A action array
        self.A = np.array(list((r,th) for (r,th) in itertools.product(self.r, self.th))).T


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

    # pol2cart transfroms rho and theta action values to delta x and delta y values
    # v must be a numpy array of any size with v[0] = rho, v[1] = theta
    @staticmethod
    def pol2cart(v):
        u = np.empty_like(v)
        u[0] = v[0] * cos(radians(v[1]))
        u[1] = v[0] * sin(radians(v[1]))
        return u

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
        self.goal_pos = np.reshape(np.array(g.ind2pos(self.goal_ind)), (2, -1))

        # possible flag positions (x,y) (an unnormalized distribution)
        self.flag_dist = np.ones((g.nx, g.ny), dtype=np.bool)
        self.flag_dist_interp = RegularGridInterpolator((g.x, g.y), self.flag_dist,\
            method='linear', bounds_error=False, fill_value=0)
        
        # the position-action collision matrix (x,y,r,th) binary to save memory
        self.C = np.zeros((g.nx, g.ny, g.nr, g.nth), dtype=np.uint8)
        self.Cinterp = RegularGridInterpolator((g.x, g.y, g.r, g.th), self.C,\
            method='linear', bounds_error=False, fill_value=np.Inf)
        
        self.dth = np.maximum(self.sagrid.rdef[2] / 2, 10)
        
    def updateFlagDist(self, obs_pnts, flag_pos):
        if flag_pos == None: # no flag found: rule out positions within distance "horizon"       
            self.flag_dist &= np.logical_not(obs_pnts)
        else: # the flag is at flag_pos
            self.flag_dist.flat = False
            self.flag_pos = np.reshape(flag_pos, (2, -1))

        # reset the interpolator
        self.flag_dist_interp.values = self.flag_dist

    ############################# COLLISION SECTION ############################
    # THIS SECTION determines whether some (s,a) pair collides with the map
    
    # TODO: collision returns a boolean if we collide with an object via regional count
    # PC is a 2 x N numpy array of the point cloud
    # P is a 2 x S numpy array of the state space
    # RTh is a 2 x A numpy array of the action space
    # returns an S x A array of counts
    def collision(self, P, RTh, PC):        
        PC = np.reshape(PC,     (2,-1,1,1)) # reshape to 2 x N x 1 x 1
        P = np.reshape(P,       (2,1,-1,1)) # reshape to 2 x 1 x S x 1
        RTh = np.reshape(RTh,   (2,1,1,-1)) # reshape to 2 x 1 x 1 x A
        m = PC - P # broadcasts to 2 x N x S x 1
        pcangle = (np.rad2deg(np.arctan2(m[1],m[0])) % 360) - RTh[1] # positive angle: broadcasts to N x S x A
        count = np.sum(np.logical_and(np.logical_and(norm(m, axis=0) < RTh[0], -self.dth < pcangle), pcangle < self.dth), axis=0, dtype=np.uint8) # count as S x A
        return count

    def getCollisionCount(self, point_cloud):
        # compute the collision function for all state-action pairs
        g = self.sagrid        
        return np.reshape(self.collision(g.P, g.A, point_cloud), self.C.shape)

    # code to update the collision matrix; it is stored in binary to save memory
    def updateCollisionMatrix(self, point_cloud):
        # compute whether the line collides for each state-action pair 
        self.C = np.maximum(self.C, self.getCollisionCount(point_cloud))
        self.Cinterp.values = self.C
    ############################## END COLLISION SECTION ######################
    
    ##############################  REWARDS ######################################################
    # reward for distance to the goal
    # P is a 2 x P numpy array
    # returns a P x _ numpy array
    def goalReward(self):
        return self.goal_reward * (norm(self.sagrid.P - self.goal_pos, axis=0) < self.pos_thres)

    # reward for distance to flag (known or unknown position)
    # P is a 2 x P numpy array
    # returns a P x _ numpy array
    def flagReward(self):
        return self.flag_reward * ( \
                self.flag_dist_interp(self.sagrid.P.T) / np.sum(self.flag_dist) \
                if self.flag_pos == None \
                else norm(self.sagrid.P.T - self.flag_pos, axis=0) < self.pos_thres \
                    )

    # computes the sampled reward matrix defined on the grid
    def getRewardMatrix(self):
        # initialize the matrix
        g = self.sagrid
        
        # get the flag rewards (doesn't care about action) X x Y x F x 1 x 1
        Rpf = np.reshape(self.flagReward(), (g.nx, g.ny, 1, 1, 1))
        Rpg = np.reshape(self.goalReward(), (g.nx, g.ny, 1, 1, 1))
        Rf = np.reshape(np.concatenate((Rpf, Rpg), axis=2), (g.nx, g.ny, g.nf, 1, 1))

        # get the collision rewards (doesn't care about flag) X x Y x 1 x R x TH
        Rc = np.reshape(self.wall_penalty * self.C, (g.nx, g.ny, 1, g.nr, g.nth))

        # add flag to collision and return the matrix 
        return Rc + Rf


# Q-learning object
class QLN:

    # static constants
    alpha = 0.1 # learning rate
    gamma = 0.95 # discount factor

    # constructor
    def __init__(self, reward_model=RewardModel(SAGrid())):
        self.reward_model = reward_model
        self.sagrid = reward_model.sagrid
        self.R = self.reward_model.getRewardMatrix()
        self.Q = np.zeros_like(self.R)
        g = self.sagrid
        self.Qinterp = RegularGridInterpolator((g.x, g.y, tuple(map(int, g.f)), g.r, g.th), self.Q,\
             method='linear', bounds_error=False, fill_value=-np.Inf)

    # define the max utility kernel over all actions at the next state
    # returns a S x 2 x A numpy array of the bellman next state expected utility
    def QsPrimeMax(self):
        g = self.sagrid
        
        # reshape to broadcastable shapes
        P = np.reshape(g.P, (2, -1, 1, 1)) # 2 x S x 1 x 1
        A = np.reshape(g.A, (2, 1, 1, -1)) # 2 x 1 x 1 x A

        # next state, s-prime; ignores flag switching
        Pp = P + g.pol2cart(A) # 2 x S x 1 x A

        # compute the best utility as the max over all subsequent actions
        Qmat = (lambda f: self.Qinterp((Pp[0], Pp[1], f, *a)) for a in itertools.product((g.r, g.th))) # S x 1 x A matrix generator (A copies)
        Qmax = [reduce(np.maximum, Qmat(f), -np.Inf) for f in [False, True]] # list of S x 1 x A maximum utility matrices
        Qmax = np.concatenate((Qmax[0], Qmax[1]), axis=2) # list of S x F x A maximum utility matrices
    
        return Qmax
    
    # perform an update iteration on the Q matrix 
    def iterateQUpdate(self):
        # max utility over all subsequent states    
        Qp = np.reshape(self.QsPrimeMax(), self.Q.shape)

        # update Q
        dQ = self.alpha * (self.R + self.gamma * Qp - self.Q)
        self.Q += dQ
        self.Qinterp.values = self.Q

        return norm(dQ.flatten())

    # update the reward model based on a new observation
    # point_cloud is a 2 x N numpy array of points
    # obs_pnts is an X x Y boolean numpy array of the observed points
    # flag_pos is a 2 x 1 numpy array if the flag is found or None
    def updateRewardModel(self, point_cloud, obs_pnts, flag_pos=None):
        self.reward_model.updateFlagDist(obs_pnts, flag_pos) # update flag position
        self.reward_model.updateCollisionMatrix(point_cloud) # update collision Map
        self.R = self.reward_model.getRewardMatrix() # update R matrix
    
    # return an optimal action based at the given state 
    # state is a tuple of (x, y, f)
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
    g = SAGrid()
    
    # define the reward model (on the grid)
    rm = RewardModel(g)

    # define the Q-learning object
    alg = QLN(rm)

    # define a test case point cloud
    pc = np.transpose(np.array([[4.5,y/10] for y in range(90)]))

    # define the observed positions as an X x Y boolean matrix
    obs_pnts = np.reshape(norm(g.P - [[5],[5]], axis=0) < 2, (g.nx, g.ny))

    # update the reward model
    alg.updateRewardModel(pc, obs_pnts)
    
    """ 
    Here is where we start computing stuff:
        at the same time we want to
        - call alg.updateRewardModel(point_cloud(2xN), points_obs(XxY)[, pos_flag(2x1)]) whenever we observe something
        - call alg.policy(state=(x,y,f)) whenever we need to take a step
        - call Qdiff = alg.iterateQUpdate() always, in order to continuously converge
    """
    
    print("I'm Mr. Meseeks, look at me!")



if __name__ == "__main__":
    main()
