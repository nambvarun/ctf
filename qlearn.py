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
        self.fdef = [False, True]
        self.rdef = rdef
        self.thdef = thdef

        self.x = self.xrng()
        self.y = self.yrng()
        self.f = self.frng()
        self.r = self.rrng()
        self.th = self.thrng()
        self.pos = (self.x, self.y)
        self.sa = (self.x, self.y, self.f, self.r, self.th)
    
    # axis in each dimension of the grid
    def xrng(self):
        return np.arange(*self.xdef)
    def yrng(self):
        return np.arange(*self.ydef)
    def frng(self):
        return [False, True]
    def rrng(self):
        return np.arange(*self.rdef)
    def thrng(self):
        return np.arange(*self.thdef)
    
    def PositionRng(self):
        return (self.x, self.y)
    def StateActionRng(self):
        return (self.x, self.y, self.f, self.r, self.th)

    # size of the set of Positions (x,y), States (x,y,f), Actions(r,th) or State-Actions (x,y,f,r,th)
    def SizeActions(self):
        return len(self.r), len(self.th)

    def SizePositions(self):
        return len(self.x), len(self.y)
    
    def SizeStates(self):
        return (*self.SizePositions(), len(self.f))

    def SizeStateActions(self):
        return (*self.SizeStates(), *self.SizeActions())

    # iterators
    def Positions(self):
        return itertools.product(self.x, self.y)
    
    def States(self):
        return itertools.product(self.x, self.y, self.f)

    def Actions(self):
        return itertools.product(self.r, self.th)

    def StateActions(self):
        return itertools.product(self.x, self.y, self.f, self.r, self.th)


# this class keeps track of the estimated reward model based on the rewards/penalties for hitting the wall, retrieving the flag, etc.
class RewardModel:
    pos_thres = 0.9
    goal_reward = 100
    flag_reward = 100
    wall_penalty = -10

    def __init__(self, sagrid, ind_goal=[0,0]):
        # state-action grid object
        self.sagrid = sagrid

        # goal position
        self.ind_goal = ind_goal

        # possible flag positions (x,y) (an unnormalized distribution)
        self.flag_dist = np.ones(sagrid.SizePositions(), dtype=np.bool)
        
        # the Map: a list of lines
        self.Map = []

        # the position-action collision matrix (x,y,r,th)
        self.C = np.zeros((*sagrid.SizePositions(), *sagrid.SizeActions()), dtype=np.uint8)


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
        f = lambda sa: self.collision(sa[0:3], sa[3:5], line)
        g = self.sagrid
        C = list(map(f, itertools.product(g.x, g.y, False, g.r, g.th)))
        C = np.reshape(C, (g.SizePositions(), g.SizeActions()))
        self.C += C

    ##############################COLLISION SECTION############################
    #THIS SECTION determines whether some (s,a) pair collides with a line in the map
    #ccw and intersection are efficient fxns to detect intersection
    def ccw(A,B,C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    def intersection(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    # pol2cart transfroms rho and theta action values to delta x and delta y
    def pol2cart(rho, theta):
        x = rho * cos(radians(theta))
        y = rho * sin(radians(theta))
        return x, y
    
    # collision returns a boolean if we collide with a line or not
    # TODO: "line" input should be x and y coordinates of endpoints (linep1 and linep2) of line from SLAM 
    def collision(self, state, action, line):
        p1 = [state[0],state[1]]
        delx,dely = pol2cart(action[0],action[1])
        p2 = [p1[0]+delx, p1[1]+dely]
        return intersection(p1,p2,linep1,linep2)
    ###############################END COLLISION SECTION#######################
    
    
    def ind2pos(self, ind):
        Pos = self.sagrid.pos
        pos = (Pos[0][ind[0]], Pos[1][ind[1]])
        return pos

    # define the reward function for any arbitrary state/action given the flag/goal pos / Map (i.e. reward @ s, a)
    def reward(self, state, action):
        r = 0
        # add the goal reward
        if state[3]: # we have the flag
            # add if within distance to the goal
            r += self.goal_reward * \
                (la.norm(np.array(state[0:1]) - np.array(self.ind2pos(self.ind_goal))) \
                    < self.pos_thres)

        # add the flag position reward  
        else: # we don't have the flag:
            # interpolated flag distribution
            flag_dist_interp = spi.RegularGridInterpolator(self.sagrid.pos, self.flag_dist)
            r += self.flag_reward * (flag_dist_interp(state[0:1]) / np.sum(self.flag_dist))

        # add collision penalty
        col = map(lambda l: self.collision(state, action, l), self.Map) # iterator
        if any(col): # short-circuit consumer
            r -= self.wall_penalty

        return r

    def getRewardMatrix(self):
        sz = self.sagrid.SizeStateActions()
        R = np.zeros(sz, dtype=np.single)
        
        # add the goal reward (pos = ind_goal, flag = True)
        R[self.ind_goal][1] += self.goal_reward

        # add the flag position reward
        R[:,:,0,:,:] += self.flag_reward * (self.flag_dist / np.sum(self.flag_dist))

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

        pass

    
    def policy(self, state):
        # the grid
        g = self.sagrid
        # create an interpolator
        Qinterp = spi.RegularGridInterpolator(g.StateActionRng(), self.Q)
        
        # for this set of actions ...
        actions = itertools.product(g.r, g.th)

        # interpolate Q from the state
        Qs = np.reshape(np.array(list(map(lambda act: Qinterp(*state, *act), actions))), (len(g.r), len(g.th)))

        # choose the best action
        a_star = np.unravel_index(np.argmax(Qs), (len(g.r), len(g.th)))
        
        return a_star

        


    
    

        

# define the action-utility functon for the Qmat (i.e. value of Q @ s, a)

# define the utility function (i.e. max of Q over a @ s)

# define the policy extraction function (i.e. max of Q over a @ s )

# compute a test version of the problem
def main():
    # define the grid to operate on
    sagrid = SAGrid()
    
    # rm = RewardModel()
    # [s for s in sagrid.getStates()]
    # [sa for sa in sagrid.getStateActions()]
    
    # define matrices for Q-learning and reward estimation
    Qmat = np.zeros(sagrid.NumStateActions(), dtype=np.single)  # sampled state action Q
    # Qinterp = spi.RegularGridInterpolator(sagrid.getStateActionRng(), Qmat)
    # Qinterp((x,y,f,r,th)) computes for the state
    
    # lines in the map
    Map = []  

    '''
    # continously update the Qmatrix
    gamma = 0.95             #discount factor
    alpha = 0.01             #learning rate
    r = Rmat[sa]             #reward from stateaction pair
    qp = 2 #interpolation from max_a of Q(s+a,a')
    for sa in sagrid.getStateActions():
        Q[sa] += alpha*(r + gamma*qp - Q[sa])
    '''
    
    print("I'm Mr. Meseeks, look at me!")



if __name__ == "__main__":
    main()
