import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sps
import scipy.interpolate as spi
import itertools

# The state action grid: all matrices are defined on this grid
class SAGrid:

    # resolution and extent of the state-action space:
    # start, stop(exclusive), step
    # xdef =  [0, 10, 1]  # state - x
    # ydef =  [0, 10, 1]  # state - y
    # flag =  [False, True] # state - flag?
    # rdef =  [0, 1, 0.25]  # action - range
    # thdef = [0, 360, 30]  # action - angle (deg)

    def __init__(self, xdef=[0,10,1], ydef=[0,10,1], rdef=[0,1,0.25], thdef=[0,360,30]):
        self.xdef = xdef
        self.ydef = ydef
        self.rdef = rdef
        self.thdef = thdef
    
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
    def getStateActionRng(self):
        return (self.xrng(), self.yrng(), self.frng(), self.rrng(), self.thrng())

    def getNumActions(self):
        return np.prod(self.getSizeActions())

    def getNumPositions(self):
        return np.prod(self.getSizePositions())
    
    def getNumStates(self):
        return 2*self.getNumPositions()

    def getNumStateActions(self):
        return self.getNumActions() * self.getNumStates()
    
    def getSizeActions(self):
        return (len(list(self.rdef())), len(list(self.thdef())))

    def getSizePositions(self):
        return (len(list(self.xrng())), len(list(self.yrng())))
    
    def getSizeStates(self):
        return (*self.getSizePositions(), 2)

    def getSizeStateActions(self):
        return (*self.getSizeStates(), *self.getSizeActions())

    def getPositions(self):
        return itertools.product(self.xrng(), self.yrng())
    
    def getStates(self):
        return itertools.product(self.xrng(), self.yrng(), [False, True])

    def getActions(self):
        return itertools.product(self.rrng(), self.thrng())

    def getStateActions(self):
        return itertools.product(self.xrng(), self.yrng(), [False, True], self.rrng(), self.thrng())

    def getIndexWeights(self, val, rngdef):
        (i0, f) = np.divmod(val, rngdef[-1])
        i0 = int(i0)
        return [i0, i0+1], [f, 1-f]

    # returns a tuple of the indices and corresponding weights for the 
    def getPositionIndexWeights(self, state):
        IxFx = self.getIndexWeights(state[1], self.xdef)
        IyFy = self.getIndexWeights(state[2], self.ydef)
        IpFp = [((ix, iy), fx*fy) for ((ix, fx), (iy, fy)) in itertools.product(IxFx, IyFy)]
        return IpFp

    def getStateIndexWeights(self, state):
        IxFx = self.getIndexWeights(state[1], self.xdef)
        IyFy = self.getIndexWeights(state[2], self.ydef)
        IsFs = [((ix, iy, int(state[3])), fx*fy) for ((ix, fx), (iy, fy)) in itertools.product(IxFx, IyFy)]
        return IsFs


class RewardModel:
    Map = []
    pos_thres = 0.9
    goal_reward = 100
    flag_reward = 100
    wall_penalty = -10

    def __init__(self, sagrid, pos_goal):
        # state-action grid object
        self.sagrid = sagrid

        # goal position
        self.pos_goal = pos_goal

        # possible flag positions
        self.flag_dist = np.ones(sagrid.getNumPositions(), dtype=np.bool)

    # TODO: # complete this functions
    def updateFlagDist(self, state, flag_pos):
        # update self.flag_dist
        if flag_pos == None: # no flag found
            # definitely no flag "near me"
            # self.flag_dist[things near state] = 0
            pass
        else:
            # the flag is at flag_pos
            # self.flag_dist = np.zeros(sagrid.getNumPositions(), dtype=np.bool)
            # self.flag_dist[index near flag pos] = 1
            pass

    def updateMap(self, Map):
        self.Map.append(Map)

    # TODO: # determine whether 
    def collision(self, state, action):
        pass

    # define the reward function for the Rmat given flag/goal pos and map (i.e. reward @ s, a)        
    def reward(self, state, action):
        r = 0
        if state[3]: # we have the flag
            goal_dist = la.norm(state[1:2] - self.pos_goal)
            if goal_dist < self.pos_thres: # we have the flag and are at the goal
                r += self.goal_reward
        else: # we don't have the flag: get the interpolated flag distribution reward 
            
            # index and weights of neighboring positions
            inds_vals = self.sagrid.getPositionIndexWeights(state[1:2])
            
            # interpolated flag distribution (vals are a prob. dist.)
            for (ind, val) in inds_vals:
                r += val * self.flag_dist[ind]
            
            r /= np.sum(self.flag_dist)

        if self.collision(state, action):
            r -= self.wall_penalty

        return r

    def getRewardMatrix(self):
        sz = self.sagrid.getSizeStateActions()
        R = np.empty(*sz)
        state_actions = itertools.product(*map(lambda N: iter(range(N)), sz))
        for sa in state_actions:
            state = sa[0:3]
            action = sa[3:5]
            R[sa] = self.reward(state, action)    



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
    Qmat = np.zeros(sagrid.getNumStateActions(), dtype=np.single)  # sampled state action Q
    Qinterp = spi.RegularGridInterpolator(sagrid.getStateActionRng(), Qmat)
    # Qinterp((x,y,f,r,th))
    
    # lines in the map
    Map = []  

    # continously update the Qmatrix
    print("I'm Mr. Meseeks, look at me!")


if __name__ == "__main__":
    main()
