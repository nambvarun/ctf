import numpy as np
import numpy.linalg as la
import scipy as sp
import itertools

# The state action grid: all matrices are defined on this grid
class SAGrid:

    # resolution and extent of the state-action space:
    # start, stop(exclusive), step
    # xdef =  [0, 10, 1]  # state - x
    # ydef =  [0, 10, 1]  # state - y
    # flag =  [true, false] # state - flag?
    # rdef =  [0, 1, 0.25]  # action - range
    # thdef = [0, 360, 30]  # action - angle (deg)

    def __init__(self, xdef=[0,10,1], ydef=[0,10,1], rdef=[0,1,0.25], thdef=[0,360,30]):
        self.xdef = xdef
        self.ydef = ydef
        self.rdef = rdef
        self.thdef = thdef

    def getNumActions(self):
        return len(list(np.arange(*self.rdef))) * len(list(np.arange(*self.thdef)))

    def getNumPositions(self):
        return len(list(np.arange(*self.xdef))) * len(list(np.arange(*self.ydef)))
    
    def getNumStates(self):
        return 2*self.getNumPositions()

    def getNumStateActions(self):
        return self.getNumActions() * self.getNumStates()

    def getPositions(self):
        return itertools.product(np.arange(*self.xdef), np.arange(*self.ydef))
    
    def getStates(self):
        return itertools.product(np.arange(*self.xdef), np.arange(*self.ydef), [False, True])

    def getActions(self):
        return itertools.product(np.arange(*self.rdef), np.arange(*self.thdef))

    def getIndexWeights(self, val, rngdef):
        (i0, f) = np.divmod(val, rngdef[-1])
        i0 = round(i0)
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
        IpFp = [((ix, iy, int(state[3])), fx*fy) for ((ix, fx), (iy, fy)) in itertools.product(IxFx, IyFy)]
        return IpFp



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
            None
        else:
            # the flag is at flag_pos
            # self.flag_dist = np.zeros(sagrid.getNumPositions(), dtype=np.bool)
            # self.flag_dist[index near flag pos] = 1
            None

        
    def reward(self, state, action):
        r = 0
        if state[3]: # we have the flag
            goal_dist = la.norm(state[1:2] - self.pos_goal)
            if goal_dist < self.pos_thres: # we have the flag and are at the goal
                r += self.goal_reward
        else: # we don't have the flag
                
        
        return r



# define the action-utility functon for the Qmat (i.e. value of Q @ s, a)

# define the reward function for the Rmat given flag/goal pos and map (i.e. reward @ s, a)

# define the utility function (i.e. max of Q over a @ s)

# define the policy extraction function (i.e. max of Q over a @ s )

# compute a test version of the problem
def main():
    # define the grid to operate on
    sagrid = SAGrid()

    # define matrices for Q-learning and reward estimation
    Qmat = np.zeros(sagrid.getNumStateActions(), dtype=np.single)  # sampled state action Q
    
    # sampled reward function
    Rmat = sp.sparse.coo_matrix(sagrid.getNumStates(), dtype=np.single)

    # scaled belief on the flag position
    Pf = np.ones(sagrid.getNumStates(), dtype=np.bool)
    
    # lines in the map
    Map = []  

    # define function for the reward of a state(/action)

    # continously update the Qmatrix


if __name__ == "__main__":
    main()
