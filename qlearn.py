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

    def getNumStates(self):
        return len(list(np.arange(*self.xdef))) * len(list(np.arange(*self.ydef))) * 2

    def getNumStateActions(self):
        return self.getNumActions() * self.getNumStates()

    def getStates(self):
        return itertools.product(np.arange(*self.xdef), np.arange(*self.ydef), [False, True])

    def getActions(self):
        return itertools.product(np.arange(*self.rdef), np.arange(*self.thdef))



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
        self.pos_flag = np.ones(sagrid.getNumStates(), dtype=np.bool)

    def reward(self, state, action):
        r = 0
        if state[3]: # we have the flag
            goal_dist = la.norm(state[1:2] - self.pos_goal)
            if goal_dist < self.pos_thres: # we have the flag and are at the goal
                r += self.goal_reward
        else: # we don't have the flag
            r += None
        
        return r



# define the action-utility functon for the Qmat (i.e. value of Q @ s, a)

# define the reward function for the Rmat given flag/goal pos and map (i.e. reward @ s, a)

# define the utility function (i.e. max of Q over a @ s)

# define the policy extraction function (i.e. max of Q over a @ s )

# define the flag position belief update given the current position and extent

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
