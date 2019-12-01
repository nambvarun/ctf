import numpy as np
import scipy as sp
import itertools

# The state action grid: all matrices are defined on this grid
class SAGrid:
    
    # resolution and extent of the state-action space:
    # start, stop(exclusive), step
    xdef = [0, 10, 1] # state - x: 
    ydef = [0, 10, 1] # state - y
    rdef = [0, 1, 0.25] # action - range
    thdef = [0, 360, 30] # action - angle (deg)

    
    def getNumActions(self):
        return len(list(np.arange(*self.rdef))) * len(list(np.arange(*self.thdef)]))        

    def getNumStates(self):
        return len(list(np.arange(*self.xdef))) * len(list(np.arange(*self.ydef)]))
    
    def getNumStateActions(self):
        return self.getNumActions() * self.getNumStates()
    
    def getStates(self):
        return itertools.product(np.arange(*self.xdef), np.arange(*self.ydef))

    def getActions(self):
        return itertools.product(np.arange(*self.rdef), np.arange(*self.thdef))
    



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
    Qmat = np.zeros(sagrid.getNumStateActions(), dtype=np.single) # sampled state action Q 
    Rmat = sp.sparse.coo_matrix(sagrid.getNumStates(), dtype=np.single) # sampled reward function
    Pf   = np.ones(sagrid.getNumStates(), dtype=np.bool) # scaled belief on the flag position
    Map = [] # lines in the map

    # define function for the reward of a state (action)
    

    # continously update the Qmatrix
    

if __name__ == "__main__":
    main()

