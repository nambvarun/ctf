import numpy as np
import scipy as sp

# The state action grid: all matrices are defined on this grid
class SAGrid:
    
    # resolution and extent of the state-action space:
    # start, resolution, stop(exclusive)
    xdef = [0, 1, 10]
    ydef = [0, 1, 10]
    rdef = [0, 0.25, 1]
    thdef = [0, 30, 360]

    def defSize(self, obdef):
        return ((obdef[2] - obdef[0]) / obdef[1]) - 1

    def getSSize(self):
        return (self.defSize(self.xdef), self.defSize(self.ydef))
    
    def getSASize(self):
        return (self.defSize(self.xdef), self.defSize(self.ydef), self.defSize(self.rdef), self.defSize(self.thdef))


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
    Qmat = np.zeros(sagrid.getSASize(), dtype=np.single) # sampled state action Q 
    Rmat = sp.sparse.coo_matrix(sagrid.getSSize(), dtype=np.single) # sampled reward function
    Pf   = np.ones(sagrid.getSASize(), dtype=np.bool) # scaled belief on the flag position
    Map = [] # lines in the map

    # define function for the reward of a state (action)
    

    # continously update the Qmatrix
    

if __name__ == "__main__":
    main()

