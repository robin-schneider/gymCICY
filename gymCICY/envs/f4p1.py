"""
Created on Wed Jun 05 11:12:20 2019

CICYlbmodels is an OpenAI gym environment to train 
agents finding 'realistic' string compactifications on
CICYs using sums of line bundles. 

In this flipping environment the agent is only allowed
to change the charges of the first four line bundles.
The last one, then automatically adopts such that 
c_1(V) = 0 is always true.

self.sun = True.

Authors
-------
Robin Schneider (robin.schneider@physics.uu.se)

Version
-------
v0.0 - created

"""

# libraries
import numpy as np
from pyCICY import CICY
import gym
import logging
#from gym_CICYlbmodels.envs.stability import *
from gym import spaces
from gymCICY.envs.flipping import flipping


logger = logging.getLogger(__name__)


class f4p1(flipping):

    def __init__(self, M, r=2, max=5, 
                    rewards={'fermion': 1e7, 'doublet': 1e6, 'triplet': 1e4, 'wstability': 2,
                                'index': 100, 'bianchi': 1e4, 'sun': 5, 'stability': 1e6, 'negative': True, 'wolfram': False},
                    fname = '', max_steps = -1):
        super().__init__(M, r, max, rewards, fname, max_steps)

        #action space is one line bundles smaller
        self.n_linebundles = 5-1
        self.action_space = spaces.Discrete(self.n_linebundles*self.M.len*2)
        # define V and initialize
        self.V = np.random.randint(-1, 2, (self.n_linebundles+1, self.M.len)).astype(np.int16)
        self.V[self.n_linebundles] = -1*np.sum(self.V[0:self.n_linebundles], axis = 0)

    def _take_action(self, action):
        r"""Takes action, changes observation space, i.e.
        for action > dim(observation) reduces a charge by 1,
        for action < dim(observation) increases a charge by 1.

        If the action brings us out of the max bound, imposes 
        cyclic boundary condition, i.e
        q_max = 5: q_{0,0} = 5 and A = 0 -> q_{0,0} = -5.

        Further, adjusts the last line bundle so, as to always
        have c_1(V) = 0.
        
        Parameters
        ----------
        action : int
            charge to be changed
        
        Returns
        -------
        None
            only changes self.V
        """
        # find line bundle and charge
        sign = 1
        tmp = action - self.n_linebundles*self.M.len
        if tmp >= 0:
            line = int(tmp/self.M.len)
            charge = tmp%self.M.len
            sign = -1
        else:
            line = int(action/self.M.len)
            charge = action%self.M.len

        #increase by one step with cyclic bnds
        if self.V[line][charge] != self.max and sign == 1:
            self.V[line][charge] += sign
            self.V[self.n_linebundles][charge] -= sign
        elif self.V[line][charge] != -self.max and sign == -1:
            self.V[line][charge] += sign
            self.V[self.n_linebundles][charge] -= sign
        else:
            self.V[line][charge] = -1*sign*self.max
            self.V[self.n_linebundles][charge] += sign*2*self.max

        return None

    def _sun(self):
        r"""In this environment c_1(V) = 0.
        
        Returns
        -------
        tuple(bool, float)
            (True, 0)
        """
        return True,  0

    def reset(self):
        r"""Resets the observation space to some
        random charges q \in {-1,0,1} with c_1(V) = 0.
        Increases nEpisode.
        
        Returns
        -------
        int_array((5, h^11))
            charge matrix of the vector bundle.
        """
        # reset V
        self.V = np.random.randint(-1,2,(self.n_linebundles+1, self.M.len)).astype(np.int16)
        self.V[self.n_linebundles] = -1*np.sum(self.V[0:self.n_linebundles], axis = 0)
        # reset other variables
        self.found_sm = False
        self.nsteps = 0
        self.index = np.zeros(5)
        self.findex = 0
        self.nEpisode += 1

        return self._get_state()
