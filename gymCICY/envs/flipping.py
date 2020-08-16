"""
Created on Wed Jun 05 11:12:20 2019

CICYlbmodels is an OpenAI gym environment to train 
agents finding 'realistic' string compactifications on
CICYs using sums of line bundles. 

This is the mother class for the flipping environments
from which the others inherit.
Here, the action space is twice as large and the agents 
can either increase or decrease a charge.

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
from gymCICY.envs.lb_model import lbmodel


logger = logging.getLogger(__name__)


class flipping(lbmodel):

    def __init__(self, M, r=2, max=5, 
                    rewards={'fermion': 1e7, 'doublet': 1e6, 'triplet': 1e4, 'wstability': 2,
                                'index': 100, 'bianchi': 1e4, 'sun': 5, 'stability': 1e6, 'negative': True, 'wolfram': False},
                    fname = ''):
        super().__init__(M, r, max, rewards, fname)

        #action space is twice as large: up and down
        self.action_space = spaces.Discrete(self.n_linebundles*self.M.len*2)

    
    def _take_action(self, action):
        r"""Takes action, changes observation space, i.e.
        for action > dim(observation) reduces a charge by 1,
        for action < dim(observation) increases a charge by 1.

        If the action brings us out of the max bound, imposes 
        cyclic boundary condition, i.e
        q_max = 5: q_{0,0} = 5 and A = 0 -> q_{0,0} = -5.
        
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
        elif self.V[line][charge] != -self.max and sign == -1:
            self.V[line][charge] += sign
        else:
            self.V[line][charge] = -1*sign*self.max

        return None
