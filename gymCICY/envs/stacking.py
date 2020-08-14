"""
Created on Wed Jun 05 11:12:20 2019

CICYlbmodels is an OpenAI gym environment to train 
agents finding 'realistic' string compactifications on
CICYs using sums of line bundles. 

This is the mother class for the stacking environments
from which the others inherit.
Here, the action space is greatly enlarged. At each step the
agent can pick form a precompiled list of line bundles and 
add a new one to V replacing an old one.

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
from gym import spaces
from gymCICY.envs.stack import create_stack
from gymCICY.envs.lb_model import lbmodel


logger = logging.getLogger(__name__)


class stacking(lbmodel):

    def __init__(self, M, r=2, max=5, pre=False, stacks=[None, None],
                    rewards={'fermion': 1e7, 'doublet': 1e6, 'triplet': 1e4, 'wstability': 2,
                                'index': 100, 'bianchi': 1e4, 'sun': 5, 'stability': 1e6, 'negative': True, 'wolfram': False}):
        super().__init__(M, r, max, rewards)

        # be careful the number of stacks grows with M.len (exponentially)
        # and still significantly with max and r
        if pre:
            self.stacks = stacks[0]
            self.istacks = stacks[1]
        else:
            self.stacks = create_stack(M, self.max, self.r)
            # index is faster determined from Leray
            self.istacks = np.array([np.round(self.M.line_co_euler(l)).astype(np.int16) for l in self.stacks])
        self.nstacks = len(self.stacks)
        self.action_space = spaces.Discrete(self.nstacks)

        #define V; include some randomness
        self.index = np.zeros(self.n_linebundles, dtype=np.int16)
        self.V = np.zeros((self.n_linebundles, self.M.len), dtype=np.int16)
        for i in range(self.n_linebundles):
            random = np.random.randint(self.nstacks)
            self.V[i] += self.stacks[random]
            self.index[i] += self.istacks[random]
        self.findex = np.sum(self.index)
    
    def _take_action(self, action):
        r"""Takes action, changes observation space, i.e.
        replaces the self.nsteps%self.n_linebundles-th line bundle
        with a new one.
        
        Parameters
        ----------
        action : int
            charge to be changed
        
        Returns
        -------
        None
            only changes self.V
        """
        #self.memory[self.nEpisode].append(action)
        # find new line bundle
        new_line = self.stacks[action]
        #determine which line bundle changes
        i = self.nsteps%self.n_linebundles
        # update index
        self.index[i] = self.istacks[action]
        # update line bundle
        self.V[i] = new_line
        self.findex = np.sum(self.index)

        return None

    def _wstability(self):
        r"""In stacking every line bundle is automatically wstable.
        
        Returns
        -------
        tuple(bool, float)
            (True, 0)
        """
        return True, 0

    def _index(self):
        r"""Determines if the index tells us that there are three 
        Fermion generations.
        
        All line bundle automatically satisfy the index.
        1) ind(V) =/= 3 \Gamma: np.amax([0.2*self.reward_index-abs(self.findex+3*self.r), 0])
        2) ind(V) = 3 \Gamma: self.reward_index

        Returns
        -------
        tuple(bool, float)
            (satisfied, reward)
        """
        if self.findex != (-3)*self.r:
            return False, np.amax([0.2*self.reward_index-abs(self.findex+3*self.r), 0])
        else:
            return True, self.reward_index
    
    def reset(self):
        r"""Resets the observation space to some
        random line bundle combination. Increases nEpisode.
        
        Returns
        -------
        int_array((5, h^11))
            charge matrix of the vector bundle.
        """
        # reset V
        for i in range(self.n_linebundles):
            random = np.random.randint(self.nstacks)
            self.V[i] = self.stacks[random]
            self.index[i] = self.istacks[random]
        self.findex = np.sum(self.index)
        # reset other variables
        self.found_sm = False
        self.nsteps = -1
        self.nEpisode += 1

        return self._get_state()