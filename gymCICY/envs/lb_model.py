"""
Created on Wed Jun 05 11:12:20 2019

CICYlbmodels is an OpenAI gym environment to train 
agents finding 'realistic' string compactifications on
CICYs using sums of line bundles. 

This is the mother class from which the other envs inherit.
The agents can pick a charge to increase. 
The observation space has 'cyclic' boundary conditions.

Authors
-------
Robin Schneider (robin.schneider@physics.uu.se)

Version
-------
v0.1 - Some Major inheritance updates. 4.11.2019
v0.0 - created

"""

# libraries
import numpy as np
import itertools
import scipy as sp
from pyCICY import CICY
import gym
import logging
from gymCICY.envs.stability import *
from gym import spaces
from gym.utils import seeding
import traceback
import json
import os as os

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class lbmodel(gym.Env):

    def __init__(self, M, r=2, max=5, 
                    rewards={'fermion': 1e7, 'doublet': 1e6, 'triplet': 1e4, 'wstability': 2,
                                'index': 100, 'bianchi': 1e4, 'sun': 5, 'stability': 1e6, 'negative': True, 'wolfram': False},
                    fname = '', max_steps = -1):

        #define underlying CICY
        self.M = M
        self.c2 = M.second_chern()
        self.c2t = M.c2_tensor
        self.d = M.triple

        #define symmetry properties
        #rank
        self.r = r

        #set rewards/punishments for differnt satisfied conditions
        self.reward_fermion = rewards['fermion']
        self.reward_doublet = rewards['doublet']
        self.reward_triplet = rewards['triplet']
        #weak stability, i.e. each line bundle by itself
        self.reward_wstability = rewards['wstability']
        self.reward_stability = rewards['stability']
        self.wolfram = rewards['wolfram']
        self.reward_index = rewards['index']
        self.reward_bianchi = rewards['bianchi']
        #vanishing first chern, su(n) bundle
        self.reward_sun = rewards['sun']
        self.negative_reward = rewards['negative']

        # max range of integer charges
        self.max = max
        # number of line bundles in V
        self.n_linebundles = 5

        # action space
        # This one needs to be defined for each envs
        self.action_space = spaces.Discrete(self.n_linebundles*self.M.len)
        self.observation_space = spaces.Box(low=-self.max, high=self.max, 
                    shape=(self.n_linebundles, self.M.len), dtype=np.int16)

        #some properties of V; include some randomness
        self.V = np.zeros((self.n_linebundles, self.M.len), dtype=np.int16)
        # improve this by saving the poly and just substitute
        self.index = np.zeros(5)
        self.findex = 0
        self.seed()

        #some tracking
        self.nEpisode = -1
        self.max_steps = max_steps
        self.nsteps = 0#episode steps
        self.tsteps = 0#total steps
        self.found_sm = False
        self.fname = fname
        if self.fname != '':
            self.directory = os.path.dirname(self.fname)
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
        
    def step(self, action):
        r"""Performs a step in action space.
        Determines reward, checks if a SM has been found.

        Parameters
        ----------
        action : int
            integer specifying the charge to be changed
        
        Returns
        -------
        tuple(int_array[(observation_space)], float, bool, set)
            tuple of observation space, reward, episode termination condition
            and additional information passed to the agent interacting with
            the environment
        
        Raises
        ------
        RuntimeError
            if configuration is already a standard model.
        """
        if self.found_sm:
            raise RuntimeError("Episode is done; found a SM.")

        self.nsteps += 1
        self.tsteps += 1
        self._take_action(action)
        reward = self._get_reward()
        obs = self._get_state()

        info = {}
        done = False
        if self.found_sm:
            done = True
            info = {'V': self.V, 'nEpisode': self.nEpisode, 'nsteps': self.nsteps,
                     'tsteps': self.tsteps, 'id': self.id}
            if self.fname != '':
                with open(self.fname, 'a') as f:
                    f.write(json.dumps(info, cls=NumpyEncoder)+'\n')
        
        if self.max_steps > 0 and self.nsteps == self.max_steps:
            # add maximal step
            done = True

        if self.negative_reward:
            return obs, reward, done, info
        else:
            return obs, np.amax([reward,0]), done, info

    def _take_action(self, action):
        r"""Takes action, changes observation space, i.e.
        adds +1 to the corresponding charge.
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
        line = int(action/self.M.len)
        charge = action%self.M.len

        #increase by one step with cyclic bnds
        if self.V[line][charge] != self.max:
            self.V[line][charge] += 1
        else:
            self.V[line][charge] = -self.max

        return None

    def _get_reward(self):
        r"""Determines the reward for the current observation space.
        Checks a list of conditions in the following order:

        1) c_1 = 0
        2) \mu(L) = 0
        3) index(L) < 3 \Gamma
        4) index(V) = 3 \Gamma
        5) bianchi
        6) no higgs triplets
        7) at lest one higgs doublet
        8) exactly three fermion generations
        ( 9) Potentially full stability check ) 
        Each condition increases the reward by a set amount.
        
        Returns
        -------
        float
            Reward
        """
        reward = 0

        # we start with if it is an SU(N)bundle
        sun, rsun = self._sun()
        reward += rsun
        if not sun:
            return reward

        # then check, the weak stability constraint
        stab, rstab = self._wstability()
        reward += rstab
        if not stab:
            return reward

        # check index contains constraints 3+4
        index, rindex = self._index()
        reward += rindex
        if not index:
            return reward

        # and the bianchi identity
        if self._bianchi():
            reward += self.reward_bianchi
        else:
            return reward

        # we continue with finding triplets, which is still quick
        if self._index_triplet():
            reward += self.reward_triplet
        else:
            return reward

        # existence of some higgs doublets
        try:
            hd = self._higgs_doublet()
        except Exception as e:
            #most likely meomry error
            logger.warning('There has been a problem. Most likely in allocating sufficient memory.')
            logger.error(traceback.format_exc())
            logger.warning('The reward computation has terminated at Higgs doublets.')
            return reward
        if hd:
            reward += self.reward_doublet
        else:
            return reward

        # no anti generations
        try:
            tf = self._three_fermions()
        except Exception as e:
            #most likely meomry error
            logger.warning('There has been a problem. Most likely in allocating sufficient memory.')
            logger.error(traceback.format_exc())
            logger.warning('The reward computation has terminated at no anti generations.')
            return reward
        if tf:
            reward += self.reward_fermion
        else:
            return reward

        # full stability appears to be not working in python; 
        # maybe use a mathematica kernel
        if self.wolfram:
            if self._stability():
                reward += self.reward_stability
            else:
                return reward
        
        # if we made it till here we checked all conditions and found an sm
        #print('found a SM. Episode: '+str(self.nEpisode)+' and step '+str(self.nsteps))
        self.found_sm = True
        
        return reward

    def _wstability(self):
        r"""Checks if each line bundle in V is slope stable by itself. 
        The returned reward is variable.
        All are satisfied: self.reward_wstability
        Otherwise: -0.2*#not satisfied.
        
        Returns
        -------
        tuple(bool, float)
            (satisfied, reward)
        """
        #weak stability, i.e. is every line bundle slope stable by itself.
        count = 0
        for line in self.V:
            signs = np.einsum('ijk,i->jk', self.M.triple, line)
            signs = np.sign(signs+signs.T)
            if not (-1 in signs and 1 in signs):
                count += 1
        if count != 0:
            # we punish with 0.2 a point for every line bundle that doesn't satisfy
            return False, -0.2*count
        else:
            return True, self.reward_wstability

    def _stability(self):
        r"""Determines if V is slope stable.
        Uses external function.
        
        Returns
        -------
        bool
            True if satisfied
        """
        stable = False
        # need to solve numerically

        #stable = scipy_stability(self.V, self.M)
        #stable = nlopt_stability(self.V, self.M)
        stable = wolfram_stability(self.V, self.M)
        
        return stable

    def _three_fermions(self):
        r"""Determines if the model has exactly three fermion
        generations.

        Returns
        -------
        bool
            True if satisfied
        """
        for entry in self.V:
            h = np.array(self.M.line_co(entry)).astype(np.int16)
            if h[2] > 0 or h[0] > 0 or h[3] > 0 or h[1]%self.r != 0:
                # we found antifamilies/stability problems/no equivariant
                return False
        return True

    def _higgs_doublet(self):
        r"""Determines if at least one Higgs doublet exist.
        
        Returns
        -------
        bool
            True if satisfied
        """
        h = np.array([0 for i in range(4)], dtype=np.int16)

        for e1, e2 in itertools.combinations(self.V, 2):
            l = np.add(list(e1), list(e2))
            h = np.add(h, np.array(self.M.line_co(l)).astype(np.int16))

        if h[2] == 0:
            return False
        else:
            return True

    def _index_triplet(self):
        r"""Determines if there are no Higgs triplets.
        
        Returns
        -------
        bool
            True if satisfied.
        """
        # check that the index of two L_a x L_b is always smaller than zero.
        # Necessary condition to project out all Higgs triplets
        for e1, e2 in itertools.combinations(self.V, 2):
            l = np.add(list(e1), list(e2))
            # we round and convert to int otherwise floating point issues
            index = np.round(self.M.line_co_euler(l)).astype(np.int16)
            if index > 0:
                return False

        return True

    def _sun(self):
        r"""Determines if c_1(V) = 0.
        Reward is variable and punishes for a far away distance from
        a vanishing first Chern class.
        
        Returns
        -------
        tuple(bool, float)
            (satisfied, reward)
        """
        x = np.sum(self.V, axis=0)
        if np.all(x == [0 for i in range(self.M.len)]):
            return True, self.reward_sun
        else:
            return False,  -0.2*np.sum(abs(x))

    def _bianchi(self):
        r"""Checks anomaly cancellation, i.e. c_2(X) - c_2(V) >= 0.
        Also checks necessary stability i.e. c2(V) must not be negative in all entries.

        Returns
        -------
        bool
            True if satisfied
        """
        c2 = -1/2*np.einsum('rst,st -> r', self.M.triple, np.einsum('is,it->st', self.V, self.V))
        if np.all(np.subtract(self.M.second_chern(), c2) > 0) and not np.array_equal(np.sign(c2), np.zeros(len(c2)-1)):
            return True
        else:
            return False
    
    def _index(self):
        r"""Determines if the index tells us that there are three 
        Fermion generations.
        
        Rewards depend on how many index constraint are satisfied
        1) Some: satisfied/50*self.reward_index
        2) All: 0.1*self.reward_index
        3) full V: self.reward_index

        Returns
        -------
        tuple(bool, float)
            (satisfied, reward)
        """
        satisfied = 5
        for i in range(5):
            #round and convert to int because of floating point issues.
            self.index[i] = np.round(self.M.line_co_euler(self.V[i])).astype(np.int16)
            # check if index in range and divisible by rank for equivariant structure
            if self.index[i] > 0 or self.index[i] < (-3)*self.r or self.index[i]%self.r != 0:
                satisfied -= 1
        
        if satisfied != 5:
            return False, satisfied/50*self.reward_index

        # sum of index has to be == -3*self.r
        self.findex = np.sum(self.index)
        # index has to be <= 0
        if self.findex != (-3)*self.r:
            return False, 0.1*self.reward_index
        else:
            return True, self.reward_index

    def reset(self):
        r"""Resets the observation space to some
        random charge q \in {-1,0,1}. Increases nEpisode.
        
        Returns
        -------
        int_array((h^11,5,3))
            charge matrix of the vector bundle.
        """
        self.V = np.random.randint(-1,2,(self.n_linebundles, self.M.len)).astype(np.int16)
        self.found_sm = False
        self.nsteps = 0
        self.index = np.zeros(5)
        self.findex = 0
        self.nEpisode += 1

        return self._get_state()

    def _get_state(self):
        r"""returns observation space.
        
        Returns
        -------
        int_array((n_linebundles,h^[1,1]))
            charge matrix of the vector bundle.
        """
        return self.V

    def seed(self, seed=42):
        r"""Takes a seed and sets the seeds accordingly
        
        Parameters
        ----------
        seed : int, optional
            seed, by default 42
        
        Returns
        -------
        [int]
            seed
        """
        self.np_random, seed = seeding.np_random(seed)
        self.id = seed
        np.random.seed(seed)
        return [seed]


        