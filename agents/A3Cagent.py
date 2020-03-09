"""An implementation of A3C.

This implementation is essentially copy pasted
together from the two chainer examples with some
minor modifications adjusted to our gym environment.

https://github.com/chainer/chainerrl/blob/master/examples/atari/a3c/train_a3c.py
https://github.com/chainer/chainerrl/blob/master/examples/ale/train_a3c_ale.py

The network architecture is inspired by the Branes with Brains paper.

Authors
-------
Robin Schneider (robin.schneider@physics.uu.se)

"""
#some chainer functions
import chainer
import chainerrl
import chainer.functions as F
from chainerrl.agents import a3c
from chainerrl import experiments
import chainer.links as L
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policy
from chainerrl import v_function

# for loading the CICY list
import ast as ast

import argparse
import logging

import multiprocessing as mp
import os

import gym
import gymCICY
from gymCICY.envs.stack import create_stack

import numpy as np
from pyCICY import CICY

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_hidden_channels):
        
        self.n_output_channels = n_hidden_channels+150
        self.n_input_channels = obs_size
        
        super().__init__()
        
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3 = L.Linear(n_hidden_channels, n_hidden_channels+150)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        #h = F.flatten(x)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        return F.relu(self.l3(h))

class A3Cagent(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_input, n_actions, n_hidden):

        self.head = QFunction(n_input, n_hidden)

        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)

def make_env(process_idx, test):
    
    process_seed = process_seeds[process_idx]
    env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
    env = gym.make('CICY-v0')
    env.seed(int(env_seed))
    return env

def load_CICY(number):
    # we parse through the cicylist
    cicys = []
    with open('cicylist.txt', 'r') as f:
        cicy = []
        for lines in f.readlines():
            if lines[0] == '{':
                cicy += [np.fromstring(lines[1:-1], dtype=int, sep=',')]
            if lines[0] == '\n':
                cicys += [np.array(cicy)]
                cicy = []
    # -1 since the list starts with 0
    M = cicys[number-1]
    # add the projective spaces in first col
    M = np.insert(M, 0, [sum(a)-1 for a in M], axis=1)
    # Create CICY object
    M = CICY(M)
    return M

def lr_setter(env, agent, value):
    agent.optimizer.lr = value


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # physics parameters
    parser.add_argument('--env', type=str, default='flip', 
                        help='Specify env. (flip, f4p1, stack, s4p1, s4p1r, normal)')
    parser.add_argument('--cicy', type=int, default=5302, help='CICY number')
    parser.add_argument('--rank', type=int, default=2, help='rank of freely acting symmetry')
    parser.add_argument('--qmax', type=int, default= 2,
                        help='maximal lb charge.')

    #general parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--stload', type=bool, default=False)

    # RL hyperparamerter/A3C
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--steps', type=int, default= 50000000)
    parser.add_argument('--stop', type=float, default= 1e12)

    # NN parameters
    parser.add_argument('--nHidden', type=int, default=100, 
                            help='# of hidden units in the first three layers last is nHidden+150')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')

    # TO DO: Implement different archictectures.
    parser.add_argument('--lr', type=float, default= 0.0005)
    parser.add_argument('--eval-interval', type=int, default= 50000)
    parser.add_argument('--eval-n-runs', type=int, default= 10)
    parser.add_argument('--max-episode-len', type=int, default = 200)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--gclipping', type=float, default=5.0)

    # Reward hyperparameters
    parser.add_argument('--nreward', type=int, default=1,
                         help='Allows for punishment of the agent, default = 1-True.')
    parser.add_argument('--r-fermion', type=int, default= 1e7)
    parser.add_argument('--r-doublet', type=int, default= 1e6)
    parser.add_argument('--r-triplet', type=int, default= 1e4)
    parser.add_argument('--r-wstability', type=int, default= 2)
    parser.add_argument('--r-index', type=int, default= 100)
    parser.add_argument('--r-bianchi', type=int, default= 1e4)
    parser.add_argument('--r-sun', type=int, default= 5)
    parser.add_argument('--r-stability', type=int, default= 1e6)

    args = parser.parse_args()

    if args.nreward == 0:
        nreward = False
    else:
        nreward = True

    rewards={'fermion': args.r_fermion, 'doublet': args.r_fermion,
             'triplet': args.r_triplet, 'wstability': args.r_wstability,
             'index': args.r_index, 'bianchi': args.r_bianchi,
             'sun': args.r_sun, 'stability': args.r_stability, 'negative': nreward}
    
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    logger = logging.getLogger('A3Cagent')
    logger.setLevel(args.logging_level)
    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    process_seeds = np.arange(args.threads) + args.seed * args.threads
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    logger.info('Output files are saved in {}'.format(args.outdir))
    logger.info('Setting up gym: {}'.format(args.env))
    logger.debug('Loading configuration matrix.')
    M = load_CICY(args.cicy)
    obs_size = 5*M.len

    # determine entrypoint + n_actions
    if args.env == 'flip':
        entry_point = 'gymCICY.envs.flipping:flipping'
        n_actions = 2*5*M.len
    elif args.env == 'f4p1':
        entry_point = 'gymCICY.envs.f4p1:f4p1'
        n_actions = 2*4*M.len
    elif args.env == 'stack' or args.env == 's4p1':
        logger.info('Creating stack.')
        stack = create_stack(M, args.qmax, args.rank)
        d = M.triple
        c = M.c2_tensor
        istack = np.array([np.round(M.line_co_euler(line)).astype(np.int16) for line in stack])
        if args.env == 'stack':
            entry_point = 'gymCICYl.envs.stacking:stacking'
        else:
            entry_point = 'gymCICY.envs.s4p1:s4p1'
        n_actions = len(stack)
    elif args.env == 'normal':
        entry_point='gymCICY.envs.lb_model:lbmodel'
        n_actions = obs_size
    else:
        logger.error('Do not recognize environment {}. Pick one of (flipping, f4p1, stacking, s4p1, normal).'.format(args.env))

    # register gym
    if args.env == 'stack' or args.env == 's4p1':
        gym.envs.register(
                    id='CICY-v0',
                    entry_point=entry_point,
                    kwargs={'M': M, 'r': args.rank, 'max': args.qmax, 'pre': True, 'stacks': [stack, istack], 'rewards': rewards},
            )
    else:
        gym.envs.register(
            id='CICY-v0',
            entry_point=entry_point,
            kwargs={'M': M, 'r': args.rank, 'max': args.qmax, 'rewards': rewards},
            )

    logger.info('Gym is set up.')

    # Define agents to be used
    model = A3Cagent(obs_size, n_actions, args.nHidden)

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(np.zeros(obs_size, dtype=np.float32)[None], name='observation')
    with chainerrl.recurrent.state_reset(model):
        # The state of the model is reset again after drawing the graph
        chainerrl.misc.draw_computational_graph(
                [model(fake_obs)],
                os.path.join(args.outdir, 'model'))

    opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=args.eps, alpha=args.alpha)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(args.gclipping))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    phi = lambda x: x.astype(np.float32, copy=False)

    agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=args.gamma,
                    beta=args.beta, phi=phi)

    lr_decay_hook = experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)

    training = experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.threads,
            make_env=make_env,
            profile=False,
            steps=args.steps,
            eval_interval=args.eval_interval,
            eval_n_episodes=args.eval_n_runs,
            max_episode_len=args.max_episode_len,
            successful_score=args.stop,
            global_step_hooks=[lr_decay_hook],
            save_best_so_far_agent=False,
            logger=logger,
        )
    
    print('done')
