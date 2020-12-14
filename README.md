# gymCICY

A set of OpenAI environments for studying heterotic line bundle models on CICYs based on the [pyCICY](https://github.com/robin-schneider/CICY/) toolkit. For an explanation of physical setting and the different environments check out our paper on [arxiv](https://arxiv.org/abs/2003.04817).

## Set up

Simply install with pip:

```console
pip install git+https://github.com/robin-schneider/gymCICY
```

## Registering the environment

To register the environment use the following (here, Quintic and *f4p1*):

```python
import gym
import gymCICY
from pyCICY import CICY

M = CICY([[4,5]])
#rewards = {}
max_steps = 300
seed = 2020
env_id = 'CICY-v0'

gym.envs.register(
        id = env_id,
        entry_point = 'gymCICY.envs.f4p1:f4p1',
        kwargs={'M': M, 'r': 5, 'max': 2, #'rewards' : rewards,
                    'fname': os.path.join('results', 'models'), 'max_steps': max_steps},
        )
env = gym.make(env_id)
env.seed(seed)
```

Note that *fname* is the file name into which *env* will write all models it finds. *env* keeps track of episodes and steps and has a unique id corresponding to its seed.

For a basic introduction to gym environments and in particular the flipping environment used in our paper check out the jupyter notebook [tutorial](https://github.com/robin-schneider/gymCICY/blob/master/agents/Tutorial.ipynb) (Note: The notebook was for an older verison.) in the agent folder.

Furthermore, you might want to use the latest pyCICY version. Update with

```console
pip install --upgrade git+https://github.com/robin-schneider/CICY
```

## Environments

Currently there are five different environments:

1. **lbmodel** which is the mother class all other environments inherit from. The agent can increase a single charge of the five line bundle with cyclic boundary conditions.
2. **flipping** the agent can either decrease or increase a single charge in V.
3. **f4p1** the agent can either decrease or increase a single charge in the first four line bundles the fifth compensated the change such that c1(V) = 0. This is the flipping environment used in the paper.
4. **stacking** the agent picks a line bundle from a precompiled list of line bundles satisfying slope and index constraint replacing the one of t-5 ago.
5. **s4p1** similar to stacking with the difference being that the last line bundle is fixed by the condition c1(V) = 0. This is the stacking environment used in the paper.

## More Information

1. gymCICY can check for stability using the wolfram language. This can be activated by setting reward['wolfram'] = 1. Otherwise it uses a necessary check to check for stable sums.
2. Cohomology computations of the pyCICY library can fail due to a lack of available memory. This results in configurations being skipped, as the reward can not be successfully computed. It happens particularly often for manifolds with increasing h11.
3. Rewards can be disable by setting them to -1.

## References and Literature

We consider line bundle models as presented in [A Comprehensive Scan for Heterotic SU(5) GUT models](https://arxiv.org/abs/1307.4787v1) by Anderson et al.. A well written introduction to machine learning applications in string theory by F. Ruehle can be found [here](https://www.sciencedirect.com/science/article/pii/S0370157319303072). This project is inspired by the recent *Branes with Brains* [paper](https://arxiv.org/abs/1903.11616).
