# gymCICY

A set of OpenAI environments for studying heterotic line bundle models on CICYs based on the [pyCICY](https://github.com/robin-schneider/CICY/) toolkit. For an explanation of physical setting and the different environments check out our paper on [arxiv](https://arxiv.org/abs/2003.04817).

## Set up

Simply install with pip:

```console
pip install git+https://github.com/robin-schneider/gymCICY
```

## ChainerRL and A3C

To run the A3C agent in the agent directory you will need the [cicylist](http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt) in the same folder.
You can then run the agent from the terminal with, e.g.

```console
python A3Cagent.py --cicy=5256 --rank=2 --qmax=2
```

For a list of possible arguments and default hyperparameters check out the [A3C file](https://github.com/robin-schneider/gymCICY/agents/A3Cagent.py).

Unfortunately ChainerRL only provides a pretty dataset with min, max and mean rewards, entropy and some more details from the evaluation runs. In order to collect all found models during training and especially when they have been found, you will have to modify the chainer training loop. Go to your ChainerRL installation and modify the file *train_agent_async.py*.

You can probably find it somewhere here: *~/conda/envs/myEnv/lib/python3.X/site-packages/chainerrl/experiments/train_agent_async.py*.

Add to the top of the file the following class

```python
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
```

Now add to the training loop after

```python
if done or ... :
```

these lines

```python
if info:
    filename = os.path.join(outdir, 'model_info')
    info['gt'] = global_t
    with open(filename, 'a') as f:
        f.write(json.dumps(info, cls=NumpyEncoder)+'\n')
```

Alternatively you can also just replace the whole file with the one provided in the agent folder.

For a basic introduction to gym environments and in particular the flipping environment used in our paper check out the jupyter notebook [tutorial](https://github.com/robin-schneider/gymCICY/blob/master/agents/Tutorial.ipynb) in the agent folder.

Furthermore, you might want to use the latest pyCICY version. There is a bigger update planned which should cut down the cohomology computation time significanty. Update with

```console
pip install git+https://github.com/robin-schneider/CICY
```

## Environments

Currently there are five different environments:

1. **lbmodel** which is the mother class all other environments inherit from. The agent can increase a single charge of the five line bundle with cyclic boundary conditions.
2. **flipping** the agent can either decrease or increase a single charge in V.
3. **f4p1** the agent can either decrease or increase a single charge in the first four line bundles the fifth compensated the change such that c1(V) = 0. This is the flipping environment used in the paper.
4. **stacking** the agent picks a line bundle from a precompiled list of line bundles satisfying slope and index constraint replacing the one of t-5 ago.
5. **s4p1** similar to stacking with the difference being that the last line bundle is fixed by the condition c1(V) = 0. This is the stacking environment used in the paper.

## Future Updates

1. Currently gymCICY does not check for stability as we were unable to find a library solving the quadratic inequalities reliably enough. We hope to fix this problem in the future and are open for suggestions regarding potential libraries. For the moment we recommend using mathematicas powerful noptimize to solve the inequalities.
2. The pyCICY library can be fairly memory hungry, which sometimes results in configurations being skipped, as the reward can not be successfully computed. This happens particularly often for manifolds with increasing h11.

## References and Literature

We consider line bundle models as presented in [A Comprehensive Scan for Heterotic SU(5) GUT models](https://arxiv.org/abs/1307.4787v1) by Anderson et al.. A well written introduction to machine learning applications in string theory by F. Ruehle can be found [here](https://www.sciencedirect.com/science/article/pii/S0370157319303072). This project is inspired by the recent *Branes with Brains* [paper](https://arxiv.org/abs/1903.11616).
