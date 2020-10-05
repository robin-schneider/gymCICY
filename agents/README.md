# ChainerRL and A3C

Info: This is for an older version. It is easier to give the environment a filename to save the models inside.

To run the A3C agent in the agent directory you will need the [cicylist](http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt) in the same folder.
You can then run the agent from the terminal with, e.g.

```console
python A3Cagent.py --cicy=5256 --rank=2 --qmax=2
```

For a list of possible arguments and default hyperparameters check out the [A3C file](https://github.com/robin-schneider/gymCICY/blob/master/agents/A3Cagent.py).

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