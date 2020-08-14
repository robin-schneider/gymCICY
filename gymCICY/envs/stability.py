from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from pyCICY import CICY
import numpy as np

def wolfram_stability(model, M):
    r"""Use the wolfram Noptimize to numerically compute
    if the given model is stable somewhere in the Kähler cone

    Parameters
    ----------
    model : np.array[5, M.len]
        sum of line bundles
    M : pyCICY.CICY
        CICY object

    Returns
    -------
    bool
        True if stable else False
    """
    session = WolframLanguageSession()#kernel_loglevel=logging.ERROR
    line_slope = []
    for line in model:
        tmp = M.line_slope()
        for i in range(M.len):
            tmp = tmp.subs({'m'+str(i): line[i]})
        line_slope += [tmp]
    str_slope = str(line_slope).replace('**', '^').replace('[','{').replace(']', '}')
    str_cone = str(['t'+str(i)+'> 1' for i in range(M.len)]).replace('\'', '').replace('[','{').replace(']', '}')
    str_vars = str(['t'+str(i) for i in range(M.len)]).replace('\'', '').replace('[','{').replace(']', '}')
    success = False
    full_string = 'NMinimize[Join[{{Plus @@ (#^2 & /@ {})}}, '.format(str_slope)
    full_string += '{}],'.format(str_cone)
    full_string += '{}, AccuracyGoal -> 20, PrecisionGoal -> 20, WorkingPrecision -> 20]'.format(str_vars)
    optimize = wlexpr(full_string)
    results = session.evaluate(optimize)
    if np.allclose([0.], [results[0].__float__()], atol=0.01):
        success = True
    session.terminate()
    return success
