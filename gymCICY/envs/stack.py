import numpy as np
import itertools

def create_stack(M, kmax, r):
    r"""Takes a CICY and determines all weak_stable
    line bundles with charges in range (-max,max) satisfying
    the index constraints.
    
    Parameters
    ----------
    M : pyCICY
        CICY to be considered
    max : int
        max charge
    r : int
        rank of freely acting symmetry
    
    Returns
    -------
    np.array((_,M.len))
        List of all line bundles satisfying these constraints.
    """
    all_lines = []
    for line_bundle in itertools.product(range(-1*kmax, kmax+1), repeat=M.len):
        line = list(line_bundle)
        if M.l_slope(line)[0]:
            index = np.round(M.line_co_euler(line))
            if index <= 0 and index >= -3*r and index%r == 0:
                h = M.line_co(line).astype(np.int)
                if h[0] == 0 and h[-1] == 0 and h[1]%r == 0:
                    all_lines += [line]
    all_lines = np.unique(all_lines, axis=-1)
    return all_lines
