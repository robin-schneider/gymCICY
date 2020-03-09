import numpy as np
import itertools


def create_stack(M, max, r):
    """Takes a CICY and determines all weak_stable
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

    all_line = itertools.product(range(-1*max, max+1), repeat=M.len)
    relevant = set()
    d = M.triple
    c = M.c2_tensor
    for line_bundle in all_line:
        line = list(line_bundle)
        signs = np.einsum('ijk,i->jk', M.triple, line)
        signs = np.sign(signs+signs.T)
        if -1 in signs and 1 in signs:
            index = _quick_index(line, d, c)
            if index <= 0 and index >= -3*r:
                relevant.add(tuple(line))

    return np.array(list(relevant))

def create_stack_h(M, max, r):
    """Takes a CICY and determines all weak_stable + hodge numbers
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

    all_line = itertools.product(range(-1*max, max+1), repeat=M.len)
    lines = []
    hodge = []
    indices = []
    d = M.triple
    c = M.c2_tensor
    for line_bundle in all_line:
        line = list(line_bundle)
        signs = np.einsum('ijk,i->jk', M.triple, line)
        signs = np.sign(signs+signs.T)
        if -1 in signs and 1 in signs:
            index = _quick_index(line, d, c)
            if index <= 0 and index >= -3*r:
                h = M.line_co(line)
                if h[0] == 0 and h[-1] == 0:
                    lines += [line]
                    indices += [index]
                    hodge += [h]
    
    l, idx = np.unique(np.array(lines), axis=0, return_index=True)
    h = np.array(hodge)[idx]
    i = np.array(indices)[idx]

    return l, h, i

def _quick_index(line, d, c):
    r"""Determines the index of a line bundle.
        
    Parameters
    ----------
    line : int_array(h^11)
        line bundle
    
    Returns
    -------
    int
        ind(L)
    """
    line_tensor = 1/6*np.einsum('i,j,k -> ijk', line, line, line)
    chern_tensor = 1/12*np.einsum('i, jk -> ijk', line, c)
    t = np.add(line_tensor, chern_tensor)
    return np.round(np.einsum('rst, rst', d, t)).astype(np.int16)
