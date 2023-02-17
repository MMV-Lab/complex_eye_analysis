# Adapted from https://github.com/MMV-Lab/cell_movie_analysis

import numpy as np
from matplotlib import cm

def random_colormap(nn: int = 65535):
    """generate a random colormap with nn different colors
    Parameter:
    ----------
    nn: int
        the number of random colors needed
    Example:
    ----------
    >>> import matplotlib.pyplot as plt
    >>> # img_label is output of a label function and represent all connected components
    >>> plt.imshow(img_label, cmap=random_colormap())
    """
    
    viridis = cm.get_cmap("viridis", nn)
    for ii in range(nn):
        for jj in range(3):
            viridis.colors[ii][jj] = np.random.rand()

    # always set first color index as black
    viridis.colors[0][0] = 0
    viridis.colors[0][1] = 0
    viridis.colors[0][2] = 0

    return viridis