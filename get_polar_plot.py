# Adapted from https://github.com/MMV-Lab/cell_movie_analysis

import numpy as np
import math
import matplotlib.pyplot as plt

import pdb
import os

movies = os.listdir('./data/tracks/')


for movie in movies:
    if not movie.endswith('dict.npy'):
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    #ax.set_ylim([0,600])

    traj = np.load('./data/tracks/' + movie, allow_pickle=True)
    lineage = traj[1]

    for _, single_trace in lineage.items():
        v_r = []
        v_theta = []

        try:
            for idx in np.arange(0, len(single_trace)):
                    v_r.append(math.dist(single_trace[0], single_trace[idx]))
                    v_theta.append(math.atan2(
                        single_trace[idx][0]-single_trace[0][0],
                        single_trace[idx][1]-single_trace[0][1]
                    ))
        except Exception:
            pass

        ax.plot(v_theta, v_r)
    plt.savefig('./results/polar_plots/' + movie.replace('_dict.npy','.png'), bbox_inches='tight')