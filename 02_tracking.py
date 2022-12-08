# ADAPTED FROM https://github.com/MMV-Lab/cell_movie_analysis/blob/main/run_tracking.py

import os
import numpy as np
from scipy import optimize, spatial, ndimage
from aicsimageio import AICSImage
import numpy as np
import os
import pandas as pd


# params
max_matching_dist = 45
approx_inf = 65535
track_display_length = 20
min_obj_size = 20


path_to_movies = './example/segmentation/'
save_path_tracks = './example/tracks/'
movies= os.listdir(path_to_movies)

for movie in movies:
    reader = AICSImage(path_to_movies+movie)
    img = reader.get_image_data("ZYX")
    if img.shape[0] < 2:
        img = reader.get_image_data("TYX")
    ##### paths ######
    well_name = movie[:-5]
    traj = dict()
    lineage = dict()
    ##### tracking loop ####
    total_time = img.shape[0]
    for tt in range(total_time):
        seg = img[tt,:,:]
        # get label image
        _, num_cells = ndimage.label(seg)
        seg_label = seg
        # calculate center of mass
        centroid = ndimage.center_of_mass(seg, labels=seg, index=np.arange(1, num_cells + 1))     # sometimes returns nans, TBD why ?? => propably fixed
        ## remove nans                                                                                   # this can probably be deleted
        # list_to_remove = []                  
        # for index in range(len(centroid)):
        #     if np.isnan(centroid[index][0]):
        #         list_to_remove.append(index)
        #         import pdb; pdb.set_trace()
        #         print("nan in frame " + str(tt) + " of movie " + well_name)
        # centroid = [i for j, i in enumerate(centroid) if j not in list_to_remove]
        # generate cell information of this frame
        traj.update({
            tt : {"centroid": centroid, "parent": [], "child": [], "ID": []}
        })
    # initialize trajectory ID, parent node, track pts for the first frame
    max_cell_id =  len(traj[0].get("centroid"))
    traj[0].update(
        {"ID": np.arange(0, max_cell_id, 1)}
    )
    traj[0].update(
        {"parent": -1 * np.ones(max_cell_id, dtype=int)}
    )
    centers = traj[0].get("centroid")
    pts = []
    for ii in range(max_cell_id):
        pts.append([centers[ii]])
        lineage.update({ii: [centers[ii]]})
    traj[0].update({"track_pts": pts})
    for tt in np.arange(1, total_time):
        p_prev = traj[tt-1].get("centroid")
        p_next = traj[tt].get("centroid")
        ###########################################################
        # simple LAP tracking
        ###########################################################
        num_cell_prev = len(p_prev)
        num_cell_next = len(p_next)

        # calculate distance between each pair of cells
        cost_mat = spatial.distance.cdist(p_prev, p_next)

        # if the distance is too far, change to approx. Inf.
        cost_mat[cost_mat > max_matching_dist] = approx_inf

        # add edges from cells in previous frame to auxillary vertices
        # in order to accomendate segmentation errors and leaving cells
        cost_mat_aug = max_matching_dist * 1.2 * np.ones(
            (num_cell_prev, num_cell_next + num_cell_prev), dtype=float
        )
        cost_mat_aug[:num_cell_prev, :num_cell_next] = cost_mat[:, :]

        # solve the optimization problem
        if sum(sum(1*np.isnan(cost_mat))) > 0:  # check if there is at least one np.nan in cost_mat
            print(well_name + ' terminated at frame ' + str(tt))
            break   
        row_ind, col_ind = optimize.linear_sum_assignment(cost_mat_aug)
        #########################################################
        # parse the matching result
        #########################################################
        prev_child = np.ones(num_cell_prev, dtype=int)
        next_parent = np.ones(num_cell_next, dtype=int)
        next_ID = np.zeros(num_cell_next, dtype=int)
        next_track_pts = []
        # assign child for cells in previous frame
        for ii in range(num_cell_prev):
            if col_ind[ii] >= num_cell_next:
                prev_child[ii] = -1
            else:
                prev_child[ii] = col_ind[ii]
        # assign parent for cells in next frame, update ID and track pts
        prev_pt = traj[tt-1].get("track_pts")
        prev_id = traj[tt-1].get("ID")
        for ii in range(num_cell_next):
            if ii in col_ind:
                # a matched cell is found
                next_parent[ii] = np.where(col_ind == ii)[0][0]
                next_ID[ii] = prev_id[next_parent[ii]]
                current_pts = prev_pt[next_parent[ii]].copy()
                current_pts.append(p_next[ii])
                if len(current_pts) > track_display_length:
                    current_pts.pop(0)
                next_track_pts.append(current_pts)
                # attach this point to the lineage
                single_lineage = lineage.get(next_ID[ii])
                try:
                    single_lineage.append(p_next[ii])
                except Exception:
                    pdb.set_trace()
                lineage.update({next_ID[ii]: single_lineage})
            else:
                # a new cell
                next_parent[ii] = -1
                next_ID[ii] = max_cell_id
                next_track_pts.append([p_next[ii]])
                lineage.update({max_cell_id: [p_next[ii]]})
                max_cell_id += 1
        # update record
        traj[tt-1].update({"child": prev_child})
        traj[tt].update({"parent": next_parent})
        traj[tt].update({"ID": next_ID})
        traj[tt].update({"track_pts": next_track_pts})
    import pdb; pdb.set_trace()    
    np.save(save_path_tracks + well_name + '_dict.npy', [traj, lineage])

    # get right format for napari tracks layer
    tracks_layer = np.round(np.asarray(traj[0]['centroid'][0])) 
    tracks_layer = np.append(tracks_layer, [0])
    tracks_layer = np.append(tracks_layer, [traj[0]['ID'][0]])
    tracks_layer=tracks_layer[[3,2,0,1]]


    tracks_layer = np.expand_dims(tracks_layer, axis=1)
    tracks_layer = tracks_layer.T
    

    for i in range(len(traj[0]['ID'])-1):
        track = np.round(np.asarray(traj[0]['centroid'][i+1]))
        track = np.append(track, [0])
        track = np.append(track, [traj[0]['ID'][i+1]])
        track = track[[3,2,0,1]]
        track = np.expand_dims(track, axis=1)
        track = track.T
        tracks_layer = np.concatenate((tracks_layer, track), axis=0)


    for i in range(len(traj)-1):                                                    # all images    
        for cell_ID in range(len(traj[i+1]['ID'])):
            track = np.round(np.asarray(traj[i+1]['centroid'][cell_ID]))            # centroid
            track = np.append(track, [i+1])                                         # frame
            track = np.append(track, [traj[i+1]['ID'][cell_ID]])                    # ID
            track = track[[3,2,0,1]]
            track = np.expand_dims(track, axis=1)
            track = track.T
            tracks_layer = np.concatenate((tracks_layer, track), axis=0)   
    
    df = pd.DataFrame(tracks_layer, columns=['ID', 'Z', 'Y', 'X'])
    df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
    tracks_formated = df.values
    np.save(save_path_tracks + well_name + '.npy', tracks_formated)