import numpy as np

### define average speed
def calculate_speed(tracks):
    speeds = []
    for unique_id in tracks[:,0]:
        track = np.delete(tracks,np.where(tracks[:,0] != unique_id),0)
        for i in range(0,len(track)-1):
            speeds.append(np.hypot(track[i,2] - track[i+1,2],track[i,3] - track[i+1,3]))
    
    return np.around(np.average(speeds),3), np.around(np.std(speeds),3)



### calculate euclidean distances
def get_euclidean_distances(tracks): # calculates euclidean distances for all cells
    distances = []
    for unique_id in np.unique(tracks[:,0]):
        track = np.delete(tracks,np.where(tracks[:,0] != unique_id),0)
        x = track[-1,3] - track[0,3]
        y = track[0,2] - track[-1,2]
        euclidean_distance = np.sqrt(np.square(x) + np.square(y))
        distances.append(euclidean_distance)
    euclidean_distances = distances
    return np.squeeze(np.asarray([[np.unique(tracks[:,0])], [distances]])).T    


### calculate accumulated distances
def get_accumulated_distances(tracks):
    distances = []
    for unique_id in np.unique(tracks[:,0]):
        track = np.delete(tracks,np.where(tracks[:,0] != unique_id),0)
        distance = 0
        for i in range(0,len(track)-1):
            distance += (np.hypot(track[i,2] - track[i+1,2],track[i,3] - track[i+1,3]))
        distances.append(distance)
    return np.squeeze(np.asarray([[np.unique(tracks[:,0])], [distances]])).T    



### define accumulated distance
def calculate_accumulated_distance(tracks, each_cell=False):
    distances = []
    for unique_id in np.unique(tracks[:,0]):
        track = np.delete(tracks,np.where(tracks[:,0] != unique_id),0)
        distance = []
        for i in range(0,len(track)-1):
            distance.append(np.hypot(track[i,2] - track[i+1,2],track[i,3] - track[i+1,3]))
        accumulated_distance = np.sum(distance)
        distances.append(accumulated_distance)
    #accumulated_distances = distances
    if not each_cell:
        return np.around(np.average(distances),3), np.around(np.std(distances),3)
    else:
        return np.around(np.average(distances),3), np.around(np.std(distances),3), distances    



### define directed speed
def calculate_directed_speed(tracks):
    euclidean_distances = get_euclidean_distances(tracks)
    directed_speeds = []
    for i in range(len(euclidean_distances)):
        directed_speeds.append(euclidean_distances[i,1] / len(np.where(tracks[:,0] == euclidean_distances[i,0])[0]))  
    return np.around(np.average(directed_speeds),3), np.around(np.std(directed_speeds),3)        


### define euclidean distance
def calculate_euclidean_distance(tracks):
    distances = get_euclidean_distances(tracks)[:,1]
    return np.around(np.average(distances),3), np.around(np.std(distances),3)    


### define directness
def calculate_directness(tracks):
    euclidean_distances = get_euclidean_distances(tracks)
    _,_,accumulated_distances = calculate_accumulated_distance(tracks, each_cell=True)
    return np.around(np.average(euclidean_distances[:,1]/accumulated_distances[:]),3), np.around(np.std(euclidean_distances[:,1]/accumulated_distances[:]),3)


### calculate size of cells
def calculate_size(tracks, segmentation):
    sizes = []
    centroid_outside_cell = False
    for unique_id in np.unique(tracks[:,0]):
        track = np.delete(tracks,np.where(tracks[:,0] != unique_id),0)
        for i in range(0,len(track)-1):
            seg_id = segmentation[track[i,1],track[i,2],track[i,3]]
            if seg_id == 0:
                centroid_outside_cell = True
                continue
            sizes.append(len(np.where(segmentation[track[i,1]] == seg_id)[0]))
    #import pdb; pdb.set_trace()        
    return (np.around(np.average(sizes),3), np.around(np.std(sizes),3)), centroid_outside_cell

def calculate_euclidean_over_threshold(tracks, move_thresh):
    movement_mask = [] #np.asarray([])
    #movement_mask = np.ndarray((tracks.shape[0],2))
    for track_id in np.unique(tracks[:,0]):
        track = tracks[np.where(tracks[:,0]==track_id)]
        
        for frame_2 in range(1,len(track)):
            if get_euclidean_distances(track[0:frame_2])[1] > move_thresh:
                #np.append(movement_mask,track_id)
                movement_mask.append(track_id)
                break
    return movement_mask

def filter_tracks(tracks,move_thresh, min_track):
    duration_mask = np.unique(tracks[:,0])[np.where(np.unique(tracks[:,0],return_counts=True)[1] >= min_track)]
    #accumulated_distance = get_accumulated_distances(tracks)
    #movement_mask = np.ndarray.astype(accumulated_distance[np.where(accumulated_distance[:,1] >= move_thresh)][:,0],int)
    #euclidean_distance = get_euclidean_distances(tracks)
    #movement_mask = np.ndarray.astype(euclidean_distance[np.where(euclidean_distance[:,1] >= move_thresh)][:,0], int)
    movement_mask = calculate_euclidean_over_threshold(tracks, move_thresh)
    #import pdb; pdb.set_trace()
    combined_mask = np.intersect1d(movement_mask,duration_mask)
    tracks = np.asarray([tracks[i] for i in range(0,len(tracks)) if tracks[i,0] in combined_mask])
    return tracks    
