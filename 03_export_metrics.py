import numpy as np
import os
import pandas as pd
from aicsimageio import AICSImage
#from glob import glob
from metrics import filter_tracks, calculate_speed


movies = os.listdir('./data/segmentation/')

### microscop settings
microscopic_resolution = 1.56                       # sqrt(pix)/micrometer
frameperiod = 8                                     # seconds/frame

### params for (moving) cells
min_track_duration_to_be_considered_as_a_cell = 5   # in frames
movement_threshold = 12                             # in sqrt(pix)
min_track_duration = 8                              # in frames, should be minimum of 2

### convert units
frameperiod_minutes = frameperiod/60                # minutes/frame
pixel_edge_length_to_µm = 1/microscopic_resolution  # [µm]
pixel_edge_length_per_frame_in_µm_per_minute = (pixel_edge_length_to_µm/frameperiod_minutes)           # [µm/minute]

### init lists for export
wells = []
wells.append('well')
total_tracks = []
total_tracks.append('total_tracks')
total_cells = []
total_cells.append('total_cells')
moving_cells = []
moving_cells.append('moving_cells')
moving_cells_perc = []
moving_cells_perc.append('moving_cells_perc [%]')
mean_speed = []
mean_speed.append('speed_mean [µm/min]')
std_speed = []
std_speed.append('speed_std [µm/min]')
min_track_dur = []
min_track_dur.append('min_track_duration [s]')
movement_thresh = []
movement_thresh.append('movement_threshold [µm]')
min_track_to_be_cell = []
min_track_to_be_cell.append('track_threshold_for_cells [s]')
frame_periods = []
frame_periods.append('frame_period [s/frame]')
microscopic_resolutions = []
microscopic_resolutions.append('microscopic resolution [sqrt(pix)/µm]')

### get metrics 
for movie in movies:
    tracks = np.load('./data/tracks/' + movie.replace('.tiff', '') + '_trackslayer.npy').astype(int)    
    reader_segmentation = AICSImage('./data/segmentation/' + movie)
    segmentation = reader_segmentation.get_image_data("ZYX")
    if segmentation.shape[0] < 2:
        segmentation = reader_segmentation.get_image_data("TYX")    
    
    filtered_tracks = filter_tracks(tracks, movement_threshold, min_track_duration)
    min_track_dur.append(min_track_duration*frameperiod)
    movement_thresh.append(np.round(movement_threshold*pixel_edge_length_to_µm,2))
    
    speed = calculate_speed(filtered_tracks)
    mean_speed.append(speed[0]*pixel_edge_length_per_frame_in_µm_per_minute)
    std_speed.append(speed[1]*pixel_edge_length_per_frame_in_µm_per_minute)  

    amount_total_tracks = len(np.unique(tracks[:,0]))
    amount_total_cells = amount_total_tracks - sum(np.unique(tracks[:,0], return_counts=True)[1] < min_track_duration_to_be_considered_as_a_cell)
    amount_moving_cells = len(np.unique(filtered_tracks[:,0]))

    total_tracks.append(amount_total_tracks)
    total_cells.append(amount_total_cells)
    moving_cells.append(amount_moving_cells)
    moving_cells_perc.append(np.round(amount_moving_cells/amount_total_cells*100,3))
    
    wells.append(movie.replace('.npy', ''))    
    min_track_to_be_cell.append(min_track_duration_to_be_considered_as_a_cell*frameperiod)
    frame_periods.append(frameperiod)
    microscopic_resolutions.append(microscopic_resolution)

### export    
df = pd.DataFrame(zip(wells, total_tracks, total_cells, moving_cells, moving_cells_perc, mean_speed, std_speed, min_track_dur, movement_thresh, min_track_to_be_cell, microscopic_resolutions, frame_periods))
df.to_csv('./results/metrics.csv', index=False, header=False)

