import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from metrics import filter_tracks, calculate_speed_single_cell, calculate_size_single_cell
from pathlib import Path
from tqdm import tqdm

movies = Path('./data/raw').glob('*')
Path('./results/single_cell_metrics').mkdir(parents=True, exist_ok=True)

### filter params
movement_threshold = 0                             # in sqrt(pix)
min_track_duration = 0                              # in frames, should be minimum of 2


### get metrics 
for movie in tqdm(movies):
    mean_speed = []
    std_speed = []
    track_ids = []
    track_duration = []
    mean_size = []
    std_size = []

    track_ids.append('track_id')
    track_duration.append('track_duration [# frames]')    
    mean_speed.append('speed_mean [px/frame]')
    std_speed.append('speed_std [px/frame]')
    mean_size.append('size_mean [px]')
    std_size.append('size_std [px]')

    tracks = np.load(Path('./data/tracks/', str(movie.stem) + '_trackslayer.npy')).astype(int)    
    reader_segmentation = AICSImage(Path('data', 'segmentation', movie.with_suffix('.tiff').name))
    if reader_segmentation.dims.T > 1:
        segmentation = reader_segmentation.get_image_data("TYX")
    else:
        segmentation = reader_segmentation.get_image_data("ZYX")
    
    filtered_tracks = filter_tracks(tracks, movement_threshold, min_track_duration)
    
    for unique_id in np.unique(filtered_tracks[:,0]):
        track_ids.append(unique_id)
        track = np.delete(filtered_tracks,np.where(filtered_tracks[:,0] != unique_id),0)        
        track_duration.append(len(track))
        
        speed = calculate_speed_single_cell(track)
        mean_speed.append(speed[0])
        std_speed.append(speed[1])

        size, _ = calculate_size_single_cell(track, segmentation)
        mean_size.append(size[0])
        std_size.append(size[1])

    ### export

    df = pd.DataFrame(zip(track_ids, track_duration, mean_speed, std_speed, mean_size, std_size))
    df.to_excel('./results/single_cell_metrics/' + movie.stem + '.xlsx', index=False, header=False)
