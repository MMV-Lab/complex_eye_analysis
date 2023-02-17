import zarr
import numpy as np
from aicsimageio import AICSImage
import os
from natsort import natsorted

data_path = './data/raw/'
files = natsorted(os.listdir(data_path))

if not os.path.exists('./data/zarrs/'):
    os.mkdir('./data/zarrs/')

for file in files:
    reader_raw = AICSImage(data_path + file)
    img_raw = reader_raw.get_image_data("ZYX")
    reader_seg = AICSImage('./data/segmentation/'+ file)
    img_seg = reader_seg.get_image_data("ZYX")
    tracking = np.load('./data/tracks/' + file.replace('.tiff','_trackslayer.npy'))
    
    root = zarr.open('./data/zarrs/' + file.replace('.tiff','.zarr'),mode='w')
    r1 = root.create_dataset('raw_data', shape = img_raw.shape, dtype = 'f8', data=img_raw) #(400,150,150), dtype = 'f8') # ZYX
    s1 = root.create_dataset('segmentation_data', shape = img_seg.shape, dtype = 'i4', data=img_seg)
    t1 = root.create_dataset('tracking_data', shape = tracking.shape, dtype = 'i4', data=tracking)