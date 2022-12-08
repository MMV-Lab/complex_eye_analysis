import zarr
import numpy as np
from aicsimageio import AICSImage
import os
from natsort import natsorted

data_path = './example/raw/'
files = natsorted(os.listdir(data_path))

if not os.path.exists('./example/zarrs/'):
    os.mkdir('./example/zarrs/')
for file in files:
    im_raw = AICSImage(data_path + file)
    img_raw = im_raw.get_image_data("ZYX")
    im_seg = AICSImage('./example/segmentation/'+ file)
    img_seg = im_seg.get_image_data("ZYX")
    tracking = np.load('./example/tracks/' + file.replace('.tiff','.npy'))
    root = zarr.open('./example/zarrs/' + file.replace('.tiff','.zarr'),mode='w')
    import pdb; pdb.set_trace()
    r1 = root.create_dataset('raw_data', shape = img_raw.shape, dtype = 'f8', data=img_raw) #(400,150,150), dtype = 'f8') # ZYX
    s1 = root.create_dataset('segmentation_data', shape = img_seg.shape, dtype = 'i4', data=img_seg)
    t1 = root.create_dataset('tracking_data', shape = tracking.shape, dtype = 'i4', data=tracking)