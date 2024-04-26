# Adapted from https://github.com/MouseLand/cellpose

# !nvcc --version
# !nvidia-smi

import os
import numpy as np
from cellpose import core, models
from glob import glob
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path


use_GPU = core.use_gpu()
yn = ["NO", "YES"]
print(f">>> GPU activated? {yn[use_GPU]}")

dir = Path('data','raw')
movies = dir.glob()

for movie in movies:
    well_name = movie.stem
    savedir = Path('data', 'segmentation')
    model_path = Path('models', 'model_neutrophils')
    chan = 0
    chan2 = 0
    flow_threshold = 0.4
    cellprob_threshold = 0

    reader_image_stack = AICSImage(movie)
    if reader_image_stack.dims.T > 1:
        image_stack = reader_image_stack.get_image_data("TYX")
    else: 
        image_stack = reader_image_stack.get_image_data("ZYX")
    images = []
    for i in range(image_stack.shape[0]):
        images.append(image_stack[i])

    # declare model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    # run model on test images
    masks, flows, styles = model.eval(
        images,
        channels=[chan, chan2],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    masks_3d = np.vstack(np.expand_dims(masks, axis=0))
    OmeTiffWriter.save(masks_3d, Path(savedir, well_name + ".tiff"), dim_order="TYX")
