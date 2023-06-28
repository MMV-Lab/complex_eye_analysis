# Adapted from https://github.com/MouseLand/cellpose

# !nvcc --version
# !nvidia-smi

import os
import numpy as np
from cellpose import core, models
from glob import glob
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter


use_GPU = core.use_gpu()
yn = ["NO", "YES"]
print(f">>> GPU activated? {yn[use_GPU]}")

dir = "./data/raw/"
movies = glob(dir + "*.tiff")

for movie in movies:
    well_name = os.path.basename(movie)[:-5]
    savedir = "./data/segmentation/"
    model_path = "./models/model_neutrophils"
    diameter = 15
    chan = 0
    chan2 = 0
    flow_threshold = 0.4
    cellprob_threshold = 0

    reader_image_stack = AICSImage(movie)
    image_stack = reader_image_stack.get_image_data("ZYX")
    if image_stack.shape[0] == 1:
        image_stack = reader_image_stack.get_image_data("TYX")    
    images = []
    for i in range(image_stack.shape[0]):
        images.append(image_stack[i])

    # declare model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    # use model diameter if user diameter is 0
    diameter = model.diam_labels if diameter == 0 else diameter

    # run model on test images
    masks, flows, styles = model.eval(
        images,
        channels=[chan, chan2],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    masks_3d = np.vstack(np.expand_dims(masks, axis=0))
    OmeTiffWriter.save(masks_3d, savedir + well_name + ".tiff", dim_order="ZYX")
