# complex_eye_analysis

## Introduction
This is the Python code related to the manuscript "ComplexEye - a multi lens array microscope for High-Throughput embedded immune cell migration analysis". Our code has been tested on Linux, but since Cellpose is also supported for Windows and Mac OS, our code should work on these platforms as well.

## Installation
We recommend following the instructions provided within the [Cellpose repo](https://github.com/MouseLand/cellpose) and then executing 
`
pip install aicsimageio matplotlib scikit-image
`.

Alternatively, install the provided Anaconda environment with
`
conda env create -f environment.yml
`.

The entire installation should not take more than a few minutes.
## Example
First, we need to segment our 2D grayscale movie of migrating neutrophils. For this we execute the script '01_tracking.py' (`python 01_tracking.py`), the generated segmentation can be checked in the segmentation folder. Afterwards the scripts '02_tracking.py' and '03_export_metrics.py' have to be executed to get the trajectories and to export the metrics as a .csv file to the results folder. '02_tracking.py' will automatically generate an overlay of the raw movie + the trajectories to verfiy the accuracy of our tracking, the .tiff file will be exported to the tracks folder. To generate polar plots, simply execute script 'get_polar_plot.py'.

Note: To analyze own movies, simply add them as .tiff files to the data/raw folder. But the model we used is a fine-tuned Cellpose model trained with movies with the settings specified in the manuscript. Performance with other data acquisition settings may vary, so if you want to train your own model [this](https://cellpose.readthedocs.io/en/latest/train.html) may be helpful. 