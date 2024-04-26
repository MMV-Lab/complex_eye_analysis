# complex_eye_analysis

## Introduction
This is the Python code related to the manuscript [ComplexEye - a multi lens array microscope for High-Throughput embedded immune cell migration analysis](https://doi.org/10.1038/s41467-023-43765-3). Our code has only been tested on Ubuntu 20.04.5 LTS, but since Cellpose is also supported for Windows and Mac OS, our code should work on these platforms as well.

We will continuously update and optimize our code. To reproduce the results of the manuscript, we refer to the branch [paper_version](https://github.com/MMV-Lab/complex_eye_analysis/tree/paper_version).

## Installation
To reproduce the results, we recommend to install the provided Anaconda environment by executing:
~~~
conda env create -f environment.yml
~~~

Alternatively, you can follow the instructions provided within the [Cellpose repo](https://github.com/MouseLand/cellpose) (GPU support strongly recommended) and then install missing packages via: 
~~~
pip install aicsimageio matplotlib scikit-image
~~~

The entire installation should not take more than a few minutes.
## Example
First, we need to segment our 2D grayscale movie of migrating neutrophils stored in `data/raw/`. For this we execute the script `01_segmentation.py` by typing `python 01_segmentation.py`, the generated segmentation mask can be checked in `data/segmentation/`. This step can take up to several minutes per movie depending on the number of timestamps and the hardware, but without GPU support this step can take even longer. Afterwards the scripts `02_tracking.py` and `03_export_metrics.py` have to be executed to get the trajectories and to export the metrics as a .csv file to the results folder. `02_tracking.py` will automatically generate an overlay of the raw movie + the trajectories to verfiy the accuracy of our tracking, the .tiff file will be exported to `data/tracks/`. To generate polar plots, simply execute script `get_polar_plot.py`. Performing all steps with the provided sample movie should take no more than a few minutes in total.

Note: To analyze own movies, simply add them as .tiff files to `data/raw/`. But the model we use is a fine-tuned Cellpose model trained using movies with the settings specified in the manuscript. Performance with other data acquisition settings may vary, so if you plan to train your own model [this](https://cellpose.readthedocs.io/en/latest/train.html) may be helpful. 
