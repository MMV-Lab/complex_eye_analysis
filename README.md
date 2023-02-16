# complex_eye_analysis

## Introduction
This is the Python code related to the manuscript "ComplexEye - a multi lens array microscope for High-Throughput embedded immune cell migration analysis".

## Installation
We recommend following the instructions provided within the [Cellpose repo](https://github.com/MouseLand/cellpose) and then executing 
`
pip install aicsimageio
`.

Alternatively, install the provided Anaconda environment with
`
conda env create -f environment.yml
`.

## Example tracking
First, we need to segment our example image. For this we execute the script '01_tracking.py' (`python 01_tracking.py`), the generated segmentation can be checked in the segmentation folder. Afterwards the scripts '02_tracking.py' and '03_export_metrics.py' have to be executed to export the metrics as a .csv file to the results folder. To generate polar plots, simply execute script 'get_polar_plot.py'.