# UNET_COLLISION_DETECTOR

Developed by Yunong Wang

This script is designed to automate the localization of partical collision over micro- / nano-electrodes in Array Micro-cell Method (AMCM) setup invented by Sasha E. Alden et al. (S. E. Alden, N. P. Siepser, J. A. Patterson, G. S. Jagdale, M. Choi, L. A. Baker, ChemElectroChem 2020, 7, 1084.) by applying an adapted U-Net deep learning model.

## Usage

Download the source code. Install Python 3.9 or above and run the "setup.py" to install dependencies.
Download UNET model parameters "fine_tunning_01062023_7.stat" from release and put under folder "model/training_model".

Use "collision_view.py" and "collision_batch.py" for data processing. Change the "data_path" in both "collision_view.py" (line 10) and "collision_batch.py" (line 51) to your own .csv data file, and "tar_path" in "collision_batch.py" (line 52) for the output file path.

The input data file should contain chronoamperometric signal traces in columns as a group of three. First column: time vector; Second column: potential vector; third column: current response vector. Please read the code in "collision_batch.py" (line 55 - line 69) to get a view of how the data is loaded.

Output from "collision_batch.py" includes a waveform.csv and peaks.csv. "_waveform.csv" contains traces of each event, and "_peaks.csv" contains time point, event intensity, averaged stair baseline and step current ratio. Please read code in "collision_batch.py" (line 110 - line 111) for more info.

Signals input for this model should have a current sampling rate of 5000 samples/second at least.

### Please at least cite the following papers if you use this script:

[1] Sasha E. Alden, Lingjie Zhang, Yunong Wang, Nickolay V. Lavrik, Scott N. Thorgaard, Lane A. Baker, High-Throughput Single-Entity Electrochemistry with Microelectrode Arrays, Analytical Chemistry, 10.1021/acs.analchem.4c01092, (2024).

[2] Ronneberger, O.;  Fischer, P.; Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation, Medical Image Computing and Computer-Assisted Intervention –MICCAI 2015, Cham, 2015//; Navab, N.;  Hornegger, J.;  Wells, W. M.; Frangi, A. F., Eds. Springer International Publishing: Cham, 2015; pp 234-241.