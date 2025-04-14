# Groupwise image registration with edge-based loss for low-SNR cardiac MRI

This repo contains the official implementation for the paper Groupwise image registration with edge-based loss for low-SNR cardiac MRI.

## Overview

AiM-ED is designed to perform registration for single-shot cardiac images with low signal-to-noise ratio (SNR), especially at low field strengths. Our method:
- Jointly registers multiple noisy source images to a noisy target image
- Utilizes a noise-robust pre-trained edge detector to define the training loss
- Produces high-quality cardiac MR images from free-breathing acquisitions

## Setup

Clone the repository:
`git clone https://github.com/OSU-MR/aimed.git cd aimed`

Create and activate a conda environment:
`conda env create -f environment.yml -n aimed`


## Usage

To get started:

1. Navigate to the notebooks directory
2. Launch Jupyter:
`jupyter notebook`

3. Open the `demo.iypnb` to explore the implementation


## Structure

- `demo.ipynb`: Jupyter notebooks demonstration of the AiM-ED framework
- `data/`: Example data
- `utils/`: Utility functions and helper scripts
- `pre_trained_weights/`: Pre-trained weights for digital phantom and healthy subjects from scanner

## Method Validation

AiM-ED has been validated using:
- Synthetic late gadolinium enhanced (LGE) images from the MRXCAT phantom
- Free-breathing single-shot LGE images from healthy subjects and patients on 3T/1.5T scanners
- Clinical data from patients scanned on a 0.55T scanner

## Model Training
1. Preparing your own training data. (intensity correction with [SCC](https://github.com/OSU-MR/SCC) then normalization)
2. Save your datasets into .nii.gz files and add the name to predefined_dataset_idx.py file.
3. Train your own model using: 
`python trainer.py --device_number 0`
(don't forget to adjust the device number if you have more than one device)

>[!TIP]
Check the example datasets in niidata_c folder for more information (dimension, maximum magnitude, etc)


## Contact
For questions or issues regarding this repository, please contact: 
Xuan Lei  Email: lei.337{at}osu.edu


## Citation

@article{lei2024image,
  title={Image Registration with Averaging Network and Edge-Based Loss for Low-SNR Cardiac MRI},
  author={Lei, Xuan and Schniter, Philip and Chen, Chong and Ahmad, Rizwan},
  journal={arXiv preprint arXiv:2409.02348},
  year={2024}
}
