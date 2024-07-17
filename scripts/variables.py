import numpy as np

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 16]  # decoder features
]

device = {}

##########kernel size############


import torch
######################!!!!!!!!!!!!
device[0] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#####################!!!!!!!!!!!!!!!!!
device[1] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

loss_name_list = ['loss_l1','loss_l2','loss_ssim', 'loss_ncc', 'loss_wl', 'loss_LDC', 'loss_NCCFED', 'loss_NCCLDC', 'loss_canny']  #NCCFED == NCC+finite difference edge detector