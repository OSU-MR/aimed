# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 16]  # decoder features
]


##########kernel size############

import torch
num_cuda_devices = torch.cuda.device_count()
device = [torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu") for i in range(num_cuda_devices)]

loss_name_list = ['loss_L1','loss_L2','loss_SSIM', 'loss_NCC', 'loss_WL', 'loss_LDC', 'loss_NCCFED', 'loss_NCCLDC', 'loss_CANNY']  #NCCFED == NCC+finite difference edge detector