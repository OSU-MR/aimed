import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from skimage.metrics import structural_similarity as ssim

# from DISTS_pytorch import DISTS
from batchgenerators.utilities.file_and_folder_operations import *

####
from scripts.helpful_functions import *
from scripts.nn_modules import *
from scripts.variables import *
from scripts.loss import *
from scripts.data_loader import load_data
from scripts.utils import *
from scripts.generator import create_generators

import torch.optim as optim
from tqdm import tqdm

class Dataset_Args:
    def __init__(self):
        self.dataset_shape = "256by192"
        self.view_list = ['SAX']
        self.suffix = '_c'




def training(data_train, data_val,data_test,
             path2save_weights, path2save_loss , 
             dataset_train_idx = None,
             path2weights=None, para_lists = None, 
             device_number = None,save_result = False):

    assert device_number is not None #training
    assert para_lists is not None #training

    int_steps = 0


    for params in para_lists:
        p_standard_deviation = params.get('standard_deviation', 0.0)  # Provide a default value if key not found
        p_lambda = params.get('lambda_value', 0.0)
        p_model = params.get('model', 'default_model')
        p_loss = params.get('loss', 'default_loss')
        p_mode = params.get('mode', 'default_mode')
        p_batch_size = params.get('batch_size', 1)
        p_learning_rate = params.get('learning_rate', 0.001)
        p_epochs = params.get('epochs', 1)
        p_SGDmomentum = params.get('SGDmomentum', 0.9)
        p_notes = params.get('notes', '')

        print(params)
        training_losses, field_regularization , D, lpips , LDC_loss_val= get_losses(device_number)

        
        if ('ncc' in p_loss or 'NCC' in p_loss) and any(n < 0 for n in dataset_train_idx):
            #adding noise for digital patients (ncc only, since ncc loss doesn work well with data without testures)
            _, _, dataset, dataset_val, _, _ = add_noise_of_certain_SNR(data_train, data_val, SNR = 50)
            _, _, dataset_with_noise, dataset_with_noise_val, _,_ = add_noise_of_certain_SNR(data_train, data_val, mimic_sd = p_standard_deviation)
        else:
            #for other patients
            dataset, dataset_val, dataset_with_noise, dataset_with_noise_val = add_noise_of_certain_SNR(data_train, data_val,mimic_sd = p_standard_deviation)
            
        #verifiying on 6 dB
        _, _, dataset_with_noise_xdB, dataset_with_noise_val_xdB = add_noise_of_certain_SNR(data_train, data_val, SNR = 6)
        #testing on 6 dB
        dataset_tst, _, dataset_with_noise_xdB_tst, _  = add_noise_of_certain_SNR(data_test, data_test, SNR = 6)
        
        model = create_model(p_model, device_number, mode2train= p_mode , int_steps = int_steps, pre_trained_weight_path = path2weights, vol_shape=vol_shape).to(device[device_number])#.eval()
        
        #base_optimizer = torch.optim.Adam(model.parameters(), lr=p_learning_rate)#1e-4)
        base_optimizer = optim.SGD(model.parameters(), lr=p_learning_rate, momentum= p_SGDmomentum )#0.9)#1e-4)
        #annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max= 100, eta_min=0, last_epoch=-1)


        losses, weights = losses_weights(model_name=p_model,
                                        reg_lambda=p_lambda,
                                        training_losses=training_losses[loss_name_list.index(p_loss)],
                                        training_losses2=field_regularization)

        #generator is for training, generator_val is for validation, generator_val_training is for recording the training losses, generator_test is for testing
        generator, generator_val, generator_val_training, generator_test  = create_generators(p_model, 
                                                                                             dataset,                dataset_val, 
                                                                                             dataset_with_noise,     dataset_with_noise_val,
                                                                                             dataset_with_noise_xdB, dataset_with_noise_val_xdB, 
                                                                                             dataset_tst,            dataset_with_noise_xdB_tst,
                                                                                             batch_size= p_batch_size)

    ###########################################################################
        losses2record = {}

        for epoch in tqdm(range(p_epochs), unit = "epoch", ncols=80):
            
            ############################################# 
            model.train()
            for inputs, y_true in tqdm(generator, unit = "batch", ncols=80, leave = False):

                # generate inputs (and true outputs) and convert them to tensors
                inputs = [torch.from_numpy(d).to(device[device_number]).float() for d in inputs]
                y_true = [torch.from_numpy(d).to(device[device_number]).float() for d in y_true]

                base_optimizer.zero_grad()

                # run inputs through the model to produce a warped image and flow field
                y_pred = model(*inputs)

                # calculate total loss
                loss_local = 0
                for n, loss_function in enumerate(losses):
                    #print(y_true[n].shape, y_pred[n].shape)
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    loss_local += curr_loss

                # backpropagate and optimize
                loss_local.backward()
                base_optimizer.step()

                
            if epoch%10 == 0:
                with torch.no_grad():
                        model.eval()
                        #here we record the losses for the training, validation and testing sets (Although we are not supposed to use the testing set for this purpose)
                        _,_ = calculate_record_losses_hist(model, generator_val_training, losses2record, loss_type= 'tr', lpips = lpips, D= D,LDC=LDC_loss_val,vol_shape=vol_shape,device_number= device_number)
                        _,_ = calculate_record_losses_hist(model, generator_val     , losses2record, loss_type= 'val' ,lpips = lpips, D= D, LDC=LDC_loss_val,vol_shape=vol_shape,device_number= device_number)
                        _,_ = calculate_record_losses_hist(model, generator_test , losses2record, loss_type= 'ts', lpips = lpips, D= D,LDC=LDC_loss_val,vol_shape=vol_shape,device_number= device_number)
                        
            scheduler.step()  

            if (save_result and epoch%10 == 0 and epoch > 0) or epoch == p_epochs - 1:
                save_model_loss_hist(params, model = model, losses2record = losses2record,  path2save_weights = path2save_weights, path2save_loss = path2save_loss, epoch = epoch)
                tqdm.write('model saved at epoch: {}'.format(epoch))






#define main function
if __name__ == '__main__':

    model_lists = [
        {
            'standard_deviation': 0.08,
            'lambda_value': 0.05,
            'model': 'simple_avg_model',
            'loss': 'loss_LDC',
            'mode': 'bicubic',
            'batch_size': 28,
            'learning_rate': 3,
            'epochs': 2500,
            'SGDmomentum': 0.9,
            'notes': ''
        }
    ]


    #pre_defined dataset index
    from predefined_dataset_idx import dataset_idx
    #get device number from input args
    import argparse
    parser = argparse.ArgumentParser()
    #dataset type
    parser.add_argument('--dataset_type', type=int, default=0 , help = 'idx for datasets, check trainer.py for details')
    parser.add_argument('--device_number', type=int, default=0)
    #root path
    parser.add_argument('--root_path', type=str, default='experiment_run')
    #path2weights
    parser.add_argument('--path2weights', type=str, default=None, help='pretrained weights for warm start')
    args = parser.parse_args()

    root_path = args.root_path
    path2save_weights = "weights_"+root_path
    path2save_loss = "loss_hist_"+root_path
    path2weights = args.path2weights


    dataset_args = Dataset_Args()
    dataset_args.patients_train = dataset_idx[args.dataset_type]['train']
    dataset_args.patients_val = dataset_idx[args.dataset_type]['val']
    dataset_args.patients_test = dataset_idx[args.dataset_type]['test']
    #load data
    data_train, data_val, data_test, vol_shape = load_data(dataset_args)


    training(data_train, data_val, data_test,
             path2save_weights = path2save_weights, 
             path2save_loss = path2save_loss , 
             dataset_train_idx = dataset_args.patients_train,
             path2weights=path2weights, 
             para_lists = model_lists, 
             device_number = args.device_number,
             save_result = True)
