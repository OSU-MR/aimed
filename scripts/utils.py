import os
import pickle
import time
import numpy as np
import torch
from scripts.variables import *
from scripts.helpful_functions import *
import torchgeometry.losses.ssim as torch_ssim_loss


def SNR2standard_deviation(SNR, img):
    #return (  np.sum(img**2)  /  (np.product(img.shape)*(10**(SNR/10)))  )**0.5
    energy_img = np.mean(( abs(img[...,0,:,:]+1j*img[...,1,:,:]) )**2)
    sd = np.sqrt(energy_img/((2)*(10**(SNR/10))))
    return sd


def add_noise_of_certain_SNR(x_train_twochannels, x_test_twochannels, SNR = None , mimic_sd = None, is_phase_ref = 0, to_print = True):
    if mimic_sd is not None:
        SNR = [1000, 19, 14, 11][[0, 0.033, 0.0565, 0.08].index(mimic_sd)]
    try:
        dataset     = np.vstack((x_train_twochannels['LAX'],x_train_twochannels['SAX'],x_train_twochannels['2CH']))
        dataset_val = np.vstack((x_test_twochannels['LAX'],x_test_twochannels['SAX'],x_test_twochannels['2CH']))
    except:
        dataset     = x_train_twochannels['SAX']
        dataset_val = x_test_twochannels['SAX']


    if SNR > 999:
        dataset_with_noise = dataset
        dataset_with_noise_val = dataset_val
        if to_print:
            print("large SNR means no noise is added")
        return dataset, dataset_val, dataset_with_noise, dataset_with_noise_val

    len_of_patient = dataset.shape[0]

    if dataset.shape[0]%8 == 0:
        len_of_patient = 8
    if dataset.shape[0]%9 == 0:
        len_of_patient = 9
    if dataset.shape[0]%10 == 0:
        len_of_patient = 10

    #print("added SNR: ", SNR)
    total_sd_training = []
    dataset_with_noise = np.zeros(dataset.shape)

    for i in range(dataset.shape[0]//len_of_patient):
        standard_deviation = SNR2standard_deviation(SNR=SNR, img = dataset[len_of_patient*i:len_of_patient*(i+1),:,is_phase_ref,...] )
        total_sd_training.append(standard_deviation)
        #print(standard_deviation)
    #print(dataset.shape) #(60, 29, 2, 2, 192, 192)
    print(dataset_with_noise.shape)
    for set_ind in range(dataset.shape[0]):
        for slice_ind in range(dataset.shape[1]):
            dataset_with_noise[set_ind,slice_ind,is_phase_ref,...] = noisy("gauss",dataset[set_ind,slice_ind,is_phase_ref,...],standard_deviation=total_sd_training[set_ind//len_of_patient])
   

    #print("\n")
    total_sd_testing = []
    dataset_with_noise_val = np.zeros(dataset_val.shape)
    for i in range(dataset_val.shape[0]//len_of_patient):
        standard_deviation = SNR2standard_deviation(SNR=SNR, img = dataset_val[len_of_patient*i:len_of_patient*(i+1),:,is_phase_ref,...] )
        total_sd_testing.append(standard_deviation)
        #print(standard_deviation)
    for set_ind in range(dataset_val.shape[0]):
        for slice_ind in range(dataset_val.shape[1]):
            dataset_with_noise_val[set_ind,slice_ind,is_phase_ref,...] = noisy("gauss",dataset_val[set_ind,slice_ind,is_phase_ref,...],standard_deviation=total_sd_testing[set_ind//len_of_patient])


    #print("avg of the ", ,"SNRs: ",np.mean(total_SNR))
    #print(SNR, ",",np.mean(total_SNR),)
    if to_print:
        print("input SNR:", SNR, "  avg standard deviation added (tr,ts):", np.mean(total_sd_training) , np.mean(total_sd_testing))
    return dataset, dataset_val, dataset_with_noise, dataset_with_noise_val


def add_noise_of_certain_SNR4training(dataset, SNR = None , mimic_sd = None, to_print = False, selected_set = None):

    dataset_with_noise = np.zeros(dataset.shape)

    set_range = range(dataset.shape[0]) if selected_set is None else list(set(selected_set))

    for set_ind in set_range:
        #get one element from list mimic_sd randomly
        mimic_sd_temp = mimic_sd[np.random.randint(0,len(mimic_sd))] if mimic_sd is not None else None
        for slice_ind in range(dataset.shape[1]):
            dataset_with_noise[set_ind,slice_ind,...] = noisy("gauss",dataset[set_ind,slice_ind,...],standard_deviation=mimic_sd_temp)#total_sd_training[set_ind//8])

    if to_print:
        print("input sd:", mimic_sd_temp)

    return dataset_with_noise


def create_SNR_list(base_SNR, P, N):
    lower_bound = base_SNR - (base_SNR * (P/100))
    upper_bound = base_SNR + (base_SNR * (P/100))
    return list(np.linspace(lower_bound, upper_bound, N))


def save_model_loss_hist(para2train, model = None,losses2record = None,  path2save_weights = "weights2", path2save_loss = "loss_hist", epoch = ''):
    # Access parameters from the dictionary
    p_standard_deviation = para2train.get('standard_deviation', -1)
    p_lambda = para2train.get('lambda_value', -1)
    p_model = para2train.get('model', '-1')
    p_loss = para2train.get('loss', '-1')
    p_mode = para2train.get('mode', '-1')
    p_steps_per_epoch = para2train.get('batch_size', -1)  # Assuming batch_size corresponds to steps_per_epoch
    p_learning_rate = para2train.get('learning_rate', -1)
    p_epochs = para2train.get('epochs', -1)
    p_SGDmomentum = para2train.get('SGDmomentum', 0.9)
    p_notes = para2train.get('notes', '')

    name2save = '_'.join([str(p_standard_deviation),
                        str(p_lambda), str(p_model),
                        str(p_loss), str(p_mode), str(p_steps_per_epoch),
                        str(p_learning_rate), str(p_epochs), str(p_SGDmomentum), str(p_notes)])

    
    t = time.localtime()
    current_time = time.strftime("%Y%m%d%H%M%S", t)#time.strftime("%Y_%m_%d_%H_%M_%S", t)
    epoch = str(epoch)+'_' if epoch != '' else ''

    os.makedirs(path2save_weights, exist_ok=True)
    os.makedirs(path2save_loss, exist_ok=True)

    if model is not None:
        os.makedirs(path2save_weights+'/'+name2save, exist_ok=True)
        torch.save(model.state_dict(), './'+path2save_weights+'/'+name2save+'/'+epoch+current_time+'.weights')

    if losses2record is not None:
        os.makedirs(path2save_loss+'/'+name2save, exist_ok=True)
        with open('./'+path2save_loss+'/'+name2save+'/'+epoch+current_time+'.pickle', 'wb') as handle:
            pickle.dump(losses2record, handle, protocol=pickle.HIGHEST_PROTOCOL)  



def get_current_score(model, inputs, y_true, lpips , D,LDC,vol_shape =[192,192] ,device_number = None):
    assert device_number is not None

    inputs = [torch.from_numpy(d).to(device[device_number]).float() for d in inputs]
    y_true = [torch.from_numpy(d).to(device[device_number]).float() for d in y_true]

    y_pred = model(*inputs)
            
    target_hats_avg = two2one(np.mean((y_pred[0].detach().cpu().numpy()),0))
    target_true     = two2one(y_true[0][0,...].detach().cpu().numpy())
    

    ########### LDC edge loss
    target_hats_avg_LDC = torch.unsqueeze(torch.from_numpy(one2two(target_hats_avg)),0).to(device[device_number])
    target_true_LDC = torch.unsqueeze(torch.from_numpy(one2two(target_true)),0).to(device[device_number])
    current_LDC_loss = LDC(target_hats_avg_LDC,target_true_LDC).item()

    ###########SNR
    current_snr_loss = SNR(target_true,target_hats_avg)
            
    ############SSIM
    reference_e = one2two(target_true)
    target_hats = one2two(target_hats_avg)
            
    reference_t   = torch.FloatTensor(reference_e[np.newaxis,...]).to(device[device_number])
    target_hats_t = torch.FloatTensor(target_hats[np.newaxis,...]).to(device[device_number])                                            
    try:
        current_ssim_loss = 1- torch_ssim_loss(reference_t, target_hats_t ,reduction = 'mean',window_size=11 , 
                                    max_val = torch.max(torch.max(reference_t),torch.max(target_hats_t))   ).cpu().numpy()
    except:
        current_ssim_loss = 0

    #############LPIPS
    placeholder = torch.zeros([1,1]+vol_shape).to(device[device_number])

    reference_t_3ch = torch.column_stack((reference_t,placeholder))
    target_hats_t_3ch = torch.column_stack((target_hats_t,placeholder))

    reference_t_3ch = reference_t_3ch/torch.max(   torch.abs(torch.min(reference_t_3ch))   ,   torch.abs(torch.max(reference_t_3ch))   )
    target_hats_t_3ch = target_hats_t_3ch/torch.max(   torch.abs(torch.min(target_hats_t_3ch))   ,   torch.abs(torch.max(target_hats_t_3ch))   )

    try:
        current_lpips_loss = lpips(reference_t_3ch,target_hats_t_3ch).item()
    except:
        current_lpips_loss = -1

    #############DISTS
    reference_t_3ch_normalized = (reference_t_3ch-torch.min(reference_t_3ch))/(torch.max(reference_t_3ch)-torch.min(reference_t_3ch))
    target_hats_t_3ch_normalized = (target_hats_t_3ch-torch.min(target_hats_t_3ch))/(torch.max(target_hats_t_3ch)-torch.min(target_hats_t_3ch))

    try:
        current_dists_loss = D(reference_t_3ch_normalized, target_hats_t_3ch_normalized).item()
    except:
        current_dists_loss = -1

    return [current_snr_loss, current_ssim_loss , current_lpips_loss, current_dists_loss, current_LDC_loss]


def calculate_record_losses_hist(model, generator_val, losses2record, loss_type , lpips, D, LDC ,vol_shape =[192,192] ,device_number = None):
    assert device_number is not None #calculate_record_losses_hist

    n = 0
    current_losses = 0
    for inputs, y_true in generator_val:
        current_losses += np.array(get_current_score(model, inputs, y_true, lpips=lpips,D=D, LDC=LDC,vol_shape = vol_shape , device_number= device_number))
        n += 1
    current_losses = (current_losses/n).tolist()

    for i, loss_name2record in enumerate(['snr','ssim','lpips','dists','LDC']):
        try:
            losses2record['_'.join([loss_name2record,loss_type])].append(current_losses[i])
        except:
            try:
                losses2record['_'.join([loss_name2record,loss_type])] = []
                losses2record['_'.join([loss_name2record,loss_type])].append(current_losses[i])
            except:
                losses2record['_'.join([loss_name2record,loss_type])] = {}
                losses2record['_'.join([loss_name2record,loss_type])] = []
                losses2record['_'.join([loss_name2record,loss_type])].append(current_losses[i])

    return current_losses, ['snr','ssim','lpips','dists','LDC']
