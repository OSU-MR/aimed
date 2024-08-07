import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from skimage.metrics import structural_similarity as ssim

# from DISTS_pytorch import DISTS
from batchgenerators.utilities.file_and_folder_operations import *
import csv

####
from scripts.helpful_functions import *
from scripts.nn_modules import *
from scripts.variables import *
from scripts.loss import *
from scripts.data_loader import load_data
from scripts.utils import *
from scripts.generator import create_generators_val

class Args:
    def __init__(self):
        self.dataset_shape = "256by192"
        self.view_list = ['SAX']
        self.suffix = '_c'


args = Args()

#train: healthy, validate: healthy, test: patient
args.patients_train = [10]
args.patients_val = [2]
args.patients_test = [99, 98, 97, 96, 93]

#train: healthy, validate: healthy, test: healthy
#args.patients_train = [10]
#args.patients_val = [2]
#args.patients_test = [19, 20, 21]

#train: digital, validate: digital, test: digital
#args.patients_train = [-15]
#args.patients_val = [-14]
#args.patients_test = [-18, -19, -24]

#load data
data_train, data_val, data_test, vol_shape = load_data(args)


import torch.optim as optim
from tqdm import tqdm



def verification(data_train,
                 data_test, 
                 para_lists, path2weights_folder,
                 target2display = [],
                 device_number=None,
                 save_cvs_result_name=None, save_result_name=None,
                 mimic_sd=None, SNR = None,
                 selected_target = 14,interpolation4display='bilinear',
                 prefix_val = '',
                brightness_augmentation = False, data_shift = False, change_noise = False):
    assert device_number is not None #verification
    result2cvs = []

    if save_cvs_result_name is not None:
        if not os.path.isfile(save_cvs_result_name+'.csv'):
            result2cvs.append(['SNRofsource(dB)', 'lambda', 'model type', 'loss function', 'interpolation', 'steps', 'lr', 'iters', 'mask for losses', 'mask for deformation field', 'SNR', 'SSIM', 'LPIPS', 'DISTS', 'LDC', 'TV', 'wavelet(trg_hats - ref)'])
            
    int_steps = 0

    _, dataset_test, _, dataset_with_noise_test = add_noise_of_certain_SNR(data_train, data_test, mimic_sd = mimic_sd, SNR = SNR,to_print = True)
    #dataset, _, dataset_with_noise, _ = add_noise_of_certain_SNR(data_train, data_test, mimic_sd = mimic_sd, SNR = SNR,given_sd = given_sd,to_print = True)
    
    #print("dataset_with_noise_val",dataset_with_noise_val.shape)

    for i, para2train in enumerate(para_lists):
        result2cvs_details = []        

        p_standard_deviation, p_lambda, p_model, p_loss, p_mode , p_steps_per_epoch , p_learning_rate, p_epochs,  p_mask_type, p_lambda_mask_type= para2train


        name2save = '_'.join([str(p_standard_deviation),
                                str(p_lambda),str(p_model),
                                str(p_loss),str(p_mode),str(p_steps_per_epoch),
                                str(p_learning_rate),str(p_epochs),
                                str(p_mask_type),str(p_lambda_mask_type)])

        path2weights = './'+path2weights_folder+'/'+name2save+'/'


        try:
            file_names = os.listdir(path2weights)
        
            ##########
            path2weights = path2weights + file_names[0]
            print("path2weights",path2weights)
            ###########
            _, _ , D, lpips , LDC_loss_val= get_losses(device_number)
            
            #dataset, dataset_val, dataset_with_noise, dataset_with_noise_val = add_noise(data_train, data_test, p_standard_deviation)
            model = create_model(p_model, device_number, mode2train= p_mode , int_steps = int_steps, weights_path4val = path2weights).to(device[device_number])
            generator_val = create_generators_val(p_model, dataset_test, dataset_with_noise_test, selected_target= selected_target, prefix_val = '_'+str(SNR)+'dB_'+prefix_val,
                                                  brightness_augmentation = brightness_augmentation, data_shift = data_shift, change_noise = change_noise) 
            ###########################################################################
            losses2show = []

            idx_recorder = 0
            for inputs, y_true in generator_val:
                model.eval()
                
                with torch.no_grad():

                    #current_snr_loss, current_ssim_loss , current_lpips_loss, current_dists_loss = get_current_score(model, inputs, y_true)
                    *current_loss, target_hats_avg, target_true, source_avg = get_current_score_val(model, inputs,
                                                                                                    y_true, D=D, 
                                                                                                    lpips=lpips, LDC_loss_val = LDC_loss_val,
                                                                                                    vol_shape = vol_shape, device_number=device_number, 
                                                                                                    title=para2train )
                    
                    

                    #current_loss = [0,0,0,0,0,0,0]
                    losses2show.append(current_loss)
                    result2cvs_details.append(current_loss)
                    if target2display != []:
                        if save_result_name is not None:
                            display_result_imgs(target_hats_avg, target_true, source_avg, idx_recorder, 
                                                target2display, current_results= current_loss, 
                                                para2train = para2train,id_ind=i,img_ind=idx_recorder,
                                                save_result_name = save_result_name,interpolation4display=interpolation4display, title_fontsize=3)
                        else:
                            display_result_imgs(target_hats_avg, target_true, source_avg, idx_recorder,
                                                target2display, current_results= current_loss, para2train = para2train, title_fontsize=3)
                        
                    # if save_result_name is not None:
                    #     display_result_imgs(target_hats_avg, target_true, source_avg, idx_recorder, 
                    #                         list2display, current_results= current_loss, 
                    #                         para2train = para2train,id_ind=i,img_ind=idx_recorder,
                    #                         save_result_name = save_result_name,interpolation4display=interpolation4display)
                    # else:
                    #     display_result_imgs(target_hats_avg, target_true, source_avg, idx_recorder,
                    #                          list2display, current_results= current_loss, para2train = para2train)
                
                idx_recorder += 1
            #print(idx_recorder)
            result2cvs.append(list(para2train) + np.mean(losses2show,0).tolist())

        except Exception as e:
            print(e)
            #result2cvs.append(list(para2train).append("nil"))
            print(list(para2train).append("nil"))


        result2cvs_details_len = len(result2cvs_details)
        avg_result2cvs_details = np.mean(result2cvs_details,0).tolist()
        std_result2cvs_details = np.std(result2cvs_details,0).tolist()
        error_std = np.round(np.array(std_result2cvs_details)/np.sqrt(result2cvs_details_len),5).tolist()

        #append avg_result2cvs_details to the first row of result2cvs_details
        #result2cvs_details = [result2cvs_details_len] + [avg_result2cvs_details] + [std_result2cvs_details] + [error_std] + [' '] + result2cvs_details
        result2cvs_details =  [avg_result2cvs_details] + [std_result2cvs_details] + [error_std] + result2cvs_details

        #print("para2train",para2train, save_cvs_result_name,save_cvs_result_name+str(para2train)+'_details')
        print("para2train",para2train, save_cvs_result_name,save_cvs_result_name)
        try:
            save2csv(save_cvs_result_name+str(para2train)+'_details', result2cvs_details)
        except:
                ...

        #

    save2csv(save_cvs_result_name, result2cvs) 

    #return result2cvs

    





#get device number from input args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device_number', type=int, default=0)
#root path
parser.add_argument('--root_path', type=str, default='experiment_run')
#path2weights
parser.add_argument('--path2weights', type=str, default=None)
args = parser.parse_args()

root_path = args.root_path
path2save_weights = "weights_"+root_path
path2save_loss = "loss_hist_"+root_path
path2weights = args.path2weights



para_lists2evaluate = [        

                      #real
                    #   (0.08,   0.05,        'vxm_model', 'loss_ncc', 'bicubic',  392,   0.2,    2500,  None,  None),
                    #   (0.08,   0.05,        'vxm_model', 'loss_LDC', 'bicubic',   70,     3,    2500,  None,  None),
                    #   (0.08,   0.05, 'simple_avg_model', 'loss_ncc', 'bicubic',   28,   0.2,    2500,  None,  None),
                      (0.08,   0.05, 'simple_avg_model', 'loss_LDC', 'bicubic',   28,     3,    2500,  None,  None),


                      #digital
                      #(0.08,   0.05,        'vxm_model', 'loss_ncc', 'bicubic',  392,   0.2,    2500,  None,  None),
                      #(0.08,   0.05,        'vxm_model', 'loss_LDC', 'bicubic',   70,     1,    2500,  None,  None),
                      #(0.08,   0.05, 'simple_avg_model', 'loss_ncc', 'bicubic',   28,   0.2,    2550,  None,  None),
                      #(0.08,   0.05, 'simple_avg_model', 'loss_LDC', 'bicubic',   28,     2,    2500,  None,  None),
    


                      ] 

given_sd = None
for given_SNR_name in ["11dB","6dB","1dB"]:
    for st in ['all']:
        loss2show = verification(data_train, data_test,
                                path2weights_folder = 'weights_candidates_3.6',
                                save_cvs_result_name = './results_experiment_run/result_SNR_'+given_SNR_name+'_t'+str(st),
                                SNR = int(given_SNR_name[:-2]),
                                para_lists = para_lists2evaluate, 
                                device_number = args.device_number,
                                selected_target= st,
                                prefix_val='test_2.16_healthy_patient_all_slices')
        

for target_img in [7]:#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
    for given_SNR_name in ["11dB","6dB","1dB"]:

        loss2show = verification(data_train, data_test,
                                path2weights_folder = 'weights_candidates_3.6',
                                target2display = [5],
                                SNR=int(given_SNR_name[:-2]),
                                para_lists = para_lists2evaluate, 
                                device_number = args.device_number,
                                save_result_name = "experiment_run_"+given_SNR_name+"_target_idx_"+str(target_img),
                                selected_target= target_img,
                                prefix_val='test_2.16_healthy_patient_all_slices')