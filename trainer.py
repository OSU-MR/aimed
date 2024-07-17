import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.io as sio
from scipy.io import savemat
import time


from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import os
import glob

from itertools import combinations
from itertools import product
import random
from numpy.random import default_rng

from torchsummary import summary
from torchviz import make_dot

# imports
import sys

# local imports
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm 
import neurite as ne

from skimage.metrics import structural_similarity as ssim
import cv2

import json
import pickle

#from ignite.metrics import SSIM as SSIM_loss
#from ignite.engine import *
import torchgeometry.losses.ssim as torch_ssim_loss
from DISTS_pytorch import DISTS

import scipy

from itertools import product

from batchgenerators.utilities.file_and_folder_operations import *

import torch.nn.functional as F

import threading

import matplotlib.transforms as mtransforms

import torchvision.transforms as transforms

import csv


_ = torch.manual_seed(321)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


####
from scripts.helpful_functions import *
from scripts.nn_modules import model_avg_
from scripts.variables import *
from scripts.loss_functions_gpu import *
from scripts.data_loader import load_data

class Args:
    def __init__(self):
        self.dataset_shape = "256by192"
        self.view_list = ['SAX']
        self.suffix = '_c'


args = Args()

#train: healthy, validate: healthy, test: patient
args.patients_train = [10, 3, 4, 6, 11, 1]
args.patients_val = [2, 5, 18, 19, 20, 21]
args.patients_test = [99, 98, 97, 96, 93]

#train: healthy, validate: healthy, test: healthy
#args.patients_train = [10, 3, 4, 6, 11, 1]
#args.patients_val = [2, 5, 18]
#args.patients_test = [19, 20, 21]

#train: digital, validate: digital, test: digital
#args.patients_train = [-15, -16, -17, -20, -21, -22]
#args.patients_val = [-14, -23, -13]
#args.patients_test = [-18, -19, -24]

#load data
data_train, data_val, data_test, vol_shape = load_data(args)

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


from numba import jit, prange
import numpy as np
@jit(nopython=True, parallel=True)
def shift_parallel(x_data_slice, clear_data_slice, set_idx):
    x_data_slice_out = np.empty(x_data_slice.shape, dtype=x_data_slice.dtype)
    clear_data_slice_out = np.empty(clear_data_slice.shape, dtype=x_data_slice.dtype)

    for idx in prange(len(set_idx)):
        i = set_idx[idx]

        # Circular shift for the third dimension of the data
        y_shift = int(np.random.uniform(low=-16, high=16)) #32
        x_shift = int(np.random.uniform(low=-22, high=22)) #44

        # Manually perform roll operation for y_shift and x_shift
        if y_shift > 0:
            x_data_slice_out[i, ..., y_shift:, :] = x_data_slice[i, ...,:-y_shift, :]
            x_data_slice_out[i, ..., :y_shift, :] = 0
            clear_data_slice_out[i, ..., y_shift:, :] = clear_data_slice[i, ..., :-y_shift, :]
            clear_data_slice_out[i, ..., :y_shift, :] = 0
        elif y_shift < 0:
            x_data_slice_out[i, ..., :y_shift, :] = x_data_slice[i, ..., -y_shift:, :]
            x_data_slice_out[i, ..., y_shift:, :] = 0
            clear_data_slice_out[i, ..., :y_shift, :] = clear_data_slice[i, ..., -y_shift:, :]
            clear_data_slice_out[i, ..., y_shift:, :] = 0
        else:
            x_data_slice_out[i, ...] = x_data_slice[i, ...]
            clear_data_slice_out[i, ...] = clear_data_slice[i, ...]
        
        if x_shift != 0:
            x_data_slice[i, ..., :, -x_shift:] = x_data_slice_out[i, ..., :, :x_shift]
            x_data_slice[i, ..., :, :-x_shift] = x_data_slice_out[i, ..., :, x_shift:]
            clear_data_slice[i, ..., :, -x_shift:] = clear_data_slice_out[i, ..., :, :x_shift]
            clear_data_slice[i, ..., :, :-x_shift] = clear_data_slice_out[i, ..., :, x_shift:]
        else:
            x_data_slice[i, ...] = x_data_slice_out[i, ...]
            clear_data_slice[i, ...] = clear_data_slice_out[i, ...]

    return x_data_slice, clear_data_slice

@jit(nopython=True, parallel=True)
def add_noise_of_certain_SNR4training_parallel(clear_data_slice,mimic_sd,selected_set):
    dataset_with_noise = np.zeros(clear_data_slice.shape, dtype=clear_data_slice.dtype)
    
    for set_ind in prange(len(selected_set)):
        idx = selected_set[set_ind]
        mimic_sd_temp = mimic_sd[np.random.randint(len(mimic_sd))]
        for slice_ind in range(clear_data_slice.shape[1]):
            dataset_with_noise[idx, slice_ind, ...] = noisy_parallel("gauss", clear_data_slice[idx, slice_ind, ...], standard_deviation=mimic_sd_temp)
            
    return dataset_with_noise

@jit(nopython=True)
def noisy_parallel(noise_typ, image, standard_deviation):
    ch, row, col = image.shape
    if noise_typ == "gauss":
        mean = 0
        gauss = np.random.normal(mean, standard_deviation, (ch, row, col))
        noisy_img = image + gauss
        return noisy_img
    else:
        raise ValueError("Unsupported noise type")
    

@jit(nopython=True, parallel=True)
def brightness_augmentation_parallel(clear_data_slice, x_datasets, set_idx, beta, percentiles_list):
    for i in prange(len(set_idx)):
        idx = set_idx[i]
        beta_idx = np.random.randint(0, len(beta))
        percentile_num = percentiles_list[idx, beta_idx]
        clear_data_slice[idx, ...] = normalize_energy_brightness_augmentation_parallel(clear_data_slice[idx, ...], percentile_num)

        if x_datasets is not None:
            x_datasets[idx, ...] = normalize_energy_brightness_augmentation_parallel(x_datasets[idx, ...], percentile_num)

    if x_datasets is None:
        return clear_data_slice, None#, np.empty_like(clear_data_slice)  # Return a placeholder
    else:
        return clear_data_slice, x_datasets


@jit(nopython=True)
def normalize_energy_brightness_augmentation_parallel(arr_o, percentile_num):
    real_part = arr_o[:, 0, ...]
    imag_part = arr_o[:, 1, ...]

    # Calculate magnitude and angle without forming complex numbers
    arr_m = np.sqrt(real_part**2 + imag_part**2)
    arr_angle = np.arctan2(imag_part, real_part)

    arr_m /= percentile_num

    # Recompose the complex number from magnitude and angle
    real_normalized = arr_m * np.cos(arr_angle)
    imag_normalized = arr_m * np.sin(arr_angle)

    arr_normalized = np.stack((real_normalized, imag_normalized), 1)
    return arr_normalized


@jit(nopython=True)
def calculate_percentile_parallel(arr_o, percentage):
    arr = np.abs(arr_o[:,0,...] + 1j * arr_o[:,1,...])
    arr_flattened = arr.ravel()
    arr_sorted = np.sort(arr_flattened)
    index = int(len(arr_sorted) * (percentage / 100.0))
    return arr_sorted[index]


@jit(nopython=True, parallel = True)
def precalculate_percentiles_parallel(clear_data_slice, beta):
    num_datasets = clear_data_slice.shape[0]
    num_beta_values = len(beta)
    percentiles = np.empty((num_datasets, num_beta_values), dtype=np.float64)
    
    for i in prange(num_datasets):
        for j in range(num_beta_values):
            beta_value = beta[j]
            temp = calculate_percentile_parallel(clear_data_slice[i, ...], beta_value)
            percentiles[i, j] = temp

    return percentiles




class vxm_data_generator_parallel:
    def __init__(self, x_data, phase_ref_or_not = 0, batch_size = 10, debug = 0, replace = False, 
                       clear_data = None, val = None, brightness_augmentation = True,
                       data_shift = True, change_noise = True, prefix_val = None):

        #save a copy of the dataset for later use 
        #only use middle 15 images for training

        ###!!!! save with different names!! the names for training and testing should be different
        if val == None:
            self.prefix = 'train_'
        else:
            self.prefix = 'val_'
        #generate a time stamp with 
        t = time.localtime()
        current_time = time.strftime("%Y%m%d%H%M%S", t)
        self.prefix = self.prefix + current_time + '_' if prefix_val is None else self.prefix + prefix_val + '_'
        self.prefix_val = prefix_val

        #if self.prefix+'self.dataset_temp.npy' does not exist, save it
        if not os.path.exists(self.prefix+'self.dataset_temp.npy'):
            np.save(self.prefix+'self.dataset_temp.npy',x_data[:,7:22,phase_ref_or_not,...])
            np.save(self.prefix+'self.clear_data_temp.npy',clear_data[:,7:22,phase_ref_or_not,...])
        ###!!!! save with different names!! the names for training and testing should be different
        self.dataset = np.load(self.prefix+'self.dataset_temp.npy')
        self.clear_data = np.load(self.prefix+'self.clear_data_temp.npy')   
        #print data loaded path
        print("data loaded, path:")
        print(self.prefix+'self.dataset_temp.npy')
        print(self.prefix+'self.clear_data_temp.npy') 

        # self.dataset_copy = self.dataset.copy()
        # self.clear_data_copy = self.clear_data.copy()

        self.vol_shape = [self.dataset.shape[-2],self.dataset.shape[-1]] # extract data shape
        self.ndims = len(self.vol_shape)

        self.batch_size = batch_size
        self.val = val
        self.debug = debug
        self.replace = replace
        self.brightness_augmentation = brightness_augmentation
        self.data_shift = data_shift
        self.change_noise = change_noise

        #self.zero_phi = np.zeros([self.batch_size, *self.vol_shape, self.ndims])
        #self.zero_phi = np.zeros([self.batch_size, self.ndims, *self.vol_shape])
        #actually we only need a placeholder here
        self.zero_phi = np.zeros([self.batch_size, self.ndims, 1,1])

        self.beta = list(np.linspace(87, 95, 9))

        base_SNR = SNR2standard_deviation(11,self.clear_data)
        print("base_SNR: "+str(base_SNR))
        #0.1403170870464488#for healthy training 10 3 4 6 11 1 => 11.3dB
        #0.08853409662802346#15dB  #0.1403170870464488
        #base_SNR = 0.13803942758081308
        self.delta = np.array(create_SNR_list(base_SNR, P = 30, N = 21))

        
        self.set_num = self.dataset.shape[0]
        self.rep_num = self.dataset.shape[1]
        #print("self.dataset.shape",self.dataset.shape)
        if self.val == None:
            self.slice_idx =  [i for i in range(self.set_num) for _ in range((self.rep_num)*(self.rep_num-1))]
            self.target_idx = [j for j in range(self.rep_num) for _ in range(self.rep_num-1)]*self.set_num
            self.source_idx = [k for j in range(self.rep_num) for k in range(self.rep_num) if k != j]*self.set_num
        else:
            self.slice_idx =  [i for i in range(self.set_num) for _ in range((self.rep_num-1))]
            self.target_idx = [self.val] * self.set_num * (self.rep_num-1)
            self.source_idx = [k for j in range(self.set_num) for k in range(self.rep_num) if k != self.val]

        self.dataset_len = len(self.slice_idx)
        self.length = self.dataset_len//self.batch_size

        self.percentiles_list = precalculate_percentiles_parallel(self.clear_data, self.beta)

    def __iter__(self):
        self.epoch_index = list(zip(self.slice_idx, self.target_idx, self.source_idx))
        if self.val == None:
            random.shuffle(self.epoch_index)
        #print("resetting the generator, the epoch lengh",self.dataset_len, "iterations per epoch ",self.length)

        return self

    def __len__(self):
        return self.length
    
    def __del__(self):
        if self.prefix_val is None:
            print("deleting the temporary files")
            os.remove(self.prefix+'self.dataset_temp.npy')
            os.remove(self.prefix+'self.clear_data_temp.npy')

    def __next__(self):
        #stop iteration when epoch_index is empty
        if len(self.epoch_index) == 0:
            raise StopIteration

        if len(self.epoch_index) >= self.batch_size:
            current_index = [self.epoch_index.pop() for _ in range(self.batch_size)]  # Get the last 'batch_size' elements
        else:
            current_index = self.epoch_index
            self.epoch_index = []
        #print(len(current_index))


        set_idx, *_ = zip(*current_index)
        set_idx = list(set(set_idx))
        set_idx_array = np.array(set_idx, dtype=np.int64)  # Convert to NumPy array
        clear_data_slice = np.load(self.prefix+'self.clear_data_temp.npy')
        beta_array = np.array(self.beta, dtype=np.float64)

        if self.change_noise:
            if self.brightness_augmentation == True:
                clear_data_slice, _ = brightness_augmentation_parallel(clear_data_slice, None, set_idx_array, beta_array, self.percentiles_list)
            x_data_slice = add_noise_of_certain_SNR4training_parallel(clear_data_slice, mimic_sd= self.delta, selected_set = set_idx_array)

        else:
            x_data_slice = np.load(self.prefix+'self.dataset_temp.npy')
            if self.brightness_augmentation == True:
                clear_data_slice,x_data_slice = brightness_augmentation_parallel(clear_data_slice,x_data_slice, set_idx_array, beta_array, self.percentiles_list)

        if self.data_shift:
            x_data_slice,clear_data_slice = shift_parallel(x_data_slice,clear_data_slice, set_idx)


        # Initialize lists for source images and target images
        inputs, outputs = process_images(x_data_slice, clear_data_slice, current_index, self.zero_phi)

        return (inputs, outputs)


@jit(nopython=True, parallel=True)
def process_images(x_data_slice, clear_data_slice, current_index, zero_phi):
    num_images = len(current_index)

    # Assuming x_data_slice and clear_data_slice have dimensions [slice, image, ...]
    image_shape = x_data_slice.shape[2:]
    source_images = np.empty((num_images,) + image_shape, dtype=x_data_slice.dtype)
    target_images = np.empty((num_images,) + image_shape, dtype=x_data_slice.dtype)
    clear_target_images = np.empty((num_images,) + image_shape, dtype=clear_data_slice.dtype)

    for i in prange(num_images):
        slice_idx, target_idx, source_idx = current_index[i]
        source_images[i] = x_data_slice[slice_idx, source_idx, ...]
        target_images[i] = x_data_slice[slice_idx, target_idx, ...]
        clear_target_images[i] = clear_data_slice[slice_idx, target_idx, ...]

    inputs = [source_images, target_images]
    outputs = [clear_target_images, zero_phi]

    return inputs, outputs


class simple_avg_data_generator_parallel:
    def __init__(self, x_data, phase_ref_or_not = 0, batch_size = 10, debug = 0, replace = False, 
                       clear_data = None, val = None, brightness_augmentation = True,
                       data_shift = True, change_noise = True, prefix_val = None):
        #save a copy of the dataset for later use 
        #only use middle 15 images for training

        ###!!!! save with different names!! the names for training and testing should be different
        if val == None:
            self.prefix = 'train_'
        else:
            self.prefix = 'val_'
        #generate a time stamp with 
        t = time.localtime()
        current_time = time.strftime("%Y%m%d%H%M%S", t)
        self.prefix = self.prefix + current_time + '_' if prefix_val is None else self.prefix + prefix_val + '_'
        self.prefix_val = prefix_val

        #if self.prefix+'self.dataset_temp.npy' does not exist, save it
        if not os.path.exists(self.prefix+'self.dataset_temp.npy'):
            np.save(self.prefix+'self.dataset_temp.npy',x_data[:,7:22,phase_ref_or_not,...])
            np.save(self.prefix+'self.clear_data_temp.npy',clear_data[:,7:22,phase_ref_or_not,...])

        ###!!!! save with different names!! the names for training and testing should be different
        self.dataset = np.load(self.prefix+'self.dataset_temp.npy')
        self.clear_data = np.load(self.prefix+'self.clear_data_temp.npy')    
        #print data loaded path
        print("data loaded, path:")
        print(self.prefix+'self.dataset_temp.npy')
        print(self.prefix+'self.clear_data_temp.npy')


        self.vol_shape = [self.dataset.shape[-2],self.dataset.shape[-1]] # extract data shape
        self.ndims = len(self.vol_shape)

        self.batch_size = batch_size
        self.val = val
        self.debug = debug
        self.replace = replace
        self.brightness_augmentation = brightness_augmentation
        self.data_shift = data_shift
        self.change_noise = change_noise

        #self.zero_phi = np.zeros([self.batch_size, *self.vol_shape, self.ndims])
        #self.zero_phi = np.zeros([self.batch_size, self.ndims, *self.vol_shape])
        #actually we only need a placeholder here
        self.zero_phi = np.zeros([self.batch_size, self.ndims, 1,1])

        self.beta = list(np.linspace(87, 95, 9)) #9

        base_SNR = 0.1403170870464488#0.08853409662802346#15dB  #0.1403170870464488
        #base_SNR = 0.13803942758081308
        self.delta = np.array(create_SNR_list(base_SNR, P = 30, N = 21)) #21

        
        self.set_num = self.dataset.shape[0]
        self.rep_num = self.dataset.shape[1]

        if self.val == None:
            self.slice_idx =  [i for i in range(self.set_num) for _ in range(self.rep_num)]
            self.target_idx = [k for j in range(self.set_num) for k in range(self.rep_num)]
        else:
            self.slice_idx =  [i for i in range(self.set_num)]
            self.target_idx = [self.val] * self.set_num
        
        self.dataset_len = len(self.slice_idx)

        self.percentiles_list = precalculate_percentiles_parallel(self.clear_data, self.beta)

    def __iter__(self):
        self.epoch_index = list(zip(self.slice_idx, self.target_idx))
        random.shuffle(self.epoch_index)
        #print("resetting the generator, the epoch lengh",self.dataset_len, "iterations per epoch ",self.dataset_len//self.batch_size)
        return self
    
    def __len__(self):
        return self.dataset_len//self.batch_size
    
    def __del__(self):
        if self.prefix_val is None:
            print("deleting the temporary files")
            os.remove(self.prefix+'self.dataset_temp.npy')
            os.remove(self.prefix+'self.clear_data_temp.npy')

    def __next__(self):
        #stop iteration when epoch_index is empty
        if len(self.epoch_index) == 0:
            raise StopIteration

        if len(self.epoch_index) >= self.batch_size:
            current_index = [self.epoch_index.pop() for _ in range(self.batch_size)]  # Get the last 'batch_size' elements
        else:
            current_index = self.epoch_index
            self.epoch_index = []
        #print(len(current_index))


        set_idx, _ = zip(*current_index)
        set_idx = list(set(set_idx))
        set_idx_array = np.array(set_idx, dtype=np.int64)  # Convert to NumPy array
        clear_data_slice = np.load(self.prefix+'self.clear_data_temp.npy')
        beta_array = np.array(self.beta, dtype=np.float64)

        if self.change_noise:
            if self.brightness_augmentation == True:
                clear_data_slice, _ = brightness_augmentation_parallel(clear_data_slice, None, set_idx_array, beta_array, self.percentiles_list)
            x_data_slice = add_noise_of_certain_SNR4training_parallel(clear_data_slice, mimic_sd= self.delta, selected_set = set_idx_array)

        else:
            x_data_slice = np.load(self.prefix+'self.dataset_temp.npy')
            if self.brightness_augmentation == True:
                clear_data_slice,x_data_slice = brightness_augmentation_parallel(clear_data_slice,x_data_slice, set_idx_array, beta_array, self.percentiles_list)

        if self.data_shift:
            x_data_slice,clear_data_slice = shift_parallel(x_data_slice,clear_data_slice, set_idx)


        # Initialize lists for source images and target images
        source_images = [[] for _ in range(self.rep_num-1)]  
        target_images = []
        clear_target_images = []

        for slice_idx, target_idx in current_index:
            target_images.append((x_data_slice[slice_idx, target_idx, ...]))
            clear_target_images.append((clear_data_slice[slice_idx, target_idx, ...]))

            # Generate a list of indices for source images (excluding the current target_idx)
            source_list = [x for x in range(self.rep_num) if x != target_idx]

            # Append data to each source image list
            for i, src_idx in enumerate(source_list):
                source_images[i].append((x_data_slice[slice_idx, src_idx, ...]))

        # After the loop, stack the arrays in each list to create single arrays
        target_images_array = np.stack(target_images)
        clear_target_images_array = np.stack(clear_target_images)
        source_images_arrays = [np.stack(source) for source in source_images]

        # Assembling the inputs and outputs
        inputs = [target_images_array] + source_images_arrays
        outputs = [clear_target_images_array] + [self.zero_phi] * (self.rep_num-1)

        if self.debug > 1:
            print(source_images.shape,target_images.shape)

        return (inputs, outputs)


def create_model(model_name = None, device_number = None, mode2train = None, 
                weights_path4load = None, int_steps = 5):
    assert model_name != None
    #############"vxm_model"                                                      
    if model_name == "vxm_model":
        model = vxm.networks.VxmDense(inshape=vol_shape,nb_unet_features=nb_features,src_feats=2, trg_feats=2, int_steps=int_steps)
        model.transformer = vxm.layers.SpatialTransformer(size = tuple(vol_shape),mode=mode2train).to(device[device_number]) #'bilinear' | 'nearest' | 'bicubic'
        
        if weights_path4load is not None:
            model.load_state_dict(torch.load(weights_path4load, map_location = device[device_number]))

    #############

    #############"simple_avg_model"
    if model_name == "simple_avg_model":
        model = model_avg_(vol_shape = vol_shape, nb_features=nb_features, src_feats=2, trg_feats=2, int_steps=int_steps).to(device[device_number])
        model.vxm_block.transformer = vxm.layers.SpatialTransformer(size = tuple(vol_shape),mode=mode2train).to(device[device_number]) #'bilinear' | 'nearest' | 'bicubic'
        if weights_path4load is not None:
            try:
                model.vxm_block.load_state_dict(torch.load(weights_path4load, map_location = device[device_number]))
            except:
                model.load_state_dict(torch.load(weights_path4load, map_location = device[device_number]))


 
    return model


def losses_weights(model_name, reg_lambda, training_losses, training_losses2 = vxm.losses.Grad('l2').loss, batch_size = 13):

    losses  = [training_losses2]
    weights = [1,reg_lambda]
    if model_name != "vxm_model":
        for i in range(batch_size):
            losses.insert(-1, training_losses2)
            weights.insert(-1, reg_lambda)
        weights[1:] = (np.array(weights[1:])/batch_size).tolist()
    losses.insert(0,training_losses)
    return losses, weights


def save_model_loss_hist(para2train, model = None, model2 = None,losses2record = None,  path2save_weights = "weights2", path2save_loss = "loss_hist", epoch = ''):
    # Access parameters from the dictionary
    p_standard_deviation = para2train.get('standard_deviation', -1)
    p_lambda = para2train.get('lambda_value', -1)
    p_model = para2train.get('model', '-1')
    p_loss = para2train.get('loss', '-1')
    p_mode = para2train.get('mode', '-1')
    p_steps_per_epoch = para2train.get('batch_size', -1)  # Assuming batch_size corresponds to steps_per_epoch
    p_learning_rate = para2train.get('learning_rate', -1)
    p_epochs = para2train.get('epochs', -1)
    p_ReduceLROnPlateau_factor = para2train.get('ReduceLROnPlateau_factor', 0.1)
    p_ReduceLROnPlateau_patience = para2train.get('ReduceLROnPlateau_patience', 10)
    p_ReduceLROnPlateau_cooldown = para2train.get('ReduceLROnPlateau_cooldown', 0)
    p_ReduceLROnPlateau_threshold = para2train.get('ReduceLROnPlateau_threshold', 0.0001)
    p_ReduceLROnPlateau_eps = para2train.get('ReduceLROnPlateau_eps', 1e-08)
    p_SGDmomentum = para2train.get('SGDmomentum', 0.9)
    p_note = para2train.get('note', '')

    name2save = '_'.join([str(p_standard_deviation),
                        str(p_lambda), str(p_model),
                        str(p_loss), str(p_mode), str(p_steps_per_epoch),
                        str(p_learning_rate), str(p_epochs),
                        str(p_ReduceLROnPlateau_factor), str(p_ReduceLROnPlateau_patience),
                        str(p_ReduceLROnPlateau_cooldown), str(p_ReduceLROnPlateau_threshold),
                        str(p_ReduceLROnPlateau_eps), str(p_SGDmomentum), str(p_note)])

    
    t = time.localtime()
    current_time = time.strftime("%Y%m%d%H%M%S", t)#time.strftime("%Y_%m_%d_%H_%M_%S", t)
    epoch = str(epoch)+'_' if epoch != '' else ''

    maybe_mkdir_p(path2save_weights)
    maybe_mkdir_p(path2save_loss)

    if model is not None:
        maybe_mkdir_p(path2save_weights+'/'+name2save)
        torch.save(model.state_dict(), './'+path2save_weights+'/'+name2save+'/'+epoch+current_time+'.weights')

    if model2 is not None:
        maybe_mkdir_p(path2save_weights+'_SAW'+'/'+name2save)
        torch.save(model2.state_dict(), './'+path2save_weights+'_SAW'+'/'+name2save+'/'+epoch+current_time+'.weights')

    if losses2record is not None:
        maybe_mkdir_p(path2save_loss+'/'+name2save)
        with open('./'+path2save_loss+'/'+name2save+'/'+epoch+current_time+'.pickle', 'wb') as handle:
            pickle.dump(losses2record, handle, protocol=pickle.HIGHEST_PROTOCOL)  



def get_current_score(model, inputs, y_true, lpips , D,LDC,device_number = None):
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


def calculate_record_losses_hist(model, generator_val, losses2record, loss_type , lpips, D, LDC ,device_number = None):
    assert device_number is not None #calculate_record_losses_hist

    n = 0
    current_losses = 0
    for inputs, y_true in generator_val:
        current_losses += np.array(get_current_score(model, inputs, y_true, lpips=lpips,D=D, LDC=LDC,device_number= device_number))
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


def get_training_losses(device_number):
    fuse_or_avg = 'avg'
    #layer_select = [0,4]#[0,1,2,3,4]#[0,1,4]#[0,1,2,3,4]
    #layer_weights = [3/4,1,1,1,1/4]

    #layer_select = [0,3,4]#[0,1,2,3,4]#[0,1,4]#[0,1,2,3,4]
    #layer_weights = [6/8,1,1,1/8,1/8]

    layer_select = [0]#[0,1,2,3,4]
    layer_weights = [1]#[1/5,1/5,1/5,1/5,1/5]

    print(layer_select)

    training_losses =  [ l1().loss, 
                        vxm.losses.MSE().loss, 
                        ssim_loss().loss, 
                        NCC( device_number = device_number).loss, 
                        Wavelet_loss( device_number = device_number).loss, 
                        LDC_loss(device_number = device_number, fuse_or_avg = fuse_or_avg, layer_select = layer_select,layer_weights=layer_weights).loss,
                        Finite_different_loss( device_number = device_number).loss,
                        NCC_LDC_Loss(fuse_or_avg = fuse_or_avg, layer_select = layer_select,layer_weights=layer_weights, device_number = device_number).loss,
                        CannyFilterLoss(device_number = device_number).loss]  #L1 L2 SSIM NCC wavelet LDC NCCLDC

    #['loss_l1','loss_l2','loss_ssim', 'loss_ncc', 'loss_wl', 'loss_LDC', 'loss_NCCFED']
        
    field_regularization = Grad(penalty='l2').loss

    D = DISTS().to(device[device_number])
    lpips = LearnedPerceptualImagePatchSimilarity(net_type = 'vgg').to(device[device_number])
    #LDC_loss_val = LDC_loss(device_number = device_number, fuse_or_avg= 'fuse').loss
    LDC_loss_val = LDC_loss(device_number = device_number, fuse_or_avg = fuse_or_avg, layer_select = layer_select,layer_weights=layer_weights).loss

    return training_losses, field_regularization , D , lpips , LDC_loss_val


def create_generators(model_name, 
                      dataset, dataset_val, 
                      dataset_with_noise, dataset_with_noise_val,
                      dataset_with_noise_xdB, dataset_with_noise_val_xdB, 
                      dataset_tst, dataset_with_noise_tst_xdB,
                      batch_size,trg = 7, SNR = 6
                      ):
    
    prefix_train = 'train_'+'_'+str(SNR)+'dB_'+'_tr_val_ts'+'healthy_192021'
    prefix_val   = 'val_'+'_'+str(SNR)+'dB_'+'_tr_val_ts'+'healthy_192021'
    prefix_test  = 'test_'+'_'+str(SNR)+'dB_'+'_tr_val_ts'+'healthy_192021'
    # prefix_train = 'train_'+'_'+str(SNR)+'dB_'+'_tr_val_ts'+'digital_voxelsize_'+str(2)
    # prefix_val   = 'val_'+'_'+str(SNR)+'dB_'+'_tr_val_ts'+'digital_voxelsize_'+str(2)
    # prefix_test  = 'test_'+'_'+str(SNR)+'dB_'+'_tr_val_ts'+'digital_voxelsize_'+str(2)


    if model_name == "vxm_model":
        generator             = vxm_data_generator_parallel(dataset_with_noise,clear_data = dataset,batch_size=batch_size)# brightness_augmentation = False)#, data_shift=False, change_noise = False)
        generator_training    = vxm_data_generator_parallel(    dataset_with_noise_xdB,     clear_data = dataset,batch_size=14,    val=trg, change_noise = False, brightness_augmentation = False, data_shift=False, prefix_val = prefix_train)
        generator_val         = vxm_data_generator_parallel(dataset_with_noise_val_xdB, clear_data = dataset_val,batch_size=14,    val=trg, change_noise = False, brightness_augmentation = False, data_shift=False, prefix_val = prefix_val)#'val_')
        generator_test        = vxm_data_generator_parallel(dataset_with_noise_tst_xdB, clear_data = dataset_tst,batch_size=14,    val=trg, change_noise = False, brightness_augmentation = False, data_shift=False, prefix_val = prefix_test)#'val_')
        return generator, generator_val, generator_training, generator_test
    else:
        generator             = simple_avg_data_generator_parallel(dataset_with_noise,clear_data = dataset,batch_size=batch_size)# brightness_augmentation = False)#, data_shift=False, change_noise = False) #no brightness augmentation for digital phantom
        generator_training    = simple_avg_data_generator_parallel(    dataset_with_noise_xdB,     clear_data = dataset,batch_size=1,    val=trg,change_noise = False, brightness_augmentation = False, data_shift=False, prefix_val = prefix_train)
        generator_val         = simple_avg_data_generator_parallel(dataset_with_noise_val_xdB, clear_data = dataset_val,batch_size=1,    val=trg,change_noise = False, brightness_augmentation = False, data_shift=False, prefix_val = prefix_val)#'val_')
        generator_test        = simple_avg_data_generator_parallel(dataset_with_noise_tst_xdB, clear_data = dataset_tst,batch_size=1,    val=trg,change_noise = False, brightness_augmentation = False, data_shift=False, prefix_val = prefix_test)

        return generator, generator_val, generator_training, generator_test



import torch.optim as optim
from tqdm import tqdm

def training(data_train, data_val,data_test,
             path2save_weights, path2save_loss , path2weights=None, para_lists = None, device_number = None,save_result = False):

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
        p_note = params.get('note', '')

        print(params)
        training_losses, field_regularization , D, lpips , LDC_loss_val= get_training_losses(device_number)

        #adding noise for digital patients (ncc only)
        # _, _, dataset, dataset_val, _, _ = add_noise_of_certain_SNR(x_train_twochannels, x_test_twochannels, SNR = 50)
        # _, _, dataset_with_noise, dataset_with_noise_val, _,_ = add_noise_of_certain_SNR(x_train_twochannels, x_test_twochannels, mimic_sd = p_standard_deviation)

        #for other patients
        dataset, dataset_val, dataset_with_noise, dataset_with_noise_val = add_noise_of_certain_SNR(data_train, data_val,mimic_sd = p_standard_deviation)
        
        #verifiying on 6 dB
        _, _, dataset_with_noise_xdB, dataset_with_noise_val_xdB = add_noise_of_certain_SNR(data_train, data_val, SNR = 6)
        #testing on 6 dB
        dataset_tst, _, dataset_with_noise_xdB_tst, _  = add_noise_of_certain_SNR(data_test, data_test, SNR = 6)
        
        model = create_model(p_model, device_number, mode2train= p_mode , int_steps = int_steps, weights_path4load = path2weights).to(device[device_number])#.eval()
        
        #base_optimizer = torch.optim.Adam(model.parameters(), lr=p_learning_rate)#1e-4)
        base_optimizer = optim.SGD(model.parameters(), lr=p_learning_rate, momentum= p_SGDmomentum )#0.9)#1e-4)
        #annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max= 100, eta_min=0, last_epoch=-1)





        losses, weights = losses_weights(model_name=p_model,
                                        reg_lambda=p_lambda,
                                        training_losses=training_losses[loss_name_list.index(p_loss)],
                                        training_losses2=field_regularization)

        generator, generator_val, generator_training, generator_testing  = create_generators(p_model, 
                                                                                             dataset,                dataset_val, 
                                                                                             dataset_with_noise,     dataset_with_noise_val,
                                                                                             dataset_with_noise_xdB, dataset_with_noise_val_xdB, 
                                                                                             dataset_tst,            dataset_with_noise_xdB_tst,
                                                                                             batch_size= p_batch_size)

    ###########################################################################
        epoch_total_loss = []
        losses2record = {}
        #load losses2record from /home/Xuan/LGE_project/git_rewriten/loss_hist_meetingDec6_real_training256x192_c_with_noise_with_bri_with_shift_lambda_0.05_SWA_mask_bigbatchsize_parallel_general_mask_val_6dB_tr_val_ts/0.08_0.05_vxm_model_loss_LDC_bicubic_70_3_2500_None_None_0_0_0_0_0_0.9_annealing_scheduler_100_no_SWA/1150_20240301075540.pickle
        # with open('/home/Xuan/LGE_project/git_rewriten/loss_hist_meetingDec6_real_training256x192_c_with_noise_with_bri_with_shift_lambda_0.05_SWA_mask_bigbatchsize_parallel_general_mask_val_6dB_tr_val_ts/0.08_0.05_vxm_model_loss_LDC_bicubic_70_3_2500_None_None_0_0_0_0_0_0.9_annealing_scheduler_100_no_SWA/1150_20240301075540.pickle', 'rb') as handle:
        #     losses2record = pickle.load(handle)

        for epoch in tqdm(range(p_epochs), unit = "epoch"):
            
            ############################################# 
            model.train()
            for inputs, y_true in generator:

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
                        current_losses,current_loss_names = calculate_record_losses_hist(model, generator_val, losses2record, loss_type= 'val' ,lpips = lpips, D= D, LDC=LDC_loss_val,device_number= device_number)
                        _,_ = calculate_record_losses_hist(model, generator_training, losses2record, loss_type= 'tr', lpips = lpips, D= D,LDC=LDC_loss_val,device_number= device_number)
                        _,_ = calculate_record_losses_hist(model, generator_testing, losses2record, loss_type= 'ts', lpips = lpips, D= D,LDC=LDC_loss_val,device_number= device_number)
                        #calculate_record_losses_hist(model, generator_val, losses2record_SWA, losst_type= 'ts' ,lpips = lpips, D= D, LDC=LDC_loss_val,device_number= device_number)
                        #calculate_record_losses_hist(model, generator_training, losses2record_SWA, losst_type= 'tr', lpips = lpips, D= D,LDC=LDC_loss_val,device_number= device_number)
                        
            scheduler.step()  


            if save_result and epoch%10 == 0 and epoch > 0:
                save_model_loss_hist(params, model = model, losses2record = losses2record,  path2save_weights = path2save_weights, path2save_loss = path2save_loss, epoch = epoch)


        if save_result:
            save_model_loss_hist(params, model = model, losses2record = losses2record,  path2save_weights = path2save_weights, path2save_loss = path2save_loss)
            


        
def create_and_start_threading(target = training, 
                              data_train = data_train, 
                              data_val = data_val, 
                              data_test = data_test,
                              para_lists = None, device_number = None, save_result = False):

   assert device_number is not None #create_and_start_threadings

   thread1 = threading.Thread( target = target, args=(data_train, data_val, data_test,path2save_weights,path2save_loss, path2weights,para_lists, device_number,save_result) )

   thread1.start()
   
   return thread1


para_lists_candi = [

    {'standard_deviation': 0.08, 'lambda_value': 0.05, 'model': 'simple_avg_model', 'loss': 'loss_LDC', 'mode': 'bicubic', 'batch_size': 5, 'learning_rate': 3, 'epochs': 25,  
     'ReduceLROnPlateau_factor': 0, 'ReduceLROnPlateau_patience': 0, 'ReduceLROnPlateau_cooldown': 0, 'ReduceLROnPlateau_threshold': 0, 'ReduceLROnPlateau_eps': 0, 'SGDmomentum': 0.9, 'note': 'annealing_scheduler_100_no_SWA'},


]



para_lists = para_lists_candi

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


training(data_train, data_val, data_test,path2save_weights = path2save_weights, path2save_loss = path2save_loss , path2weights=path2weights, para_lists = para_lists[0:1], device_number = args.device_number,save_result = True)