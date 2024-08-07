import os
import time
from numba import jit, prange
import numpy as np
import random
from scripts.utils import SNR2standard_deviation, create_SNR_list

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning

# Ignore Numba deprecation warnings
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)


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


        base_SNR = SNR2standard_deviation(11,self.clear_data)
        print("base_SNR: "+str(base_SNR))
        #base_SNR = 0.1403170870464488#0.08853409662802346#15dB  #0.1403170870464488
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


def create_generators_val(model_name, 
                          dataset_test, dataset_with_noise_test, 
                          selected_target = 14, 
                          prefix_val = None,
                          brightness_augmentation = False, 
                          data_shift = False, 
                          change_noise = False):
    if model_name == "vxm_model":
        generator_val         = vxm_data_generator_parallel(dataset_with_noise_test,clear_data=dataset_test, val=selected_target, batch_size = 14, brightness_augmentation=brightness_augmentation, data_shift=data_shift, change_noise=change_noise, prefix_val=prefix_val)#'_'+str(SNR)+'dB_'+'digital'+'_')
        return generator_val
    else:
        generator_val = simple_avg_data_generator_parallel(dataset_with_noise_test,clear_data=dataset_test, val=selected_target, batch_size = 1, brightness_augmentation=brightness_augmentation, data_shift=data_shift, change_noise=change_noise, prefix_val=prefix_val)#'_'+str(SNR)+'dB_'+'digital'+'_')
        return generator_val
