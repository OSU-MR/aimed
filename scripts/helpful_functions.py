import numpy as np
import time

def two2one(arr):
    return arr[0,...]+1j*arr[1,...]
def two2one2(arr):
    return arr[:,0,...]+1j*arr[:,1,...]

def one2two(arr):
    return np.stack((np.real(arr),np.imag(arr)))

def normalize_energy(arr , percentage = 95):
    arr_angle = np.angle(arr)
    arr_m     = np.abs(arr)
    length = len(arr_m.shape)
    arr_m_shape = arr_m.shape
    percentile = np.percentile(arr_m,percentage,axis=[length-2,length-1])
    arr_m = arr_m.reshape(arr_m_shape[0],-1)
    for i in range(arr_m_shape[0]):
        arr_m[i] = np.divide(arr_m[i],percentile[i])
        arr_normalized = arr_m.reshape(arr_m_shape)*np.exp(1j*arr_angle)

    return arr_normalized

def normalize_complex_arr(arr):
    arr_angle = np.angle(arr)
    arr_m     = np.abs(arr)
    arr_m_max = np.max(arr_m)
    arr_m_min = np.min(arr_m)
    arr_m = (arr_m - arr_m_min)/(arr_m_max - arr_m_min)
    arr_normalized = arr_m*np.exp(1j*arr_angle)

    return arr_normalized

def delete_zero_sets(arr):
    arr_cleaned = arr.copy()
    arr = arr.reshape(arr.shape[0],arr.shape[1],arr.shape[2], -1)
    arr = np.sum(arr,axis = len(arr.shape) - 1)
    arr = abs(np.prod(arr,axis = (1,2)))
    arr = np.where(arr == 0)[0]
    arr_cleaned = np.delete(arr_cleaned,arr,axis = 0)
    return arr_cleaned

def put_the_best_to_the_first_arr(arr):
    arr_shape = arr.shape
    for dataset in range(arr_shape[0]):
        for data_or_ref in range(arr_shape[2]):
            arr_mean  = np.mean(arr[dataset, :, data_or_ref,:,:],axis = 0)
            arr_min_value = np.zeros((8))
            for images in range(arr_shape[1]):
                arr_min_value[images] = np.sum(arr_mean - arr[dataset, images, data_or_ref,:,:])
            
            first_image = arr[dataset, 0 , data_or_ref,:,:].copy()
            arr[dataset, 0 , data_or_ref,:,:]  = arr[dataset, np.argmin(arr_min_value) , data_or_ref,:,:].copy()
            arr[dataset, np.argmin(arr_min_value) , data_or_ref,:,:] = first_image.copy()   
            
    return arr

def SNR(image_without_noise,image_with_noise):
    return 10*np.log10((np.sum(( abs(image_without_noise) )**2) / np.sum(( abs(image_without_noise-image_with_noise) )**2)))


def noisy(noise_typ,image,standard_deviation, correction_map = None):
    np.random.seed(round((time.time()*1e7)%100000))
    ch,row,col = image.shape
    if noise_typ == "gauss":
        mean = 0
        gauss = np.random.normal(mean,standard_deviation,(ch,row,col))
        gauss = gauss.reshape(ch,row,col)
        if correction_map is not None:
            gauss = gauss*correction_map
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        gauss = np.random.randn(ch,row,col)
        gauss = gauss.reshape(ch,row,col)        
        noisy = image + image * gauss
