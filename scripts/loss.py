import torch
import torchgeometry.losses.ssim as torch_ssim_loss
from DISTS_pytorch import DISTS
from scripts.variables import device


class l1:
    """
    L1 error loss.
    """
    def __init__(self, mask = None):
        self.mask = mask

    def loss(self, y_true, y_pred):

        y_true = torch.unsqueeze(torch.abs(y_true[:,0,...] + 1j*y_true[:,1,...]),1)
        y_pred = torch.unsqueeze(torch.abs(y_pred[:,0,...] + 1j*y_pred[:,1,...]),1)

        if self.mask is not None:
            return torch.mean(torch.abs(y_true - y_pred)*self.mask)
        else:
            return torch.mean(torch.abs(y_true - y_pred))
    
    

class ssim_loss:
    """
    SSIM error loss.
    """

    def loss(self, y_true, y_pred):
        #y_true = torch.abs(y_true)
        #y_pred = torch.abs(y_pred)
        return torch_ssim_loss(y_true, y_pred ,reduction = 'mean',window_size=11 , max_val = torch.max(torch.max(y_true),torch.max(y_pred)) )


class DISTS_loss:
    """
    DISTS loss.
    """
    def __init__(self, win=None, mask = None, device_number = None):
        self.device_number = device_number

    def loss(self, y_true, y_pred):
        y_true = (y_true-torch.min(y_true))/(torch.max(y_true)-torch.min(y_true))
        y_pred = (y_pred-torch.min(y_pred))/(torch.max(y_pred)-torch.min(y_pred))
        #y_true = torch.permute(torch.vstack((torch.permute(y_true,[1,0,2,3]),torch.permute(torch.zeros(y_true.shape),[1,0,2,3]))),[1,0,2,3])[:,0:3,...]
        #y_pred = torch.permute(torch.vstack((torch.permute(y_pred,[1,0,2,3]),torch.permute(torch.zeros(y_pred.shape),[1,0,2,3]))),[1,0,2,3])[:,0:3,...]

        #y_true = torch.cat((y_true,torch.zeros((y_true.shape[0], 1, y_true.shape[2], y_true.shape[3]))),dim=1)
        #y_pred = torch.cat((y_pred,torch.zeros((y_pred.shape[0], 1, y_pred.shape[2], y_pred.shape[3]))),dim=1)
        dummuy = torch.zeros((y_true.shape[0], 1, y_true.shape[2], y_true.shape[3])).to(device[device_number])
        y_true = torch.cat((y_true,dummuy),dim=1)
        y_pred = torch.cat((y_pred,dummuy),dim=1)      
        
        return D(y_true ,y_pred, require_grad=True, batch_average=True)        


import torch.nn.functional as F
import numpy as np
import math



class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, mask = None, device_number = None, debug = False):
        self.win = win
        self.mask = mask
        self.device_number = device_number
        self.debug = debug

    def loss(self, y_true, y_pred):
        
        Ii_all = torch.unsqueeze(torch.abs(y_true[:,0,...] + 1j*y_true[:,1,...]),1)
        Ji_all = torch.unsqueeze(torch.abs(y_pred[:,0,...] + 1j*y_pred[:,1,...]),1)
        
        #Ii_all = y_true
        #Ji_all = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, channel , *vol_shape]
        ch = list(Ii_all.size())[1]
        cc_all = None
        for ch_i in range(ch):
            Ii = torch.unsqueeze(Ii_all[:,ch_i,...],1)
            Ji = torch.unsqueeze(Ji_all[:,ch_i,...],1)
            ndims = len(list(Ii.size())) - 2
            assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

            # set window size
            win = [9] * ndims if self.win is None else self.win

            # compute filters
            sum_filt = torch.ones([1, 1, *win]).to(device[self.device_number])

            pad_no = math.floor(win[0] / 2)

            if ndims == 1:
                stride = (1)
                padding = (pad_no)
            elif ndims == 2:
                stride = (1, 1)
                padding = (pad_no, pad_no)
            else:
                stride = (1, 1, 1)
                padding = (pad_no, pad_no, pad_no)

            # get convolution function
            conv_fn = getattr(F, 'conv%dd' % ndims)

            # compute CC squares
            I2 = Ii * Ii
            J2 = Ji * Ji
            IJ = Ii * Ji

            I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = np.prod(win)
            win_size = win_size*ch
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + 1e-5)
            if self.mask is not None:
                cc = cc*self.mask
            cc = -torch.mean(cc)
            
            #print(cc)

            cc_all = cc if cc_all is None else (cc+cc_all)

        return cc_all


class Wavelet_loss:
    """
    Harr wavelet loss.
    """

    def __init__(self, mask = None, device_number = None):
        self.mask = mask
        self.device_number = device_number
        self.LL = torch.tensor([[[[1.,1.],[1.,1.]]]]).to(device[self.device_number])
        self.LH = torch.tensor([[[[1.,1.],[-1.,-1.]]]]).to(device[self.device_number])
        self.HL = torch.tensor([[[[1.,-1.],[1.,-1.]]]]).to(device[self.device_number])
        self.HH = torch.tensor([[[[1.,-1.],[-1.,1.]]]]).to(device[self.device_number])
        # self.LL = torch.tensor([[[[1., 1.],[ 1., 1.]],[[1., 1.],[ 1., 1.]]]]).to(device[self.device_number])
        # self.LH = torch.tensor([[[[1., 1.],[-1.,-1.]],[[1., 1.],[-1.,-1.]]]]).to(device[self.device_number])
        # self.HL = torch.tensor([[[[1.,-1.],[ 1.,-1.]],[[1.,-1.],[ 1.,-1.]]]]).to(device[self.device_number])
        # self.HH = torch.tensor([[[[1.,-1.],[-1., 1.]],[[1.,-1.],[-1., 1.]]]]).to(device[self.device_number])
        self.conv_fn = getattr(F, 'conv2d')

    def loss(self, y_true, y_pred):
        
        #Ii_all = torch.unsqueeze(torch.abs(y_true[:,0,...] + 1j*y_true[:,1,...]),1)
        #Ji_all = torch.unsqueeze(torch.abs(y_pred[:,0,...] + 1j*y_pred[:,1,...]),1)
        #[batch_size, channel , *vol_shape]
        img_wavelet = y_true - y_pred
        
        img_wavelet = torch.unsqueeze(torch.abs(img_wavelet[:,0,...] + 1j*img_wavelet[:,1,...]),1)
        
        img_LL = self.conv_fn(img_wavelet, self.LL, stride=(1,1), padding='same')
        img_LH = self.conv_fn(img_wavelet, self.LH, stride=(1,1), padding='same')
        img_HL = self.conv_fn(img_wavelet, self.HL, stride=(1,1), padding='same')
        img_HH = self.conv_fn(img_wavelet, self.HH, stride=(1,1), padding='same')
        if self.mask is not None:
            current_wavelet_loss_mean = 0*torch.mean(torch.abs(img_LL*self.mask))+ 1*torch.mean(torch.abs(img_LH*self.mask))+ 1*torch.mean(torch.abs(img_HL*self.mask))+ 1*torch.mean(torch.abs(img_HH*self.mask))
        else:
            current_wavelet_loss_mean = 0*torch.mean(torch.abs(img_LL))+ 1*torch.mean(torch.abs(img_LH))+ 1*torch.mean(torch.abs(img_HL))+ 1*torch.mean(torch.abs(img_HH))

        return current_wavelet_loss_mean

class Finite_different_loss:
    """
    Finite_different_loss.
    """

    def __init__(self, win=None, mask = None, device_number = None):
        self.win = win
        self.mask = mask
        self.device_number = device_number
        # self.V = torch.tensor([[[[1.,-2.,1.]]]]).to(device[self.device_number]) #[1,-1]
        # self.H = torch.tensor([[[[1.],[-2.],[1.]]]]).to(device[self.device_number])
        self.V = torch.tensor([[[[1.,-1.]]]]).to(device[self.device_number]) #[1,-1]
        self.H = torch.tensor([[[[1.],[-1.]]]]).to(device[self.device_number])
        self.conv_fn = getattr(F, 'conv2d')


    def loss(self, y_true, y_pred):

        cc_all = self.ncc_loss(y_true, y_pred)
        FD_all = self.FD_loss(y_true, y_pred)

        #print("cc",cc_all)
        #print("FD",FD_all)

        return cc_all+FD_all*10


    def FD_loss(self, y_true, y_pred):

        y_true_edge = self.edge_map(y_true)
        y_pred_edge = self.edge_map(y_pred)



        delta =  torch.abs(y_true_edge - y_pred_edge)

        
        if self.mask is not None:
            current_FD_loss_mean = torch.mean( delta*self.mask )
        else:
            current_FD_loss_mean = torch.mean( delta )

        return current_FD_loss_mean

    def edge_map(self, images):
        #[batch_size, channel , *vol_shape]
        images = images.to(torch.float32)
        image_m = torch.unsqueeze(torch.abs(images[:,0,...] + 1j*images[:,1,...]),1)

        img_V = self.conv_fn(image_m, self.V, stride=(1,1), padding='same')
        img_H = self.conv_fn(image_m, self.H, stride=(1,1), padding='same')

        return ((img_V)**2+(img_H)**2)**0.5 

    def ncc_loss(self, y_true, y_pred):
        
        Ii_all = torch.unsqueeze(torch.abs(y_true[:,0,...] + 1j*y_true[:,1,...]),1).to(torch.float32)
        Ji_all = torch.unsqueeze(torch.abs(y_pred[:,0,...] + 1j*y_pred[:,1,...]),1).to(torch.float32)
        
        #Ii_all = y_true
        #Ji_all = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, channel , *vol_shape]
        ch = list(Ii_all.size())[1]
        cc_all = None
        for ch_i in range(ch):
            Ii = torch.unsqueeze(Ii_all[:,ch_i,...],1)
            Ji = torch.unsqueeze(Ji_all[:,ch_i,...],1)
            ndims = len(list(Ii.size())) - 2
            assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

            # set window size
            win = [9] * ndims if self.win is None else self.win

            # compute filters
            sum_filt = torch.ones([1, 1, *win]).to(device[self.device_number])

            pad_no = math.floor(win[0] / 2)

            if ndims == 1:
                stride = (1)
                padding = (pad_no)
            elif ndims == 2:
                stride = (1, 1)
                padding = (pad_no, pad_no)
            else:
                stride = (1, 1, 1)
                padding = (pad_no, pad_no, pad_no)

            # get convolution function
            conv_fn = getattr(F, 'conv%dd' % ndims)

            # compute CC squares
            I2 = Ii * Ii
            J2 = Ji * Ji
            IJ = Ii * Ji

            I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = np.prod(win)
            win_size = win_size*ch
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + 1e-5)
            if self.mask is not None:
                cc = cc*self.mask
            cc = -torch.mean(cc)
            
            cc_all = cc if cc_all is None else (cc+cc_all)

        return cc_all




        
import torchvision.transforms as transforms
import os
from LDC.modelB4 import LDC


class LDC_loss:
    """
    LDC edge loss.
    """

    def __init__(self, win=None, mask = None,device_number = None, interpolation = None, test = False,
                fuse_or_avg = None, layer_select = [0,1,2,3,4], 
                layer_weights = [3/4,1,1,1,1/4],#[6/8,1,1,1/8,1/8],#[1,0.9,0.825,0.83,0.81],
                path2checkpoint = os.getcwd()+"/LDC/checkpoints/BRIND/16/16_model.pth"):
        self.win = win
        self.mask = mask
        self.device_number = device_number
        if interpolation is None or interpolation == "bilinear":
            self.LDC_resize =transforms.Resize((512,512),interpolation=transforms.InterpolationMode.BILINEAR).to(device[device_number])
        elif interpolation == "bicubic":
            self.LDC_resize =transforms.Resize((512,512),interpolation=transforms.InterpolationMode.BICUBIC).to(device[device_number])
        elif interpolation == "nearest":
            self.LDC_resize =transforms.Resize((512,512),interpolation=transforms.InterpolationMode.NEAREST).to(device[device_number])
        self.test = test
        self.LDC_model = LDC().to(device[device_number])
        self.LDC_model.load_state_dict(torch.load(path2checkpoint, map_location=device[device_number]))
        for param in self.LDC_model.parameters():
            param.requires_grad = False
        ####freeze the weights!!!!!!!
        self.fuse_or_avg = fuse_or_avg
        self.layer_select = layer_select
        self.layer_weights = layer_weights
        # self.layer_weights = []
        # for weights_iter in layer_select:
        #     self.layer_weights.append(layer_weights[weights_iter])


    def loss(self, y_true, y_pred):
        #[batch_size, channel , *vol_shape] #[14,2,192,192]

        loss_all = 0
        #y_true_fuse_all = {}
        #y_pred_fuse_all = {}
        for i in self.layer_select:
            loss_all = loss_all + torch.mean(torch.abs(self.edge_map(y_true,i)-self.edge_map(y_pred,i)))

        if self.test:
            y_true_fuse = self.edge_map(y_true)
            y_pred_fuse = self.edge_map(y_pred)
            return torch.mean(torch.abs(y_true_fuse-y_pred_fuse)) , y_true_fuse.detach().cpu().numpy() , y_pred_fuse.detach().cpu().numpy()
            #return torch.mean(torch.abs(y_true_fuse-y_pred_fuse)) - torch.mean(torch.abs(y_pred_fuse)), y_true_fuse.detach().cpu().numpy() , y_pred_fuse.detach().cpu().numpy()
        else:
            return loss_all
            #return torch.mean(torch.abs(y_true_fuse0-y_pred_fuse0)) + torch.mean(torch.abs(y_true_fuse3-y_pred_fuse3)) + torch.mean(torch.abs(y_true_fuse4-y_pred_fuse4)) 
            #return torch.mean(torch.abs(y_true_fuse-y_pred_fuse))  #plus
            #return torch.mean(torch.abs(y_true_fuse-y_pred_fuse)) - torch.mean(torch.abs(y_pred_fuse))

    def edge_map(self, images, single_layer = None):
        images = images.to(torch.float32)

        images = torch.abs(images[:,0,...]+1j*images[:,1,...])

        #set values about to 1 to 1 with torch.clamp
        #images = torch.clamp(images, min=0, max=1)

        images = torch.stack((images,images,images),1)

        images = ( (images - torch.min(images))*255 ) / ( (torch.max(images) - torch.min(images) + 1e-12)  )

        images = self.LDC_resize(images)

        images = self.LDC_model(images)

        
        if self.fuse_or_avg == 'avg':
        #avg:
            if single_layer == None:
                images_out = 0
                for layer_iter in self.layer_select:
                    images_out = images_out + torch.sigmoid(images[layer_iter])*(self.layer_weights[layer_iter])#*(1/self.layer_weights[layer_iter])
                images = images_out#/(len(self.layer_select))
            else:
                images = torch.sigmoid(images[single_layer])*(self.layer_weights[single_layer])#*(1/self.layer_weights[layer_iter])

        elif self.fuse_or_avg == 'fuse':
        #fues:
            images = torch.sigmoid(images[4])
        else:
            print("plase select fuse mode or avg mode for LDC loss")

        if self.mask is not None:
            return images*self.mask
        else:
            return images


class NCC_LDC_Loss:
    def __init__(self, mask = None, mask2 = None, device_number = None ,
                 
                fuse_or_avg = None, 
                layer_select  = [0,1,2,3,4], 
                layer_weights = [3/4,1,1,1,1/4], 
                
                weight_ncc=0.14/2,#0.0333,#0.107/2,
                weight_ldc=1/2,#1/2,#1/2,
                
                debug = False):
        """
        Initialize the combined loss class with NCC and LDC loss instances.

        Parameters:
        ncc_params (dict): Parameters to initialize the NCC loss.
        ldc_params (dict): Parameters to initialize the LDC loss.
        weight_ncc (float): Weight for the NCC loss component.
        weight_ldc (float): Weight for the LDC loss component.
        """
        self.ncc_loss = NCC(mask = mask, device_number = device_number)
        self.ldc_loss = LDC_loss(mask = mask2, device_number = device_number, fuse_or_avg = fuse_or_avg, layer_select = layer_select,layer_weights=layer_weights)
        self.weight_ncc = weight_ncc
        self.weight_ldc = weight_ldc

        self.debug = debug



    def loss(self, y_true, y_pred):
        """
        Compute the combined loss as a weighted sum of NCC and LDC losses.

        Parameters:
        y_true (torch.Tensor): The ground truth images.
        y_pred (torch.Tensor): The predicted images.

        Returns:
        torch.Tensor: The combined loss value.
        """
        loss_ncc = self.ncc_loss.loss(y_true, y_pred)
        loss_ldc = self.ldc_loss.loss(y_true, y_pred)

        if self.debug:
            try:
                print(self.weight_ncc * loss_ncc.item(),self.weight_ldc *loss_ldc.item())
            except:
                print(self.weight_ncc * loss_ncc, self.weight_ldc * loss_ldc)


        combined_loss = self.weight_ncc * loss_ncc + self.weight_ldc * loss_ldc
        return combined_loss



def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_thin_kernels(start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels

import torch.nn as nn
import cv2

class CannyFilterLoss(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device_number=None):
        super(CannyFilterLoss, self).__init__()
        # device

        if device_number is not None:
            self.device = torch.device("cuda:"+str(device_number) if torch.cuda.is_available() else "cpu")

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        gaussian_2D = gaussian_2D.reshape(1, 1, *gaussian_2D.shape).astype(np.float32)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight.data = self.gaussian_filter.weight.data.type(torch.float32)
        self.gaussian_filter.weight = nn.Parameter(torch.from_numpy(gaussian_2D).to(self.device), requires_grad=False)  # Disable gradient computations
        self.gaussian_filter.weight.data = self.gaussian_filter.weight.data.type(torch.float32)
        
        # sobel

        sobel_2D = get_sobel_kernel(k_sobel).astype(np.float32)
        sobel_2D_x = sobel_2D.reshape(1, 1, *sobel_2D.shape)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False).to(self.device)
        self.sobel_filter_x.weight.data = self.sobel_filter_x.weight.data.type(torch.float32)
        self.sobel_filter_x.weight = nn.Parameter(torch.from_numpy(sobel_2D_x).to(self.device), requires_grad=False)
        self.sobel_filter_x.weight.data = self.sobel_filter_x.weight.data.type(torch.float32)


        sobel_2D_T = sobel_2D.T
        sobel_2D_y = sobel_2D_T.reshape(1, 1, *sobel_2D_T.shape).astype(np.float32)
        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False).to(self.device)
        self.sobel_filter_y.weight.data = self.sobel_filter_y.weight.data.type(torch.float32)
        self.sobel_filter_y.weight = nn.Parameter(torch.from_numpy(sobel_2D_y).to(self.device), requires_grad=False)
        self.sobel_filter_y.weight.data = self.sobel_filter_y.weight.data.type(torch.float32)



    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian
        for c in range(C):
            #print(blurred[:, c:c+1].shape, self.gaussian_filter)
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        #grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = torch.pow(grad_x, 2) + torch.pow(grad_y, 2)
 


        return blurred, grad_x, grad_y, grad_magnitude
    
    def loss(self, y_true, y_pred,low_threshold=None, high_threshold=None, hysteresis=False, result_index = 3, use_magnitude = True):
        if use_magnitude:
            y_true = torch.abs(y_true[:,0:1,...]+1j*y_true[:,1:2,...])
            y_pred = torch.abs(y_pred[:,0:1,...]+1j*y_pred[:,1:2,...])
        y_true_edge = self.forward(y_true, low_threshold, high_threshold, hysteresis)
        y_pred_edge = self.forward(y_pred, low_threshold, high_threshold, hysteresis)

        #plot 
        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(10,10))

        # plt.subplot(2, 2, 1)
        # plt.imshow(y_true[0,0,...].detach().cpu().numpy(), cmap='gray')
        # plt.title("y_true")

        # plt.subplot(2, 2, 2)
        # plt.imshow(y_pred[0,0,...].detach().cpu().numpy(), cmap='gray')
        # plt.title("y_pred")

        # plt.subplot(2, 2, 3)
        # plt.imshow(y_true_edge[result_index][0,0,...].detach().cpu().numpy(), cmap='gray')
        # plt.title("y_true_edge")

        # plt.subplot(2, 2, 4)
        # plt.imshow(y_pred_edge[result_index][0,0,...].detach().cpu().numpy(), cmap='gray')
        # plt.title("y_pred_edge")

        # plt.tight_layout()
        # plt.show()


        loss = torch.mean(torch.abs((y_true_edge[result_index] - y_pred_edge[result_index])))
        #print(loss)
        return loss




class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None, mask = None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        if mask is not None:
            self.mask_dx = mask[:,1:]
            self.mask_dy = mask[1:,:]
        else:
            self.mask_dx = None
            self.mask_dy = None      

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        #dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            #dz = dz * dz
        #print(dx.shape) #torch.Size([10, 2, 192, 191])  dy torch.Size([10, 2, 191, 192])
        if self.mask_dx is None:
            d = torch.mean(dx) + torch.mean(dy) #+ torch.mean(dz)
        else:
            d = torch.mean(dx*self.mask_dx) + torch.mean(dy*self.mask_dy) #+ torch.mean(dz)
        grad = d / 2.0 #3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
    














#######################
import os
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm 
_ = torch.manual_seed(321)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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



def get_losses(device_number):
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
