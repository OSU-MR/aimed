import torch
import torch.nn as nn
import os 
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from scripts.variables import *

class model_avg_(nn.Module):
    def __init__(self, vol_shape, nb_features, src_feats, trg_feats, int_steps):
        super(model_avg_, self).__init__()
        self.vxm_block = vxm.networks.VxmDense(
            inshape=vol_shape, 
            nb_unet_features=nb_features, 
            src_feats=src_feats, 
            trg_feats=trg_feats, 
            int_steps=int_steps
        )

    def forward(self, t, *sources):
        outputs = [self.vxm_block(source, t) for source in sources]
        
        # Prepare the outputs in the exact required format
        mean_image = torch.mean(torch.stack([out[0] for out in outputs]), 0)
        flows = [out[1] for out in outputs]

        # Return the mean image and then each flow separately
        return [mean_image, *flows]


# # old code for creating the model, if the above code doesn't work, use this one
# class model_avg_(nn.Module):

#     def __init__(self, vol_shape, nb_features, src_feats, trg_feats, int_steps):
#         super(model_avg_, self).__init__()
#         self.vxm_block = vxm.networks.VxmDense(inshape=vol_shape,nb_unet_features=nb_features,src_feats=src_feats, trg_feats=trg_feats, int_steps=int_steps)

#     def forward(self, t, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14):
#         Unet1_out = self.vxm_block(s1, t)
#         Unet2_out = self.vxm_block(s2, t)
#         Unet3_out = self.vxm_block(s3, t)
#         Unet4_out = self.vxm_block(s4, t)
#         Unet5_out = self.vxm_block(s5, t)
#         Unet6_out = self.vxm_block(s6, t)
#         Unet7_out = self.vxm_block(s7, t)
#         Unet8_out = self.vxm_block(s8, t)
#         Unet9_out = self.vxm_block(s9, t)
#         Unet10_out = self.vxm_block(s10, t)
#         Unet11_out = self.vxm_block(s11, t)
#         Unet12_out = self.vxm_block(s12, t)
#         Unet13_out = self.vxm_block(s13, t)
#         Unet14_out = self.vxm_block(s14, t)
        
#         out = torch.stack((Unet1_out[0],
#                            Unet2_out[0],
#                            Unet3_out[0],
#                            Unet4_out[0],
#                            Unet5_out[0],
#                            Unet6_out[0],
#                            Unet7_out[0],
#                            Unet8_out[0],
#                            Unet9_out[0],
#                            Unet10_out[0],
#                            Unet11_out[0],
#                            Unet12_out[0],
#                            Unet13_out[0],
#                            Unet14_out[0]))
        
#         out = torch.mean(out,0)
#         return out,Unet1_out[1],Unet2_out[1],Unet3_out[1],Unet4_out[1],Unet5_out[1],Unet6_out[1],Unet7_out[1],Unet8_out[1],Unet9_out[1],Unet10_out[1],Unet11_out[1],Unet12_out[1],Unet13_out[1],Unet14_out[1]





def create_model( model_name = None, device_number = None, mode2train = None, 
                pre_trained_weight_path = None, int_steps = 5, device = device, vol_shape = (192,192), nb_features = nb_features):
    assert model_name != None
    #############"vxm_model"                                                      
    if model_name == "vxm_model":
        model = vxm.networks.VxmDense(inshape=vol_shape,nb_unet_features=nb_features,src_feats=2, trg_feats=2, int_steps=int_steps)
        model.transformer = vxm.layers.SpatialTransformer(size = tuple(vol_shape),mode=mode2train).to(device[device_number]) #'bilinear' | 'nearest' | 'bicubic'
        
        if pre_trained_weight_path is not None:
            model.load_state_dict(torch.load(pre_trained_weight_path, map_location = device[device_number]))

    #############

    #############"simple_avg_model"
    if model_name == "simple_avg_model" or model_name == "AiM_model":
        model = model_avg_(vol_shape = vol_shape, nb_features=nb_features, src_feats=2, trg_feats=2, int_steps=int_steps).to(device[device_number])
        model.vxm_block.transformer = vxm.layers.SpatialTransformer(size = tuple(vol_shape),mode=mode2train).to(device[device_number]) #'bilinear' | 'nearest' | 'bicubic'
        if pre_trained_weight_path is not None:
            try:
                model.vxm_block.load_state_dict(torch.load(pre_trained_weight_path, map_location = device[device_number]))
            except:
                model.load_state_dict(torch.load(pre_trained_weight_path, map_location = device[device_number]))

    return model



