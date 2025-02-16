import segmentation_models_pytorch as smp
import torch
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from nnunetv2.mymodel.unetr import UNETR
from nnunetv2.mymodel.unet_3d2 import UNet
from nnunetv2.mymodel.unet_3d2 import DoubleConv3D, Down3D, Up3D, Tail3D
from nnunetv2.mymodel.CoTr.ResTranUnet import ResTranUnet
from nnunetv2.mymodel.TransBTS.TransBTS import my_TransBTS
from nnunetv2.mymodel.umamba.umamba_bot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.mymodel.segmamba.segmamba import SegMamba
from nnunetv2.mymodel.umamba.umamba_enc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.mymodel.dual_mamba import get_dual_mamba_from_plans



def get_my_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           model: str, enable_deep_supervision: bool = False):
    label_manager = plans_manager.get_label_manager(dataset_json)
    
    if(model == '3dunet'):

        model = UNet(stem=DoubleConv3D,
                    down=Down3D,
                    up=Up3D,
                    tail=Tail3D,
                    width=[64,128,256,512],
                    conv_builder=DoubleConv3D,
                    n_channels=num_input_channels,
                    n_classes=label_manager.num_segmentation_heads)
        
    elif(model == 'unetr'): 
        # only support 3D image
        model = UNETR(in_channels=num_input_channels,
                    out_channels=label_manager.num_segmentation_heads,
                    img_size=configuration_manager.patch_size)   
   
    elif(model== 'transbts'):
        # The patch size here represents image size
        model = my_TransBTS(num_classes=label_manager.num_segmentation_heads,in_channels=num_input_channels,patch_size=[configuration_manager.patch_size[0],configuration_manager.patch_size[1],configuration_manager.patch_size[2]])
    
    elif(model == 'CoTr'):
        # only support 3D image
        # verse 和 totalseg 数据集上用lr=0.01
        model = ResTranUnet(norm_cfg= 'IN' , activation_cfg= 'LeakyReLU' ,img_size=configuration_manager.patch_size, in_channels=num_input_channels ,num_classes=label_manager.num_segmentation_heads)


    elif(model == 'umamba'):
        model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels,deep_supervision=False)
        
        
    elif(model == 'segmamba'):
        model = SegMamba(in_chans=num_input_channels, out_chans=label_manager.num_segmentation_heads,)

     
    elif(model == 'umamba_enc'):
        model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels,deep_supervision=False)
        
    elif(model == 'dual_mamba'):
        model = get_dual_mamba_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels,deep_supervision = enable_deep_supervision)


    return model

### important:need pip install einops==0.3.0 to run unetr

