from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import segmentation_models_pytorch as smp
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.mymodel.mymodel import get_my_network_from_plans



class nnUNetTrainer_segmamba(nnUNetTrainer):
    def initialize(self):
        if not self.was_initialized:
            ### Some hyperparameters for you to fiddle with
            self.initial_lr = 1e-4
            # 权重衰减用于控制正则化项的强度，权重衰减可以帮助防止模型过拟合
            self.weight_decay = 3e-5
            # 用于控制正样本（foreground）的过采样比例
            self.oversample_foreground_percent = 0.33
            self.num_iterations_per_epoch = 250
            self.num_val_iterations_per_epoch = 50
            self.num_epochs = 400
            self.current_epoch = 0
            # self.batch_size = 2
            
            # print(self.configuration_manager.patch_size)
            # self.configuration_manager.patch_size[0]=64
            # self.configuration_manager.patch_size[1]=64
            # self.configuration_manager.patch_size[2]=64
            # 针对ACDC数据集中，pathc_size不能被32整除导致报错：
            if((self.configuration_manager.patch_size[0] % 32)!=0 ):
                self.configuration_manager.patch_size[0]=self.configuration_manager.patch_size[0] +(32 - self.configuration_manager.patch_size[0] % 32)
            print(self.configuration_manager.patch_size)
            # print(self.configuration_manager.patch_size)

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            self.network = get_my_network_from_plans(self.plans_manager, self.dataset_json,
                                                    self.configuration_manager,
                                                    self.num_input_channels,
                                                    model = self.model).to(self.device)
            # from nnunetv2.torchsummary import summary
            # summary(self.network,input_size=(1,128,128,128))
            # exit()
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            #self.loss = my_get_dice_loss
            #self.loss = None
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")