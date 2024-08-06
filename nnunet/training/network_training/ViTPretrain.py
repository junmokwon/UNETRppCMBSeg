#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params, get_default_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from simmim.vit import SimMIM
from simmim.mask_generator import MaskGenerator


class SimMIMLoss(nn.Module):
    """
    SimMIMLoss based on: "Xie et al.,
    SimMIM: A Simple Framework for Masked Image Modeling"
    """
    def forward(self, output, data, masks, patch_size=16, eps=1e-5):
        masks = masks.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = torch.nn.functional.l1_loss(data, output, reduction='none')
        loss = (loss_recon * masks).sum() / (masks.sum() + eps)
        return loss


class ViTPretrain(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = SimMIMLoss()
        self.mask_generator = None

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()
        self.plans['plans_per_stage'][self.stage]['patch_size'] = np.array([384, 320])
        self.crop_size = np.array([384, 320])

        self.process_plans(self.plans)

        self.setup_DA_params()

        if training:
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr, self.dl_val,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def initialize_network(self):
        self.network = SimMIM(in_channels=self.num_input_channels,
                              img_size=self.plans['plans_per_stage'][self.stage]['patch_size'].tolist(),
                              )
        if torch.cuda.is_available():
            self.network.cuda()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']

        data = maybe_to_torch(data)

        if torch.cuda.is_available():
            data = to_cuda(data)

        self.optimizer.zero_grad()
        mask_generator = self.mask_generator

        if self.fp16:
            with autocast():
                masks = to_cuda(torch.as_tensor(np.stack([
                    mask_generator() for _ in range(data.size(0))
                ], axis=0))).float()
                output = self.network(data, masks)
                l = self.loss(output, data, masks)
                del masks

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            masks = to_cuda(torch.as_tensor(np.stack([
                mask_generator() for _ in range(data.size(0))
            ], axis=0))).float()
            output = self.network(data, masks)
            l = self.loss(output, data, masks)
            del masks

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, data)

        del data

        return l.detach().cpu().numpy()
    
    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """
        pass
    
    def on_epoch_end(self):
        self.finish_online_evaluation()

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.epoch < self.max_num_epochs
        return continue_training


class ViTPretrainPatch16Ratio5(ViTPretrain):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_generator = MaskGenerator(mask_patch_size=16, model_patch_size=16, mask_ratio=0.5)


class ViTPretrainPatch16Ratio6(ViTPretrain):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_generator = MaskGenerator(mask_patch_size=16, model_patch_size=16, mask_ratio=0.6)


class ViTPretrainPatch32Ratio5(ViTPretrain):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_generator = MaskGenerator(mask_patch_size=32, model_patch_size=16, mask_ratio=0.5)


class ViTPretrainPatch32Ratio6(ViTPretrain):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_generator = MaskGenerator(mask_patch_size=32, model_patch_size=16, mask_ratio=0.6)


class ViTPretrainPatch64Ratio5(ViTPretrain):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_generator = MaskGenerator(mask_patch_size=64, model_patch_size=16, mask_ratio=0.5)


class ViTPretrainPatch64Ratio6(ViTPretrain):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_generator = MaskGenerator(mask_patch_size=64, model_patch_size=16, mask_ratio=0.6)
