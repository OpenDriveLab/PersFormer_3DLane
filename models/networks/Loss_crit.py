# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# Loss_crit.py       The PersFormer Authors        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
# 
# See:
# https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection/blob/master/networks/Loss_crit.py
#
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax


class Laneline_loss_3D(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is based on real 3D X, Y, Z.

    loss = loss1 + loss2 + loss2
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, anchor_dim, pred_cam):
        super(Laneline_loss_3D, self).__init__()
        self.num_types = num_types
        self.anchor_dim = anchor_dim
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (2K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :-1]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :-1]

        loss1 = -torch.sum(gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
                           (torch.ones_like(gt_class)-gt_class) *
                           torch.log(torch.ones_like(pred_class)-pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*(pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss1+loss2
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss1+loss2+loss3


class Laneline_loss_gflat(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam):
        super(Laneline_loss_gflat, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class)-gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss0+loss1+loss2
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3

class Laneline_loss_gflat_withdict(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam):
        super(Laneline_loss_gflat_withdict, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class)-gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss0+loss1+loss2, {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2}
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3, {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2, 'cam_pred_loss': loss3}

class Laneline_loss_gflat_multiclass(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam, num_category, no_3d, loss_dist):
        super(Laneline_loss_gflat_multiclass, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.num_category = num_category
        self.no_3d = no_3d
        self.loss_dist = loss_dist      # a list, weight for loss0, loss1, loss2, loss3
        if no_3d:
            self.anchor_dim = self.num_y_steps + num_category
        else:
            self.anchor_dim = 3*self.num_y_steps + num_category
        self.pred_cam = pred_cam
        
    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam=None, gt_hcam=None, pred_pitch=None, gt_pitch=None):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_category  = pred_3D_lanes[:, :, :, self.anchor_dim - self.num_category:]
        gt_category_onehot  = gt_3D_lanes[:, :, :, self.anchor_dim - self.num_category:]
        if self.no_3d:
            pred_anchors = pred_3D_lanes[:, :, :, :self.num_y_steps]
            gt_anchors = gt_3D_lanes[:, :, :, :self.num_y_steps]
            pred_visibility = torch.ones_like(pred_anchors)
            gt_visibility = torch.ones_like(gt_anchors)
        else:
            pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
            pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
            gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
            gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # valid_category_weight = torch.sum(torch.mul(pred_category, gt_category_onehot), dim=-1)
        valid_category_weight = torch.sum(gt_category_onehot[:, :, :, 1:], dim=-1).unsqueeze(-1)

        # cross-entropy loss for visibility
        # loss0 = -torch.sum(
        #     gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
        #     (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
        #     torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        loss0 = -torch.sum(
            valid_category_weight * gt_visibility * torch.log(pred_visibility + torch.tensor(1e-9)) +
            valid_category_weight * (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) * 
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9))) / self.num_y_steps
        # weights = np.linspace(1.0, 2.0, num=self.num_y_steps)
        # weights = torch.from_numpy(weights).to(gt_visibility.device)
        # weights = weights.expand(sizes[0], sizes[1], self.num_types, -1)
        # binary_cross_entropy_loss = nn.BCELoss(weight=weights, reduction='sum')
        # loss0 = binary_cross_entropy_loss(pred_visibility, gt_visibility)
        
        # cross-entropy loss for lane probability
        # loss1 = -torch.sum(
        #     gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
        #     (torch.ones_like(gt_class)-gt_class) *
        #     torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))

        # balance categories
        num_category = pred_category.shape[-1]
        weight_category = 1.0 - torch.sum(gt_category_onehot.reshape(-1, num_category), dim=0) / torch.sum(gt_category_onehot)
        cross_entropy_loss = nn.CrossEntropyLoss(weight=weight_category, reduction='sum')
        # cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        gt_category_onehot2class = torch.argmax(gt_category_onehot, dim=-1)
        pred_category = pred_category.reshape(-1, pred_category.shape[-1])
        gt_category_onehot2class = gt_category_onehot2class.reshape(-1)
        loss1 = cross_entropy_loss(pred_category, gt_category_onehot2class)

        # print("pred_category: ", pred_category)
        # print("gt_category: ", gt_category_onehot)
        # print("gt_visibility: ", gt_visibility)
        # log_prob = nn.functional.log_softmax(pred_category, dim=-1)
        # loss1 = -torch.sum(log_prob * gt_category_onehot)

        # applying L1 norm does not need to separate X and Z
        # loss2 = torch.sum(torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3) *
        #                              (pred_anchors-gt_anchors), p=1, dim=3))
        if self.no_3d:
            # only x offsets
            loss2 = torch.sum(torch.norm(valid_category_weight*gt_visibility*(pred_anchors-gt_anchors), p=1, dim=3))
        else:
            # x/z offsets
            loss2 = torch.sum(torch.norm(valid_category_weight*torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors-gt_anchors), p=1, dim=3))
        
        # Batch mean
        # loss0 /= sizes[0]
        # loss1 /= sizes[0]
        # loss2 /= sizes[0]
        # loss0 = self.loss_dist[0] * loss0
        # loss1 = self.loss_dist[1] * loss1
        # loss2 = self.loss_dist[2] * loss2

        if not self.pred_cam:
            return self.loss_dist[0]*loss0 + self.loss_dist[1]*loss1 + self.loss_dist[2]*loss2, {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2}
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3, {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2, 'cam_pred_loss': loss3}

class Laneline_loss_gflat_novis_withdict(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam):
        super(Laneline_loss_gflat_novis_withdict, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam=None, gt_hcam=None, pred_pitch=None, gt_pitch=None):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        # print("pred_3D_lanes shape: ", pred_3D_lanes.size())
        # print("gt_3D_lanes shape: ", gt_3D_lanes.size())
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class)-gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss1+loss2, {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2}
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3, {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2, 'cam_pred_loss': loss3}


class Laneline_loss_gflat_3D(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.
    The X' Y' and Z estimation will be transformed to real X, Y to compare with ground truth. An additional loss in
    X, Y space is expected to guide the learning of features to satisfy the geometry constraints between two spaces

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, batch_size, num_types, anchor_x_steps, anchor_y_steps, x_off_std, y_off_std, z_std, pred_cam=False, no_cuda=False):
        super(Laneline_loss_gflat_3D, self).__init__()
        self.batch_size = batch_size
        self.num_types = num_types
        self.num_x_steps = anchor_x_steps.shape[0]
        self.num_y_steps = anchor_y_steps.shape[0]
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam

        # prepare broadcast anchor_x_tensor, anchor_y_tensor, std_X, std_Y, std_Z
        tmp_zeros = torch.zeros(self.batch_size, self.num_x_steps, self.num_types, self.num_y_steps)
        self.x_off_std = torch.tensor(x_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.y_off_std = torch.tensor(y_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.z_std = torch.tensor(z_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = torch.tensor(anchor_x_steps.astype(np.float32)).reshape(1, self.num_x_steps, 1, 1) + tmp_zeros
        self.anchor_y_tensor = torch.tensor(anchor_y_steps.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = self.anchor_x_tensor/self.x_off_std
        self.anchor_y_tensor = self.anchor_y_tensor/self.y_off_std

        if not no_cuda:
            self.z_std = self.z_std.cuda()
            self.anchor_x_tensor = self.anchor_x_tensor.cuda()
            self.anchor_y_tensor = self.anchor_y_tensor.cuda()

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor values N x ipm_w/8 x 3 x 2K, visibility N x ipm_w/8 x 3 x K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class) - gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(
            torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3)*(pred_anchors-gt_anchors), p=1, dim=3))

        # compute loss in real 3D X, Y space, the transformation considers offset to anchor and normalization by std
        pred_Xoff_g = pred_anchors[:, :, :, :self.num_y_steps]
        pred_Z = pred_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        gt_Xoff_g = gt_anchors[:, :, :, :self.num_y_steps]
        gt_Z = gt_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        pred_hcam = pred_hcam.reshape(self.batch_size, 1, 1, 1)
        gt_hcam = gt_hcam.reshape(self.batch_size, 1, 1, 1)

        pred_Xoff = (1 - pred_Z * self.z_std / pred_hcam) * pred_Xoff_g - pred_Z * self.z_std / pred_hcam * self.anchor_x_tensor
        pred_Yoff = -pred_Z * self.z_std / pred_hcam * self.anchor_y_tensor
        gt_Xoff = (1 - gt_Z * self.z_std / gt_hcam) * gt_Xoff_g - gt_Z * self.z_std / gt_hcam * self.anchor_x_tensor
        gt_Yoff = -gt_Z * self.z_std / gt_hcam * self.anchor_y_tensor
        loss3 = torch.sum(
            torch.norm(
                gt_class * torch.cat((gt_visibility, gt_visibility), 3) *
                (torch.cat((pred_Xoff, pred_Yoff), 3) - torch.cat((gt_Xoff, gt_Yoff), 3)), p=1, dim=3))

        if not self.pred_cam:
            return loss0+loss1+loss2+loss3
        loss4 = torch.sum(torch.abs(gt_pitch-pred_pitch)) + torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3+loss4


# unit test
if __name__ == '__main__':
    num_types = 3

    # for Laneline_loss_3D
    print('Test Laneline_loss_3D')
    anchor_dim = 2*6 + 1
    pred_cam = True
    criterion = Laneline_loss_3D(num_types, anchor_dim, pred_cam)
    criterion = criterion.cuda()

    pred_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(8).float().cuda()
    gt_pitch = torch.ones(8).float().cuda()
    pred_hcam = torch.ones(8).float().cuda()
    gt_hcam = torch.ones(8).float().cuda()

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)
    print(loss)

    # for Laneline_loss_gflat
    print('Test Laneline_loss_gflat')
    num_y_steps = 6
    anchor_dim = 3*num_y_steps + 1
    pred_cam = True
    criterion = Laneline_loss_gflat(num_types, num_y_steps, pred_cam)
    criterion = criterion.cuda()

    pred_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(8).float().cuda()
    gt_pitch = torch.ones(8).float().cuda()
    pred_hcam = torch.ones(8).float().cuda()
    gt_hcam = torch.ones(8).float().cuda()

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)

    print(loss)

    # for Laneline_loss_gflat_3D
    print('Test Laneline_loss_gflat_3D')
    batch_size = 8
    anchor_x_steps = np.linspace(-10, 10, 26, endpoint=True)
    anchor_y_steps = np.array([3, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    num_y_steps = anchor_y_steps.shape[0]
    x_off_std = np.ones(num_y_steps)
    y_off_std = np.ones(num_y_steps)
    z_std = np.ones(num_y_steps)
    pred_cam = True
    criterion = Laneline_loss_gflat_3D(batch_size, num_types, anchor_x_steps, anchor_y_steps, x_off_std, y_off_std, z_std, pred_cam, no_cuda=False)
    # criterion = criterion.cuda()

    anchor_dim = 3*num_y_steps + 1
    pred_3D_lanes = torch.rand(batch_size, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(batch_size, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(batch_size).float().cuda()
    gt_pitch = torch.ones(batch_size).float().cuda()
    pred_hcam = torch.ones(batch_size).float().cuda()*1.5
    gt_hcam = torch.ones(batch_size).float().cuda()*1.5

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)

    print(loss)