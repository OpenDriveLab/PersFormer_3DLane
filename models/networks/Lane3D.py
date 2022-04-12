# ==============================================================================
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

import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import *
from .libs.layers import *

# initialize base_grid with different sizes can adapt to different sizes
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        # self.N, self.H, self.W = size_ipm
        self.H, self.W = size_ipm
        # self.im_h = im_h
        # self.im_w = im_w
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        # use M only to decide the type not value
        # self.base_grid = M.new(self.N, self.H, self.W, 3)
        # self.base_grid[:, :, :, 0] = torch.ger(
        #         torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        # self.base_grid[:, :, :, 1] = torch.ger(
        #         linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        # self.base_grid[:, :, :, 2] = 1
        self.base_grid = torch.zeros(self.H, self.W, 3)
        self.base_grid[:, :, 0] = torch.ger(torch.ones(self.H), linear_points_W)
        self.base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(self.W))
        self.base_grid[:, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()
            # self.im_h = self.im_h.cuda()
            # self.im_w = self.im_w.cuda()

    def forward(self, M):
        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa
        # grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        # grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        grid = torch.matmul(self.base_grid.view(self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((-1, self.H, self.W, 2))
        #
        """
        output grid to be used for grid_sample. 
            1. grid specifies the sampling pixel locations normalized by the input spatial dimensions.
            2. pixel locations need to be converted to the range (-1, 1)
        """
        grid = (grid - 0.5) * 2
        return grid


# initialize base_grid with different sizes can adapt to different sizes
class RefPntsGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        # self.N, self.H, self.W = size_ipm
        self.H, self.W = size_ipm
        # self.im_h = im_h
        # self.im_w = im_w
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        # use M only to decide the type not value
        # self.base_grid = M.new(self.N, self.H, self.W, 3)
        # self.base_grid[:, :, :, 0] = torch.ger(
        #         torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        # self.base_grid[:, :, :, 1] = torch.ger(
        #         linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        # self.base_grid[:, :, :, 2] = 1
        self.base_grid = torch.zeros(self.H, self.W, 3)
        self.base_grid[:, :, 0] = torch.ger(torch.ones(self.H), linear_points_W)
        self.base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(self.W))
        self.base_grid[:, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()
            # self.im_h = self.im_h.cuda()
            # self.im_w = self.im_w.cuda()

    def forward(self, M):
        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa
        # grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        # grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        grid = torch.matmul(self.base_grid.view(self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        # this is (w, h) version of grid, above is (h, w)
        # grid = torch.index_select(grid, 2, torch.LongTensor([1,0]))
        return grid


# initialize base_grid with different sizes can adapt to different sizes
class RefPntsNormGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        self.H, self.W = size_ipm
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        self.base_grid = torch.zeros(self.H, self.W, 3)
        self.base_grid[:, :, 0] = torch.ger(torch.ones(self.H), linear_points_W)
        self.base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(self.W))
        self.base_grid[:, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()

    def forward(self, M):
        grid = torch.matmul(self.base_grid.view(self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        eps = 1e-5
        x_max = torch.max(grid[:, :, 0])
        y_max = torch.max(grid[:, :, 1])
        grid[:, :, 0] = torch.div(grid[:, :, 0], torch.maximum(x_max, torch.ones_like(grid[:, :, 0]) * eps))
        grid[:, :, 1] = torch.div(grid[:, :, 1], torch.maximum(y_max, torch.ones_like(grid[:, :, 1]) * eps))
        return grid



# initialize base_grid with different sizes can adapt to different sizes
class RefPntsNoGradGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        self.H, self.W = size_ipm
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        self.base_grid = torch.zeros(self.H, self.W, 3)
        self.base_grid[:, :, 0] = torch.ger(torch.ones(self.H), linear_points_W)
        self.base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(self.W))
        self.base_grid[:, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()

    def forward(self, M):
        with torch.no_grad():
            grid = torch.matmul(self.base_grid.view(self.H * self.W, 3), M.transpose(1, 2))
            grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        return grid


# Sub-network corresponding to the top view pathway
# class TopViewPathway(nn.Module):
#     def __init__(self, batch_norm=False, init_weights=True):
#         super(TopViewPathway, self).__init__()
#         self.features1 = make_layers(['M', 128, 128, 128], 128, batch_norm)
#         self.features2 = make_layers(['M', 256, 256, 256], 256, batch_norm)
#         self.features3 = make_layers(['M', 256, 256, 256], 512, batch_norm)

#         if init_weights:
#             self._initialize_weights()

#     def forward(self, a, b, c, d):
#         x = self.features1(a)
#         feat_1 = x
#         x = torch.cat((x, b), 1)
#         x = self.features2(x)
#         feat_2 = x
#         x = torch.cat((x, c), 1)
#         x = self.features3(x)
#         feat_3 = x
#         x = torch.cat((x, d), 1)
#         return x, feat_1, feat_2, feat_3

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 # if m.bias is not None:
#                 #     nn.init.constant_(m.bias, 0)
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

# Sub-network corresponding to the top view pathway
class TopViewPathway(nn.Module):
    def __init__(self, input_channels, num_proj, batch_norm=False, init_weights=True, use_proj=True):
        super(TopViewPathway, self).__init__()

        self.input_channels = input_channels
        self.num_proj = num_proj
        self.use_proj = use_proj

        self.features = nn.ModuleList()
        for i in range(num_proj - 1):
            layers = []
            if self.use_proj:
                output_channels = input_channels if i < num_proj - 2 else input_channels // 2
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                if i < num_proj - 2:
                    input_channels *= 2 # because concatenate projected features
            else:
                output_channels = input_channels * 2 if i < num_proj - 2 else input_channels
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                input_channels = output_channels
            self.features.append(nn.Sequential(*layers))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         print("Max weight in 3D TopViewPathway: ", torch.max(torch.abs(m.weight)))
        #         if m.weight.grad != None:
        #             print("Max weight grad in 3D TopViewPathway: ", torch.max(torch.abs(m.weight.grad)))

        outs = []
        x = input[0]
        if len(self.features) > 0:
            for i, feature in enumerate(self.features):
                x = feature(x)
                feat = x
                if self.use_proj:
                    x = torch.cat((x, input[i + 1]), 1)
                # print("MD features[{}] shape: ".format(str(i)), x.size())
                outs.append(feat)
                # if self.features[i][0].weight.grad != None and torch.max(self.features[i][0].weight.grad) > 100:
                #     print("feature[{}] weights grad > 100".format(i))
                #     # print("feature[{}] weights grad: ".format(i), self.features[i][0].weight.grad)
        else:
            outs.append(x)
        return x, outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N × 1 × 3 ·(3 · K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, input_channels, num_lane_type, num_y_steps, num_category,
                 fmap_mapping_interp_index, fmap_mapping_interp_weight, no_3d=False, batch_norm=False, no_cuda=False):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.num_y_steps = num_y_steps
        self.no_3d = no_3d
        if no_3d:
            self.anchor_dim = self.num_y_steps + num_category   # (x) + category conf
        else:
            # self.anchor_dim = 3*self.num_y_steps + 1
            self.anchor_dim = 3*self.num_y_steps + num_category # (x, z, vis) + category conf
        self.num_category = num_category  

        layers = []
        layers += make_one_layer(input_channels, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
        # TODO: use large kernel_size in x or fc layer to estimate z with global parallelism
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(128, self.num_lane_type*self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))]
        self.dim_rt = nn.Sequential(*dim_rt_layers)
        
        self.use_default_anchor = True
        if fmap_mapping_interp_index is not None and fmap_mapping_interp_weight is not None:
            self.use_default_anchor = False
            self.fmap_mapping_interp_index = torch.tensor(fmap_mapping_interp_index)
            self.fmap_mapping_interp_weight = torch.tensor(fmap_mapping_interp_weight)
            # if not no_cuda:
            #     self.fmap_mapping_interp_index = self.fmap_mapping_interp_index.cuda()
            #     self.fmap_mapping_interp_weight = self.fmap_mapping_interp_weight.cuda()

    def forward(self, x):
        if not self.use_default_anchor:
            # multi-gpu setting
            batch_size, channel, fmap_h, fmap_w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            sheared_feature_map = torch.zeros((batch_size, channel, fmap_h, fmap_w*6)).to(x.device)
            v_arange = torch.arange(fmap_h).unsqueeze(dim=1).repeat(1,fmap_w*6).to(x.device)
            self.fmap_mapping_interp_index = self.fmap_mapping_interp_index.to(x.device)
            self.fmap_mapping_interp_weight = self.fmap_mapping_interp_weight.to(x.device)

            for batch_idx, x_feature_map in enumerate(x):
                # if True:
                # print("v_arange device: " + str(v_arange.device))
                # print("self.fmap_mapping_interp_index device: " + str(self.fmap_mapping_interp_index.device))
                # print("self.fmap_mapping_interp_weight device: " + str(self.fmap_mapping_interp_weight.device))
                # print("sheared_feature_map device: " + str(sheared_feature_map.device))
                # print("batch_idx device: " + str(v_arange.device))
                # print("x_feature_map device: " + str(x_feature_map.device))

                sheared_feature_map[batch_idx] = \
                    x_feature_map[:, v_arange, self.fmap_mapping_interp_index[:,:,0]] * self.fmap_mapping_interp_weight[:,:,0] + \
                    x_feature_map[:, v_arange, self.fmap_mapping_interp_index[:,:,1]] * self.fmap_mapping_interp_weight[:,:,1]
            x = torch.cat((x, sheared_feature_map), dim=3)

        x = self.features(x)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        # TODO: this only works with fpn_out_channels=128 & num_proj=4
        sizes = x.shape
        # print("x shape: ", x.size())
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        if self.no_3d:
            return x

        # apply sigmoid to the visbility terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            # x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim] = \
            #     torch.sigmoid(x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim])
            x[:, :, i*self.anchor_dim + 2*self.num_y_steps:i*self.anchor_dim + 3*self.num_y_steps] = \
                torch.sigmoid(x[:, :, i*self.anchor_dim + 2*self.num_y_steps:i*self.anchor_dim + 3*self.num_y_steps])
            # x[:, :, i*self.anchor_dim + 2*self.num_y_steps : i*self.anchor_dim + 3*self.num_y_steps] = \
            #     torch.sigmoid(
            #         x[:, :, i*self.anchor_dim + 2*self.num_y_steps : i*self.anchor_dim + 3*self.num_y_steps])
        return x


# Sub-network corresponding to the top view pathway
class SingleTopViewPathway(nn.Module):
    def __init__(self, input_channels, batch_norm=False, init_weights=True):
        super(SingleTopViewPathway, self).__init__()

        self.input_channels = input_channels
        # self.num_proj = num_proj
        # self.use_proj = use_proj
        layers = []
        output_channels = input_channels
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):

        return self.feature(input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Sub-network corresponding to the top view pathway
class EasyTopViewPathway(nn.Module):
    def __init__(self, input_channels, batch_norm=False, init_weights=True):
        super(EasyTopViewPathway, self).__init__()

        self.input_channels = input_channels
        # self.num_proj = num_proj
        # self.use_proj = use_proj
        layers = []
        # output_channels = input_channels
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        # layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        # conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        # layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        # layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):

        return self.feature(input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class EasyDown2TopViewPathway(nn.Module):
    def __init__(self, input_channels, batch_norm=False, init_weights=True):
        super(EasyDown2TopViewPathway, self).__init__()

        self.input_channels = input_channels
        # self.num_proj = num_proj
        # self.use_proj = use_proj
        layers = []
        output_channels = input_channels // 2
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):

        return self.feature(input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MSTopViewPathway(nn.Module):
    def __init__(self, input_channels, num_proj, batch_norm=False, init_weights=True, use_proj=True):
        super(TopViewPathway, self).__init__()

        self.input_channels = input_channels
        self.num_proj = num_proj
        self.use_proj = use_proj

        self.features = nn.ModuleList()
        for i in range(num_proj - 1):
            layers = []
            if self.use_proj:
                output_channels = input_channels if i < num_proj - 2 else input_channels // 2
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                if i < num_proj - 2:
                    input_channels *= 2 # because concatenate projected features
            else:
                output_channels = input_channels * 2 if i < num_proj - 2 else input_channels
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                input_channels = output_channels
            self.features.append(nn.Sequential(*layers))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         print("Max weight in 3D TopViewPathway: ", torch.max(torch.abs(m.weight)))
        #         if m.weight.grad != None:
        #             print("Max weight grad in 3D TopViewPathway: ", torch.max(torch.abs(m.weight.grad)))

        outs = []
        x = input[0]
        if len(self.features) > 0:
            for i, feature in enumerate(self.features):
                x = feature(x)
                feat = x
                if self.use_proj:
                    x = torch.cat((x, input[i + 1]), 1)
                # print("MD features[{}] shape: ".format(str(i)), x.size())
                outs.append(feat)
                # if self.features[i][0].weight.grad != None and torch.max(self.features[i][0].weight.grad) > 100:
                #     print("feature[{}] weights grad > 100".format(i))
                #     # print("feature[{}] weights grad: ".format(i), self.features[i][0].weight.grad)
        else:
            outs.append(x)
        return x, outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)