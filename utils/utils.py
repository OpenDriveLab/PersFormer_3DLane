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

import argparse
import errno
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.init as init
import torch.optim
from torch.optim import lr_scheduler
import os.path as ops
from nms import nms
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import softmax
plt.rcParams['figure.figsize'] = (35, 30)


def define_args():
    parser = argparse.ArgumentParser(description='PersFormer_3DLane_Detection')
    # Paths settings
    parser.add_argument('--dataset_name', type=str, help='the dataset name')
    parser.add_argument('--data_dir', type=str, help='The path of dataset json files (annotations)')
    parser.add_argument('--dataset_dir', type=str, help='The path of dataset image files (images)')
    parser.add_argument('--save_path', type=str, default='data_splits/', help='directory to save output')
    parser.add_argument('--use_memcache', type=str2bool, nargs='?', const=True, default=True, help='if use memcache')
    # Dataset settings
    parser.add_argument('--org_h', type=int, default=1080, help='height of the original image')
    parser.add_argument('--org_w', type=int, default=1920, help='width of the original image')
    parser.add_argument('--crop_y', type=int, default=0, help='crop from image')
    parser.add_argument('--cam_height', type=float, default=1.55, help='height of camera in meters')
    parser.add_argument('--pitch', type=float, default=3, help='pitch angle of camera to ground in centi degree')
    parser.add_argument('--fix_cam', type=str2bool, nargs='?', const=True, default=False, help='if to use fix camera')
    parser.add_argument('--no_3d', action='store_true', help='if a dataset include laneline 3D attributes')
    parser.add_argument('--no_centerline', action='store_true', help='if a dataset include centerline')
    parser.add_argument('--num_category', type=int, default=2, help='number of lane category, including background')
    # PersFormer settings
    parser.add_argument('--mod', type=str, default='PersFormer', help='model to train')
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=True, help="use pretrained model to start training")
    parser.add_argument("--batch_norm", type=str2bool, nargs='?', const=True, default=True, help="apply batch norm")
    parser.add_argument("--pred_cam", type=str2bool, nargs='?', const=True, default=False, help="use network to predict camera online?")
    parser.add_argument('--ipm_h', type=int, default=208, help='height of inverse projective map (IPM)')
    parser.add_argument('--ipm_w', type=int, default=128, help='width of inverse projective map (IPM)')
    parser.add_argument('--resize_h', type=int, default=360, help='height of the resized image (input of net)')
    parser.add_argument('--resize_w', type=int, default=480, help='width of the resized image (input of net)')
    parser.add_argument('--y_ref', type=float, default=20.0, help='the reference Y distance in meters from where lane association is determined')
    parser.add_argument('--prob_th', type=float, default=0.5, help='probability threshold for selecting output lanes')
    parser.add_argument('--encoder', type=str, default='ResNext101', help='feature extractor:'
                                                                          'ResNext101/VGG19/DenseNet161/InceptionV3/MobileNetV2/ResNet101/EfficientNet-Bx')
    parser.add_argument('--feature_channels', type=int, default=128, help='number of channels after encoder')
    parser.add_argument('--num_proj', type=int, default=4, help='number of projection layers')
    parser.add_argument('--num_att', type=int, default=3, help='number of attention encoding layers')
    parser.add_argument('--use_proj', type=str2bool, nargs='?', const=True, default=True, help='proj features in 2D pathway')
    parser.add_argument('--use_fpn', type=str2bool, nargs='?', const=True, default=False, help='use FPN features')
    parser.add_argument('--use_default_anchor', type=str2bool, nargs='?', const=True, default=False, help='use default anchors in 2D and 3D')
    parser.add_argument('--nms_thres_3d', type=float, default=1.0, help='nms threshold to filter detections in BEV, unit: meter')
    parser.add_argument('--new_match', type=str2bool, nargs='?', const=True, default=False, help='Allow multiple anchors to match the same GT during 3D data loading')
    parser.add_argument('--match_dist_thre_3d', type=float, default=2.0, help='Threshold to match an anchor to GT when using new_match, unit: meter')
    # LaneATT settings
    parser.add_argument('--max_lanes', type=int, default=6, help='max lane number detection in LaneATT')
    parser.add_argument('--S', type=int, default=72, help='max sample number in img height')
    parser.add_argument('--anchor_feat_channels', type=int, default=64, help='number of anchor feature channels')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='cls loss weight w.r.t. reg loss in 2D prediction')
    parser.add_argument('--reg_vis_loss_weight', type=float, default=1.0, help='reg vis loss weight w.r.t. reg loss in 2D prediction')
    parser.add_argument('--nms_thres', type=float, default=45.0, help='nms threshold')
    parser.add_argument('--conf_th', type=float, default=0.1, help='confidence threshold for selecting output 2D lanes')
    parser.add_argument('--vis_th', type=float, default=0.1, help='visibility threshold for output 2D lanes')
    parser.add_argument('--loss_att_weight', type=float, default=100.0, help='2D lane losses weight w.r.t. 3D lane losses')
    # General model settings
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--nepochs', type=int, default=100, help='total numbers of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--nworkers', type=int, default=0, help='num of threads')
    parser.add_argument('--seg_start_epoch', type=int, default=1, help='Number of epochs to perform segmentation pretraining')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    parser.add_argument('--vgg_mean', type=float, default=[0.485, 0.456, 0.406], help='Mean of rgb used in pretrained model on ImageNet')
    parser.add_argument('--vgg_std', type=float, default=[0.229, 0.224, 0.225], help='Std of rgb used in pretrained model on ImageNet')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw', help='adam/adamw/sgd/rmsprop')
    parser.add_argument('--weight_init', type=str, default='normal', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 weight decay/regularisation on?')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=900, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', default=None, help='learning rate policy: lambda|step|cosine|cosine_warm')
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
    parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--T_max', type=int, default=10, help='maximum number of iterations')
    parser.add_argument('--T_0', type=int, default=500, help='number of iterations for the first restart')
    parser.add_argument('--T_mult', type=int, default=1, help='a factor increases T_i after a restart')
    parser.add_argument('--eta_min', type=float, default=1e-3, help='minimum learning rate')
    parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
    # CUDNN usage
    parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    # Tensorboard settings
    parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=False, help="Use tensorboard logging by tensorflow")
    # Print and Save settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--world_size', type=int, default = 1)
    return parser


class Visualizer:
    def __init__(self, args):
        self.save_path = args.save_path
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline
        self.vgg_mean = args.vgg_mean
        self.vgg_std = args.vgg_std
        self.ipm_w = args.ipm_w
        self.ipm_h = args.ipm_h
        self.num_y_steps = args.num_y_steps
        self.h_net = args.resize_h
        self.w_net = args.resize_w

        self.dataset_name = args.dataset_name
        self.use_default_anchor = args.use_default_anchor

        self.num_category = args.num_category
        self.category_dict = {0: 'invalid',
                              1: 'white-dash',
                              2: 'white-solid',
                              3: 'double-white-dash',
                              4: 'double-white-solid',
                              5: 'white-ldash-rsolid',
                              6: 'white-lsolid-rdash',
                              7: 'yellow-dash',
                              8: 'yellow-solid',
                              9: 'double-yellow-dash',
                              10: 'double-yellow-solid',
                              11: 'yellow-ldash-rsolid',
                              12: 'yellow-lsolid-rdash',
                              13: 'fishbone',
                              14: 'others',
                              20: 'roadedge'}

        if args.no_3d:
            # self.anchor_dim = args.num_y_steps + 1
            self.anchor_dim = args.num_y_steps + self.num_category
        else:
            if 'no_visibility' in args.mod:
                # self.anchor_dim = 2 * args.num_y_steps + 1
                self.anchor_dim = 2 * args.num_y_steps + self.num_category
            else:
                # self.anchor_dim = 3 * args.num_y_steps + 1
                self.anchor_dim = 3 * args.num_y_steps + self.num_category

        x_min = args.top_view_region[0, 0]
        x_max = args.top_view_region[1, 0]
        if self.use_default_anchor:
            self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w / 8), endpoint=True)
        else:
            self.anchor_x_steps = args.anchor_grid_x
        self.anchor_y_steps = args.anchor_y_steps

        # transformation from ipm to ground region
        H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                          [self.ipm_w-1, 0],
                                                          [0, self.ipm_h-1],
                                                          [self.ipm_w-1, self.ipm_h-1]]),
                                              np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(H_ipm2g)

        # probability threshold for choosing visualize lanes
        self.prob_th = args.prob_th

    def draw_on_img_category(self, img, pred_anchors, gt_anchors, P_g2im, draw_type='laneline'):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:
        """
        fig = plt.figure(dpi=120, figsize=(4,3))
        plt.imshow(img)
        plot_lines = {}
        plot_lines["pred"] = []
        plot_lines["gt"] = []
        lane_anchor = pred_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            line_pred = {}
            lane_cate = np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim])
            if draw_type == 'laneline' and lane_cate != 0:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    if not self.use_default_anchor:
                        anchor_x_2d, _ = homographic_transformation(P_g2im, self.anchor_x_steps[j], self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                visibility = lane_anchor[j, 2 * self.num_y_steps:3 * self.num_y_steps]
                if not self.use_default_anchor:
                    anchor_x_2d = anchor_x_2d.astype(np.int)
                
                x_2d = [x for i, x in enumerate(x_2d) if visibility[i] > self.prob_th]
                y_2d = [y for i, y in enumerate(y_2d) if visibility[i] > self.prob_th]

                line_pred["x_2d"] = x_2d
                line_pred["y_2d"] = y_2d
                line_pred["lane_cate"] = int(lane_cate)
                plot_lines["pred"].append(line_pred)

                if lane_cate == 1: # white dash
                    plt.plot(x_2d, y_2d, 'mediumpurple', lw=4, alpha=0.6)
                    plt.plot(x_2d, y_2d, 'white', linestyle=(0,(10,10)), lw=2, alpha=0.5)
                elif lane_cate == 2: # white solid
                    plt.plot(x_2d, y_2d, 'mediumturquoise', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 3: # double-white-dash
                    plt.plot(x_2d, y_2d, 'mediumorchid', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                elif lane_cate == 4: # double-white-solid
                    plt.plot(x_2d, y_2d, 'lightskyblue', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 5: # white-ldash-rsolid
                    plt.plot(x_2d, y_2d, 'hotpink', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 6: # white-lsolid-rdash
                    plt.plot(x_2d, y_2d, 'cornflowerblue', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=0.75, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=0.5, alpha=0.5)
                elif lane_cate == 7: # yellow-dash
                    plt.plot(x_2d, y_2d, 'yellowgreen', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', linestyle=(0,(20,10)), lw=2, alpha=0.5)
                elif lane_cate == 8: # yellow-solid
                    plt.plot(x_2d, y_2d, 'dodgerblue', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 9: # double-yellow-dash
                    plt.plot(x_2d, y_2d, 'salmon', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                elif lane_cate == 10: # double-yellow-solid
                    plt.plot(x_2d, y_2d, 'lightcoral', lw=4, alpha=0.6)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 11: # yellow-ldash-rsolid
                    plt.plot(x_2d, y_2d, 'coral', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 12: # yellow-lsolid-rdash
                    plt.plot(x_2d, y_2d, 'lightseagreen', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                elif lane_cate == 13: # fishbone
                    plt.plot(x_2d, y_2d, 'royalblue', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 14: # others
                    plt.plot(x_2d, y_2d, 'forestgreen', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 20 or lane_cate == 21: # road
                    plt.plot(x_2d, y_2d, 'gold', lw=4, alpha=0.3)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                else:
                    plt.plot(x_2d, y_2d, lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)

        lane_anchor = gt_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            line_gt = {}
            lane_cate = np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim])
            if draw_type == 'laneline' and lane_cate != 0:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    if not self.use_default_anchor:
                        anchor_x_2d, _ = homographic_transformation(P_g2im, self.anchor_x_steps[j], self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                visibility = lane_anchor[j, 2 * self.num_y_steps:3 * self.num_y_steps]
                if not self.use_default_anchor:
                    anchor_x_2d = anchor_x_2d.astype(np.int)
                
                x_2d = [x for i, x in enumerate(x_2d) if visibility[i] > self.prob_th]
                y_2d = [y for i, y in enumerate(y_2d) if visibility[i] > self.prob_th]

                # plt.plot(x_2d, y_2d, 'yellowgreen', lw=2, alpha=0.7)
                # plt.plot(x_2d, y_2d, 'white', lw=1, alpha=0.5)

                line_gt["x_2d"] = x_2d
                line_gt["y_2d"] = y_2d
                line_gt["lane_cate"] = int(lane_cate)
                plot_lines["gt"].append(line_gt)

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.xlim(0, self.w_net)
        plt.ylim(self.h_net-1, -1)

        return fig

    def draw_on_ipm_category(self, im_ipm, pred_anchors, gt_anchors, draw_type='laneline'):
        fig = plt.figure(dpi=16, figsize=(8,13))
        plt.imshow(im_ipm)
        plot_lines = {}
        plot_lines["pred"] = []
        plot_lines["gt"] = []
        lane_anchor = pred_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            line_pred = {}
            lane_cate = np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim])
            if draw_type == 'laneline' and lane_cate != 0:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                if not self.use_default_anchor:
                    anchor_x_ipm, _ = homographic_transformation(self.H_g2ipm, self.anchor_x_steps[j], self.anchor_y_steps)
                x_2d = [x for i, x in enumerate(x_ipm) if visibility[i] > self.prob_th]
                y_2d = [y for i, y in enumerate(y_ipm) if visibility[i] > self.prob_th]

                line_pred["x_ipm"] = x_2d
                line_pred["y_ipm"] = y_2d
                line_pred["lane_cate"] = int(lane_cate)
                plot_lines["pred"].append(line_pred)

                if lane_cate == 1: # white dash
                    plt.plot(x_2d, y_2d, 'mediumpurple', lw=25, alpha=0.6)
                    plt.plot(x_2d, y_2d, 'white', linestyle=(0,(10,10)), lw=10, alpha=0.5)
                elif lane_cate == 2: # white solid
                    plt.plot(x_2d, y_2d, 'mediumturquoise', lw=20, alpha=0.8)
                    plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 3: # double-white-dash
                    plt.plot(x_2d, y_2d, 'mediumorchid', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', linestyle=(0,(20,10)), lw=10, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', linestyle=(0,(20,10)), lw=10, alpha=0.5)
                elif lane_cate == 4: # double-white-solid
                    plt.plot(x_2d, y_2d, 'lightskyblue', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', lw=10, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 5: # white-ldash-rsolid
                    plt.plot(x_2d, y_2d, 'hotpink', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', linestyle=(0,(20,10)), lw=5, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 6: # white-lsolid-rdash
                    plt.plot(x_2d, y_2d, 'cornflowerblue', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', lw=10, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', linestyle=(0,(20,10)), lw=5, alpha=0.5)
                elif lane_cate == 7: # yellow-dash
                    plt.plot(x_2d, y_2d, 'yellowgreen', lw=20, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', linestyle=(0,(20,10)), lw=10, alpha=0.5)
                elif lane_cate == 8: # yellow-solid
                    plt.plot(x_2d, y_2d, 'dodgerblue', lw=20, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 9: # double-yellow-dash
                    plt.plot(x_2d, y_2d, 'salmon', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', linestyle=(0,(20,10)), lw=5, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', linestyle=(0,(20,10)), lw=5, alpha=0.5)
                elif lane_cate == 10: # double-yellow-solid
                    plt.plot(x_2d, y_2d, 'lightcoral', lw=25, alpha=0.6)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', lw=10, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 11: # yellow-ldash-rsolid
                    plt.plot(x_2d, y_2d, 'coral', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', linestyle=(0,(20,10)), lw=5, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 12: # yellow-lsolid-rdash
                    plt.plot(x_2d, y_2d, 'lightseagreen', lw=25, alpha=0.4)
                    plt.plot(np.array(x_2d) - 2, y_2d, 'white', lw=10, alpha=0.5)
                    plt.plot(np.array(x_2d) + 2, y_2d, 'white', linestyle=(0,(20,10)), lw=5, alpha=0.5)
                elif lane_cate == 13: # fishbone
                    plt.plot(x_2d, y_2d, 'royalblue', lw=20, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 14: # others
                    plt.plot(x_2d, y_2d, 'forestgreen', lw=20, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)
                elif lane_cate == 20 or lane_cate == 21: # road
                    plt.plot(x_2d, y_2d, 'gold', lw=25, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)
                else:
                    plt.plot(x_2d, y_2d, lw=20, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)

        lane_anchor = gt_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            line_gt = {}
            lane_cate = np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim])
            if draw_type == 'laneline' and lane_cate != 0:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                if not self.use_default_anchor:
                    anchor_x_ipm, _ = homographic_transformation(self.H_g2ipm, self.anchor_x_steps[j], self.anchor_y_steps)
                x_2d = [x for i, x in enumerate(x_ipm) if visibility[i] > self.prob_th]
                y_2d = [y for i, y in enumerate(y_ipm) if visibility[i] > self.prob_th]

                line_gt["x_ipm"] = x_2d
                line_gt["y_ipm"] = y_2d
                line_gt["lane_cate"] = int(lane_cate)
                plot_lines["gt"].append(line_gt)

                # plt.plot(x_2d, y_2d, 'yellowgreen', lw=20, alpha=0.7)
                # plt.plot(x_2d, y_2d, 'white', lw=10, alpha=0.5)

            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.xlim(0, self.ipm_w)
            plt.ylim(self.ipm_h-1, -1)

        return fig

    def draw_3d_curves_category(self, pred_anchors, gt_anchors, h_cam, draw_type='laneline'):
        fig = plt.figure(dpi=100, figsize=(12,6))
        plt.style.use('seaborn-white')
        plt.rc('font', family='Times New Roman', size=10)
        ax = fig.gca(projection='3d')
        line1, line2 = None, None
        plot_lines = {}
        plot_lines["pred"] = []
        plot_lines["gt"] = []
        lane_anchor = pred_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            lane_cate = np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim])
            if draw_type == 'laneline' and lane_cate != 0:
                line_pred = {}
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    line_pred["x_3d"] = x_g.tolist()
                    line_pred["y_3d"] = y_g.tolist()
                    line_pred["z_3d"] = z_g.tolist()
                    line_pred["lane_cate"] = int(lane_cate)
                    plot_lines["pred"].append(line_pred)
                    fit1 = np.polyfit(y_g,x_g,2)
                    fit2 = np.polyfit(y_g,z_g,2)
                    f_xy = np.poly1d(fit1)
                    f_zy = np.poly1d(fit2)
                    y_g = np.linspace(min(y_g), max(y_g), 5 * len(y_g))
                    x_g = f_xy(y_g)
                    z_g = f_zy(y_g)
                    
                    if lane_cate == 1: # white dash
                        line1, = ax.plot(x_g, y_g, z_g, 'mediumpurple', lw=3, alpha=0.8, label='pred')
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
                    elif lane_cate == 2: # white solid
                        line1, = ax.plot(x_g, y_g, z_g, 'mediumturquoise', lw=3, alpha=0.8, label='pred')
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
                    elif lane_cate == 3: # double-white-dash
                        line1, = ax.plot(x_g, y_g, z_g, 'mediumorchid', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                    elif lane_cate == 4: # double-white-solid
                        line1, = ax.plot(x_g, y_g, z_g, 'lightskyblue', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                    elif lane_cate == 5: # white-ldash-rsolid
                        line1, = ax.plot(x_g, y_g, z_g, 'hotpink', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                    elif lane_cate == 6: # white-lsolid-rdash
                        line1, = ax.plot(x_g, y_g, z_g, 'cornflowerblue', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                    elif lane_cate == 7: # yellow-dash
                        line1, = ax.plot(x_g, y_g, z_g, 'lawngreen', lw=3, alpha=0.8)
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.8, linestyle=(0,(20,10)))
                    elif lane_cate == 8: # yellow-solid
                        line1, = ax.plot(x_g, y_g, z_g, 'dodgerblue', lw=3, alpha=0.8)
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
                    elif lane_cate == 9: # double-yellow-dash
                        line1, = ax.plot(x_g, y_g, z_g, 'salmon', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                    elif lane_cate == 10: # double-yellow-solid
                        line1, = ax.plot(x_g, y_g, z_g, 'lightcoral', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                    elif lane_cate == 11: # yellow-ldash-rsolid
                        line1, = ax.plot(x_g, y_g, z_g, 'coral', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                    elif lane_cate == 12: # yellow-lsolid-rdash
                        line1, = ax.plot(x_g, y_g, z_g, 'lightseagreen', lw=4, alpha=0.8, label='pred')
                        ax.plot(np.array(x_g)+0.15, y_g, z_g, 'white', lw=0.5, alpha=1)
                        ax.plot(np.array(x_g)-0.15, y_g, z_g, 'white', lw=0.5, alpha=1, linestyle=(0,(20,10)))
                    elif lane_cate == 13: # fishbone
                        line1, = ax.plot(x_g, y_g, z_g, 'royalblue', lw=3, alpha=0.8)
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
                    elif lane_cate == 14: # others
                        line1, = ax.plot(x_g, y_g, z_g, 'forestgreen', lw=3, alpha=0.8)
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
                    elif lane_cate == 20 or lane_cate == 21: # road
                        line1, = ax.plot(x_g, y_g, z_g, 'gold', lw=2, alpha=0.8)
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
                    else:
                        line1, = ax.plot(x_g, y_g, z_g, lw=3, alpha=0.8)
                        ax.plot(x_g, y_g, z_g, 'white', lw=1, alpha=0.5)
        z_min = -1
        z_max = 1
        lane_anchor = gt_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and \
                np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim]) != 0:
                line_gt = {}
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    line_gt["x_3d"] = x_g.tolist()
                    line_gt["y_3d"] = y_g.tolist()
                    line_gt["z_3d"] = z_g.tolist()
                    plot_lines["gt"].append(line_gt)
                    fit1 = np.polyfit(y_g,x_g,2)
                    fit2 = np.polyfit(y_g,z_g,2)
                    f_xy = np.poly1d(fit1)
                    f_zy = np.poly1d(fit2)
                    y_g = np.linspace(min(y_g), max(y_g), 5 * len(y_g))
                    x_g = f_xy(y_g)
                    z_g = f_zy(y_g)
                    if z_min == -1 and z_max == 1:
                        z_max = max(z_g)
                        z_min = min(z_g)
                    else:
                        if max(z_g) > z_max:
                            z_max = max(z_g)
                        if min(z_g) < z_min:
                            z_min = min(z_g)
                    line2, = ax.plot(x_g, y_g, z_g, lw=2, c='yellowgreen', alpha=1, label='gt')
        ax.set_xlabel('x-axis', labelpad=10)
        # ax.set_xlim(-10, 10)
        ax.set_ylabel('y-axis', labelpad=10)
        # ax.set_ylim(0, 100)
        ax.set_zlabel('z-axis')
        ax.set_zlim(2*z_min-z_max, 2*z_max-z_min)
        ax.zaxis.set_major_locator(plt.MultipleLocator(max(0.1, round((z_max-z_min)*2/5,1))))
        ax.view_init(20, -60)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # plt.gca().set_box_aspect((1,1.5,0.5))
        ax.set_box_aspect((1,1.5,0.5))
        if line2:
            plt.legend([line2], ['gt'], loc=(0.75,0.7), fontsize=15)
        plt.tick_params(pad=0)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0,0)

        return fig

    def save_result_new(self, dataset, train_or_val, epoch, idx, images, gt, pred, pred_cam_pitch, pred_cam_height,
                        aug_mat=np.identity(3, dtype=np.float), evaluate=False,
                        laneatt_gt=None, laneatt_pred=None, laneatt_pos_anchor=None, intrinsics=None, extrinsics=None, seg_name=None, img_name = None):
        if not dataset.data_aug:
            aug_mat = np.repeat(np.expand_dims(aug_mat, axis=0), idx.shape[0], axis=0)

        assert(idx.shape[0] == images.shape[0])
        assert(idx.shape[0] == gt.shape[0])
        assert(idx.shape[0] == pred.shape[0])
        if not 'openlane' in self.dataset_name:
            assert(idx.shape[0] == pred_cam_pitch.shape[0])
            assert(idx.shape[0] == pred_cam_height.shape[0])
        if laneatt_gt is not None:
            # print("laneatt_gt size: ", len(laneatt_gt))
            assert(idx.shape[0] == len(laneatt_gt))
        if laneatt_pred is not None:
            # print("laneatt_pred size: ", len(laneatt_pred))
            assert(idx.shape[0] == len(laneatt_pred))
        if laneatt_pos_anchor is not None:
            assert(idx.shape[0] == len(laneatt_pos_anchor))

        for i in range(idx.shape[0]):
            # during training, only visualize the first sample of this batch
            if i > 0 and not evaluate:
                break
            # seg_name
            if seg_name is not None:
                seg_name_i = seg_name[i]
            im = images.permute(0, 2, 3, 1).data.cpu().numpy()[i]
            # the vgg_std and vgg_mean are for images in [0, 1] range
            im = im * np.array(self.vgg_std)
            im = im + np.array(self.vgg_mean)
            im = np.clip(im, 0, 1)

            gt_anchors = gt[i]
            pred_anchors = pred[i]

            # apply nms to avoid output directly neighbored lanes
            # consider w/o centerline cases
            if self.use_default_anchor:
                for j in range(1, self.num_category+1):
                    pred_anchors[:, self.anchor_dim - j] = nms_1d(pred_anchors[:, self.anchor_dim - j])
                    if not self.no_centerline:
                        pred_anchors[:, 2*self.anchor_dim - j] = nms_1d(pred_anchors[:, 2*self.anchor_dim - j])
                        pred_anchors[:, 3*self.anchor_dim - j] = nms_1d(pred_anchors[:, 3*self.anchor_dim - j])

            H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
            P_gt = np.matmul(H_crop, H_g2im)
            if 'openlane' in self.dataset_name:
                P_pred = P_gt
            else:
                H_g2im_pred = homograpthy_g2im(pred_cam_pitch[i],
                                               pred_cam_height[i], intrinsics[i])
                P_pred = np.matmul(H_crop, H_g2im_pred)

            # consider data augmentation
            P_gt = np.matmul(aug_mat[i, :, :], P_gt)
            P_pred = np.matmul(aug_mat[i, :, :], P_pred)

            # update transformation with image augmentation
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat[i, :, :]))
            im_ipm = cv2.warpPerspective(im, H_im2ipm, (self.ipm_w, self.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)

            im_laneline = im.copy()
            fig = self.draw_on_img_category(im_laneline, pred_anchors, gt_anchors, P_gt, 'laneline')
            ipm_laneline = im_ipm.copy()
            fig2 = self.draw_on_ipm_category(ipm_laneline, pred_anchors, gt_anchors, 'laneline')
            fig3 = self.draw_3d_curves_category(pred_anchors, gt_anchors, extrinsics[i][2,3], 'laneline')

            if evaluate:
                filename = self.save_path + '/vis_2d/' + '{}'.format(img_name[i])
                mkdir_if_missing(os.path.dirname(filename))
                fig.savefig(filename)

                filename = self.save_path + '/vis_ipm/' + '{}'.format(img_name[i])
                mkdir_if_missing(os.path.dirname(filename))
                fig2.savefig(filename)

                filename = self.save_path + '/vis_3d/' + '{}'.format(img_name[i])
                mkdir_if_missing(os.path.dirname(filename))
                fig3.savefig(filename, bbox_inches='tight')
                # fig3.savefig(filename[:-4]+'.svg', bbox_inches='tight')
            else:
                filename = self.save_path + '/example/{}/epoch-{}/segment-{}_{}'.format(train_or_val, epoch,
                                                                                             seg_name_i if seg_name is not None else "segment",
                                                                                             img_name[i])
                mkdir_if_missing(os.path.dirname(filename))
                fig.savefig(filename)
            plt.clf()
            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d


def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d


def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values


def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T


def homography_im2ipm_norm(top_view_region, org_img_size, crop_y, resize_img_size, cam_pitch, cam_height, K, E=None):
    """
        Compute the normalized transformation such that image region are mapped to top_view region maps to
        the top view image's 4 corners
        Ground coordinates: x-right, y-forward, z-up
        The purpose of applying normalized transformation: 1. invariance in scale change
                                                           2.Torch grid sample is based on normalized grids

    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                            top-left, top-right, bottom-left, bottom-right
    :param org_img_size: the size of original image size: [h, w]
    :param crop_y: pixels croped from original img
    :param resize_img_size: the size of image as network input: [h, w]
    :param cam_pitch: camera pitch angle wrt ground plane
    :param cam_height: camera height wrt ground plane in meters
    :param K: camera intrinsic parameters
    :return: H_im2ipm_norm: the normalized transformation from image to IPM image
    """

    if isinstance(E, np.ndarray):
        H_g2im = homograpthy_g2im_extrinsic(E, K)
    else:
        # compute homography transformation from ground to image (only this depends on cam_pitch and cam height)
        H_g2im = homograpthy_g2im(cam_pitch, cam_height, K)
    # transform original image region to network input region
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)
    H_g2im = np.matmul(H_c, H_g2im)

    # compute top-view corners' coordinates in image
    x_2d, y_2d = homographic_transformation(H_g2im, top_view_region[:, 0], top_view_region[:, 1])
    border_im = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1)

    # compute the normalized transformation
    border_im[:, 0] = border_im[:, 0] / resize_img_size[1]
    border_im[:, 1] = border_im[:, 1] / resize_img_size[0]
    border_im = np.float32(border_im)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # img to ipm
    H_im2ipm_norm = cv2.getPerspectiveTransform(border_im, dst)
    # ipm to im
    H_ipm2im_norm = cv2.getPerspectiveTransform(dst, border_im)
    return H_im2ipm_norm, H_ipm2im_norm


def homography_ipmnorm2g(top_view_region):
    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g


def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im


def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im


def homograpthy_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0,1,3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im


def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c


def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def transform_lane_gflat2g(h_cam, X_gflat, Y_gflat, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_g = X_gflat - X_gflat * Z_g / h_cam
    Y_g = Y_gflat - Y_gflat * Z_g / h_cam

    return X_g, Y_g


def transform_lane_g2gflat(h_cam, X_g, Y_g, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_gflat = X_g * h_cam / (h_cam - Z_g)
    Y_gflat = Y_g * h_cam / (h_cam - Z_g)

    return X_gflat, Y_gflat


def nms_1d(v):
    """

    :param v: a 1D numpy array
    :return:
    """
    v_out = v.copy()
    len = v.shape[0]
    if len < 2:
        return v
    for i in range(len):
        if i != 0 and v[i - 1] > v[i]:
            v_out[i] = 0.
        elif i != len-1 and v[i+1] > v[i]:
            v_out[i] = 0.
    return v_out

def nms_bev(batch_output_net, args):
    """apply nms to filter predictions of same GT from different anchors"""
    if not args.no_centerline:
        raise NotImplementedError
    if args.no_3d:
        anchor_dim = args.num_y_steps + args.num_category
    else:
        anchor_dim = 3*args.num_y_steps + args.num_category
    anchor_x_steps = args.anchor_grid_x \
        if not args.use_default_anchor \
        else np.linspace(args.top_view_region[0, 0], args.top_view_region[1, 0], np.int(args.ipm_w/8))

    batch_output_net = batch_output_net.reshape(batch_output_net.shape[0], batch_output_net.shape[1], anchor_dim)
    # print("cate before softmax: ", batch_output_net[:, :, anchor_dim-args.num_category:].shape)
    # print(batch_output_net[0, :16, :])
    batch_output_net[:, :, anchor_dim-args.num_category:] = \
        softmax(batch_output_net[:, :, anchor_dim-args.num_category:], axis=2)
    # print("cate after softmax: ", batch_output_net[:, :, anchor_dim-args.num_category:].shape)
    # print(batch_output_net[0, :16, :])
    for i, output_net in enumerate(batch_output_net):
        # pack prediction data to nms library format
        scores = torch.zeros(len(output_net)).cuda()
        output_net_nms = torch.zeros(len(output_net), 2 + 3 + args.S).cuda()
        pre_nms_valid_anchor_id = []
        valid_count = 0
        for j, output_anchor in enumerate(output_net):
            # print("max cate id: ", np.argmax(output_anchor[anchor_dim-args.num_category:]))
            visible_yid = np.where(output_anchor[2*args.num_y_steps: 3*args.num_y_steps] > args.prob_th)[0]
            # print("vis points: ", len(visible_yid))
            if np.argmax(output_anchor[anchor_dim-args.num_category:]) == \
                anchor_dim-args.num_category or len(visible_yid) < 2:
                # print("skip")
                continue
            pre_nms_valid_anchor_id.append(j)
            
            scores[valid_count] = np.max(output_anchor[anchor_dim-args.num_category+1:]).item()
            yid_start = visible_yid[0]
            # approximation here, since there can be invisible points in between
            yid_num = visible_yid[-1] - visible_yid[0] + 1
            output_net_nms[valid_count, 2] = (yid_start / (args.S-1)).item()
            output_net_nms[valid_count, 4] = yid_num.item()
            output_net_nms[valid_count, 5 : 5+args.num_y_steps] = \
                torch.from_numpy(output_anchor[:args.num_y_steps] + anchor_x_steps[j])
            
            valid_count = valid_count + 1
        scores = scores[:valid_count]
        output_net_nms = output_net_nms[:valid_count]
        # print("scores: ", scores)
        # print(output_net_nms[:, :16])
        keep, num_to_keep, _ = nms(output_net_nms, scores, overlap=args.nms_thres_3d, top_k=args.max_lanes)
        # print("keep before reduction: ", keep)
        keep = keep[:num_to_keep].tolist()
        # print("keep after reduction: ", keep)
        for jj, anchor_id in enumerate(pre_nms_valid_anchor_id):
            if jj not in keep:
                # update category as invalid, so that it can be filted in compute_3d_lanes()
                batch_output_net[i][anchor_id][anchor_dim-args.num_category] = 1.0

    return batch_output_net

def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=args.T_max, eta_min=args.eta_min)
    elif args.lr_policy == 'cosine_warm':
        '''
        lr_config = dict(
            policy='CosineAnnealing',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=1.0 / 3,
            min_lr_ratio=1e-3)
        '''
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    # elif args.lr_policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                factor=args.gamma,
    #                                                threshold=0.0001,
    #                                                patience=args.lr_decay_iters)
    elif args.lr_policy == 'None':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    # print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        try:
            init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        except:
            print("{} not support init".format(str(classname)))
    elif classname.find('Linear') != -1:
        try:
            init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        except:
            print("{} not support init".format(str(classname)))
    elif classname.find('BatchNorm2d') != -1:
        try:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        except:
            print("{} not support init".format(str(classname)))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# multi-gpu helper function
def unit_update_projection_extrinsic(args, extrinsics, intrinsics):
    """
        Unit function to Update transformation matrix based on ground-truth extrinsics
        This function is "Mutually Exclusive" to the updates of M_inv from network prediction
    """
    batch_size = extrinsics.shape[0]
    M_inv = torch.zeros(batch_size, 3, 3)
    for i in range(batch_size):
        _M, _M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
                                            args.crop_y, np.array([args.resize_h, args.resize_w]),
                                            None, None, intrinsics[i].data.cpu().numpy(), extrinsics[i].data.cpu().numpy())
        M_inv[i] = torch.from_numpy(_M_inv).type(torch.FloatTensor)
    if not args.no_cuda:
        M_inv = M_inv.cuda()
    return M_inv

def unit_update_projection(args, cam_height, cam_pitch, intrinsics=None, extrinsics=None):
    """
        Unit function to Update transformation matrix based on ground-truth cam_height and cam_pitch
        This function is "Mutually Exclusive" to the updates of M_inv from network prediction
    :param args:
    :param cam_height:
    :param cam_pitch:
    :return:
    """
    M_inv = torch.zeros(args.batch_size, 3, 3)
    for i in range(args.batch_size):
        _M, _M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
                                            args.crop_y, np.array([args.resize_h, args.resize_w]),
                                            cam_pitch[i].data.cpu().numpy(), cam_height[i].data.cpu().numpy(), args.K)
        M_inv[i] = torch.from_numpy(_M_inv).type(torch.FloatTensor)
    if not args.no_cuda:
        M_inv = M_inv.cuda()
    cam_height = cam_height
    cam_pitch = cam_pitch
    return M_inv, cam_height, cam_pitch

def unit_update_projection_for_data_aug(args, aug_mats, _M_inv, _S_im_inv=None, _S_im=None):
    """
        Unit function to update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
        Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
                                            2. when cam_height, cam_pitch are online estimated, update H_c for later use
    """
    if not args.no_cuda:
        aug_mats = aug_mats.cuda()

    if _S_im_inv is None:
        _S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                                    [                        0, 1/np.float(args.resize_h), 0],
                                                    [                        0,                         0, 1]], dtype=np.float32)).cuda()
    
    if _S_im is None:
        _S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                                [            0,  args.resize_h, 0],
                                                [            0,              0, 1]], dtype=np.float32)).cuda()

    for i in range(aug_mats.shape[0]):
        # update H_c directly
        # self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
        # augmentation need to be applied in unnormalized image coords for M_inv
        aug_mats[i] = torch.matmul(torch.matmul(_S_im_inv, aug_mats[i]), _S_im)
        _M_inv[i] = torch.matmul(aug_mats[i], _M_inv[i])

    return _M_inv