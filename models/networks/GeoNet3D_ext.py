"""
3D-GeoNet with new anchor: predict 3D lanes from segmentation input. The geometry-guided anchor design is based on:
    "Gen-laneNet: a generalized and scalable approach for 3D lane detection"

New Anchor:
    1. Prediction head's lane representation is in X_g, Y_g in flat ground space and Z in real 3D ground space.
    Y_g is sampled equally, X_g, Z is regressed from network output.
    2. In addition, visibility of each point is added into the anchor representation and regressed from network.

Overall dimension of the output tensor would be: N * W * 3 *(3 * K + 1), where
    K          : number of y samples.
    (3 * K + 1): Each lane includes K attributes for X_g offset + K attributes for Z + K attributes for visibility + 1 lane probability
    3          : Each anchor column include one laneline and two centerlines --> 3
    W          : Number of columns for the output tensor each corresponds to a IPM X_g location
    N          : batch size

Use of this network requires to use its corresponding data loader and loss criterion.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from config import sim3d_config
from utils.utils import define_args, define_init_weights, homography_im2ipm_norm
# from tools.utils import define_args, define_init_weights, homography_im2ipm_norm, homography_crop_resize, homography_ipmnorm2g, tusimple_config, sim3d_config


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv2d, nn.ReLU(inplace=True)]
    return layers


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

# initialize base_grid with different sizes can adapt to different sizes
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, no_cuda):
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

        # # use M only to decide the type not value
        # self.base_grid = M.new(self.N, self.H, self.W, 3)
        self.base_grid = torch.zeros(self.H, self.W, 3)
        self.base_grid[:, :, 0] = torch.ger(torch.ones(self.H), linear_points_W)
        self.base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(self.W))
        self.base_grid[:, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()

    def forward(self, M):
        if self.base_grid.device != M.device:
            self.base_grid = self.base_grid.to(M.device)
        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa
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


# Sub-network corresponding to the top view pathway
class TopViewPathway(nn.Module):
    def __init__(self, batch_norm=False, init_weights=True):
        super(TopViewPathway, self).__init__()
        self.features1 = make_layers(['M', 128, 128, 128], 128, batch_norm)
        self.features2 = make_layers(['M', 256, 256, 256], 256, batch_norm)
        self.features3 = make_layers(['M', 256, 256, 256], 512, batch_norm)

        if init_weights:
            self._initialize_weights()

    def forward(self, a, b, c, d):
        x = self.features1(a)
        feat_1 = x
        x = torch.cat((x, b), 1)
        x = self.features2(x)
        feat_2 = x
        x = torch.cat((x, c), 1)
        x = self.features3(x)
        feat_3 = x
        x = torch.cat((x, d), 1)
        return x, feat_1, feat_2, feat_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
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
    def __init__(self, num_lane_type, num_y_steps, batch_norm=False):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        layers = []
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
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

    def forward(self, x):
        x = self.features(x)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim] = \
                torch.sigmoid(x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim])
        return x


# The 3D-lanenet composed of image encode, top view pathway, and lane predication head
class Net(nn.Module):
    def __init__(self, args, input_dim=1, debug=False):
        super().__init__()

        self.no_cuda = args.no_cuda
        self.debug = debug
        self.pred_cam = args.pred_cam
        if args.no_centerline:
            self.num_lane_type = 1
        else:
            self.num_lane_type = 3

        self.num_y_steps = args.num_y_steps
        if args.no_3d:
            self.anchor_dim = args.num_y_steps + 1
        else:
            self.anchor_dim = 3*args.num_y_steps + 1

        # define required transformation matrices
        # define homographic transformation between image and ipm
        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])

        # image scale matrix
        self.S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                               [            0,  args.resize_h, 0],
                                               [            0,              0, 1]], dtype=np.float32))
        self.S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                                   [                        0, 1/np.float(args.resize_h), 0],
                                                   [                        0,                         0, 1]], dtype=np.float32))

        # # image transform matrix
        # H_c = homography_crop_resize(org_img_size, args.crop_y, resize_img_size)
        # self.H_c = torch.from_numpy(H_c).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        if not self.no_cuda:
            self.S_im = self.S_im.cuda()
            self.S_im_inv = self.S_im_inv.cuda()
            # self.H_c = self.H_c.cuda()

            # Define network
            # the grid considers both src and dst grid normalized
            size_top = torch.Size([np.int(args.ipm_h), np.int(args.ipm_w)])
            self.project_layer = ProjectiveGridGenerator(size_top, args.no_cuda)

            # Conv layers to convert original resolution binary map to target resolution with high-dimension
            self.encoder = make_layers([8, 'M', 16, 'M', 32, 'M', 64], input_dim, batch_norm=args.batch_norm)

            self.lane_out = LanePredictionHead(self.num_lane_type, self.num_y_steps, args.batch_norm)

    def forward(self, input, _M_inv=None):
        # compute image features from multiple layers

        # cam_height = self.cam_height
        # cam_pitch = self.cam_pitch

        # spatial transfer image features to IPM features
        if _M_inv is not None:
            grid = self.project_layer(_M_inv)
        else:
            grid = self.project_layer(self.M_inv)
        x_proj = F.grid_sample(input, grid) #[1, 1, 208, 128]

        # conv layers to convert original resolution binary map to target resolution with high-dimension
        x_feat = self.encoder(x_proj) #[1, 64, 26, 16]

        # convert top-view features to anchor output
        out = self.lane_out(x_feat)

        if self.debug:
            return out, cam_height, cam_pitch, x_proj, x_feat

        return out

    def update_projection(self, args, cam_height, cam_pitch, intrinsics=None):
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        batch_size = cam_height.shape[0]
        self.M_inv = torch.zeros(batch_size, 3, 3)
        for i in range(batch_size):
            M, M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
                                              args.crop_y, np.array([args.resize_h, args.resize_w]),
                                              cam_pitch[i].data.cpu().numpy(), cam_height[i].data.cpu().numpy(), intrinsics[i].data.cpu().numpy())
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        if not self.no_cuda:
            self.M_inv = self.M_inv.cuda()
        self.cam_height = cam_height
        self.cam_pitch = cam_pitch
    
    def update_projection_extrinsic(self, args, extrinsics, intrinsics=None):
        """
            Update transformation matrix based on ground-truth extrinsics
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        """
        batch_size = extrinsics.shape[0]
        self.M_inv = torch.zeros(batch_size, 3, 3)
        for i in range(batch_size):
            M, M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
                                              args.crop_y, np.array([args.resize_h, args.resize_w]),
                                              None, None, intrinsics[i].data.cpu().numpy(), extrinsics[i].data.cpu().numpy())
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        if not self.no_cuda:
            self.M_inv = self.M_inv.cuda()

    def update_projection_for_data_aug(self, aug_mats):
        """
            update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
            Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
                                               2. when cam_height, cam_pitch are online estimated, update H_c for later use
        """
        if not self.no_cuda:
            aug_mats = aug_mats.cuda()

        for i in range(aug_mats.shape[0]):
            # NOT supporting image crop for now # update H_c directly
            # self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
            # augmentation need to be applied in unnormalized image coords for M_inv
            aug_mats[i] = torch.matmul(torch.matmul(self.S_im_inv, aug_mats[i]), self.S_im)
            self.M_inv[i] = torch.matmul(aug_mats[i], self.M_inv[i])


# unit test
if __name__ == '__main__':
    import os
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as F2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global args
    parser = define_args()
    args = parser.parse_args()

    # args.dataset_name = 'tusimple'
    # tusimple_config(args)
    args.dataset_name = 'sim3d'
    sim3d_config(args)
    args.pred_cam = True
    args.batch_size = 1

    # construct model
    model = Net(args)
    print(model)

    # initialize model weights
    define_init_weights(model, args.weight_init)

    # load in vgg pretrained weights on ImageNet
    if args.pretrained:
        model.load_pretrained_vgg(args.batch_norm)
        print('vgg weights pretrained on ImageNet loaded!')
    model = model.cuda()

    # prepare input
    image = torch.randn(1, 1, args.resize_h, args.resize_w)
    image = image.cuda()

    # test update of camera height and pitch
    cam_height = torch.tensor(1.65).unsqueeze_(0).expand([args.batch_size, 1]).type(torch.FloatTensor)
    cam_pitch = torch.tensor(0.1).unsqueeze_(0).expand([args.batch_size, 1]).type(torch.FloatTensor)
    # model.update_projection(args, cam_height, cam_pitch)

    # inference the model
    output_net, pred_height, pred_pitch = model(image)

    print(output_net.shape)
    print(pred_height)
    print(pred_pitch)
