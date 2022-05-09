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

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
from utils.utils import *
from models.networks.feature_extractor import *
from models.networks import Lane2D, Lane3D
from models.networks.libs.layers import *
from models.networks.PE import PositionEmbeddingLearned
from models.networks.Layers import EncoderLayer
from models.networks.Unet_parts import Down, Up

# overall network
class PersFormer(nn.Module):
    def __init__(self, args):
        super(PersFormer, self).__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        self.num_proj = args.num_proj
        self.num_att = args.num_att

        # define required transformation matrices
        self.M_inv, self.cam_height, self.cam_pitch = self.get_transform_matrices(args)
        if not self.no_cuda:
            self.M_inv = self.M_inv.cuda()

        # Define network
        # backbone: feature_extractor
        self.encoder = self.get_encoder(args)
        self.neck = nn.Sequential(*make_one_layer(self.encoder.dimList[0], args.feature_channels, batch_norm=True),
                                  *make_one_layer(args.feature_channels, args.feature_channels, batch_norm=True))
        # 2d lane detector
        self.shared_encoder = Lane2D.FrontViewPathway(args.feature_channels, args.num_proj)
        stride = 2
        self.laneatt_head = Lane2D.LaneATTHead(stride * pow(2, args.num_proj - 1),
                                               args.feature_channels * pow(2, args.num_proj - 2), # no change in last proj
                                               args.im_anchor_origins,
                                               args.im_anchor_angles,
                                               img_w=args.resize_w,
                                               img_h=args.resize_h,
                                               S=args.S,
                                               anchor_feat_channels=args.anchor_feat_channels,
                                               num_category=args.num_category)
        # Perspective Transformer: get better bev feature
        self.pers_tr = PerspectiveTransformer(args,
                                              channels=args.feature_channels, # 128
                                              bev_h=args.ipm_h,  # 208
                                              bev_w=args.ipm_w,  # 128
                                              uv_h=args.resize_h//stride,  # 180
                                              uv_w=args.resize_w//stride,  # 240
                                              M_inv=self.M_inv, 
                                              num_att=self.num_att, 
                                              num_proj=self.num_proj, 
                                              nhead=args.nhead,
                                              npoints=args.npoints)
        # BEV feature extractor
        self.bev_head = BEVHead(args, channels=args.feature_channels)
        # 3d lane detector
        self.lane_out = Lane3D.LanePredictionHead(args.feature_channels * pow(2, self.num_proj - 2),
                                                  self.num_lane_type,
                                                  self.num_y_steps,
                                                  args.num_category,
                                                  args.fmap_mapping_interp_index,
                                                  args.fmap_mapping_interp_weight,
                                                  args.no_3d,
                                                  args.batch_norm,
                                                  args.no_cuda)
        # segmentation head
        self.segment_head = SegmentHead(channels=args.feature_channels)
        # uncertainty loss weight
        self.uncertainty_loss = nn.Parameter(torch.tensor([args._3d_vis_loss_weight,
                                                            args._3d_prob_loss_weight,
                                                            args._3d_reg_loss_weight,
                                                            args._2d_vis_loss_weight,
                                                            args._2d_prob_loss_weight,
                                                            args._2d_reg_loss_weight,
                                                            args._seg_loss_weight]), requires_grad=True)
        self._initialize_weights(args)

    def forward(self, input, _M_inv = None):
        out_featList = self.encoder(input)
        neck_out = self.neck(out_featList[0])
        frontview_features = self.shared_encoder(neck_out)
        '''
            frontview_features_0 size: torch.Size([4, 128, 180, 240])
            frontview_features_1 size: torch.Size([4, 256, 90, 120])
            frontview_features_2 size: torch.Size([4, 512, 45, 60])
            frontview_features_3 size: torch.Size([4, 512, 22, 30])
        '''
        frontview_final_feat = frontview_features[-1]

        laneatt_proposals_list = self.laneatt_head(frontview_final_feat)

        projs = self.pers_tr(input, frontview_features, _M_inv)
        '''
            projs_0 size: torch.Size([4, 128, 208, 128])
            projs_1 size: torch.Size([4, 256, 104, 64])
            projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])
        '''

        bev_feat = self.bev_head(projs)
        '''
            bev_feat size: torch.Size([4, 512, 26, 16])
        '''

        out = self.lane_out(bev_feat)

        cam_height = self.cam_height.to(input.device)
        cam_pitch = self.cam_pitch.to(input.device)

        pred_seg_bev_map = self.segment_head(projs[0])

        # seperate loss weight
        uncertainty_loss = torch.tensor(1.0).to(input.device) * self.uncertainty_loss.to(input.device)

        return laneatt_proposals_list, out, cam_height, cam_pitch, pred_seg_bev_map, uncertainty_loss

    def _initialize_weights(self, args):
        define_init_weights(self.neck, args.weight_init)
        define_init_weights(self.shared_encoder, args.weight_init)
        define_init_weights(self.laneatt_head, args.weight_init)
        define_init_weights(self.pers_tr, args.weight_init)
        define_init_weights(self.bev_head, args.weight_init)
        define_init_weights(self.lane_out, args.weight_init)
        define_init_weights(self.segment_head, args.weight_init)

    def get_transform_matrices(self, args):
        # define homographic transformation between image and ipm
        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])
        cam_pitch = np.pi / 180 * args.pitch

        # image scale matrix
        S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                          [            0,  args.resize_h, 0],
                                          [            0,              0, 1]], dtype=np.float32))
        S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                              [                        0, 1/np.float(args.resize_h), 0],
                                              [                        0,                         0, 1]], dtype=np.float32))
        S_im_inv_batch = S_im_inv.unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # image transform matrix
        H_c = homography_crop_resize(org_img_size, args.crop_y, resize_img_size)
        H_c = torch.from_numpy(H_c).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # camera intrinsic matrix
        K = torch.from_numpy(args.K).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # homograph ground to camera
        H_g2cam = np.array([[1,                             0,               0],
                            [0, np.sin(-cam_pitch), args.cam_height],
                            [0, np.cos(-cam_pitch),               0]])
        H_g2cam = torch.from_numpy(H_g2cam).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # transform from ipm normalized coordinates to ground coordinates
        H_ipmnorm2g = homography_ipmnorm2g(args.top_view_region)
        H_ipmnorm2g = torch.from_numpy(H_ipmnorm2g).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # compute the tranformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(H_g2cam, H_ipmnorm2g)
        M_ipm2im = torch.bmm(K, M_ipm2im)
        M_ipm2im = torch.bmm(H_c, M_ipm2im)
        M_ipm2im = torch.bmm(S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(M_ipm2im,  M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]))
        M_inv = M_ipm2im

        cam_height = torch.tensor(args.cam_height).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)
        cam_pitch = torch.tensor(cam_pitch).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)

        return M_inv, cam_height, cam_pitch

    def get_encoder(self, args):
        if args.encoder == 'ResNext101':
            return deepFeatureExtractor_ResNext101(lv6=False)
        elif args.encoder == 'VGG19':
            return deepFeatureExtractor_VGG19(lv6=False)
        elif args.encoder == 'DenseNet161':
            return deepFeatureExtractor_DenseNet161(lv6=False)
        elif args.encoder == 'InceptionV3':
            return deepFeatureExtractor_InceptionV3(lv6=False)
        elif args.encoder == 'MobileNetV2':
            return deepFeatureExtractor_MobileNetV2(lv6=False)
        elif args.encoder == 'ResNet101':
            return deepFeatureExtractor_ResNet101(lv6=False)
        elif 'EfficientNet' in args.encoder:
            return deepFeatureExtractor_EfficientNet(args.encoder, lv6=False, lv5=False, lv4=False, lv3=False)
        else:
            raise Exception("encoder model in args is not supported")


class PerspectiveTransformer(nn.Module):
    def __init__(self, args, channels, bev_h, bev_w, uv_h, uv_w, M_inv, num_att, num_proj, nhead, npoints):
        super(PerspectiveTransformer, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.uv_h = uv_h
        self.uv_w = uv_w
        self.M_inv = M_inv
        self.num_att = num_att
        self.num_proj = num_proj
        self.nhead = nhead
        self.npoints = npoints

        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        self.project_layers = nn.ModuleList()
        self.ref_2d = []
        self.input_spatial_shapes = []
        self.input_level_start_index = []

        uv_feat_c = channels
        for i in range(self.num_proj):
            if i > 0:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
                uv_h = uv_h // 2
                uv_w = uv_w // 2
                if i != self.num_proj-1:
                    uv_feat_c = uv_feat_c * 2

            bev_feat_len = bev_h * bev_w
            query_embed = nn.Embedding(bev_feat_len, uv_feat_c)
            self.query_embeds.append(query_embed)
            position_embed = PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=uv_feat_c//2)
            self.pe.append(position_embed)

            ref_point = self.get_reference_points(H=bev_h, W=bev_w, dim='2d', bs=1)
            self.ref_2d.append(ref_point)

            size_top = torch.Size([bev_h, bev_w])
            project_layer = Lane3D.RefPntsNoGradGenerator(size_top, self.M_inv, args.no_cuda)
            self.project_layers.append(project_layer)

            spatial_shape = torch.as_tensor([(uv_h, uv_w)], dtype=torch.long)
            self.input_spatial_shapes.append(spatial_shape)

            level_start_index = torch.as_tensor([0.0,], dtype=torch.long)
            self.input_level_start_index.append(level_start_index)

            for j in range(self.num_att):
                encoder_layers = EncoderLayer(d_model=uv_feat_c, dim_ff=uv_feat_c*2, num_levels=1, 
                                              num_points=self.npoints, num_heads=self.nhead)
                self.el.append(encoder_layers)

    def forward(self, input, frontview_features, _M_inv = None):
        projs = []
        for i in range(self.num_proj):
            if i == 0:
                bev_h = self.bev_h
                bev_w = self.bev_w
            else:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
            bs, c, h, w = frontview_features[i].shape
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            src = frontview_features[i].flatten(2).permute(0, 2, 1)
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(query_embed.dtype)
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            ref_2d = self.ref_2d[i].repeat(bs, 1, 1, 1).to(input.device)
            ref_pnts = self.project_layers[i](_M_inv).unsqueeze(-2)
            input_spatial_shapes = self.input_spatial_shapes[i].to(input.device)
            input_level_start_index = self.input_level_start_index[i].to(input.device)
            for j in range(self.num_att):
                query_embed = self.el[i*self.num_att+j](query=query_embed, value=src, bev_pos=bev_pos, 
                                                        ref_2d = ref_2d, ref_3d=ref_pnts,
                                                        bev_h=bev_h, bev_w=bev_w, 
                                                        spatial_shapes=input_spatial_shapes,
                                                        level_start_index=input_level_start_index)
            query_embed = query_embed.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()
            projs.append(query_embed)
        return projs

    @staticmethod
    def get_reference_points(H, W, Z=8, D=4, dim='3d', bs=1, device='cuda', dtype=torch.long):
        """Get the reference points used in decoder.
        Args:
            H, W spatial shape of bev
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # 2d to 3d reference points, need grid from M_inv
        if dim == '3d':
            raise Exception("get reference poitns 3d not supported")
            zs = torch.linspace(0.5, Z - 0.5, D, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(-1, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(D, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(D, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H  # ?
            ref_x = ref_x.reshape(-1)[None] / W  # ?
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d   


class BEVHead(nn.Module):
    def __init__(self, args, channels=128):
        super(BEVHead, self).__init__()
        self.size_reduce_layer_1 = Lane3D.SingleTopViewPathway(channels)            # 128 to 128
        self.size_reduce_layer_2 = Lane3D.SingleTopViewPathway(channels*2)          # 256 to 256
        self.size_dim_reduce_layer_3 = Lane3D.EasyDown2TopViewPathway(channels*4)   # 512 to 256

        self.dim_reduce_layers = nn.ModuleList()
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*2,     # 256
                                                        channels,                   # 128
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*4,     # 512
                                                        channels*2,                 # 256
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*4,     # 512
                                                        channels*2,                 # 256
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))

    def forward(self, projs):
        '''
            projs_0 size: torch.Size([4, 128, 208, 128])
            projs_1 size: torch.Size([4, 256, 104, 64])
            projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])

            bev_feat_1 size: torch.Size([4, 128, 104, 64])
            bev_feat_2 size: torch.Size([4, 256, 52, 32])
            bev_feat_3 size: torch.Size([4, 256, 26, 16])

            bev_feat   size: torch.Size([4, 512, 26, 16])
        '''
        bev_feat_1 = self.size_reduce_layer_1(projs[0])          # 128 -> 128
        rts_proj_feat_1 = self.dim_reduce_layers[0](projs[1])    # 256 -> 128
        bev_feat_2 = self.size_reduce_layer_2(torch.cat((bev_feat_1, rts_proj_feat_1), 1))     # 128+128 -> 256
        rts_proj_feat_2 = self.dim_reduce_layers[1](projs[2])    # 512 -> 256
        bev_feat_3 = self.size_dim_reduce_layer_3(torch.cat((bev_feat_2, rts_proj_feat_2), 1)) # 256+256 -> 256
        rts_proj_feat_3 = self.dim_reduce_layers[2](projs[3])    # 512 -> 256
        bev_feat = torch.cat((bev_feat_3, rts_proj_feat_3), 1)   # 256+256=512
        return bev_feat


class SegmentHead(nn.Module):
    def __init__(self, channels=128):
        super(SegmentHead, self).__init__()
        self.down1 = Down(channels, channels*2)     # Down(128, 256)
        self.down2 = Down(channels*2, channels*4)   # Down(256, 512)
        self.down3 = Down(channels*4, channels*4)   # Down(512, 512)
        self.up1 = Up(channels*8, channels*2)       # Up(1024, 256)
        self.up2 = Up(channels*4, channels)         # Up(512, 128)
        self.up3 = Up(channels*2, channels)         # Up(256, 128)
        self.segment_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, input):
        x1 = self.down1(input)                      # 128 -> 256
        x2 = self.down2(x1)                         # 256 -> 512
        x3 = self.down3(x2)                         # 512 -> 512
        x_out = self.up1(x3, x2)                    # 512+512 -> 256
        x_out = self.up2(x_out, x1)                 # 256+256 -> 128
        x_out = self.up3(x_out, input)              # 128+128 ->128
        pred_seg_bev_map = self.segment_head(x_out) # 128 -> 1

        return pred_seg_bev_map
