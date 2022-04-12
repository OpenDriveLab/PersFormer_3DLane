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
from utils.utils import *
from models.networks.feature_extractor import *
from models.networks import Lane2D, Lane3D
from models.networks.libs.layers import *
from models.networks.PE import PositionEmbeddingLearned
from models.networks.Layers import EncoderLayer
from models.networks.Unet_parts import Down, Up
from models.networks.fpn import FPN

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
        self.fpn = FPN(self.encoder.dimList, args.feature_channels, 4, add_extra_convs=False)
        self.neck = nn.Sequential(*make_one_layer(self.encoder.dimList[0], args.feature_channels, batch_norm=True),
                                  *make_one_layer(args.feature_channels, args.feature_channels, batch_norm=True))

        # 2d detector
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

        self.lane_out = Lane3D.LanePredictionHead(args.feature_channels * pow(2, self.num_proj - 2),
                                                  self.num_lane_type,
                                                  self.num_y_steps,
                                                  args.num_category,
                                                  args.fmap_mapping_interp_index,
                                                  args.fmap_mapping_interp_weight,
                                                  args.no_3d,
                                                  args.batch_norm,
                                                  args.no_cuda)

        '''
            ATTENTION RELATED
            frontview_features_0 size: torch.Size([4, 128, 180, 240])
            frontview_features_1 size: torch.Size([4, 256, 90, 120])
            frontview_features_2 size: torch.Size([4, 512, 45, 60])
            frontview_features_3 size: torch.Size([4, 512, 22, 30])
            x_0 size: torch.Size([4, 128, 208, 128])
            x_1 size: torch.Size([4, 128, 104, 64])
            x_2 size: torch.Size([4, 256, 52, 32])
            x_3 size: torch.Size([4, 256, 26, 16])
        '''
        # attn num channel
        self.uv_feat_c_1 = 128
        self.uv_feat_c_2 = self.uv_feat_c_1 * 2
        self.uv_feat_c_3 = self.uv_feat_c_2 * 2
        self.uv_feat_c_4 = self.uv_feat_c_2 * 2
        # self.uv_feat_len_4 = 22*30
        self.uv_h_1 = 180
        self.uv_w_1 = 240
        self.uv_feat_len_1 = self.uv_h_1 * self.uv_w_1

        self.uv_h_2 = self.uv_h_1 // 2
        self.uv_w_2 = self.uv_w_1 // 2
        self.uv_feat_len_2 = self.uv_h_2 * self.uv_w_2

        self.uv_h_3 = self.uv_h_2 // 2
        self.uv_w_3 = self.uv_w_2 // 2
        self.uv_feat_len_3 = self.uv_h_3 * self.uv_w_3

        self.uv_h_4 = self.uv_h_3 // 2
        self.uv_w_4 = self.uv_w_3 // 2
        self.uv_feat_len_4 = self.uv_h_4 * self.uv_w_4
        # self.bev_feat_len_4 = 26*16

        self.bev_h_1 = 208
        self.bev_w_1 = 128
        self.bev_feat_len_1 = self.bev_h_1 * self.bev_w_1
        
        self.bev_h_2 = self.bev_h_1 // 2
        self.bev_w_2 = self.bev_w_1 // 2
        self.bev_feat_len_2 = self.bev_h_2 * self.bev_w_2

        self.bev_h_3 = self.bev_h_2 // 2
        self.bev_w_3 = self.bev_w_2 // 2
        self.bev_feat_len_3 = self.bev_h_3 * self.bev_w_3

        self.bev_h_4 = self.bev_h_3 // 2
        self.bev_w_4 = self.bev_w_3 // 2
        self.bev_feat_len_4 = self.bev_h_4 * self.bev_w_4

        self.dim_ffn_4 = self.uv_feat_c_4 * 2
        self.dim_ffn_3 = self.uv_feat_c_3 * 2
        self.dim_ffn_2 = self.uv_feat_c_2 * 2
        self.dim_ffn_1 = self.uv_feat_c_1 * 2

        self.nhead = args.nhead

        # learnable query
        query_embed_1 = nn.Embedding(self.bev_feat_len_1, self.uv_feat_c_1)
        query_embed_2 = nn.Embedding(self.bev_feat_len_2, self.uv_feat_c_2)
        query_embed_3 = nn.Embedding(self.bev_feat_len_3, self.uv_feat_c_3)
        query_embed_4 = nn.Embedding(self.bev_feat_len_4, self.uv_feat_c_4)

        self.query_embeds = nn.ModuleList()
        self.query_embeds.append(query_embed_1)
        self.query_embeds.append(query_embed_2)
        self.query_embeds.append(query_embed_3)
        self.query_embeds.append(query_embed_4)

        self.npoints = args.npoints

        # Encoder layer version
        el1 = EncoderLayer(d_model=self.uv_feat_c_1, dim_ff=self.uv_feat_c_1*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el2 = EncoderLayer(d_model=self.uv_feat_c_2, dim_ff=self.uv_feat_c_2*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el3 = EncoderLayer(d_model=self.uv_feat_c_3, dim_ff=self.uv_feat_c_3*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el4 = EncoderLayer(d_model=self.uv_feat_c_4, dim_ff=self.uv_feat_c_4*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el1_1 = EncoderLayer(d_model=self.uv_feat_c_1, dim_ff=self.uv_feat_c_1*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el2_1 = EncoderLayer(d_model=self.uv_feat_c_2, dim_ff=self.uv_feat_c_2*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el3_1 = EncoderLayer(d_model=self.uv_feat_c_3, dim_ff=self.uv_feat_c_3*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el4_1 = EncoderLayer(d_model=self.uv_feat_c_4, dim_ff=self.uv_feat_c_4*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el1_2 = EncoderLayer(d_model=self.uv_feat_c_1, dim_ff=self.uv_feat_c_1*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el2_2 = EncoderLayer(d_model=self.uv_feat_c_2, dim_ff=self.uv_feat_c_2*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el3_2 = EncoderLayer(d_model=self.uv_feat_c_3, dim_ff=self.uv_feat_c_3*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el4_2 = EncoderLayer(d_model=self.uv_feat_c_4, dim_ff=self.uv_feat_c_4*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)

        self.el = nn.ModuleList()
        self.el.append(el1)
        self.el.append(el2)
        self.el.append(el3)
        self.el.append(el4)
        self.el.append(el1_1)
        self.el.append(el2_1)
        self.el.append(el3_1)
        self.el.append(el4_1)
        self.el.append(el1_2)
        self.el.append(el2_2)
        self.el.append(el3_2)
        self.el.append(el4_2)

        pe1 = PositionEmbeddingLearned(h=self.bev_h_1, w=self.bev_w_1, num_pos_feats=self.uv_feat_c_1 // 2)
        pe2 = PositionEmbeddingLearned(h=self.bev_h_2, w=self.bev_w_2, num_pos_feats=self.uv_feat_c_2 // 2)
        pe3 = PositionEmbeddingLearned(h=self.bev_h_3, w=self.bev_w_3, num_pos_feats=self.uv_feat_c_3 // 2)
        pe4 = PositionEmbeddingLearned(h=self.bev_h_4, w=self.bev_w_4, num_pos_feats=self.uv_feat_c_4 // 2)
        self.pe = nn.ModuleList()
        self.pe.append(pe1)
        self.pe.append(pe2)
        self.pe.append(pe3)
        self.pe.append(pe4)

        # 2d uniform sampling
        self.ref_2d_1 = self.get_reference_points(H=self.bev_h_1, W=self.bev_w_1, dim='2d', bs=1)
        self.ref_2d_2 = self.get_reference_points(H=self.bev_h_2, W=self.bev_w_2, dim='2d', bs=1)
        self.ref_2d_3 = self.get_reference_points(H=self.bev_h_3, W=self.bev_w_3, dim='2d', bs=1)
        self.ref_2d_4 = self.get_reference_points(H=self.bev_h_4, W=self.bev_w_4, dim='2d', bs=1)

        size_top1 = torch.Size([self.bev_h_1, self.bev_w_1])
        self.project_layer1 = Lane3D.RefPntsNoGradGenerator(size_top1, self.M_inv, args.no_cuda)
        size_top2 = torch.Size([self.bev_h_2, self.bev_w_2])
        self.project_layer2 = Lane3D.RefPntsNoGradGenerator(size_top2, self.M_inv, args.no_cuda)
        size_top3 = torch.Size([self.bev_h_3, self.bev_w_3])
        self.project_layer3 = Lane3D.RefPntsNoGradGenerator(size_top3, self.M_inv, args.no_cuda)
        size_top4 = torch.Size([self.bev_h_4, self.bev_w_4])
        self.project_layer4 = Lane3D.RefPntsNoGradGenerator(size_top4, self.M_inv, args.no_cuda)

        # input_spatial_shapes & input_level_start_index
        self.input_spatial_shapes_1 = torch.as_tensor([(self.uv_h_1, self.uv_w_1)], dtype=torch.long)
        self.input_level_start_index_1 = torch.as_tensor([0.0,], dtype=torch.long)

        self.input_spatial_shapes_2 = torch.as_tensor([(self.uv_h_2, self.uv_w_2)], dtype=torch.long)
        self.input_level_start_index_2 = torch.as_tensor([0.0,], dtype=torch.long)

        self.input_spatial_shapes_3 = torch.as_tensor([(self.uv_h_3, self.uv_w_3)], dtype=torch.long)
        self.input_level_start_index_3 = torch.as_tensor([0.0,], dtype=torch.long)

        self.input_spatial_shapes_4 = torch.as_tensor([(self.uv_h_4, self.uv_w_4)], dtype=torch.long)
        self.input_level_start_index_4 = torch.as_tensor([0.0,], dtype=torch.long)

        # dim reduce & size reduce
        self.dim_size_rts_1 = Lane3D.SingleTopViewPathway(128)   # 128 to 128
        self.dim_size_rts_2 = Lane3D.SingleTopViewPathway(256)   # 256 to 256
        self.dim_size_rts_3 = Lane3D.EasyDown2TopViewPathway(512)   # 512 to 256

        self.dim_rts = nn.ModuleList()
        self.dim_rts.append(nn.Sequential(*make_one_layer(256,
                                                        128,
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))
        self.dim_rts.append(nn.Sequential(*make_one_layer(512,
                                                        256,
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))
        self.dim_rts.append(nn.Sequential(*make_one_layer(512,
                                                        256,
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=args.batch_norm)))


        # single att-style ipm
        self.use_proj_layer = 0

        '''
            projs_0 size: torch.Size([4, 128, 208, 128])
            projs_1 size: torch.Size([4, 256, 104, 64])
            projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])
        '''
        # segmentation feature extractor
        # v3 first bev into unet
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        factor = 2
        self.down3 = Down(512, 1024//factor)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 128)

        # segmentation head
        self.segment_head = nn.Conv2d(128, 1, kernel_size=1)

        # uncertainty loss weight
        self.uncertainty_loss = nn.Parameter(torch.tensor([args._3d_vis_loss_weight,
                                                            args._3d_prob_loss_weight,
                                                            args._3d_reg_loss_weight,
                                                            args._2d_vis_loss_weight,
                                                            args._2d_prob_loss_weight,
                                                            args._2d_reg_loss_weight,
                                                            args._seg_loss_weight]), requires_grad=True)

    def forward(self, input, _M_inv = None, eval=False, use_fpn=False, use_att=True, args=None):
        out_featList = self.encoder(input)
        if not use_fpn:
            neck_out = self.neck(out_featList[0])
            frontview_features = self.shared_encoder(neck_out)
            # print("max feat number in neck_out: ", torch.max(neck_out))
        else:
            fpn_outs = self.fpn(out_featList)
            frontview_features = self.shared_encoder(fpn_outs[0])

        frontview_final_feat = frontview_features[-1]

        '''
            frontview_features_0 size: torch.Size([4, 128, 180, 240])
            frontview_features_1 size: torch.Size([4, 256, 90, 120])
            frontview_features_2 size: torch.Size([4, 512, 45, 60])
            frontview_features_3 size: torch.Size([4, 512, 22, 30])
        '''

        if eval is False:
            laneatt_proposals_list = self.laneatt_head(frontview_final_feat, nms_thres=20., eval=eval)
        else:
            # TODO: conf_threshold -> no prediction -> error in loss
            laneatt_proposals_list = self.laneatt_head(frontview_final_feat, conf_threshold=0.5,
                                                       nms_thres=36., nms_topk=self.max_lanes, eval=eval)


        cam_height = self.cam_height.to(input.device)
        cam_pitch = self.cam_pitch.to(input.device)

        projs = []
        # deform att multi scale
        for i in range(4):
            input_spatial_shapes = getattr(self, "input_spatial_shapes_{}".format(i + 1)).to(input.device)
            input_level_start_index = getattr(self, "input_level_start_index_{}".format(i + 1)).to(input.device)
            bs, c, h, w = frontview_features[i].shape
            bev_h = getattr(self, "bev_h_{}".format(i + 1))
            bev_w = getattr(self, "bev_w_{}".format(i + 1))

            src = frontview_features[i].flatten(2).permute(0, 2, 1)
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)

            # reference points generated by ipm grid
            project_layer = getattr(self, "project_layer{}".format(i + 1))
            ref_pnts = project_layer(_M_inv).unsqueeze(-2)
            # ref_pnts = getattr(self, "reference_point_{}".format(i + 1)).unsqueeze(0).repeat(bs, 1, 1, 1).to(input.device)

            # encoder layers
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(query_embed.dtype)
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            ref_2d = getattr(self, 'ref_2d_{}'.format(i + 1)).repeat(bs, 1, 1, 1).to(input.device)
            # first layer
            query_embed = self.el[i](query=query_embed, value=src, bev_pos=bev_pos, 
                                        ref_2d = ref_2d, ref_3d=ref_pnts,
                                        bev_h=bev_h, bev_w=bev_w, 
                                        spatial_shapes=input_spatial_shapes,
                                        level_start_index=input_level_start_index)
            # second layer
            query_embed = self.el[i+4](query=query_embed, value=src, bev_pos=bev_pos, 
                                        ref_2d = ref_2d, ref_3d=ref_pnts,
                                        bev_h=bev_h, bev_w=bev_w, 
                                        spatial_shapes=input_spatial_shapes,
                                        level_start_index=input_level_start_index)
            x = self.el[i+8](query=query_embed, value=src, bev_pos=bev_pos, 
                                ref_2d = ref_2d, ref_3d=ref_pnts,
                                bev_h=bev_h, bev_w=bev_w, 
                                spatial_shapes=input_spatial_shapes,
                                level_start_index=input_level_start_index)

            x = x.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()
            projs.append(x)

        proj_feat_1 = self.dim_size_rts_1(projs[0])
        rts_proj_feat_1 = self.dim_rts[0](projs[1])
        proj_feat_2 = self.dim_size_rts_2(torch.cat((proj_feat_1, rts_proj_feat_1), 1))
        rts_proj_feat_2 = self.dim_rts[1](projs[2])
        proj_feat_3 = self.dim_size_rts_3(torch.cat((proj_feat_2, rts_proj_feat_2), 1))
        rts_proj_feat_3 = self.dim_rts[2](projs[3])
        x = torch.cat((proj_feat_3, rts_proj_feat_3), 1)

        
        '''
            x_0 size: torch.Size([4, 128, 208, 128])
            x_1 size: torch.Size([4, 128, 104, 64])
            x_2 size: torch.Size([4, 256, 52, 32])
            x_3 size: torch.Size([4, 256, 26, 16])
        '''

        out = self.lane_out(x)

        # segment head v3
        x1 = self.down1(projs[0])
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x_out = self.up1(x3, x2)
        x_out = self.up2(x_out, x1)
        x_out = self.up3(x_out, projs[0])
        pred_seg_bev_map = self.segment_head(x_out)

        # seperate loss weight
        uncertainty_loss = torch.tensor(1.0).to(input.device) * self.uncertainty_loss.to(input.device)

        return laneatt_proposals_list, out, cam_height, cam_pitch, pred_seg_bev_map, uncertainty_loss

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

            # zs = torch.arange(1, Z, 2, dtype=dtype,
            #                  device=device).view(-1, 1, 1).expand(-1, H, W)/Z

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
            return deepFeatureExtractor_EfficientNet(args.encoder, lv6=False, lv5=True, lv4=True, lv3=True)
        else:
            raise Exception("encoder model in args is not supported")
