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
import torch.nn.functional as F
from utils.utils import *

from nms import nms
from .libs.lane import Lane
from .libs.focal_loss import FocalLoss
from .libs.layers import *
from .libs.matching import match_proposals_with_targets

import math
import numpy as np

class FrontViewPathway(nn.Module):
    def __init__(self, input_channels, num_proj, init_weights=True):
        super(FrontViewPathway, self).__init__()

        self.input_channels = input_channels
        self.num_proj = num_proj

        # TODO: maxpool, then conv
        self.layers = nn.ModuleList()
        output_channels = input_channels
        for i in range(num_proj - 1):
            if i < num_proj - 2:
                output_channels *= 2
            layers = []
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            self.layers.append(nn.Sequential(*layers))
            input_channels = output_channels
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         print("Max weight in 2D FrontViewPathway: ", torch.max(torch.abs(m.weight)))
        #         if m.weight.grad != None:
        #             print("Max weight grad in 2D FrontViewPathway: ", torch.max(torch.abs(m.weight.grad)))

        outs = []
        outs.append(input)
        for i, layer in enumerate(self.layers):
            input = layer(input)
            # print("proj_features[{}] shape: ".format(str(i)), input.size())
            # print("running mean for layer[{}]".format(i), layer[1].running_mean)
            # print("running var for layer[{}]".format(i), layer[1].running_var)
            outs.append(input)
            # if self.layers[i][0].weight.grad != None and torch.max(self.layers[i][0].weight.grad) > 100:
            #     print("layers[{}] weights grad > 100".format(i))
            #     # print("conv in layers[-1] weights grad: ", self.layers[-1][0].weight.grad)
        return outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FrontViewPathwayFixChannel(nn.Module):
    def __init__(self, input_channels, num_proj, init_weights=True):
        super(FrontViewPathwayFixChannel, self).__init__()

        self.input_channels = input_channels
        self.num_proj = num_proj

        # TODO: maxpool, then conv
        self.layers = nn.ModuleList()
        output_channels = input_channels
        for i in range(num_proj - 1):
            layers = []
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            self.layers.append(nn.Sequential(*layers))
            input_channels = output_channels
        
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        outs = []
        outs.append(input)
        for i, layer in enumerate(self.layers):
            input = layer(input)
            outs.append(input)
        return outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class LaneATTHead(nn.Module):
    def __init__(self, stride, input_channels, im_anchor_origins, im_anchor_angles,
                 img_w=640, img_h=360, S=72, anchor_feat_channels=64, num_category=2):
        super(LaneATTHead, self).__init__()

        self.stride = stride
        self.img_w = img_w
        self.img_h = img_h
        self.hw_ratio = img_h / img_w
        self.fmap_h = img_h // stride
        fmap_w = img_w // stride
        self.fmap_w = fmap_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.anchor_feat_channels = anchor_feat_channels
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.num_category = num_category

        self.use_default_anchor = True
        if im_anchor_origins is not None and im_anchor_angles is not None:
            self.use_default_anchor = False
            self.left_origins = im_anchor_origins[0]
            self.right_origins = im_anchor_origins[1]
            self.bottom_origins = im_anchor_origins[2]
            self.left_angles = im_anchor_angles[0]
            self.right_angles = im_anchor_angles[1]
            self.bottom_angles = im_anchor_angles[2]
            # Generate anchors
            self.anchors, self.anchors_cut = self.generate_anchors_new()
        else:
            self.left_angles = [72., 60., 49., 39., 30., 22.]
            self.right_angles = [108., 120., 131., 141., 150., 158.]
            self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]
            # Generate anchors
            self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(input_channels, self.anchor_feat_channels, kernel_size=1)
        # self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.num_category)
        # self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2 * self.n_offsets)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def forward(self, batch_features, conf_threshold=None, nms_thres=0, nms_topk=3000, eval=False):
        batch_features = self.conv1(batch_features)
        batch_anchor_features = self.cut_anchor_features(batch_features)

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Move relevant tensors to device
        self.anchors = self.anchors.to(device=batch_features.device)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(batch_features.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=batch_features.device).repeat(batch_features.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(batch_features.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(batch_features.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(batch_features.shape[0], -1, reg.shape[1])
        sigmoid = nn.Sigmoid()
        reg[:, :, self.n_offsets:] = sigmoid(reg[:, :, self.n_offsets:])

        # Add offsets to anchors
        # reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=batch_features.device)
        reg_proposals = torch.zeros((*cls_logits.shape[:2], self.num_category + 2 + 2 * self.n_offsets), 
                                        device=batch_features.device)
        reg_proposals += self.anchors
        # reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, :self.num_category] = cls_logits
        # reg_proposals[:, :, 4:] += reg
        reg_proposals[:, :, self.num_category+2:] += reg

        # Apply nms
        # proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold, eval=eval)

        proposals_list = []
        for proposals, att_matrix in zip(reg_proposals, attention_matrix):
            anchor_inds = torch.arange(reg_proposals.shape[1], device=proposals.device)
            proposals_list.append((proposals, self.anchors, att_matrix, anchor_inds))

        return proposals_list
    
    # nms_thres -> lane distance between valid portion of two lanes
    # small nms_thres -> less supression
    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]   # idxes to keep
            proposals = proposals[keep]     # proposals to keep
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))
        
        # proposals_list = torch.FloatTensor(proposals_list).to(device=batch_proposals.device)

        return proposals_list

    def nms_new(self, batch_proposals_list, nms_thres, nms_topk, conf_threshold, vis_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, _, attention_matrix, _ in batch_proposals_list:
            anchor_inds = torch.arange(proposals.shape[0], device=proposals.device)
            # proposals[:, self.num_category+2+self.n_offsets:] = \
            #     torch.sigmoid(proposals[:, self.num_category+2+self.n_offsets:])
            scores = softmax(proposals[:, :self.num_category])
            # only preserve the max prob category for one anchor
            scores_one_category = torch.max(scores[:, 1:], dim=1)[0]
            with torch.no_grad():
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores_one_category > conf_threshold
                    proposals = proposals[above_threshold]
                    scores_one_category = scores_one_category[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                
                # proposals_starts = (proposals[:, self.num_category] * self.n_strips).round().long()
                # last_vis_idx = proposals.new_zeros(proposals.shape[0])
                # for i, lane in enumerate(proposals):
                #     idx = proposals.shape[1] - 1
                #     while lane[idx]  < vis_threshold and idx > proposals_starts[i]:
                #         idx -= 1
                #     last_vis_idx[i] = idx - (self.num_category + 2 + self.n_offsets)
                
                # last_vis_idx = proposals.new_zeros(proposals.shape[0], dtype=torch.long)
                proposal_vis = proposals[:, self.num_category+2+self.n_offsets:]
                x_s, y_s = torch.nonzero(proposal_vis >= vis_threshold, as_tuple=True)
                # last_vis_idx[x_s] = y_s
                new_last_vis_idx = np.zeros(proposals.shape[0], dtype=np.int)
                new_last_vis_idx[x_s.cpu().numpy()] = y_s.cpu().numpy().astype(np.int)
                ends = torch.from_numpy(new_last_vis_idx).to(proposals.device)

                # change anchor dim back to fit in the nms pkg
                proposals_nms = proposals.new_zeros(proposals.shape[0], 2 + 3 + self.n_offsets)
                proposals_nms[:, 2:2+2] = proposals[:, self.num_category:self.num_category+2]

                # TODO: no inplace modification
                _a = ends - proposals_nms[:, 2] + 1
                proposals_nms[:, 2+2] = _a
                # proposals_nms[:, 2+2] = ends - proposals_nms[:, 2] + 1
                
                proposals_nms[:, 2+3:2+3+self.n_offsets] = \
                    proposals[:, self.num_category+2:self.num_category+2+self.n_offsets]
                keep, num_to_keep, _ = nms(proposals_nms, scores_one_category, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            
            # TODO: no inplace modification
            _b = proposals[keep]
            proposals = _b
            # proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[keep]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

            # TODO: multi-label output option
            # TODO: proposals_one_img, anchors_one_img, attention_matrix_one_img, anchor_inds_one_img
            # for i in range(1, self.num_category):
            #     scores_one_category = scores[:, i]
            #     proposals_tmp = proposals.clone()
            #     if conf_threshold is not None:
            #         # apply confidence threshold
            #         above_threshold = scores_one_category > conf_threshold
            #         proposals_tmp = proposals_tmp[above_threshold]
            #         scores_one_category = scores_one_category[above_threshold]
            #         anchor_inds = anchor_inds[above_threshold]
            #     if proposals_tmp.shape[0] == 0:
            #         # proposals_list.append((proposals_tmp[[]], self.anchors[[]], attention_matrix[[]], None))
            #         continue
            #     with torch.no_grad():
            #         last_vis_idx = proposals_tmp.new_zeros(proposals_tmp.shape[0])
            #         for i, proposal in enumerate(proposals_tmp):
            #             idx = proposal.shape[1] - 1
            #             while proposal[idx]  < vis_threshold:
            #                 idx -= 1
            #             last_vis_idx[i] = idx - (self.num_category + 2 + self.n_offsets)
            #         proposals_nms = proposals_tmp.new_zeros(proposal.shape[0], 2 + 3 + self.n_offsets)
            #         proposals_nms[:, 2:2+2] = proposals_tmp[:, self.num_category:self.num_category+2]
            #         proposals_nms[:, 2+3:2+3+self.n_offsets] = \
            #             proposals_tmp[:, self.num_category+2:self.num_category+2+self.n_offsets]
            #         keep, num_to_keep, _ = nms(proposals_nms, scores_one_category, overlap=nms_thres, top_k=nms_topk)
            #         keep = keep[:num_to_keep]
            #     proposals_tmp = proposals_tmp[keep]
            #     anchor_inds = anchor_inds[keep]
            #     attention_matrix = attention_matrix[keep]
            #     proposals_list.append((proposals_tmp, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10, reg_vis_loss_weight=100):        
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        binary_cross_entropy_loss = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        cls_loss = torch.tensor(0).float().to(targets.device)
        reg_loss = torch.tensor(0).float().to(targets.device)
        reg_vis_loss = torch.tensor(0).float().to(targets.device)
        valid_imgs = len(targets)
        total_positives = 0
        batch_anchors_positives = []
        cnt = 0
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            # target = target[target[:, 1] == 1]
            target = target[target[:, 0] == 0]
            # in case no proposals when large nms suppression for test
            if len(proposals) == 0:
                continue
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                # cls_pred = proposals[:, :2]
                cls_pred = proposals[:, :self.num_category]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                batch_anchors_positives.append([])
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                anchors = anchors.to(device=target.device)
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = \
                    match_proposals_with_targets(self, anchors, target)
                anchors_positives = anchors[positives_mask]
                batch_anchors_positives.append(anchors_positives)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            # print("num_positives, num_negatives: ", num_positives, num_negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                # cls_pred = proposals[:, :2]
                cls_pred = proposals[:, :self.num_category]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            # cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            # cls_target[:num_positives] = 1.
            # cls_pred = all_proposals[:, :2]
            cls_pred = all_proposals[:, :self.num_category]
            with torch.no_grad():
                target_positives = target[target_positives_indices]
                cls_target = proposals.new_zeros(num_positives + num_negatives, self.num_category).long()
                cls_target[:num_positives, :] = target_positives[:, :self.num_category]
                cls_target[num_positives:, 0] = 1
            # print("cls_pred: ", cls_pred)
            # print("cls_target: ", cls_target)
            log_prob = F.log_softmax(cls_pred, dim=-1)
            cls_loss = -torch.sum(log_prob * cls_target) / (num_positives + num_negatives)

            # Regression targets
            # reg_pred = positives[:, 4:]
            reg_pred = positives[:, self.num_category+2 : self.num_category+2+self.n_offsets]
            reg_vis_pred = positives[:, self.num_category+2+self.n_offsets:]
            # reg_vis_pred = sigmoid(reg_vis_pred)
            with torch.no_grad():
                target = target[target_positives_indices]
                # positive_starts = (positives[:, 2] * self.n_strips).round().long()
                # target_starts = (target[:, 2] * self.n_strips).round().long()
                # target[:, 4] -= positive_starts - target_starts
                positive_starts = (positives[:, self.num_category] * self.n_strips).round().long()
                target_starts = (target[:, self.num_category] * self.n_strips).round().long()
                all_indices = torch.arange(num_positives, dtype=torch.long)

                # ends = (positive_starts + target[:, 4] - 1).round().long()
                # invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                #                                    dtype=torch.int)  # length + S + pad
                # invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                # invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                # invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                # invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                # invalid_offsets_mask[:, 0] = False

                # Brute-force find ends with last visible point in target
                # last_vis_idx = target_starts.new_zeros(target_starts.shape)
                # for i, lane in enumerate(target):
                #     idx = target.shape[1] - 1
                #     while lane[idx] < 1e-5 and idx > target_starts[i]:
                #         idx -= 1
                #     last_vis_idx[i] = idx - (self.num_category + 2 + self.n_offsets)
                # # this actually -> ends = last_vis_idx
                # length = (last_vis_idx - target_starts + 1)
                # length -= positive_starts - target_starts
                # ends = (positive_starts + length - 1)
                # invalid_offsets_mask = torch.zeros((num_positives, self.n_offsets + 1), dtype=torch.int)  # S + pad
                # invalid_offsets_mask[all_indices, positive_starts] = 1
                # invalid_offsets_mask[all_indices, ends + 1] -= 1
                # invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0  # invalid points with True tag
                # invalid_offsets_mask = invalid_offsets_mask[:, :-1] # remove pad

                invalid_offsets_mask = target[:, self.num_category+2+self.n_offsets:] < 1e-5

                # reg_target = target[:, 4:]
                # reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]
                reg_target = target[:, self.num_category+2 : self.num_category+2+self.n_offsets]               
                reg_vis_target = target[:, self.num_category+2+self.n_offsets:]
                reg_target_length = torch.sum(reg_vis_target, dim=-1) + 1e-9

            # Loss calc
            # reg_loss += smooth_l1_loss(reg_pred, reg_target)
            # print("reg_target_length: ", reg_target_length)
            # print("reg_pred: ", reg_pred)
            # print("reg_target: ", reg_target)
            reg_loss += torch.mean(torch.norm(reg_vis_target * (reg_pred - reg_target), p=1, dim=-1) / reg_target_length)
            reg_vis_loss += binary_cross_entropy_loss(reg_vis_pred, reg_vis_target)
            # cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

            if reg_loss > 1e7:
                torch.set_printoptions(edgeitems=20)
                # print("reg_target: \n", reg_target)
                # print("reg_pred: \n", reg_pred)
                # print("reg_vis_target: \n", reg_vis_target)
                # print("idx cnt: ", cnt)
                torch.set_printoptions(edgeitems=3)
            cnt += 1
            
        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs
        reg_vis_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss + reg_vis_loss_weight * reg_vis_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss,
                      'vis_loss': reg_vis_loss, 'batch_positives': total_positives,
                      'anchors_positives': batch_anchors_positives}

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        # anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        # anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        
        # num_category scores, 1 start_y, start_x, S coordinates
        anchors = torch.zeros((n_anchors, self.num_category + 2 + 2 * self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, self.num_category + 2 + 2 * self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut
    
    def generate_anchors_new(self):
        left_anchors, left_cut = self.generate_side_anchors_new(self.left_angles, self.left_origins, x=0.)
        right_anchors, right_cut = self.generate_side_anchors_new(self.right_angles, self.right_origins, x=1.)
        bottom_anchors, bottom_cut = self.generate_side_anchors_new(self.bottom_angles, self.bottom_origins, y=1.)
        
        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])
    
    def generate_side_anchors_new(self, angles, origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in origins]
        elif x is not None and y is None:
            starts = [(x, y) for y in origins]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        # anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        # anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        
        # num_category scores, 1 start_y, start_x, S coordinates
        anchors = torch.zeros((n_anchors, self.num_category + 2 + 2 * self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, self.num_category + 2 + 2 * self.fmap_h))
        for k, (start, angle) in enumerate(zip(starts, angles)):
            anchors[k] = self.generate_anchor(start, angle)
            anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut
    
    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            # anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
            anchor = torch.zeros(self.num_category + 2 + 2 * self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            # anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
            anchor = torch.zeros(self.num_category + 2 + 2 * self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        # anchor[2] = 1 - start_y
        # anchor[3] = start_x
        # anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w
        anchor[self.num_category] = 1 - start_y
        anchor[self.num_category+1] = start_x
        if self.use_default_anchor:
            if cut:
                anchor[self.num_category+2:self.num_category+2+self.fmap_h] = \
                    (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w
            else:
                anchor[self.num_category+2:self.num_category+2+self.n_offsets] = \
                    (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w
        else:
            if cut:
                anchor[self.num_category+2:self.num_category+2+self.fmap_h] = \
                    (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle) * self.hw_ratio) * self.img_w
            else:
                anchor[self.num_category+2:self.num_category+2+self.n_offsets] = \
                    (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle) * self.hw_ratio) * self.img_w
        
        # NOTE: anchors do not need visibility assignment here

        return anchor

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        # unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = torch.flip((self.anchors_cut[:, self.num_category+2:self.num_category+2+fmaps_h] 
                                        / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def anchors_to_lanes(self, batch_anchors):
        decoded = []
        for anchors in batch_anchors:
            if isinstance(anchors, list):
                decoded.append([])
                continue
            self.anchor_ys = self.anchor_ys.to(anchors.device)
            self.anchor_ys = self.anchor_ys.double()
            lanes = []
            for anchor in anchors:
                lane_xs = anchor[self.num_category+2:self.num_category+2+self.n_offsets] / self.img_w
                lane_ys = self.anchor_ys[(lane_xs >= 0.) & (lane_xs <= 1.)]
                lane_xs = lane_xs[(lane_xs >= 0.) & (lane_xs <= 1.)]
                lane_xs = lane_xs.flip(0).double()
                lane_ys = lane_ys.flip(0)
                if len(lane_xs) <= 1:
                    continue
                points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
                points = points.data.cpu().numpy()
                if np.shape(points)[0] < 2:
                    continue
                # diff from proposals_to_lanes, directly ouput points here rather than Lane
                lanes.append(points)
            decoded.append(lanes)
        return decoded

    def proposals_to_pred(self, proposals, vis_thres):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            # lane_xs = lane[5:] / self.img_w
            lane_xs = lane[self.num_category+2:self.num_category+2+self.n_offsets] / self.img_w
            # start = int(round(lane[2].item() * self.n_strips))
            start = int(round(lane[self.num_category].item() * self.n_strips))
            # length = int(round(lane[4].item()))
            lane_vis = lane[self.num_category+2+self.n_offsets:]
            valid_vis_idxes = torch.nonzero(lane_vis >= vis_thres)
            start_vis_idx = valid_vis_idxes[0, 0].item() if len(valid_vis_idxes) else 0
            end_vis_idx = valid_vis_idxes[-1, 0].item() if len(valid_vis_idxes) else 0
            # end_vis_idx = proposals.shape[1] - 1
            # while lane[end_vis_idx] < vis_thres and end_vis_idx > self.num_category+2+self.n_offsets:
            #     end_vis_idx -= 1
            # end_vis_idx -= (self.num_category + 2 + self.n_offsets)
            length = (end_vis_idx - start + 1)
            # print("end: ", end_vis_idx)
            # print("length: ", length)
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            start = max(start, start_vis_idx)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            points = points.data.cpu().numpy()
            if np.shape(points)[0] < 2:
                continue
            pred_cat = torch.argmax(lane[1:self.num_category])
            lane = Lane(points=points,
                        metadata={
                            # 'start_x': lane[3],
                            # 'start_y': lane[2],
                            # 'conf': lane[1]
                            'start_x': lane[self.num_category+1].item(),
                            'start_y': lane[self.num_category].item(),
                            'pred_cat': pred_cat.item() + 1,
                            'conf': lane[pred_cat.item()]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, vis_thres, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            # proposals[:, :2] = softmax(proposals[:, :2])
            # proposals[:, :self.num_category] = softmax(proposals[:, :self.num_category])
            a = softmax(proposals[:, :self.num_category])
            proposals = torch.cat((a, proposals[:, self.num_category: ]), dim=1)
            # proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals, vis_thres)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded
