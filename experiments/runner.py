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
import torch.optim
import torch.nn as nn
import numpy as np
import glob
import time
import shutil
import os
from os import mkdir, write
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data.Load_Data import *
from models.PersFormer import PersFormer
from models.networks import Loss_crit
from models.networks.feature_extractor import *
from utils import eval_3D_lane
from utils.utils import *

# ddp related
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .ddp import *


class Runner:
    def __init__(self, args):
        self.args = args
        # Check GPU availability
        if args.proc_id == 0:
            if not args.no_cuda and not torch.cuda.is_available():
                raise Exception("No gpu available for usage")
            if torch.cuda.device_count() >= 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                torch.cuda.empty_cache()

        save_id = args.mod
        args.save_json_path = args.save_path
        args.save_path = os.path.join(args.save_path, save_id)
        if args.proc_id == 0:
            mkdir_if_missing(args.save_path)
            mkdir_if_missing(os.path.join(args.save_path, 'example/'))
            mkdir_if_missing(os.path.join(args.save_path, 'example/train'))
            mkdir_if_missing(os.path.join(args.save_path, 'example/valid'))

        # Get Dataset
        if args.proc_id == 0:
            print("Loading Dataset ...")
        self.val_gt_file = ops.join(args.save_path, 'test.json')
        self.train_dataset, self.train_loader, self.train_sampler = self._get_train_dataset()
        self.valid_dataset, self.valid_loader, self.valid_sampler = self._get_valid_dataset()

        self.crit_string = 'loss_gflat'
        # Define loss criteria
        if self.crit_string == 'loss_gflat_3D':
            self.criterion = Loss_crit.Laneline_loss_gflat_3D(args.batch_size, self.train_dataset.num_types,
                                                              self.train_dataset.anchor_x_steps, self.train_dataset.anchor_y_steps,
                                                              self.train_dataset._x_off_std, self.train_dataset._y_off_std,
                                                              self.train_dataset._z_std, args.pred_cam, args.no_cuda)
        else:
            self.criterion = Loss_crit.Laneline_loss_gflat_multiclass(self.train_dataset.num_types, args.num_y_steps,
                                                                      args.pred_cam, args.num_category, args.no_3d, args.loss_dist)
        if 'openlane' in args.dataset_name:
            self.evaluator = eval_3D_lane.LaneEval(args)
        else:
            self.evaluator = eval_3D_lane.LaneEval(args)
        # Tensorboard writer
        if not args.no_tb and args.proc_id == 0:
            tensorboard_path = os.path.join(args.save_path, 'Tensorboard/')
            mkdir_if_missing(tensorboard_path)
            self.writer = SummaryWriter(tensorboard_path)
        # initialize visual saver
        self.vs_saver = Visualizer(args)
        if args.proc_id == 0:
            print("Init Done!")


    def train(self):
        args = self.args

        # Get Dataset
        train_dataset = self.train_dataset
        train_loader = self.train_loader
        train_sampler = self.train_sampler

        # Define model or resume
        model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1 = self._get_model_ddp()

        criterion = self.criterion
        if not args.no_cuda:
            device = torch.device("cuda", args.local_rank)
            criterion = criterion.to(device)
        bceloss = nn.BCEWithLogitsLoss()

        # Print model basic info
        if args.proc_id == 0:
            print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
            print("Init model: '{}'".format(args.mod))
            print("Number of parameters in model {} is {:.3f}M".format(args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))

        # image matrix
        _S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                                    [                        0, 1/np.float(args.resize_h), 0],
                                                    [                        0,                         0, 1]], dtype=np.float32)).cuda()
        _S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                                [            0,  args.resize_h, 0],
                                                [            0,              0, 1]], dtype=np.float32)).cuda()
        if args.proc_id == 0:
            writer = self.writer
        vs_saver = self.vs_saver

        # Start training and validation for nepochs
        for epoch in range(args.start_epoch, args.nepochs):
            if args.proc_id == 0:
                print("\n => Start train set for EPOCH {}".format(epoch + 1))
                lr = optimizer.param_groups[0]['lr']
                print('lr is set to {}'.format(lr))

            if args.distributed:
                train_sampler.set_epoch(epoch)

            if epoch > args.seg_start_epoch:
                args.loss_seg_weight = 10.0

            # Define container objects to keep track of multiple losses/metrics
            batch_time = AverageMeter()
            data_time = AverageMeter()          # compute FPS
            losses = AverageMeter()
            losses_3d_vis = AverageMeter()
            losses_3d_prob = AverageMeter()
            losses_3d_reg = AverageMeter()
            losses_2d_vis = AverageMeter()
            losses_2d_cls = AverageMeter()
            losses_2d_reg = AverageMeter()

            # Specify operation modules
            model.train()
            # compute timing
            end = time.time()
            # Start training loop
            for i, (json_files, input, seg_maps, gt, gt_laneline_img, idx, gt_hcam, gt_pitch, gt_intrinsic, gt_extrinsic, aug_mat, seg_name, seg_bev_map) in tqdm(enumerate(train_loader)):
                # Time dataloader
                data_time.update(time.time() - end)

                # Put inputs on gpu if possible
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    seg_maps = seg_maps.cuda(non_blocking=True)
                    gt_hcam = gt_hcam.cuda()
                    gt_pitch = gt_pitch.cuda()
                    gt_intrinsic = gt_intrinsic.cuda()
                    gt_extrinsic = gt_extrinsic.cuda()
                    gt_laneline_img = gt_laneline_img.cuda()
                    seg_bev_map = seg_bev_map.cuda()
                input = input.contiguous().float()
                # print("input img: ", input, torch.isnan(input).any())
                # print("gt_laneline_img: ", gt_laneline_img, torch.isnan(gt_laneline_img).any())

                # update transformation based on gt extrinsic/intrinsic
                M_inv = unit_update_projection_extrinsic(args, gt_extrinsic, gt_intrinsic)
                # update transformation for data augmentation (only for training)
                M_inv = unit_update_projection_for_data_aug(args, aug_mat, M_inv, _S_im_inv, _S_im)

                # Run model
                optimizer.zero_grad()
                # Inference model
                laneatt_proposals_list, output_net, pred_hcam, pred_pitch, pred_seg_bev_map, uncertainty_loss = model(input=input, _M_inv=M_inv)

                # 3D loss
                loss_3d, loss_3d_dict = criterion(output_net, gt, pred_hcam, gt_hcam, pred_pitch, gt_pitch)
                # Add laneatt loss
                loss_att, loss_att_dict = model.module.laneatt_head.loss(laneatt_proposals_list, gt_laneline_img,
                                                                         cls_loss_weight=args.cls_loss_weight,
                                                                         reg_vis_loss_weight=args.reg_vis_loss_weight)
                # segmentation loss
                loss_seg = bceloss(pred_seg_bev_map, seg_bev_map)
                # overall loss
                loss = self.compute_loss(args, epoch, loss_3d, loss_att, loss_seg, uncertainty_loss, loss_3d_dict, loss_att_dict)

                if loss.data > args.loss_threshold:
                    print("Batch with idx {} skipped due to aug-caused too large loss".format(idx.numpy()))
                    loss.fill_(0.0)
                # Clip gradients (usefull for instabilities or mistakes in ground truth)
                if args.clip_grad_norm != 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # Setup backward pass
                loss.backward()

                # update params
                optimizer.step()

                # reduce loss from all gpu, then update losses
                loss_list = [losses, losses_3d_vis, losses_3d_prob, losses_3d_reg, losses_2d_vis, losses_2d_cls, losses_2d_reg]
                loss_list = self.reduce_all_loss(args, loss_list, loss, loss_3d_dict, loss_att_dict, input.size(0))

                # Time trainig iteration
                batch_time.update(time.time() - end)
                end = time.time()

                # Print info
                if (i + 1) % args.print_freq == 0 and args.proc_id == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(epoch+1, i+1, len(train_loader), 
                                            batch_time=batch_time, data_time=data_time, loss=loss_list[0]))

            # Adjust learning rate
            scheduler.step()

            # loss terms need to be all reduced, eval_stats need to be all gather
            # Do them all in validate
            loss_valid_list, eval_stats = self.validate(model, epoch, vis=False)

            # for Tensorboard
            if not args.no_tb and args.proc_id == 0:
                self.write_tensorboard(writer, loss_list, loss_valid_list, eval_stats, epoch+1)

            total_score = loss_list[0].avg
            if args.proc_id == 0:
                # File to keep latest epoch
                with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
                    f.write(str(epoch + 1))
                # Save model
                to_save = False
                if total_score < lowest_loss:
                    best_epoch = epoch + 1
                    lowest_loss = total_score
                if eval_stats[0] > best_val_f1:
                    to_save = True
                    best_f1_epoch = epoch + 1
                    best_val_f1 = eval_stats[0]
                # print validation result every epoch 
                print("===> Average {}-loss on training set is {:.8f}".format(self.crit_string, loss_list[0].avg))
                print("===> Average {}-loss on validation set is {:.8f}".format(self.crit_string, loss_valid_list[0].avg))
                print("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
                print("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
                print("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
                print("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
                print("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
                print("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
                print("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
                print("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))
                print("===> Last best {}-loss was {:.8f} in epoch {}".format(self.crit_string, lowest_loss, best_epoch))
                print("===> Last best F1 was {:.8f} in epoch {}".format(best_val_f1, best_f1_epoch))

                self.save_checkpoint({
                    'arch': args.mod,
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch + 1,
                    'loss': total_score,
                    'f1': eval_stats[0],
                    'best_epoch': best_epoch,
                    'lowest_loss': lowest_loss,
                    'best_f1_epoch': best_f1_epoch,
                    'best_val_f1': best_val_f1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, to_save, epoch+1, args.save_path)

            dist.barrier()
            torch.cuda.empty_cache()

        # at the end of training
        if not args.no_tb and args.proc_id == 0:
            writer.close()


    def validate(self, model, epoch=0, vis=False):
        args = self.args
        loader = self.valid_loader
        dataset = self.valid_dataset
        criterion = self.criterion
        if not args.no_cuda:
            device = torch.device("cuda", args.local_rank)
            criterion = criterion.to(device)
        bceloss = nn.BCEWithLogitsLoss()
        vs_saver = self.vs_saver
        val_gt_file = self.val_gt_file
        # valid_set_labels = self.valid_set_labels
        # Define container to keep track of metric and loss
        losses = AverageMeter()
        losses_3d_vis = AverageMeter()
        losses_3d_prob = AverageMeter()
        losses_3d_reg = AverageMeter()
        losses_2d_vis = AverageMeter()
        losses_2d_cls = AverageMeter()
        losses_2d_reg = AverageMeter()

        pred_lines_sub = []
        gt_lines_sub = []

        # Evaluate model
        model.eval()

        # Start validation loop
        with torch.no_grad():
            for i, (json_files, input, seg_maps, gt, gt_laneline_img, idx, gt_hcam, gt_pitch, gt_intrinsic, gt_extrinsic, seg_name, seg_bev_map) in tqdm(enumerate(loader)):
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    seg_maps = seg_maps.cuda(non_blocking=True)
                    gt_hcam = gt_hcam.cuda()
                    gt_pitch = gt_pitch.cuda()
                    gt_intrinsic = gt_intrinsic.cuda()
                    gt_extrinsic = gt_extrinsic.cuda()
                    gt_laneline_img = gt_laneline_img.cuda()
                    seg_bev_map = seg_bev_map.cuda()
                input = input.contiguous().float()

                M_inv = unit_update_projection_extrinsic(args, gt_extrinsic, gt_intrinsic)
                # Inference model
                laneatt_proposals_list, output_net, pred_hcam, pred_pitch, pred_seg_bev_map, uncertainty_loss = model(input=input, _M_inv=M_inv)

                ## compute FPS
                # iterations = 1000
                # torch.cuda.synchronize()
                # start = time.time()
                # for _ in range(iterations):
                #     laneatt_proposals_list, output_net, pred_hcam, pred_pitch, pred_seg_bev_map, uncertainty_loss = model(input=input, _M_inv=M_inv)
                # torch.cuda.synchronize()
                # end = time.time()
                # FPS = iterations / (end - start)
                # print("FPS: ", FPS)
                # break

                # 3D loss
                loss_3d, loss_3d_dict = criterion(output_net, gt, pred_hcam, gt_hcam, pred_pitch, gt_pitch)
                # Add laneatt loss
                loss_att, loss_att_dict = model.module.laneatt_head.loss(laneatt_proposals_list, gt_laneline_img,
                                                                            cls_loss_weight=args.cls_loss_weight,
                                                                            reg_vis_loss_weight=args.reg_vis_loss_weight)
                # segmentation loss
                loss_seg = bceloss(pred_seg_bev_map, seg_bev_map)
                # overall loss
                loss = self.compute_loss(args, epoch, loss_3d, loss_att, loss_seg, uncertainty_loss, loss_3d_dict, loss_att_dict)

                # reduce loss from all gpu, then update losses
                loss_list = [losses, losses_3d_vis, losses_3d_prob, losses_3d_reg, losses_2d_vis, losses_2d_cls, losses_2d_reg]
                loss_list = self.reduce_all_loss(args, loss_list, loss, loss_3d_dict, loss_att_dict, input.size(0))

                # Print info
                if (i + 1) % args.print_freq == 0 and args.proc_id == 0:
                        print('Test: [{0}/{1}]\t'
                                'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(i+1, len(loader), loss=loss_list[0]))

                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()
                gt_intrinsic = gt_intrinsic.data.cpu().numpy()
                gt_extrinsic = gt_extrinsic.data.cpu().numpy()
                output_net = output_net.data.cpu().numpy()
                gt = gt.data.cpu().numpy()
                gt_laneline_img = gt_laneline_img.data.cpu().numpy()

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(output_net[j], dataset)
                    unormalize_lane_anchor(gt[j], dataset)

                # Apply nms on network BEV output
                if not args.use_default_anchor:
                    output_net = nms_bev(output_net, args)

                # Visualization
                if (i + 1) % args.save_freq == 0 or args.evaluate:
                    gt_2d = []
                    for j in range(num_el):
                        gt_2d.append(dataset.label_to_lanes(gt_laneline_img[j]))
                    gt_decoded_2d = []
                    for p in gt_2d:
                        lanes = []
                        for l in p:
                            lanes.append(l.points)
                        gt_decoded_2d.append(lanes)
                    # Apply nms 
                    laneatt_proposals_list = model.module.laneatt_head.nms_new(laneatt_proposals_list,
                                                                                args.nms_thres,
                                                                                model.module.max_lanes,
                                                                                args.conf_th,
                                                                                args.vis_th)
                    prediction_2d = model.module.laneatt_head.decode(laneatt_proposals_list, args.vis_th, as_lanes=True)

                    pred_decoded_2d = []
                    pred_decoded_2d_cate = []
                    for p in prediction_2d:
                        lanes = []
                        cate = []
                        for l in p:
                            lanes.append(l.points)
                            cate.append(l.metadata['pred_cat'])
                        pred_decoded_2d.append(lanes)
                        pred_decoded_2d_cate.append(cate)

                    img_name_all = []
                    for j in range(num_el):
                        im_id = idx[j]
                        # json_line = copy.deepcopy(valid_set_labels[im_id])
                        json_file = json_files[j]
                        with open(json_file, 'r') as file:
                            file_lines = [line for line in file]
                            json_line = json.loads(file_lines[0])
                        img_path = json_line["file_path"]
                        img_name = os.path.basename(img_path)
                        img_name_all.append(img_name)

                    # For the purpose of vis positive anchors
                    if vis and (i + 1) % args.save_freq == 0:
                        anchors_positives = model.module.laneatt_head.anchors_to_lanes(loss_att_dict['anchors_positives'])
                        vs_saver.save_result_new(dataset, 'valid', epoch, idx,
                                                input, gt, output_net, pred_pitch, pred_hcam,
                                                evaluate=args.evaluate,
                                                laneatt_gt=gt_decoded_2d, laneatt_pred=pred_decoded_2d, laneatt_pos_anchor=anchors_positives,
                                                intrinsics=gt_intrinsic, extrinsics=gt_extrinsic, seg_name=seg_name, img_name=img_name_all)

                # Write results
                for j in range(num_el):
                    im_id = idx[j]
                    # saving json style
                    # json_line = valid_set_labels[im_id]
                    json_file = json_files[j]
                    with open(json_file, 'r') as file:
                        file_lines = [line for line in file]
                        json_line = json.loads(file_lines[0])

                    gt_lines_sub.append(copy.deepcopy(json_line))

                    lane_anchors = output_net[j]
                    # convert to json output format
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob = \
                        compute_3d_lanes_all_category(lane_anchors, dataset, args.anchor_y_steps, gt_extrinsic[j][2,3])
                    json_line["laneLines"] = lanelines_pred
                    json_line["laneLines_prob"] = lanelines_prob
                    pred_lines_sub.append(copy.deepcopy(json_line))

                    # save 2d/3d eval results
                    if args.evaluate:
                        img_path = json_line["file_path"]
                        self.save_eval_result(args, img_path, pred_decoded_2d[j], pred_decoded_2d_cate[j], lanelines_pred, lanelines_prob)

            if 'openlane' in args.dataset_name:
                eval_stats = self.evaluator.bench_one_submit_openlane_DDP(pred_lines_sub, gt_lines_sub, vis=False)
            else:
                eval_stats = self.evaluator.bench_one_submit(pred_lines_sub, gt_lines_sub, vis=False)

            gather_output = [None for _ in range(args.world_size)]
            # all_gather all eval_stats and calculate mean
            dist.all_gather_object(gather_output, eval_stats)
            dist.barrier()
            r_lane = np.sum([eval_stats_sub[8] for eval_stats_sub in gather_output])
            p_lane = np.sum([eval_stats_sub[9] for eval_stats_sub in gather_output])
            c_lane = np.sum([eval_stats_sub[10] for eval_stats_sub in gather_output])
            cnt_gt = np.sum([eval_stats_sub[11] for eval_stats_sub in gather_output])
            cnt_pred = np.sum([eval_stats_sub[12] for eval_stats_sub in gather_output])
            match_num = np.sum([eval_stats_sub[13] for eval_stats_sub in gather_output])
            Recall = r_lane / (cnt_gt + 1e-6)
            Precision = p_lane / (cnt_pred + 1e-6)
            f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
            category_accuracy = c_lane / (match_num + 1e-6)

            eval_stats[0] = f1_score
            eval_stats[1] = Recall
            eval_stats[2] = Precision
            eval_stats[3] = category_accuracy
            eval_stats[4] = np.sum([eval_stats_sub[4] for eval_stats_sub in gather_output]) / args.world_size
            eval_stats[5] = np.sum([eval_stats_sub[5] for eval_stats_sub in gather_output]) / args.world_size
            eval_stats[6] = np.sum([eval_stats_sub[6] for eval_stats_sub in gather_output]) / args.world_size
            eval_stats[7] = np.sum([eval_stats_sub[7] for eval_stats_sub in gather_output]) / args.world_size

            return loss_list, eval_stats

    def eval(self):
        args = self.args
        model = PersFormer(args)
        if args.sync_bn:
            if args.proc_id == 0:
                print("Convert model with Sync BatchNorm")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if not args.no_cuda:
            device = torch.device("cuda", args.local_rank)
            model = model.to(device)
        best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
        if os.path.isfile(best_file_name):
            if args.proc_id == 0:
                sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
                print("=> loading checkpoint '{}'".format(best_file_name))
                checkpoint = torch.load(best_file_name)
                model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(best_file_name))
        dist.barrier()
        # DDP setting
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        loss_list, eval_stats = self.validate(model, vis=True)
        if args.proc_id == 0:
            print("===> Average {}-loss on validation set is {:.8f}".format(self.crit_string, loss_list[0].avg))
            print("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
            print("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
            print("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
            print("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
            print("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
            print("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
            print("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
            print("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))

    def _get_train_dataset(self):
        args = self.args
        if 'openlane' in args.dataset_name:
            train_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'training/', args, data_aug=True, save_std=True, seg_bev=True)
        else:
            train_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'train.json'), args, data_aug=True, save_std=True)
        
        # train_dataset.normalize_lane_label()
        train_loader, train_sampler = get_loader(train_dataset, args)

        return train_dataset, train_loader, train_sampler


    def _get_valid_dataset(self):
        args = self.args
        if 'openlane' in args.dataset_name:
            if not args.evaluate_case:
                valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'validation/', args, seg_bev=True)
            else:
                valid_dataset = LaneDataset(args.dataset_dir, args.data_dir, args, seg_bev=True)
        else:
            valid_dataset = LaneDataset(args.dataset_dir, self.val_gt_file, args)

        # assign std of valid dataset to be consistent with train dataset
        valid_dataset.set_x_off_std(self.train_dataset._x_off_std)
        if not args.no_3d:
            valid_dataset.set_z_std(self.train_dataset._z_std)
        # valid_dataset.normalize_lane_label()
        valid_loader, valid_sampler = get_loader(valid_dataset, args)

        return valid_dataset, valid_loader, valid_sampler

    def _get_model_ddp(self):
        args = self.args
        # Define network
        model = PersFormer(args)

        if args.sync_bn:
            if args.proc_id == 0:
                print("Convert model with Sync BatchNorm")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.no_cuda:
            # Load model on gpu before passing params to optimizer
            device = torch.device("cuda", args.local_rank)
            model = model.to(device)

        """
            first load param to model, then model = DDP(model)
        """
        # Logging setup
        best_epoch = 0
        lowest_loss = np.inf
        best_f1_epoch = 0
        best_val_f1 = -1e-5
        optim_saved_state = None
        schedule_saved_state = None

        # resume model
        args.resume = first_run(args.save_path)
        if args.resume:
            model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, \
                optim_saved_state, schedule_saved_state = self.resume_model(args, model)
        elif args.pretrained and args.proc_id == 0:
            path = 'models/pretrain/model_pretrain.pth.tar'
            if os.path.isfile(path):
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['state_dict'])
                print("Use pretrained model in {} to start training".format(path))
            else:
                raise Exception("No pretrained model found in {}".format(path))

        dist.barrier()
        # DDP setting
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        # Define optimizer and scheduler
        '''
            Define optimizer after DDP init
        '''
        optimizer = define_optim(args.optimizer, model.parameters(),
                                args.learning_rate, args.weight_decay)
        scheduler = define_scheduler(optimizer, args)

        # resume optimizer and scheduler
        if optim_saved_state is not None:
            print("proc_id-{} load optim state".format(args.proc_id))
            optimizer.load_state_dict(optim_saved_state)
        if schedule_saved_state is not None:
            print("proc_id-{} load scheduler state".format(args.proc_id))
            scheduler.load_state_dict(schedule_saved_state)

        return model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1

    def resume_model(self, args, model):
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
        if os.path.isfile(path):
            print("=> loading checkpoint from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            if args.proc_id == 0:
                log_file_name = 'log_train_start_{}.txt'.format(args.resume)
                # Redirect stdout
                sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
                model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            lowest_loss = checkpoint['lowest_loss']
            best_f1_epoch = checkpoint['best_f1_epoch']
            best_val_f1 = checkpoint['best_val_f1']
            optim_saved_state = checkpoint['optimizer']
            schedule_saved_state = checkpoint['scheduler']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if args.proc_id == 0:
                log_file_name = 'log_train_start_0.txt'
                # Redirect stdout
                sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
                print("=> Warning: no checkpoint found at '{}'".format(path))
            best_epoch = 0
            lowest_loss = np.inf
            best_f1_epoch = 0
            best_val_f1 = -1e-5
            optim_saved_state = None
            schedule_saved_state = None
        return model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, optim_saved_state, schedule_saved_state

    def save_checkpoint(self, state, to_copy, epoch, save_path):
        filepath = os.path.join(save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
        torch.save(state, filepath)
        if to_copy:
            if epoch > 1:
                lst = glob.glob(os.path.join(save_path, 'model_best*'))
                if len(lst) != 0:
                    os.remove(lst[0])
            shutil.copyfile(filepath, os.path.join(save_path, 'model_best_epoch_{}.pth.tar'.format(epoch)))
            print("Best model copied")
        # if epoch > 1:
        #     prev_checkpoint_filename = os.path.join(save_path, 
        #             'checkpoint_model_epoch_{}.pth.tar'.format(epoch-1))
        #     if os.path.exists(prev_checkpoint_filename):
        #         os.remove(prev_checkpoint_filename)

    def compute_loss(self, args, epoch, loss_3d, loss_att, loss_seg, uncertainty_loss, loss_3d_dict, loss_att_dict):
        if args.learnable_weight_on:
            loss = 0
            _3d_vis_loss_factor = 1 / torch.exp(uncertainty_loss[0])
            loss += _3d_vis_loss_factor * loss_3d_dict['vis_loss']
            _3d_prob_loss_factor = 1 / torch.exp(uncertainty_loss[1])
            loss += _3d_prob_loss_factor * loss_3d_dict['prob_loss']
            _3d_reg_loss_factor = 1 / torch.exp(uncertainty_loss[2])
            loss += _3d_reg_loss_factor * loss_3d_dict['reg_loss']

            open_2d = 1.0
            _2d_vis_loss_factor = 1 / torch.exp(uncertainty_loss[3])
            loss += open_2d * _2d_vis_loss_factor * loss_att_dict['vis_loss']
            _2d_prob_loss_factor = 1 / torch.exp(uncertainty_loss[4])
            loss += open_2d * _2d_prob_loss_factor * loss_att_dict['cls_loss']
            _2d_reg_loss_factor = 1 / torch.exp(uncertainty_loss[5])
            loss += open_2d * _2d_reg_loss_factor * loss_att_dict['reg_loss']

            loss += uncertainty_loss[0:6].sum() * 0.5
            if epoch > args.seg_start_epoch:
                _seg_loss_factor = 1 / torch.exp(uncertainty_loss[6])
                loss += _seg_loss_factor * loss_seg
                loss += uncertainty_loss[6].sum() * 0.5
            else:
                loss += 0.0 * loss_seg
                loss += uncertainty_loss[6].sum() * 0.0
        else:
            # epoch depended loss
            if epoch > args.seg_start_epoch:
                loss = loss_3d + args.loss_att_weight * loss_att + args.loss_seg_weight * loss_seg + 0.0 * uncertainty_loss[0:6].sum()
            else:
                loss = loss_3d + args.loss_att_weight * loss_att + 0.0 * loss_seg + 0.0 * uncertainty_loss[0:6].sum()
        return loss

    def reduce_all_loss(self, args, loss_list, loss, loss_3d_dict, loss_att_dict, num):
        reduced_loss = loss.data
        reduced_loss_all = reduce_tensors(reduced_loss, world_size=args.world_size)
        losses = loss_list[0]
        losses.update(to_python_float(reduced_loss_all), num)

        reduced_vis_loss = loss_3d_dict['vis_loss'].data
        reduced_vis_loss = reduce_tensors(reduced_vis_loss, world_size=args.world_size)
        losses_3d_vis = loss_list[1]
        losses_3d_vis.update(to_python_float(reduced_vis_loss), num)

        reduced_prob_loss = loss_3d_dict['prob_loss'].data
        reduced_prob_loss = reduce_tensors(reduced_prob_loss, world_size=args.world_size)
        losses_3d_prob = loss_list[2]
        losses_3d_prob.update(to_python_float(reduced_prob_loss), num)

        reduced_reg_loss = loss_3d_dict['reg_loss'].data
        reduced_reg_loss = reduce_tensors(reduced_reg_loss, world_size=args.world_size)
        losses_3d_reg = loss_list[3]
        losses_3d_reg.update(to_python_float(reduced_reg_loss), num)

        reduce_2d_vis_loss = loss_att_dict['vis_loss'].data
        reduce_2d_vis_loss = reduce_tensors(reduce_2d_vis_loss, world_size=args.world_size)
        losses_2d_vis = loss_list[4]
        losses_2d_vis.update(to_python_float(reduce_2d_vis_loss), num)

        reduced_2d_cls_loss = loss_att_dict['cls_loss'].data
        reduced_2d_cls_loss = reduce_tensors(reduced_2d_cls_loss, world_size=args.world_size)
        losses_2d_cls = loss_list[5]
        losses_2d_cls.update(to_python_float(reduced_2d_cls_loss), num)

        reduced_2d_reg_loss = loss_att_dict['reg_loss'].data
        reduced_2d_reg_loss = reduce_tensors(reduced_2d_reg_loss, world_size=args.world_size)
        losses_2d_reg = loss_list[6]
        losses_2d_reg.update(to_python_float(reduced_2d_reg_loss), num)

        return loss_list

    def write_tensorboard(self, writer, loss_list, loss_valid_list, eval_stats, epoch):
        writer.add_scalars('3D-Lane-Loss', {'Training': loss_list[0].avg}, epoch)
        writer.add_scalars('3D-Lane-Loss', {'Validation': loss_valid_list[0].avg}, epoch)

        writer.add_scalars('3D-Lane-Separate-Loss', {'Training vis': loss_list[1].avg}, epoch)
        writer.add_scalars('3D-Lane-Separate-Loss', {'Training prob': loss_list[2].avg}, epoch)
        writer.add_scalars('3D-Lane-Separate-Loss', {'Training reg': loss_list[3].avg}, epoch)
        writer.add_scalars('3D-Lane-Separate-Loss', {'Validation vis': loss_valid_list[1].avg}, epoch)
        writer.add_scalars('3D-Lane-Separate-Loss', {'Validation prob': loss_valid_list[2].avg}, epoch)
        writer.add_scalars('3D-Lane-Separate-Loss', {'Validation reg': loss_valid_list[3].avg}, epoch)

        writer.add_scalars('2D-Lane-Separate-Loss', {'Training vis': loss_list[4].avg}, epoch)
        writer.add_scalars('2D-Lane-Separate-Loss', {'Training cls': loss_list[5].avg}, epoch)
        writer.add_scalars('2D-Lane-Separate-Loss', {'Training reg': loss_list[6].avg}, epoch)
        writer.add_scalars('2D-Lane-Separate-Loss', {'Validation vis': loss_valid_list[4].avg}, epoch)
        writer.add_scalars('2D-Lane-Separate-Loss', {'Validation cls': loss_valid_list[5].avg}, epoch)
        writer.add_scalars('2D-Lane-Separate-Loss', {'Validation reg': loss_valid_list[6].avg}, epoch)

        writer.add_scalars('Evaluation', {'laneline F-measure': eval_stats[0]}, epoch)

    def save_eval_result(self, args, img_path, pred_decoded_2d, pred_decoded_2d_cate, lanelines_pred, lanelines_prob):
        # 2d eval result
        result = {}
        result_dir = os.path.join(args.save_path, 'result_2d/')
        mkdir_if_missing(result_dir)
        mkdir_if_missing(os.path.join(result_dir, 'validation/'))
        mkdir_if_missing(os.path.join(result_dir, 'training/'))
        file_path_splited = img_path.split('/')
        mkdir_if_missing(os.path.join(result_dir, 'validation/'+file_path_splited[1]))  # segment
        result_file_path = ops.join(result_dir, 'validation/'+file_path_splited[1]+'/'+file_path_splited[-1][:-4]+'.json')

        # write img path
        result['file_path'] = img_path
        # write lane result
        lane_lines = []
        # "uv":              <float> [2, n] -- u,v coordinates of sample points in pixel coordinate
        # "category":        <int> -- lane shape category, 1 - num_category
        for k in range(len(pred_decoded_2d)):
            uv_orgsize = pred_decoded_2d[k] * np.array([args.org_w, args.org_h]) # to original uv coord
            lane_lines.append({'category': pred_decoded_2d_cate[k],
                               'uv': uv_orgsize.T.tolist()})
        result['lane_lines'] = lane_lines
        
        with open(result_file_path, 'w') as result_file:
            json.dump(result, result_file)

        # 3d eval result
        result = {}
        result_dir = os.path.join(args.save_path, 'result_3d/')
        mkdir_if_missing(result_dir)
        mkdir_if_missing(os.path.join(result_dir, 'validation/'))
        mkdir_if_missing(os.path.join(result_dir, 'training/'))
        file_path_splited = img_path.split('/')
        mkdir_if_missing(os.path.join(result_dir, 'validation/'+file_path_splited[1]))  # segment
        result_file_path = ops.join(result_dir, 'validation/'+file_path_splited[1]+'/'+file_path_splited[-1][:-4]+'.json')

        # write result
        result['file_path'] = img_path
        # write lane result
        lane_lines = []
        # "xyz":             <float> [3, n] -- x,y,z coordinates of sample points in camera coordinate
        # "category":        <int> -- lane shape category, 1 - num_category
        for k in range(len(lanelines_pred)):
            if np.max(lanelines_prob[k]) < 0.5:
                continue
            lane_lines.append({'xyz': lanelines_pred[k],
                               'category': int(np.argmax(lanelines_prob[k]))})
        result['lane_lines'] = lane_lines

        with open(result_file_path, 'w') as result_file:
            json.dump(result, result_file)