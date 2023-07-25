from functools import partial

import torch
from .roi_head_template import RoIHeadTemplate
from pytorch3d.loss import chamfer_distance as chamfer_dist
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as point_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils

class RoIHeadTemplatePG(RoIHeadTemplate):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.fps_num = self.model_cfg.LOSS_CONFIG.SAMPLE_POINTS
        self.min_cd_num = self.model_cfg.LOSS_CONFIG.get("MIN_CD_POINTS", 0)

    def get_point_cls_loss(self, forward_ret_dict, set_ignore_flag=True):
        rebuilt_points = forward_ret_dict['rebuilt_points']
        gt_boxes = forward_ret_dict['gt_boxes']
        target_gt_idx = forward_ret_dict['gt_idx_of_rois']
        batch_size, roi_size = target_gt_idx.shape
                
        # calculate point cls loss for rebuilt points
        rebuilt_points = rebuilt_points.reshape(batch_size, -1, rebuilt_points.shape[-1])
        sample_pt_idxs = point_utils.farthest_point_sample(
            rebuilt_points[..., 1:4].contiguous(), self.fps_num
        ).long()
        sampled_points = sample_pt_idxs.new_zeros((batch_size, self.fps_num, 5), dtype=torch.float)
        for bidx in range(batch_size):
            sampled_points[bidx] = rebuilt_points[bidx][sample_pt_idxs[bidx]]
        
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            sampled_points[..., 1:4], gt_boxes[..., :-1]
        )
        
        box_fg_flag = (box_idxs_of_pts >= 0)
        point_cls_labels = box_fg_flag.new_zeros(box_fg_flag.shape)
        if set_ignore_flag:
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
            
            extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                sampled_points[..., 1:4], extend_gt_boxes[..., :-1]
            )
            fg_flag = box_fg_flag
            ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
            point_cls_labels[ignore_flag] = -1
        for bidx in range(batch_size):
            gt_box_of_fg_points = torch.index_select(gt_boxes[bidx], 0, box_idxs_of_pts[bidx][fg_flag[bidx]])    
            point_cls_labels[bidx][fg_flag[bidx]] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
        point_cls_labels = point_cls_labels.view(-1)
        point_cls_preds = sampled_points[..., -1].view(-1, self.num_class)
        
        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss = point_loss_cls * loss_weights_dict['point_cls_weight']
        tb_dict = {
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item() / batch_size,
        }
        return point_loss, tb_dict

    def get_chamfer_distance(self, forward_ret_dict):
        rebuilt_points = forward_ret_dict['rebuilt_points']
        target_points = forward_ret_dict['target_points']
        
        fg_mask = forward_ret_dict['reg_valid_mask']
        gt_boxes = forward_ret_dict['gt_boxes']
        target_gt_idx = forward_ret_dict['gt_idx_of_rois']
        batch_size, roi_size = target_gt_idx.shape
        tb_dict = {}
        
        cd_loss = 0
        rebuilt_points = rebuilt_points.reshape(batch_size, -1, *rebuilt_points.shape[-2:])
        
        # filter non-foreground
        if fg_mask.sum() == 0:
            tb_dict['cd_loss'] = cd_loss
            return cd_loss, tb_dict
        # collate target points
        tp_list = []
        for bidx in range(batch_size):
            tp_list.append(target_points[target_points[:, 0] == bidx])
        tp_batch = target_points.new_zeros(batch_size, max([len(x) for x in tp_list]), target_points.shape[-1])
        for bidx, cur_tp in enumerate(tp_list):
            tp_batch[bidx, :len(cur_tp)] = cur_tp
        target_points_in_roi = roiaware_pool3d_utils.points_in_boxes_gpu(tp_batch[..., 1:4], gt_boxes[...,:-1])
        
        tpr_list = []   # target points in roi
        roi_target_num = [] 
        for (bidx, ridx) in fg_mask.nonzero():
            current_target_gt_idx = target_gt_idx[bidx, ridx]
            tpr = tp_batch[bidx, target_points_in_roi[bidx, :] == current_target_gt_idx]
            if len(tpr) >= self.min_cd_num:
                tpr_list.append(tpr)
            else:
                fg_mask[bidx, ridx] = 0
        if len(tpr_list) == 0:
            tb_dict['cd_loss'] = None
            return None, tb_dict
        roi_target = target_points.new_zeros(len(tpr_list), max([len(x) for x in tpr_list]), target_points.shape[-1])
        for ridx, cur_tpr in enumerate(tpr_list):
            roi_target[ridx, :len(cur_tpr)] = cur_tpr
            roi_target_num.append(len(cur_tpr))
        roi_source = rebuilt_points[fg_mask > 0]
        cd_loss, _ = chamfer_dist(roi_target[..., 1:4], roi_source[..., 1:4], x_lengths=torch.tensor(roi_target_num).to(roi_target.device), point_reduction='mean', batch_reduction='sum')
#        try:
#            dist_t2s, dist_s2t = chamfer_dist(roi_target[..., 1:4], roi_source[..., 1:4], x_lengths=torch.tensor(roi_target_num).to(roi_target.device))
#        except:
#            cd = chamfer_dist()
#            dist_t2s, dist_s2t, _, _ = cd(roi_target[..., 1:4], roi_source[..., 1:4])
#        cd_loss += torch.mean(dist_t2s, dim=-1).sum() + torch.mean(dist_s2t, dim=-1).sum()
            
        if fg_mask.sum() > 0:
            cd_loss /= fg_mask.sum()
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            cd_loss = cd_loss * loss_weights_dict['chamfer_dist_weight']
            tb_dict['cd_loss'] = cd_loss.item()

            if self.model_cfg.LOSS_CONFIG.get('POINT_LOSS_REGULARIZATION', False):
                rois_fg = forward_ret_dict['rois'][fg_mask > 0].unsqueeze(1)
                offset = roi_source[..., 1:4] - rois_fg[..., :3]
                if self.model_cfg.LOSS_CONFIG.get('NORMALIZE_POINT_LOSS_REGULARIZATION', False):
                    offset = offset / rois_fg[..., 3:6]
                mean_dist = -0.5 * offset.norm(dim=-1).mean()
                tb_dict['pg_regularization'] = mean_dist
                cd_loss += mean_dist
                
        return cd_loss, tb_dict
        
    def get_point_generation_loss(self, forward_ret_dict):
        
        tb_dict = {}
        point_gen_loss = 0
        cd_loss, cd_tb_dict = self.get_chamfer_distance(forward_ret_dict)
        point_loss, pc_tb_dict = self.get_point_cls_loss(forward_ret_dict)        
        tb_dict.update(cd_tb_dict)
        tb_dict.update(pc_tb_dict)
        pg_loss = cd_loss + point_loss
        tb_dict['pg_loss'] = pg_loss.item()
        return pg_loss, tb_dict
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        
        rcnn_loss_pg, pg_tb_dict =  self.get_point_generation_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_pg
        tb_dict.update(pg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict