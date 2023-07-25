import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils, loss_utils, box_utils
from .roi_head_template_pg import RoIHeadTemplatePG
from ..model_utils.attention_utils import TransformerEncoder, TransformerDecoder, get_positional_encoder
#from ..model_utils.attention_utils2 import TransformerEncoder, TransformerDecoder, get_positional_encoder
import numpy as np


class PGRCNNHead(RoIHeadTemplatePG):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.attention_cfg = model_cfg.TRANSFORMER
        self.point_cfg = model_cfg.POINT_FEATURE_CONFIG
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        
        # RoI Grid Pooling
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            c_out += sum([x[-1] for x in mlps])
            
            self.roi_grid_pool_layers.append(pool_layer)

        # Transformer Encoder
        assert self.attention_cfg.ENCODER.NUM_FEATURES == c_out, f'ATTENTION.ENCODER.NUM_FEATURES must equal voxel aggregation output dimension of {c_out}.'
        self.pos_encoder = get_positional_encoder(self.attention_cfg)
        self.attention_head = TransformerEncoder(self.attention_cfg.ENCODER)
       
        # Point Generation
        gen_fc_list = []
        for k in range(0, self.model_cfg.GEN_FC.__len__()):
            gen_fc_list.extend([
                nn.Linear(c_out, self.model_cfg.GEN_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.GEN_FC[k]),
                nn.ReLU()
            ])
            c_out = self.model_cfg.GEN_FC[k]
            if k != self.model_cfg.GEN_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                gen_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        gen_fc_list.append(nn.Linear(c_out, (3 + self.point_cfg.POINT_FEATURE_NUM) * self.point_cfg.NUM_POINTS, bias=True))
        self.gen_fc_layers= nn.Sequential(*gen_fc_list)
    
        self.point_pred_layer = nn.Linear(self.point_cfg.POINT_FEATURE_NUM, self.num_class, bias=True)
        
        # Generated Points Feature Extraction
        self.num_prefix_channels = 3        # x, y, z
        if self.point_cfg.USE_DEPTH:
            self.num_prefix_channels += 1   # d
        if self.point_cfg.USE_SCORE:
            self.num_prefix_channels += 1   # s
            
        xyz_mlps = [self.num_prefix_channels] + self.point_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        if self.point_cfg.get('MERGE_DOWN', True):
            c_out = self.point_cfg.XYZ_UP_LAYER[-1]
            self.merge_down_layer = nn.Sequential(
                nn.Conv2d(c_out + self.point_cfg.POINT_FEATURE_NUM, c_out, kernel_size=1, bias=not use_bn),
                *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
            )
        else:
            c_out = self.point_cfg.XYZ_UP_LAYER[-1] + self.point_cfg.POINT_FEATURE_NUM

        for k in range(self.point_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [c_out] + self.point_cfg.SA_CONFIG.MLPS[k]

            npoint = self.point_cfg.SA_CONFIG.NPOINTS[k] if self.point_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.point_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.point_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            c_out = mlps[-1]

        if self.model_cfg.BOX_EMBEDDING:
            self.box_embedding = nn.Sequential(
                nn.Linear(self.box_coder.code_size, c_out // 2, 1),
                nn.BatchNorm1d(c_out // 2),
                nn.ReLU(c_out // 2),
                nn.Linear(c_out // 2, c_out, 1),
            )
            
        # Confidence Head
        pre_channel = c_out
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        # Regression Head
        pre_channel = c_out
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)
        
        self.init_weights()
        
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        fc_list = [self.gen_fc_layers, self.cls_fc_layers, self.reg_fc_layers]
        if self.model_cfg.BOX_EMBEDDING:
            fc_list.append(self.box_embedding)
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
                    
        nn.init.normal_(self.point_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.point_pred_layer.bias, 0)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
                
        for p in self.attention_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, roi_grid_xyz_rel = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div((roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]), self.voxel_size[0])
        roi_grid_coords_y = torch.div((roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]), self.voxel_size[1])
        roi_grid_coords_z = torch.div((roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]), self.voxel_size[2])
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = torch.div(roi_grid_coords,  cur_stride, rounding_mode='trunc')
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features, roi_grid_xyz_rel


    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points


    def get_positional_input(self, rois, roi_labels, local_roi_grid_points):
        if self.attention_cfg.POSITIONAL_ENCODER == 'grid_points':
            positional_input = local_roi_grid_points
        elif self.attention_cfg.POSITIONAL_ENCODER == 'grid_points_corners':
            local_rois = rois.view(-1, rois.shape[-1]).clone()
            local_rois[:, 0:3] = 0
            local_corners = box_utils.boxes_to_corners_3d(local_rois)
            positional_input_corners = (local_corners.unsqueeze(1) - local_roi_grid_points.unsqueeze(2)).reshape(*local_roi_grid_points.shape[:-1], -1)
            positional_input = torch.cat([positional_input_corners, local_roi_grid_points], dim=-1)
        else:
            positional_input = None
        rois = rois.view(-1, rois.shape[-1])
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        bs, box_num = roi_labels.shape
        bidx = torch.arange(bs).repeat_interleave(box_num).to(rois.device)
        bidx = bidx.reshape(-1, 1, 1).expand(-1, positional_input.shape[1], 1)
        global_roi_grid_points = torch.cat([bidx, global_roi_grid_points], dim=-1)  # [bidx, x, y, z]
        return positional_input, global_roi_grid_points
   
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        if 'rois' not in batch_dict:
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features, roi_grid_xyz_rel = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        
        # Transformer Encoder
        positional_input, global_roi_grid_points = self.get_positional_input(batch_dict['rois'], batch_dict['roi_labels'], roi_grid_xyz_rel)
        positional_embedding = self.pos_encoder(positional_input)

        bs, _, C = pooled_features.shape
        pooled_features = pooled_features.reshape(bs, -1, C)
        positional_embedding = positional_embedding.reshape(bs, -1, positional_embedding.shape[-1])

        attention_output = self.attention_head(pooled_features, positional_embedding)
        
        # Point Generation
        gen_output = self.gen_fc_layers(attention_output.view(-1, attention_output.shape[-1]))

        gen_output = gen_output.reshape(bs, pooled_features.shape[1] * self.point_cfg.NUM_POINTS, -1)   # (BxN, 6x6x6xP, 3 + C')
        gen_points_offset = gen_output[..., :3]        # (BxN, 6x6x6xP, 3)
        if self.point_cfg.get('CANONICAL_OFFSET', False):
            gen_points_offset = common_utils.rotate_points_along_z(
                gen_points_offset, batch_dict['rois'].view(-1, batch_dict['rois'].shape[-1])[:, 6]
            )
        gen_points_features = gen_output[..., 3:]    # (BxN, 6x6x6xP, C')
        
        gen_points_score = self.point_pred_layer(gen_points_features)  # (BxN, 6x6x6xP, 1)

        # global_rebuilt_points
        gen_points_xyz = global_roi_grid_points.repeat(1, self.point_cfg.NUM_POINTS, 1).clone()
#        rebuilt_points_new[..., 1:4] = batch_dict['rois'].reshape(-1, 7)[:, :3].unsqueeze(1)
        gen_points_xyz[..., 1:4] = gen_points_xyz[..., 1:4] + gen_points_offset # (BxN, 6x6x6xP, bxyz)
   
        # canonical transform
        xyz_local = gen_points_xyz[..., 1:4] - batch_dict['rois'].reshape(bs, 1, -1)[..., :3]
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local, -batch_dict['rois'].view(-1, batch_dict['rois'].shape[-1])[:, 6]
        )
        
        rebuilt_points = torch.cat([gen_points_xyz, gen_points_score], dim=-1)  # (BxN, 6x6x6xP, bxyzs)
        batch_dict['rebuilt_points'] = rebuilt_points.reshape(-1, rebuilt_points.shape[-1])
        
        # generated points feature extraction
        xyz_input = [xyz_local]
        if self.point_cfg.USE_DEPTH:
            point_depths = gen_points_xyz[..., 1:4].norm(dim=-1) / self.point_cfg.DEPTH_NORMALIZER - 0.5
            xyz_input.append(point_depths.unsqueeze(-1))
        if self.point_cfg.USE_SCORE:
            xyz_input.append(torch.sigmoid(gen_points_score))
            
        xyz_input = torch.cat(xyz_input, dim=-1).transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input.clone().detach())

        if self.point_cfg.SCORE_WEIGHTING:
            xyz_features = xyz_features * torch.sigmoid(gen_points_score).unsqueeze(1)
        
        merged_features = torch.cat((xyz_features, gen_points_features.transpose(1, 2).unsqueeze(dim=3)), dim=1)
        if self.point_cfg.get('MERGE_DOWN', True):
            merged_features = self.merge_down_layer(merged_features)
        threshold = self.point_cfg.get('POINT_THRESHOLD', 0)
        xyz_local[torch.sigmoid(gen_points_score.squeeze(-1)) < threshold] = 0
        l_xyz, l_features = [xyz_local.contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        shared_features = l_features[-1].squeeze(dim=-1)
        if self.model_cfg.BOX_EMBEDDING:
            shared_features += self.box_embedding(batch_dict['rois'].view(bs, -1))
        
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))
        
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
                        
        else:
#            gt_points_ratio, gt_points_num = self.get_points_ratio(batch_dict['points'], targets_dict['gt_of_rois_src'])
#            targets_dict['ratio_of_points'] = gt_points_ratio
#            targets_dict['num_of_points'] = gt_points_num
            
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['rebuilt_points'] = rebuilt_points
            
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
            
            targets_dict['target_points'] = batch_dict['bm_points']
            targets_dict['gt_boxes'] = batch_dict['gt_boxes']
            self.forward_ret_dict = targets_dict

        return batch_dict