import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from .kitti_dataset import KittiDataset
from ..augmentor import database_sampler
from .kitti_object_eval_python import kitti_common

class KittiObjectDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, sample=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.dataset_cfg = dataset_cfg
        self.img_aug_type = dataset_cfg.get('IMG_AUG_TYPE', None)
        
        db_sampler = database_sampler.DataBaseSampler(
                root_path=self.root_path,
                sampler_cfg=self.dataset_cfg,
                class_names=self.class_names,
                logger=self.logger
            )
        db_infos = []
        for class_name, infos in db_sampler.db_infos.items():
            db_infos.extend(infos)
        self.db_infos = db_infos

    def collect_image_crops_kitti(self, info, obj_points, sampled_gt_box2d):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        # copy crops from images
        img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        raw_image = io.imread(img_path)
        raw_image = raw_image.astype(np.float32)
        raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_box2d.astype(np.int)
        new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int)
        raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
            new_center = new_box.reshape(2,2).mean(0)
            new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
            new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int)

        img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255

        return new_box, img_crop2d, obj_points, points_2d

    def __len__(self):
        return len(self.db_infos)

    def __getitem__(self, index):

        info = copy.deepcopy(self.db_infos[index])
        file_path = self.root_path / info['path']
        obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
            [-1, self.dataset_cfg.NUM_POINT_FEATURES])

        obj_points[:, :3] += info['box3d_lidar'][:3]
        input_dict = {}
        if self.img_aug_type is not None:
            new_box, img_crop2d, obj_points, points_2d = self.collect_image_crops_kitti(
                info, obj_points, info['bbox']
            )
            input_dict['images'] = img_crop2d
            input_dict['points_2d'] = points_2d - new_box[:2]
#            input_dict['gt_boxes_2d'] = np.expand_dims(new_box, axis=0)

        input_dict['points'] = obj_points
        input_dict['gt_boxes'] = np.expand_dims(info['box3d_lidar'], axis=0)
        input_dict['gt_names'] = np.expand_dims(info['name'], axis=0)
        
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict