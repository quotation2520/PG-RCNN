import argparse
import glob
from pathlib import Path
from skimage import io
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils


class DemoDataset(KittiDataset):
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
        self.ext = ext
        if sample == None:
            data_file_list = self.sample_id_list
        else:
            data_file_list = [self.sample_id_list[sample]]
            self.kitti_infos = [self.kitti_infos[sample]]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']

        calib = self.get_calib(self.sample_file_list[index])
        points = self.get_lidar(self.sample_file_list[index])
        image = self.get_image(self.sample_file_list[index])
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]        

        points_2d, _ = calib.lidar_to_img(points[:, 0:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=img_shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=img_shape[0]-1)
        points_2d = points_2d.astype(np.int)
        
        

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'images' : image,
#            'gt_bbox2d': gt_bbox2d,
#            'points_2d' : points_2d,
            'calib': calib,

        }

        road_plane = self.get_road_plane(sample_idx)
        if road_plane is not None:
            input_dict['road_plane'] = road_plane
            
        # GT
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
            })
#            input_dict['gt_boxes2d'] = annos["bbox"]

        
        data_dict = self.prepare_data(data_dict=input_dict)

        # To see GT RoI
        boxes = data_dict['gt_boxes'][..., :-1].copy()
        offset = np.random.uniform(-0.25, 0.25, size=(boxes.shape[0], 3))
        boxes[..., 0:3] += offset
        noise_scale = np.random.uniform(0.95, 1.05, size=(boxes.shape[0], 3))
        boxes[..., 3:6] *= noise_scale
        noise_rot = np.random.uniform(-0.78539816, 0.78539816, size=(boxes.shape[0]))
        boxes[..., 6] += noise_rot
        data_dict['rois'] = boxes
        data_dict['roi_labels'] = data_dict['gt_boxes'][:, -1]
        data_dict['has_class_labels'] = True 
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--root_path', type=str, default='../data/kitti/',
                        help='KITTI root directory')
    parser.add_argument('--split', type=str, default='training',
                        help='specify dataset split')
    parser.add_argument('--sample', type=int, default=None,
                        help='specify a sample index')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def draw_2d_view(image, points=None, keypoints=None, gt_boxes=None, ref_boxes=None):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    plt.imshow(image)
    plt.scatter(x=points[:, 0], y=points[:, 1], c='w', s=0.5)
#    plt.scatter(x=keypoints[:, 0], y=keypoints[:, 1], c='r', s=5)
    ax = plt.gca()
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', fill=False)
        ax.add_patch(rect)
    for box in ref_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='g', fill=False)
        ax.add_patch(rect)
    plt.savefig('showme2d.png', dpi=1000)
    plt.show()

def main():
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=True,
        root_path=Path(args.root_path), sample=args.sample, ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict, debug=True)

#            pred_box3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(data_dict['gt_boxes'][0,:,:-1].cpu(), data_dict['calib'][0])
#            pred_box3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_dicts[0]['rois'][0].cpu(), data_dict['calib'][0])
#            image_shape = list(data_dict['images'][0, 0].shape)
#            proj_box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(pred_box3d_camera,  data_dict['calib'][0], image_shape)
            
#            print(len(pred_dicts[0]['keypoints_2d']))
            if idx > 10:
                break
            logger.info('2d_view...')
#            draw_2d_view(image=data_dict['images'][0], points=data_dict['points_2d'][0], # keypoints=pred_dicts[0]['keypoints_2d'],
#                gt_boxes=data_dict['gt_boxes2d'][0], ref_boxes=proj_box2d#pred_dicts[0]['pred_boxes_2d']
#            )

#            logger.info('3d_view...')
#            V.draw_scenes(
#                points=data_dict['points'][:, 1:], #keypoints=pred_dicts[0]['target_points'][:, 1:], 
##                ref_boxes=pred_dicts[0]['rois'][0], #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
#                gt_boxes=data_dict['gt_boxes'][0, :, :-1]
#            )
            
            logger.info('3d_view...')
            V.draw_scenes(
                points=data_dict['points'][:, 1:], keypoints=pred_dicts[0]['rebuilt_points'][:, 1:], 
                ref_boxes=pred_dicts[0]['rois'][0], #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                gt_boxes=data_dict['gt_boxes'][0, :, :-1]
            )
            
            V.draw_scenes(
                keypoints=data_dict['bm_points'][:, 1:], #point_colors=np.ones((data_dict['bm_points'].shape[0], 3)) * [0, 1, 0],
    #            ref_boxes=pred_dicts[0]['rois'][0], #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                ref_boxes=data_dict['gt_boxes'][0, :, :-1]
            )
            
            V.draw_scenes(
                keypoints=pred_dicts[0]['rebuilt_points'][:, 1:], #point_colors=np.ones((pred_dicts[0]['rebuilt_points'].shape[0], 3)) * [1, 0, 0], 
#                ref_boxes=pred_dicts[0]['rois'][0], #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                ref_boxes=data_dict['gt_boxes'][0, :, :-1]
            )
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
