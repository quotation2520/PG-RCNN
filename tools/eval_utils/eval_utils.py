import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
####################################
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from sklearn import metrics
####################################


def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    Args:
        rois: (N, 7)
        roi_labels: (N)
        gt_boxes: (M, )
        gt_labels:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = (roi_labels == k)
        gt_mask = (gt_labels == k)
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]
    return max_overlaps, gt_assignment


def generate_point_prediction_dicts(point_dict, batch_dict):

    gt_boxes = batch_dict['gt_boxes'][..., :-1]
    gt_labels = batch_dict['gt_boxes'][..., -1].int()
    batch_size = batch_dict['batch_size']
    
    rois = point_dict['rois']
    roi_labels = point_dict['roi_labels']
    rebuilt_points = point_dict['rebuilt_points']
    rebuilt_points = rebuilt_points.view(batch_size, rois.shape[1], -1, rebuilt_points.shape[-1])

    positive_idx = roiaware_pool3d_utils.points_in_boxes_gpu(rebuilt_points.view(batch_size, -1, rebuilt_points.shape[-1])[..., 1:4], gt_boxes)
    positive_idx = (positive_idx != -1).view(batch_size, rois.shape[1], -1)

    annos = []
    for bidx in range(batch_size):
        anno = {}
        score = torch.sigmoid((rebuilt_points[bidx, :, :, -1])).view(-1)
        label = positive_idx[bidx, :].view(-1)
        anno.update({'score': score, 'label': label})
        annos.append(anno)
    return annos

def evaluate_points(gen_annos):
    return_dict = {}
    score = []
    label = []
    for gen_dict in gen_annos:
        score += [gen_dict['score'].detach().cpu()]
        label += [gen_dict['label'].detach().cpu()]
    score = torch.cat(score).numpy()
    label = torch.cat(label).numpy()
    fpr, tpr, _ = metrics.roc_curve(label, score)
    return_dict['auroc'] = metrics.auc(fpr, tpr)
    for thr in [0.5, 0.7, 0.9]:
        pos = score > thr
        return_dict['precision_%s' % thr] = label[pos].sum() / pos.sum()
        return_dict['pos_num_%s' % thr] = pos.sum() / len(gen_annos)
    return return_dict

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None, eval_points=False):

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    gen_annos = []
    
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()
    
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    # total_params = sum(param.numel() for param in model.parameters())
    # logger.info('*************** total_parameter: %s *****************' % total_params)


    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        if getattr(args, 'infer_time', False):
            start_time = time.time()
        with torch.no_grad():
            if eval_points:
                pred_dicts, ret_dict, point_dict = model(batch_dict, return_points=eval_points)
                annos2 = generate_point_prediction_dicts(point_dict, batch_dict)
                gen_annos += annos2
            else:
                try:
                    pred_dicts, ret_dict, _ = model(batch_dict)
                except:
                    pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        
        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    if getattr(args, 'infer_time', False):
        logger.info('*************** Average_time of inference time %f *****************' % infer_time_meter.avg)
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
        if eval_points:
            gen_annos = common_utils.merge_results_dist(gen_annos, len(dataset), tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    if eval_points:
        result_dict = evaluate_points(gen_annos)
        for key, value in result_dict.items():
            logger.info('%s: %f' % (key, value))
            ret_dict['point_generation/%s' % key] = value 

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass