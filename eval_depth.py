import time, argparse, os.path as osp, os, sys
import torch, numpy as np
import torch.distributed as dist
import cv2

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

from utils.feat_tools import multi2single_scale
from copy import deepcopy
from utils.config_tools import modify_for_eval


def pass_print(*args, **kwargs):
    pass

def visualize_depth(disp_resized_np):
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # im.save(save_path)
    return colormapped_im


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, args.dataset) # 这里修改了一些参数！
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20606")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, 'eval_depth_' + osp.basename(args.py_config)))
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_depth_{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(args)
    logger.info(f'Config:\n{cfg.pretty_text}')

    import model
    from dataset import get_dataloader

    # build model    
    cfg.model.head.return_max_depth = True
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=distributed,)

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', False)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    print(osp.exists(cfg.resume_from))
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        logger.info(f'successfully resumed from epoch {ckpt["epoch"]}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    print_freq = cfg.print_freq
    my_model.eval()
    if args.depth_metric:
        from utils.metric_util import DepthMetric
        if args.dataset == 'kitti' or args.dataset == 'kitti_raw':
            camera_names = ['front']
        elif args.dataset == 'nuscenes':
            camera_names = ['front', 'front_right', 'front_left', \
                            'back',  'back_left',   'back_right']
        depth_metric = DepthMetric(
            camera_names=camera_names).cuda()
        depth_metric._reset()
    if args.save_depth:
        depth_save_path = os.path.join(args.work_dir, "depth")
        depth_median_save_path = os.path.join(args.work_dir, "depth_median")
        depth_max_save_path = os.path.join(args.work_dir, "depth_max")
        os.makedirs(depth_save_path, exist_ok=True)
        os.makedirs(depth_median_save_path, exist_ok=True)
        os.makedirs(depth_max_save_path, exist_ok=True)
    if args.save_rgb:
        vis_save_path = os.path.join(args.work_dir, "depth_rgb")
        os.makedirs(vis_save_path, exist_ok=True)
        
    with torch.no_grad():
        for i_iter_val, (input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, \
                         img_metas, curr_aug, prev_aug, next_aug) in enumerate(val_dataset_loader):
            
            input_imgs = input_imgs.cuda()
            
            with torch.cuda.amp.autocast(amp):
                if cfg.get('estimate_pose'):
                    assert curr_aug is not None and prev_aug is not None and next_aug is not None
                    assert img_metas[0]['input_imgs_path'] == img_metas[0]['curr_imgs_path']
                    curr_aug, prev_aug, next_aug = curr_aug.cuda(), prev_aug.cuda(), next_aug.cuda()
                    pose_dict = my_model(pose_input=[curr_aug, prev_aug, next_aug], metas=img_metas, predict_pose=True)
                    for i_meta, meta in enumerate(img_metas):
                        meta.update(pose_dict[i_meta])

                if args.render_type == '3dgs':
                    result_dict = my_model(
                    imgs=input_imgs, metas=img_metas)
                else:
                    my_model(imgs=input_imgs, metas=img_metas, prepare=True)
                    # result_dict = my_model.module.head.render(metas=img_metas, batch=args.batch)
                    # result_dict = raw_model.head.render(metas=img_metas, batch=args.batch)
                    if distributed:
                        result_dict = my_model.module.head.render(metas=img_metas, batch=args.batch)
                    else:
                        result_dict = my_model.head.render(metas=img_metas, batch=args.batch)
                
                if args.flip:
                    input_imgs = torch.flip(input_imgs, dims=[-1])
                    for meta in img_metas:
                        meta['flip'] = True
                    my_model(imgs=input_imgs, metas=img_metas, prepare=True)
                    # result_dict_flip = my_model.module.head.render(metas=img_metas, batch=args.batch)
                    # result_dict_flip = raw_model.head.render(metas=img_metas, batch=args.batch)
                    if distributed:
                        result_dict_flip = my_model.module.head.render(metas=img_metas, batch=args.batch)
                    else:
                        result_dict_flip = my_model.head.render(metas=img_metas, batch=args.batch)
                    
                    ms_depths_flip = result_dict_flip['ms_depths'][0]
                    result_dict['ms_depths'][0] = (result_dict['ms_depths'][0] + ms_depths_flip) / 2

                    if 'ms_depths_median' in result_dict_flip:
                        ms_depths_median_flip = result_dict_flip['ms_depths_median'][0]
                        result_dict['ms_depths_median'][0] = (result_dict['ms_depths_median'][0] + ms_depths_median_flip) / 2
                    
                    ms_depths_max_flip = result_dict_flip['ms_max_depths'][0]
                    result_dict['ms_max_depths'][0] = (result_dict['ms_max_depths'][0] + ms_depths_max_flip) / 2

                #### calculate all sorts of depths
                if 'disp' in result_dict:
                    ms_depths = result_dict['disp'] # B, N, H(H/2), W(W/2)
                else:
                    ms_depths = result_dict['ms_depths'][0] # B, N, R(107008=176x608)
                    ms_depths = ms_depths.unflatten(-1, cfg.num_rays) # B, N, H(H/2), W(W/2)

                if 'ms_depths_median' in result_dict:
                    ms_depths_median = result_dict['ms_depths_median'][0]
                    ms_depths_median = ms_depths_median.unflatten(-1, cfg.num_rays)
                if 'ms_max_depths' in result_dict:
                    ms_depths_max = result_dict['ms_max_depths'][0]
                    ms_depths_max = ms_depths_max.unflatten(-1, cfg.num_rays)

                if args.save_depth:
                    # ms_depths: B, N, R
                    for i, (ms_depth, img_meta) in enumerate(zip(ms_depths, img_metas)):
                        token = img_meta['token']
                        np.save(os.path.join(depth_save_path, token + '.npy'), ms_depth.cpu().numpy())

                    if 'ms_depths_median' in result_dict:
                        for i, (ms_depth, img_meta) in enumerate(zip(ms_depths_median, img_metas)):
                            token = img_meta['token']
                            np.save(os.path.join(depth_median_save_path, token + '.npy'), ms_depth.cpu().numpy())

                    for i, (ms_depth, img_meta) in enumerate(zip(ms_depths_max, img_metas)):
                        token = img_meta['token']
                        np.save(os.path.join(depth_max_save_path, token + '.npy'), ms_depth.cpu().numpy())
                
                if args.save_rgb and int(img_metas[0]['token']) % 100 == 0:
                # if args.save_rgb:
                    for i, (color_imgs, ms_depth, img_meta) in enumerate(zip(color_imgs, ms_depths, img_metas)):
                        H, W = cfg.img_size
                        depth_H, depth_W  = ms_depth.shape[-2:]
                        color = color_imgs[0].permute(1, 2, 0).cpu().numpy() * 256 # 3, H, W --> H, W, 3
                        color = color[...,[2, 1, 0]]
                        # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                        
                        pred_depth = ms_depth.squeeze(0).cpu().numpy()
                        pred_depth = 1.0 / pred_depth
                        depth_color = visualize_depth(pred_depth) # H, W, 3
                        # depth_color = depth_color.reshape(depth_H, depth_W, 3)
                        depth_color = cv2.resize(depth_color, (W, H))

                        concated_imgs = np.concatenate((color, depth_color), axis=0)
                        token = img_meta['token']
                        cv2.imwrite(os.path.join(vis_save_path, token + '.png'), concated_imgs[...,[2, 1, 0]])
                    
                    
                if args.depth_metric:
                    depth_loc = ms_depths.new_tensor(img_metas[0]['depth_loc'])
                    depth_gt = ms_depths.new_tensor(img_metas[0]['depth_gt'])
                    depth_mask = torch.from_numpy(img_metas[0]['depth_mask']).cuda()
                    if args.depth_metric_tgt == 'raw':
                        depth_pred = ms_depths[0]
                    elif args.depth_metric_tgt == 'median':
                        depth_pred = ms_depths_median[0]
                    elif args.depth_metric_tgt == 'max':
                        depth_pred = ms_depths_max[0]
                    depth_metric._after_step(depth_loc, depth_gt, depth_mask, depth_pred)

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
    
    if args.depth_metric:
        depth_metric._after_epoch()
        # depth_metric.reset()
            

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-depth', action='store_true', default=False)
    parser.add_argument('--save-rgb', action='store_true', default=False)
    parser.add_argument('--depth-metric', action='store_true', default=False)
    parser.add_argument('--depth-metric-tgt', type=str, default='raw')
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--render_type', type=str, default='volume_render')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.hfai:
        os.environ['HFAI'] = 'true'

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
