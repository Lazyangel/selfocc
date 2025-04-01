import time, argparse, os.path as osp, os, sys
import torch, numpy as np
import torch.distributed as dist
import cv2
import torch.nn as nn

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
from model.encoder.bevformer.mappings import GridMeterMapping

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

def writeFlowKITTI(filename, uv):
    # uv: [h, w, 2]
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def visualFlowKITTI(uv):
    # uv: [h, w, 2]
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    return uv
    
def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid

def cal_2d_flow():
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, args.dataset)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20706")
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
    
    # prepare utils
    mapping_args = dict(
            nonlinear_mode='linear',
            h_size=[256, 0],
            h_range=[51.2, 0],
            h_half=True,
            w_size=[128, 0],
            w_range=[25.6, 0],
            w_half=False,
            d_size=[32, 0],
            d_range=[-2.0, 4.4, 4.4]
        )
    mapping = GridMeterMapping(
        # bev_inner,
        # bev_outer,
        # range_inner,
        # range_outer,
        # nonlinear_mode,
        # z_inner,
        # z_outer,
        # z_ranges
        **mapping_args)
        
    # eval
    print_freq = cfg.print_freq
    my_model.eval()

    if args.save_flow_rgb:
        flow_vis_save_path = os.path.join(args.work_dir, "flow_rgb")
        os.makedirs(flow_vis_save_path, exist_ok=True)
    if args.flow_metric:
        c2p_out_list, c2p_epe_list = [], []
        c2n_out_list, c2n_epe_list = [], []
        results = {}
    if args.return_loss:
        tot_flow_loss = 0.
    
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

                result_dict = my_model(imgs=input_imgs, metas=img_metas)
                # if distributed:
                #     result_dict = my_model.module.head.render(metas=img_metas, batch=args.batch)
                # else:
                #     result_dict = my_model.head.render(metas=img_metas, batch=args.batch)

                #### calculate all sorts of flows
                c2p_flow = result_dict['curr2prev_flow']
                c2n_flow = result_dict['curr2next_flow'] # bs, 3, H, W, D
                ms_rays = result_dict['ms_rays'] # HW(55*190), 2 
                depths = result_dict['ms_depths'][0] # B, N, R(107008=176x608)
                
                
                num_rays = cfg.num_rays # [176, 608]
                bs, num_cams= curr_imgs.shape[:2]
                H, W = curr_imgs.shape[-2:] 
                device = depths.device
                # prepare transformation matrices
                lidar2prevImg, lidar2nextImg = [], []
                img2lidar, lidar2img = [], []
                for meta in img_metas:
                    lidar2prevImg.append(meta['lidar2prevImg'])
                    lidar2nextImg.append(meta['lidar2nextImg'])
                    img2lidar.append(meta['img2lidar'])
                    lidar2img.append(meta['lidar2img'])

                # prepare flow gt
                c2n_flow_gt = meta['c2n_flow_gt'] # (H, W, 2)
                c2p_flow_gt = meta['c2p_flow_gt']
                flow_gt_shape = c2n_flow_gt.shape
                c2n_flow_gt = (torch.from_numpy(np.array(c2n_flow_gt).astype(np.float32))
                    .float()
                    .reshape(bs, num_cams, flow_gt_shape[0], flow_gt_shape[1], flow_gt_shape[2])
                    .to(device=device)
                    .permute(0, 1, 4, 2, 3)) # (bs, num_cams, H, W, 2)
                    
                c2p_flow_gt = (torch.from_numpy(np.array(c2p_flow_gt).astype(np.float32))
                    .float()
                    .reshape(bs, num_cams, flow_gt_shape[0], flow_gt_shape[1], flow_gt_shape[2])
                    .to(device=device)
                    .permute(0, 1, 4, 2, 3))
                
                def list2tensor(trans):
                    if isinstance(trans[0], (np.ndarray, list)):
                        trans = np.asarray(trans)
                        trans = depths.new_tensor(trans) # B, 36(6tem * 6cur), 4, 4
                    else:
                        trans = torch.stack(trans, dim=0)
                    # trans = trans.reshape(bs, num_cams, num_cams, 1, 4, 4)
                    trans = trans.reshape(bs, num_cams, 1, 4, 4)
                    # trans = trans.transpose(1, 2)
                    return trans

                lidar2prevImg = list2tensor(lidar2prevImg)
                lidar2nextImg = list2tensor(lidar2nextImg)
                img2lidar = list2tensor(img2lidar)
                lidar2img = list2tensor(lidar2img)
                
                # cal 2d flow
                for cam, d in enumerate(depths):
                    rays = ms_rays # 107008 , 2
                    # nums_rays, num_samples_per_ray = ms_rays.shape[0], rays.shape[0] // ms_rays.shape[0]
                    pixel_coords = torch.ones((bs, 1, len(rays), 4), device=device) # B, N, R, 4
                    pixel_coords[..., :2] = rays.reshape(1, 1, -1, 2)
                    pixel_coords[..., :3] *= d.reshape(1, 1, -1, 1) # t = depth
                    ## mono specific
                    pixel_coords = pixel_coords.reshape(bs, 1, len(rays), 4, 1) # 1, 1, hw*r, 4, 1

                    @torch.cuda.amp.autocast(enabled=False)
                    def cal_pixel(trans, coords):
                        trans = trans.float()
                        coords = coords.float()
                        eps = 1e-5
                        pixel = torch.matmul(trans, coords).squeeze(-1) # bs, N, R, 4
                        mask = pixel[..., 2] > 0
                        pixel = pixel[..., :2] / torch.maximum(torch.ones_like(pixel[..., :1]) * eps, pixel[..., 2:3])
                        mask = mask & (pixel[..., 0] > 0) & (pixel[..., 0] < W) & \
                                    (pixel[..., 1] > 0) & (pixel[..., 1] < H)
                        return pixel, mask
                    
                    lidar_coords = torch.matmul(img2lidar[:, cam:(cam+1), ...], pixel_coords).squeeze(-1)[..., :3] # bs, N, R, 3 (whz)
                    lidar_coords_norm = mapping.meter2grid(lidar_coords, True).unsqueeze(-2) # (hwz)
                    prev_sampled_flow = nn.functional.grid_sample( 
                        c2p_flow,  # bs, 3, H, W, Z
                        lidar_coords_norm[..., [2, 1, 0]] * 2 - 1, # bs, N, RS, 1, 3
                        mode='bilinear',
                        align_corners=True).permute(0, 2, 3, 4, 1) # bs, 3, N, RS, 1 --> bs, N, R*S, 1, 3
                    prev_lidar_coords = lidar_coords.unsqueeze(-2) + prev_sampled_flow # bs, N, R*S, 1, 3 (预测的流坐标顺序?)
                    prev_lidar_coords = torch.cat([prev_lidar_coords, torch.ones_like(prev_lidar_coords[..., :1])], dim=-1).permute(0, 1, 2, 4, 3) # bs, N, R*S, 4, 1
                    
                    next_sampled_flow = nn.functional.grid_sample(
                        c2n_flow,
                        lidar_coords_norm[..., [2, 1, 0]] * 2 - 1,
                        mode='bilinear',
                        align_corners=True).permute(0, 2, 3, 4, 1)
                    next_lidar_coords = lidar_coords.unsqueeze(-2) + next_sampled_flow
                    next_lidar_coords = torch.cat([next_lidar_coords, torch.ones_like(next_lidar_coords[..., :1])], dim=-1).permute(0, 1, 2, 4, 3)
                    
                    prev_trans, prev_coords = lidar2img[:, cam:(cam+1), ...], prev_lidar_coords
                    next_trans, next_coords = lidar2img[:, cam:(cam+1), ...], next_lidar_coords
                            
                    pixel_prev, prev_mask = cal_pixel(prev_trans, prev_coords) # bs, N, 1, RS(10450), 2
                    pixel_prev = pixel_prev.unsqueeze(2)
                    pixel_next, next_mask = cal_pixel(next_trans, next_coords)
                    pixel_next = pixel_next.unsqueeze(2)

                    pixel_curr = ms_rays.reshape(bs, num_cams, 1, -1, 2) # bs, N, 1, 10450 , 2

                    c2p_flow_2d = (pixel_prev - pixel_curr).reshape(bs, num_cams, num_rays[0], num_rays[1], 2).permute(0, 1, 4, 2, 3) # bs, num_cams, 2， H, W
                    c2n_flow_2d = (pixel_next - pixel_curr).reshape(bs, num_cams, num_rays[0], num_rays[1], 2).permute(0, 1, 4, 2, 3)
                
                def eval_flow_metric(flow_pred, flow_gt):
                    epe = torch.sum((flow_pred - flow_gt) ** 2, dim=0).sqrt()
                    mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
                    
                    epe = epe.view(-1)
                    mag = mag.view(-1)
                    out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
                    return epe.cpu().numpy(), out.cpu().numpy()
                    epe_list.append(epe.cpu().numpy())
                    out_list.append(out.cpu().numpy())
                
                if args.flow_metric:
                    gt_H, gt_W = flow_gt_shape[0], flow_gt_shape[1]
                    c2p_flow_pred = nn.functional.interpolate(c2p_flow_2d[0], (gt_H, gt_W), mode='bilinear') # 1, 2, H, W
                    c2n_flow_pred = nn.functional.interpolate(c2n_flow_2d[0], (gt_H, gt_W), mode='bilinear')
                    
                    c2p_epe, c2p_out = eval_flow_metric(c2p_flow_pred, c2p_flow_gt[0])
                    c2p_epe_list.append(c2p_epe)
                    c2p_out_list.append(c2p_out)
                    
                    c2n_epe, c2n_out = eval_flow_metric(c2n_flow_pred, c2n_flow_gt[0])
                    c2n_epe_list.append(c2n_epe)
                    c2n_out_list.append(c2n_out)
                    
                
                if args.save_flow_rgb and int(img_metas[0]['token']) % 100 == 0:
                # if args.save_flow_rgb:
                    for i, (color_imgs, c2p_flow_2d, c2n_flow_2d, img_meta) in enumerate(zip(color_imgs, c2p_flow_2d, c2n_flow_2d, img_metas)):
                        # H, W = cfg.img_size
                        color = color_imgs[0].permute(1, 2, 0).cpu().numpy() * 256 # 3, H, W --> H, W, 3
                        color = color[...,[2, 1, 0]]
                        # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    
                        c2p_flow = nn.functional.interpolate(c2p_flow_2d, (H, W), mode='bilinear')[0].permute(1, 2, 0).cpu().numpy() # 1, 2, H, W --> H, W, 2
                        c2n_flow = nn.functional.interpolate(c2n_flow_2d, (H, W), mode='bilinear')[0].permute(1, 2, 0).cpu().numpy()
                        
                        c2p_flow_color = visualFlowKITTI(c2p_flow) # H, W, 3
                        c2n_flow_color = visualFlowKITTI(c2n_flow)

                        concated_imgs = np.concatenate((color, c2p_flow_color, c2n_flow_color), axis=0)
                        token = img_meta['token']
                        cv2.imwrite(os.path.join(flow_vis_save_path, token + '.png'), concated_imgs[...,[2, 1, 0]])
                if args.return_loss:
                    def sample_pixel(pixel, imgs):
                        # imgs: B, N, 3, H, W
                        # pixel: B, N, 1, R, 2 (xy)
                        pixel_ = pixel
                        pixel = pixel_.clone()
                        pixel[..., 0] /= W
                        pixel[..., 1] /= H
                        pixel = 2 * pixel - 1
                        pixel_rgb = nn.functional.grid_sample(
                            imgs.flatten(0, 1), pixel.flatten(0, 1), align_corners=True) # BN, 3, 1, R
                        pixel_rgb = pixel_rgb.reshape(bs, 1, -1, 1, pixel_rgb.shape[-1]) # 1, 1, 3, 1, R
                        pixel_rgb = pixel_rgb.permute(0, 3, 1, 2, 4) # B, 1, N, 3, R
                        return pixel_rgb

                    def compute_flow_loss(flow_gt, pixel_pred, mask):
                        """_summary_
                        Args:
                            flow_gt (_type_): _description_ 
                            pixel_pred (_type_): _description_
                            mask (_type_): _description_
                            weights (_type_): _description_
                        """
                        pixel_curr = ms_rays.reshape(bs, num_cams, 1, -1, 2) # bs, N, 1, 10450 , 2
                        # 计算2d flow
                        flow_pred = (pixel_pred - pixel_curr).flatten(0,2).permute(0, 2, 1) # 1, 2, 10450 
                        # 1. 采样gt
                        flow_gt_ = sample_pixel(pixel_curr, flow_gt[:, cam:(cam+1), ...]) # B, 1, N, 2, R(10450 )
                        flow_gt_ = flow_gt_.reshape(bs* num_cams, 2, -1) # (bs, num_cams, 2, H, W) --> # (bs*num_cams, 2, H*W)
                        
                        abs_diff = torch.abs(flow_gt_ - flow_pred) # B*N_cur*N_tem, 2, R
                        flow_loss = abs_diff.mean(1, True) # B*N_cur*N_tem, 1, R
                        flow_loss[~mask.reshape(1, 1, -1)] = 1e3
                        
                        return flow_loss

                    # flow_loss
                    prev_flow_loss = compute_flow_loss(
                        flow_gt=c2p_flow_gt,
                        pixel_pred=pixel_prev,
                        mask=prev_mask,)
                    next_flow_loss = compute_flow_loss(
                        flow_gt=c2n_flow_gt,
                        pixel_pred=pixel_next,
                        mask=next_mask,) # B*N, 1, R
                    flow_loss = prev_flow_loss + next_flow_loss
                        
                    flow_loss_avg = torch.mean(flow_loss)
                    tot_flow_loss = tot_flow_loss + flow_loss_avg

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
    
    if args.flow_metric:
        c2p_epe_list = np.concatenate(c2p_epe_list)
        c2p_out_list = np.concatenate(c2p_out_list)
        
        c2p_epe = np.mean(c2p_epe_list)
        c2p_f1 = 100 * np.mean(c2p_out_list)
        print("Validation KITTI c2p EPE: %.3f, F1-all: %.3f" % (c2p_epe, c2p_f1))
        
        c2n_epe_list = np.concatenate(c2n_epe_list)
        c2n_out_list = np.concatenate(c2n_out_list)
        
        c2n_epe = np.mean(c2n_epe_list)
        c2n_f1 = 100 * np.mean(c2n_out_list)
        print("Validation KITTI c2n EPE: %.3f, F1-all: %.3f" % (c2n_epe, c2n_f1))
        
    if args.return_loss:
        print("Validation KITTI flow loss: %.3f" % (tot_flow_loss))
            

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--save-depth', action='store_true', default=False)
    parser.add_argument('--save-flow-rgb', action='store_true', default=False)
    parser.add_argument('--flow-metric', action='store_true', default=False)
    parser.add_argument('--depth-metric-tgt', type=str, default='raw')
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--return-loss', action='store_true', default=False)
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
