from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F
import torch.nn as nn, torch
from model.encoder.bevformer.mappings import GridMeterMapping

@OPENOCC_LOSS.register_module()
class FlowLossMonoMulti(BaseLoss):

    def __init__(self, weight=0.005, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict={
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'ray_indices': 'ray_indices',
                'weights': 'weights',
                'ts': 'ts',
                'metas': 'metas',
                'ms_rays': 'ms_rays',
                'curr2prev_flow': 'curr2prev_flow',
                'curr2next_flow': 'curr2next_flow',
            }
        else:
            self.input_dict = input_dict
            
        self.img_size = kwargs.get('img_size', [768, 1600])
        self.ray_resize = kwargs.get('ray_resize', None)
        self.mot_mask = kwargs.get('mot_mask', False)
        self.dynamic_disentanglement = kwargs.get('dynamic_disentanglement', False)
        self.loss_func = self.flow_loss
        self.iter_counter = 0
        # 测试两种计算rays_points的方法
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
        self.mapping = GridMeterMapping(
            # bev_inner,
            # bev_outer,
            # range_inner,
            # range_outer,
            # nonlinear_mode,
            # z_inner,
            # z_outer,
            # z_ranges
            **mapping_args)
    
    def flow_loss(self, 
            curr_imgs, 
            prev_imgs, 
            next_imgs, 
            ray_indices,
            weights,
            ts,
            metas, 
            ms_rays,
            deltas=None,
            sample_sdfs=None,
            curr2prev_flow=None,
            curr2next_flow=None,
            ):

        device = ts[0].device
        bs, num_cams = curr_imgs.shape[:2]
        num_rays = ms_rays.shape[0]
        assert bs == 1

        # prepare transformation matrices
        img2prevImg, img2nextImg = [], []
        lidar2prevImg, lidar2nextImg = [], []
        lidar2prevLidar, lidar2nextLidar = [], []
        img2lidar, lidar2Img = [], []
        for meta in metas:
            img2prevImg.append(meta['img2prevImg'])
            img2nextImg.append(meta['img2nextImg'])
            lidar2prevImg.append(meta['lidar2prevImg'])
            lidar2nextImg.append(meta['lidar2nextImg'])
            lidar2prevLidar.append(meta['lidar2prevLidar'])
            lidar2nextLidar.append(meta['lidar2nextLidar'])
            img2lidar.append(meta['img2lidar'])
            lidar2Img.append(meta['lidar2img'])
            
        # get flow gt
        c2n_flow_gt = meta['c2n_flow_gt'] # (H, W, 2)
        c2p_flow_gt = meta['c2p_flow_gt']
        flow_shape = c2n_flow_gt.shape
        c2n_flow_gt = (torch.from_numpy(np.array(c2n_flow_gt).astype(np.float32))
            .float()
            .permute(2, 0, 1)
            .reshape(bs, num_cams, flow_shape[2], flow_shape[0], flow_shape[1])
            .to(device=device)) # (bs, num_cams, 2, H, W)
        c2p_flow_gt = (torch.from_numpy(np.array(c2p_flow_gt).astype(np.float32))
            .float()
            .permute(2, 0, 1)
            .reshape(bs, num_cams, flow_shape[2], flow_shape[0], flow_shape[1])
            .to(device=device))
        
        def list2tensor(trans):
            if isinstance(trans[0], (np.ndarray, list)):
                trans = np.asarray(trans)
                trans = ts[0].new_tensor(trans) # B, 36(6tem * 6cur), 4, 4
            else:
                trans = torch.stack(trans, dim=0)
            # trans = trans.reshape(bs, num_cams, num_cams, 1, 4, 4)
            trans = trans.reshape(bs, num_cams, 1, 4, 4)
            # trans = trans.transpose(1, 2)
            return trans

        img2prevImg = list2tensor(img2prevImg)
        img2nextImg = list2tensor(img2nextImg)
        lidar2prevImg = list2tensor(lidar2prevImg)
        lidar2nextImg = list2tensor(lidar2nextImg)
        lidar2nextLidar = list2tensor(lidar2nextLidar)
        lidar2prevLidar = list2tensor(lidar2prevLidar)
        img2lidar = list2tensor(img2lidar)
        lidar2Img = list2tensor(lidar2Img)
        
        tot_loss = 0.
        rays = ms_rays # H*W, 2(wh)
        for cam, (ray_idx, weight, t) in enumerate(zip(ray_indices, weights, ts)):
            rays = ms_rays[ray_idx] # 10450 * 256, 2
            # nums_rays, num_samples_per_ray = ms_rays.shape[0], rays.shape[0] // ms_rays.shape[0]
            if deltas is not None:
                delta = deltas[cam].detach()
                eps = torch.finfo(delta.dtype).eps
                weight = weight.clone()
                weight[delta < eps] = 0.
                weight = weight / delta.clamp_min(eps)
                # weight = weight / (delta.detach() + 1e-6)

            pixel_coords = torch.ones((bs, 1, len(rays), 4), device=device) # B, N, R, 4
            pixel_coords[..., :2] = rays.reshape(1, 1, -1, 2)
            pixel_coords[..., :3] *= t.reshape(1, 1, -1, 1) # t = depth
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
                mask = mask & (pixel[..., 0] > 0) & (pixel[..., 0] < self.img_size[1]) & \
                              (pixel[..., 1] > 0) & (pixel[..., 1] < self.img_size[0])
                return pixel, mask
            
            def sample_pixel(pixel, imgs):
                # imgs: B, N, 3, H, W
                # pixel: B, N, 1, R, 2 (xy)
                pixel_ = pixel
                pixel = pixel_.clone()
                pixel[..., 0] /= self.img_size[1]
                pixel[..., 1] /= self.img_size[0]
                pixel = 2 * pixel - 1
                pixel_rgb = F.grid_sample(
                    imgs.flatten(0, 1), pixel.flatten(0, 1), align_corners=True) # BN, 3, 1, R
                pixel_rgb = pixel_rgb.reshape(bs, 1, -1, 1, pixel_rgb.shape[-1]) # 1, 1, 3, 1, R
                pixel_rgb = pixel_rgb.permute(0, 3, 1, 2, 4) # B, 1, N, 3, R
                return pixel_rgb
            
            # === Compute motion mask === #
            if self.mot_mask:
                pixel_curr = ms_rays[ray_idx].reshape(bs, num_cams, 1, -1, 2) # bs, N, 1, 10450 * 256, 2
                # get movable object mask
                mot_mask_ = torch.from_numpy(meta["mot_mask"]).to(device).float().reshape(bs, num_cams, 1, self.img_size[0], self.img_size[1]) # H, W
                mot_mask = sample_pixel(pixel_curr, mot_mask_).permute(0, 1, 4, 2, 3) # bs, 1, N, 1, Rs --> 1, 1, Rs, 1, 1
                if self.dynamic_disentanglement:
                    # compute independent flow mask
                    def get_residual_mask(flow_gt, trans, samples):
                        # get complete from pretrained model pred flow gt
                        complete_flow = sample_pixel(pixel_curr, flow_gt[:, cam:(cam+1), ...]).permute(0, 1, 2, 4, 3) # B, 1, N, R(10450 * 256), 2
                        # get static_flow from depth 
                        pixel_static, static_mask = cal_pixel(trans, samples) # bs, N, 1, RS(10450*256), 2
                        static_flow = pixel_static - pixel_curr
                        # compute dynamic area mask
                        residual_flow = complete_flow - static_flow # B, N, 1, R(10450 * 256), 2
                        residual_mask = static_mask & (torch.abs(residual_flow[..., 0]) > self.dynamic_threshold) & \
                                        (torch.abs(residual_flow[..., 1]) > self.dynamic_threshold) # B, N, 1, R(10450 * 256)
                        return residual_mask.float().reshape(bs, num_cams, -1, 1, 1)
                    # get dynamic mask
                    prev_residual_mask = get_residual_mask(c2p_flow_gt, img2prevImg[:, cam:(cam+1), ...], pixel_coords)
                    prev_dynamic_mask = prev_residual_mask & mot_mask # bs, num_cams, Rs, 1, 1
                    next_residual_mask = get_residual_mask(c2n_flow_gt, img2nextImg[:, cam:(cam+1), ...], pixel_coords)
                    next_dynamic_mask = next_residual_mask & mot_mask
                else:
                    prev_dynamic_mask = mot_mask
                    next_dynamic_mask = mot_mask
            else:
                mot_mask = torch.ones(bs, num_cams, len(rays), 1).to(device) # bs, N, Rs, 1, 1
            # === Compute sample for reconstruction === #
            assert curr2next_flow is not None or curr2prev_flow is not None
            lidar_coords = torch.matmul(img2lidar[:, cam:(cam+1), ...], pixel_coords).squeeze(-1)[..., :3] # bs, N, R, 3 (whz)
            lidar_coords_norm = self.mapping.meter2grid(lidar_coords, True).unsqueeze(-2) # (hwz) in grid coords

            def cal_flow(pred_flow, dynamic_flow_mask):
                # in this case regard the pred flow as object independent flow
                independ_flow = nn.functional.grid_sample( 
                    pred_flow,  # bs, 3, H, W, Z 
                    lidar_coords_norm[..., [2, 1, 0]] * 2 - 1, # bs, N, RS, 1, 3(hwz)
                    mode='bilinear',
                    align_corners=True).permute(0, 2, 3, 4, 1) # bs, 3, N, RS, 1 --> bs, N, R*S, 1, 3 
                
                if self.mot_mask:
                    dynamic_independ_flow = independ_flow * dynamic_flow_mask # bs, N, RS, 1, 3 
                    pred_lidar_coords = lidar_coords.unsqueeze(-2) + dynamic_independ_flow # bs, N, R*S, 1, 3(whz)
                else:
                    pred_lidar_coords = lidar_coords.unsqueeze(-2) + independ_flow # bs, N, R*S, 1, 3(whz)
                
                pred_lidar_coords = torch.cat([pred_lidar_coords, torch.ones_like(pred_lidar_coords[..., :1])], dim=-1).permute(0, 1, 2, 4, 3) # bs, N, R*S, 4, 1

                return independ_flow, pred_lidar_coords 

            prev_independ_flow, prev_lidar_coords = cal_flow(curr2prev_flow, prev_dynamic_mask)
            next_independ_flow, next_lidar_coords = cal_flow(curr2next_flow, next_dynamic_mask)
                
            prev_trans, prev_coords = lidar2prevImg[:, cam:(cam+1), ...], prev_lidar_coords
            next_trans, next_coords = lidar2nextImg[:, cam:(cam+1), ...], next_lidar_coords
                    
            pixel_prev, prev_mask = cal_pixel(prev_trans, prev_coords) # bs, N, 1, RS(10450*256), 2
            pixel_prev = pixel_prev.unsqueeze(2)
            pixel_next, next_mask = cal_pixel(next_trans, next_coords)
            pixel_next = pixel_next.unsqueeze(2)
            
            
            def mask_invalid(mask, weight):
                new_weight = weight.clone()
                new_weight[~mask.flatten()] = 0.
                return new_weight
            
            rgb_prev = sample_pixel(pixel_prev, prev_imgs[:, cam:(cam+1), ...]) # bs, N, 1, 3, RS(10450*256)
            prev_weight = mask_invalid(prev_mask, weight)
            rgb_next = sample_pixel(pixel_next, next_imgs[:, cam:(cam+1), ...])
            next_weight = mask_invalid(next_mask, weight)
            
            def get_acc_weight_mask(rgb_prev, prev_weight, prev_mask):
                """_summary_

                Args:
                    rgb_prev (_type_): _description_ B, N, 1, 3, RS(10450*256)
                    prev_weight (_type_): _description_ RS(10450*256)
                    prev_mask (_type_): _description_ B, N, RS(10450*256)

                Returns:
                    prev_weight: _description_ RS(10450*256)         归一化后的权重
                    rgb_prev_new: _description_ B, N, 1, 3, R(10450) 加权求和后的像素值
                    acc_prev_mask: _description_ R(10450)            加权求和后的mask
                """
                # 对prev_weight进行累积，得到累积权重，并归一化
                acc_prev_weight = torch.zeros(num_rays, device=rgb_prev.device, dtype=rgb_prev.dtype)
                acc_prev_weight.index_add_(-1, ray_idx, prev_weight) # 计算每条射线上的累积权重
                acc_prev_weight = torch.gather(acc_prev_weight, dim=0, index=ray_idx).clamp_min(torch.finfo(rgb_prev.dtype).eps) # 提取每个采样点对应的累积权重
                prev_weight = prev_weight / acc_prev_weight # 归一化
                
                rgb_prev_new = torch.zeros(
                    (*rgb_prev.shape[:-1], num_rays), device=rgb_prev.device, dtype=rgb_prev.dtype)
                rgb_prev_new.index_add_(-1, ray_idx, rgb_prev * prev_weight.reshape(1, 1, 1, 1, -1))
                acc_prev_mask = torch.zeros(num_rays, device=rgb_prev.device, dtype=rgb_prev.dtype)
                acc_prev_mask.index_add_(-1, ray_idx, prev_mask.flatten().to(rgb_prev.dtype))
                acc_prev_mask = acc_prev_mask == 0
                # acc_prev_mask = acc_prev_mask.reshape(1, 1, 1, 1, -1).expand(-1, -1, -1, 3, -1)
                # rgb_prev_new[acc_prev_mask] = 1e3
                return prev_weight, rgb_prev_new, acc_prev_mask
            
            prev_weight, rgb_prev_new, acc_prev_mask = get_acc_weight_mask(rgb_prev, prev_weight, prev_mask)
            next_weight, rgb_next_new, acc_next_mask = get_acc_weight_mask(rgb_next, next_weight, next_mask)

            def compute_flow_loss(flow_gt, pixel_pred, weight, mask):
                """_summary_
                Args:
                    flow_gt (_type_): _description_ 
                    pixel_pred (_type_): _description_
                    mask (_type_): _description_
                    weights (_type_): _description_
                """
                pixel_curr = ms_rays[ray_idx].reshape(bs, num_cams, 1, -1, 2) # bs, N, 1, 10450 * 256, 2
                # 计算2d flow
                flow_pred = (pixel_pred - pixel_curr).flatten(0,2).permute(0, 2, 1) # 1, 2, 10450 * 256
                # 1. 采样gt
                flow_gt_ = sample_pixel(pixel_curr, flow_gt[:, cam:(cam+1), ...]) # B, 1, N, 2, R(10450 * 256)
                flow_gt_ = flow_gt_.reshape(bs* num_cams, 2, -1) # (bs, num_cams, 2, H, W) --> # (bs*num_cams, 2, H*W)
                
                abs_diff = torch.abs(flow_gt_ - flow_pred) # B*N_cur*N_tem, 2, R
                flow_loss_ = abs_diff.mean(1, True) # B*N_cur*N_tem, 1, R
                flow_loss = torch.zeros(*flow_loss_.shape[:-1], num_rays, dtype=flow_loss_.dtype, device=flow_loss_.device)
                flow_loss.index_add_(dim=-1, index=ray_idx, source=flow_loss_*weight.reshape(1, 1, -1))
                flow_loss[mask.reshape(1, 1, -1)] = 1e3 #这里为0是不是没有监督作用？
                
                return flow_loss

            # flow_loss
            prev_flow_loss = compute_flow_loss(
                flow_gt=c2p_flow_gt,
                pixel_pred=pixel_prev,
                weight=prev_weight,
                mask=acc_prev_mask,)
            next_flow_loss = compute_flow_loss(
                flow_gt=c2n_flow_gt,
                pixel_pred=pixel_next,
                weight=next_weight,
                mask=acc_next_mask,) # B*N, 1, R
            flow_loss = prev_flow_loss + next_flow_loss
                
            flow_loss_avg = torch.mean(flow_loss)
            tot_loss = tot_loss + flow_loss_avg
        self.iter_counter += 1

        return tot_loss / num_cams