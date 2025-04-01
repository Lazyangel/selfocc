import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F
from model.encoder.bevformer.mappings import GridMeterMapping
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@OPENOCC_LOSS.register_module()
class ReprojLossMonoMultiNew(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'ray_indices': 'ray_indices',
                'weights': 'weights',
                'ts': 'ts',
                'metas': 'metas',
                'ms_rays': 'ms_rays'}
        else:
            self.input_dict = input_dict
        self.no_ssim = kwargs.get('no_ssim', False)
        self.img_size = kwargs.get('img_size', [768, 1600])
        self.ray_resize = kwargs.get('ray_resize', None)
        self.no_automask = kwargs.get('no_automask', False)
        self.sdf_loss = kwargs.get('sdf_loss', False)
        self.sdf_loss_weight = kwargs.get('sdf_loss_weight', 0.1)
        self.dreg_loss = kwargs.get('dreg_loss', False)
        self.dreg_loss_weight = kwargs.get('dreg_loss_weight', 0.1)
        self.dynamic_threshold = kwargs.get('threshold', 5)
        self.dynamic_disentanglement = kwargs.get('dynamic_disentanglement', False)
        self.dims = kwargs.get('dims', 3)
        self.no_ssim = self.no_ssim or (self.ray_resize is None)
        self.mot_mask = kwargs.get('mot_mask', False)
        self.flow_pred = kwargs.get('flow_pred', False)
        
        if not self.no_ssim:
            self.ssim = SSIM()
        
        self.loss_func = self.reproj_loss
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
    def reproj_loss(
            self, 
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
            curr2prev_flow=None, # 在当前lidar坐标系中的移动
            curr2next_flow=None,
            prev_warp_cams=None,
            next_warp_cams=None,):
        # curr_imgs: B, N, C, H, W
        # depth: B, N, R
        # rays: R, 2
        # curr2prev_flow: B, 3, H, W, Z
        # prev_warp_list: list[num_cam] B, N, R, S, 3 lidar coord
        # import pdb; pdb.set_trace()
        if self.sdf_loss:
            assert sample_sdfs is not None
        device = ts[0].device
        bs, num_cams = curr_imgs.shape[:2]
        num_rays = ms_rays.shape[0]
        assert bs == 1

        # prepare transformation matrices
        img2prevImg, img2nextImg = [], []
        lidar2prevImg, lidar2nextImg = [], []
        lidar2prevLidar, lidar2nextLidar = [], []
        img2lidar, lidar2img = [], []
        for meta in metas:
            img2prevImg.append(meta['img2prevImg'])
            img2nextImg.append(meta['img2nextImg'])
            lidar2prevImg.append(meta['lidar2prevImg'])
            lidar2nextImg.append(meta['lidar2nextImg'])
            lidar2prevLidar.append(meta['lidar2prevLidar'])
            lidar2nextLidar.append(meta['lidar2nextLidar'])
            img2lidar.append(meta['temImg2lidar'])
            lidar2img.append(meta['lidar2temImg'])


        
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
        lidar2img = list2tensor(lidar2img)
        
        tot_loss = 0.
        rays = ms_rays # H*W, 2(wh)
        for cam, (ray_idx, weight, t, meta) in enumerate(zip(ray_indices, weights, ts, metas)):
            rays = ms_rays[ray_idx] # 10450(HW) * 256, 2(wh)
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
            if self.flow_pred: # independent 3D flow is predicted
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
                    # # in this case regard the pred flow as scene complete flow
                    # if self.mot_mask:
                    #     cmp_flow = nn.functional.grid_sample( 
                    #         pred_flow,  # bs, 3, H, W, Z 
                    #         lidar_coords_norm[..., [2, 1, 0]] * 2 - 1, # bs, N, RS, 1, 3(hwz)
                    #         mode='bilinear',
                    #         align_corners=True).permute(0, 2, 3, 4, 1) # bs, 3, N, RS, 1 --> bs, N, R*S, 1, 3 
                    #     img2prevlidar = lidar2prevLidar[:, cam:(cam+1), ...] @ img2lidar[:, cam:(cam+1), ...]
                    #     ego_flow = (torch.matmul(img2prevlidar, pixel_coords).squeeze(-1)[..., :3] 
                    #                     - torch.matmul(img2lidar[:, cam:(cam+1), ...], pixel_coords).squeeze(-1)[..., :3]).unsqueeze(-2) # bs, N, RS, 1, 3 (whz)
                    #     residual_flow = cmp_flow - ego_flow # bs, N, R, 1, 3 (whz)
                    #     residual_flow = torch.cat([residual_flow, torch.ones_like(residual_flow[..., :1])], dim=-1).permute(0, 1, 2, 4, 3) # bs, N, RS, 4(whz), 1
                    #     residual_flow = torch.matmul(torch.inverse(lidar2prevLidar[:, cam:(cam+1), ...]), residual_flow).permute(0, 1, 2, 4, 3)[..., :3] # bs, N, RS, 1, 3(whz)
                    #     independ_flow = residual_flow * mot_mask # bs, N, RS, 1, 3
                    #     prev_lidar_coords = lidar_coords.unsqueeze(-2) + independ_flow # bs, N, R*S, 1, 3(whz)
                prev_independ_flow, prev_lidar_coords = cal_flow(curr2prev_flow, prev_dynamic_mask)
                next_independ_flow, next_lidar_coords = cal_flow(curr2next_flow, next_dynamic_mask)
                
                prev_trans, prev_coords = lidar2prevImg[:, cam:(cam+1), ...], prev_lidar_coords
                next_trans, next_coords = lidar2nextImg[:, cam:(cam+1), ...], next_lidar_coords
            else:
                prev_trans, prev_coords = img2prevImg[:, cam:(cam+1), ...], pixel_coords
                next_trans, next_coords = img2nextImg[:, cam:(cam+1), ...], pixel_coords
            
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

            pixel_curr = ms_rays.reshape(
                1, 1, 1, num_rays, 2).repeat(bs, 1, 1, 1, 1) # bs, N, 1, R, 2
            target_curr = sample_pixel(pixel_curr, curr_imgs[:, cam:(cam+1), ...]) # B, 1, N, 3, R
            target_prev = sample_pixel(pixel_curr, prev_imgs[:, cam:(cam+1), ...])
            target_next = sample_pixel(pixel_curr, next_imgs[:, cam:(cam+1), ...])

            def compute_reprojection_loss(pred, target):
                # target = target.reshape(
                #     bs, num_cams, num_cams, 3, num_rays).flatten(0, 2) # B*N*N, 3, R
                target = target.expand(*pred.shape).flatten(0, 2)
                pred = pred.flatten(0, 2)
                abs_diff = torch.abs(target - pred) # B*N_cur*N_tem, 3, R
                l1_loss = abs_diff.mean(1, True) # B*N_cur*N_tem, 1, R

                if self.no_ssim:
                    reprojection_loss = l1_loss
                else:
                    pred_reshape = pred.reshape(pred.shape[0], -1, *self.ray_resize)
                    target_reshape = target.reshape(target.shape[0], -1, *self.ray_resize)
                    ssim_loss = self.ssim(pred_reshape, target_reshape).mean(1, True)
                    ssim_loss = ssim_loss.flatten(2)
                    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

                return reprojection_loss

            def compute_reprojection_loss_fn(pred, target, pred_new, weight, mask):
                target = target.expand(*pred_new.shape).flatten(0, 2) # BN, 3, R
                pred = pred.flatten(0, 2)
                target_ = torch.gather(
                    target, dim=-1, index=ray_idx.reshape(1, 1, -1).expand(-1, self.dims, -1))
                abs_diff = torch.abs(target_ - pred) # B*N_cur*N_tem, 3, R
                l1_loss_ = abs_diff.mean(1, True) # B*N_cur*N_tem, 1, R
                l1_loss = torch.zeros(*l1_loss_.shape[:-1], num_rays, dtype=l1_loss_.dtype, device=l1_loss_.device)
                l1_loss.index_add_(dim=-1, index=ray_idx, source=l1_loss_*weight.reshape(1, 1, -1))

                if self.no_ssim:
                    reprojection_loss = l1_loss
                else:
                    pred_reshape = pred_new.reshape(pred.shape[0], -1, *self.ray_resize)
                    target_reshape = target.reshape(target.shape[0], -1, *self.ray_resize)
                    ssim_loss = self.ssim(pred_reshape, target_reshape).mean(1, True)
                    ssim_loss = ssim_loss.flatten(2)
                    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                reprojection_loss[mask.reshape(1, 1, -1)] = 1e3

                return reprojection_loss
            
            prev_proj_loss = compute_reprojection_loss_fn(rgb_prev, target_curr, rgb_prev_new, prev_weight, acc_prev_mask) # B*N_cur*N_tem, 1, R
            next_proj_loss = compute_reprojection_loss_fn(rgb_next, target_curr, rgb_next_new, next_weight, acc_next_mask)
            prev_proj_loss = prev_proj_loss.reshape(bs, 1, num_rays)
            next_proj_loss = next_proj_loss.reshape(bs, 1, num_rays)
            prev_proj_loss = prev_proj_loss.unsqueeze(2)
            next_proj_loss = next_proj_loss.unsqueeze(2)

            if self.no_automask:
                proj_loss = torch.cat([
                    prev_proj_loss, 
                    next_proj_loss], dim=2)
            else:
                mask_prev_proj_loss = compute_reprojection_loss(target_prev, target_curr) # B*N, 1, R
                mask_next_proj_loss = compute_reprojection_loss(target_next, target_curr)
                mask_prev_proj_loss = mask_prev_proj_loss.reshape(bs, 1, 1, num_rays)
                mask_next_proj_loss = mask_next_proj_loss.reshape(bs, 1, 1, num_rays)

                proj_loss = torch.cat([
                    prev_proj_loss, 
                    next_proj_loss,
                    mask_prev_proj_loss,
                    mask_next_proj_loss], dim=2)
            proj_loss, _ = proj_loss.min(dim=2) # bs, N_cur, R

            if self.sdf_loss:
                prev_weight_id = prev_weight.reshape(num_rays, -1).argmax(dim=-1, keepdim=True)
                next_weight_id = next_weight.reshape(num_rays, -1).argmax(dim=-1, keepdim=True)
                sample_sdf = sample_sdfs[cam].reshape(num_rays, -1)
                prev_sdf = torch.gather(sample_sdf, -1, prev_weight_id) # R, 1
                next_sdf = torch.gather(sample_sdf, -1, next_weight_id) # R, 1
                sdfs = torch.cat([prev_sdf, next_sdf, \
                                  torch.zeros(num_rays, 2, device=prev_sdf.device)], dim=-1) # R, 4
                sdf = torch.gather(sdfs, -1, _.reshape(num_rays, 1)) # R, 1
                sdf_loss = sdf.abs().sum() / (_ <= 1).sum()
            
            # regulate static area flow prediction
            if self.dreg_loss:
                def compute_dreg_loss(independ_flow, dynamic_flow_mask, weight):
                    abs_diff = torch.abs(independ_flow * (1-dynamic_flow_mask) * weight.reshape(1, 1, -1, 1, 1)) # bs, N, RS, 1, 3 
                    l1_loss = abs_diff.mean(-1, True).reshape(1, 1, -1) # bs, 1, RS
                    return l1_loss
                
                prev_dreg_loss = compute_dreg_loss(prev_independ_flow, prev_dynamic_mask, prev_weight)
                next_dreg_loss = compute_dreg_loss(next_independ_flow, next_dynamic_mask, next_weight)
                dreg_loss = torch.mean(prev_dreg_loss + next_dreg_loss)
                

            if self.writer and self.iter_counter % 10 == 0:
                mask = _ > 1
                self.writer.add_scalar(f'masked/{cam}', mask.sum(), self.iter_counter)
                if self.sdf_loss:
                    self.writer.add_scalar(f'sdf_loss/{cam}', sdf_loss, self.iter_counter)
                if self.dreg_loss:
                    self.writer.add_scalar(f'dreg_loss/{cam}', dreg_loss, self.iter_counter)
                
            proj_loss_avg = torch.mean(proj_loss)
            tot_loss = tot_loss + proj_loss_avg
            if self.sdf_loss:
                tot_loss = tot_loss + sdf_loss * self.sdf_loss_weight
            if self.dreg_loss:
                tot_loss = tot_loss + dreg_loss * self.dreg_loss_weight
        self.iter_counter += 1

        return tot_loss / num_cams