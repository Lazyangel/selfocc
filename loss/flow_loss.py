from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F
import torch.nn as nn, torch
from model.encoder.bevformer.mappings import GridMeterMapping

@OPENOCC_LOSS.register_module()
class FlowLoss(BaseLoss):

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
            # ts,
            ms_depths,
            metas, 
            ms_rays,
            deltas=None,
            sample_sdfs=None,
            curr2prev_flow=None,
            curr2next_flow=None,
            ):

        device = ms_depths[0].device
        bs, num_cams = curr_imgs.shape[:2]
        num_rays = ms_rays.shape[0]
        assert bs == 1

        # prepare transformation matrices
        lidar2prevImg, lidar2nextImg = [], []
        img2lidar, lidar2img = [], []
        for meta in metas:
            lidar2prevImg.append(meta['lidar2prevImg'])
            lidar2nextImg.append(meta['lidar2nextImg'])
            img2lidar.append(meta['img2lidar'])
            lidar2img.append(meta['lidar2img'])
            
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
                trans = ms_depths[0].new_tensor(trans) # B, 36(6tem * 6cur), 4, 4
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
        
        tot_loss = 0.
        rays = ms_rays # H*W, 2(wh)
        depths = ms_depths[0]
        for cam, d in enumerate(depths):
            rays = ms_rays # 10450 , 2
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
                mask = mask & (pixel[..., 0] > 0) & (pixel[..., 0] < self.img_size[1]) & \
                              (pixel[..., 1] > 0) & (pixel[..., 1] < self.img_size[0])
                return pixel, mask
            
            assert curr2next_flow is not None or curr2prev_flow is not None
            lidar_coords = torch.matmul(img2lidar[:, cam:(cam+1), ...], pixel_coords).squeeze(-1)[..., :3] # bs, N, R, 3 (whz)
            lidar_coords_norm = self.mapping.meter2grid(lidar_coords, True).unsqueeze(-2) # (hwz)
            prev_sampled_flow = nn.functional.grid_sample( 
                curr2prev_flow,  # bs, 3, H, W, Z
                lidar_coords_norm[..., [2, 1, 0]] * 2 - 1, # bs, N, RS, 1, 3
                mode='bilinear',
                align_corners=True).permute(0, 2, 3, 4, 1) # bs, 3, N, RS, 1 --> bs, N, R*S, 1, 3
            prev_lidar_coords = lidar_coords.unsqueeze(-2) + prev_sampled_flow # bs, N, R*S, 1, 3 (预测的流坐标顺序?)
            prev_lidar_coords = torch.cat([prev_lidar_coords, torch.ones_like(prev_lidar_coords[..., :1])], dim=-1).permute(0, 1, 2, 4, 3) # bs, N, R*S, 4, 1
            
            next_sampled_flow = nn.functional.grid_sample(
                curr2next_flow,
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
            tot_loss = tot_loss + flow_loss_avg
        self.iter_counter += 1

        return tot_loss / num_cams