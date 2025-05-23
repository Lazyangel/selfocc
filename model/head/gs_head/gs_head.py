# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import pdb
import time

import torch, torch.distributed as dist, torch.nn as nn, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_efficient_distloss import eff_distloss, eff_distloss_native

from utils import geom
from utils import vox
from utils import basic
from utils import render
from ._3DCNN import S3DCNN
from sys import path

from ..lib.gaussian_renderer import splatting_render, DistCUDA2

# from mmdet.models.builder import build_loss
# from mmdet.models.losses import cross_entropy_loss

# for gt occ loss
from utils.losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from utils.losses.lovasz_softmax import lovasz_softmax

# neus head 
from ..base_head import BaseTaskHead
from mmseg.models import HEADS
from mmengine.logging import MMLogger
import math
logger = MMLogger.get_instance('selfocc')
from utils.tb_wrapper import WrappedTBWriter
if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None

@HEADS.register_module()
class GSHead(BaseTaskHead):

    def __init__(self, 
                #  opt,
                dataset='kitti',
                use_semantic=False,
                semantic_classes=18,
                batch=1,
                min_depth=0.1,
                max_depth=80,
                real_size=[-25.6, 0.0, -2.0, 25.6, 51.2, 4.4], # xmin, ymin， zmin， xmax， ymax， zmax
                voxels_size=[32, 256, 256],
                stepsize=0.5,
                contracted_coord=False,
                infinite_range=False,
                contracted_ratio=0.667,
                input_channel=96,
                position='embedding',
                render_type='3dgs',
                gs_sample=0,
                gs_scale=0.20,
                last_free=True,
                render_h=55,
                render_w=190,
                **kwargs):
        super(GSHead, self).__init__()

        # self.opt = opt
        self.dataset = dataset
        self.use_semantic = use_semantic
        self.semantic_classes = semantic_classes
        self.batch = batch

        self.near = min_depth
        self.far = max_depth
        self.position = position
        
        self.contracted_ratio = contracted_ratio
        self.infinite_range = infinite_range
        self.contracted_coord = contracted_coord
        
        self.real_size = real_size
        self.voxels_size = voxels_size
        self.gs_sample = gs_sample
        self.gs_scale = gs_scale
        self.render_type = render_type
        self.last_free = last_free
        self.render_h = render_h
        self.render_w = render_w
        
        # loss
        self.weight_entropy_last = kwargs.get("weight_entropy_last", 0.1)
        self.weight_distortion = kwargs.get("weight_distortion", 0.1)
        self.weight_sparse_reg = kwargs.get("weight_sparse_reg", 0.0)
        # loss_occ=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     ignore_index=255,
        #     loss_weight=1.0)

        # self.loss_occ = build_loss(loss_occ)
        # self.loss_occ = cross_entropy_loss

        num_classes = semantic_classes


        self.register_buffer('xyz_min', torch.from_numpy(
            np.array([real_size[0], real_size[1], real_size[2]])))
        self.register_buffer('xyz_max', torch.from_numpy(
            np.array([real_size[3], real_size[4], real_size[5]])))

        self.ZMAX = real_size[4] # Xmax?

        self.Z = voxels_size[0]
        self.Y = voxels_size[1]
        self.X = voxels_size[2]

        self.Z_final = self.Z
        self.Y_final = self.Y
        self.X_final = self.X


        self.stepsize = stepsize  # sample interval
        self.num_voxels = self.Z_final * self.Y_final * self.X_final
        self.stepsize_log = self.stepsize
        self.interval = self.stepsize
        self.semantic_sample_ratio = 0.25

        if self.contracted_coord:
            # Sampling strategy for contracted coordinate
            contracted_rate = self.contracted_ratio
            num_id_voxels = int(self.num_voxels * (contracted_rate)**3)
            self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_id_voxels).pow(1 / 3)
            diagonal = (self.xyz_max - self.xyz_min).pow(2).sum().pow(1 / 2)
            self.N_samples = int(diagonal / 2 / self.stepsize / self.voxel_size / contracted_rate)
            
            if self.infinite_range:
                # depth_roi = [-self.far] * 3 + [self.far] * 3
                zval_roi = [-diagonal] * 3 + [diagonal] * 3
                fc = 1 - 0.5 / self.X  # avoid NaN
                zs_contracted = torch.linspace(0.0, fc, steps=self.N_samples)
                zs_world = vox.contracted2world(
                    zs_contracted[None, :, None].repeat(1, 1, 3),
                    # pc_range_roi=depth_roi,
                    pc_range_roi=zval_roi,
                    ratio=self.contracted_ratio)[:, :, 0]
            else:
                zs_world = torch.linspace(0.0, self.N_samples - 1, steps=self.N_samples)[None] * self.stepsize * self.voxel_size
            self.register_buffer('Zval', zs_world)

            pc_range_roi = self.xyz_min.tolist() + self.xyz_max.tolist()
            
            self.norm_func = lambda xyz: vox.world2contracted(xyz, pc_range_roi=pc_range_roi, ratio=self.opt.contracted_ratio)

        else:
            self.N_samples = int(np.linalg.norm(np.array([self.Z_final // 2, self.Y_final // 2, self.X_final]) + 1) / self.stepsize) + 1
            self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
            zs_world = torch.linspace(0.0, self.N_samples - 1, steps=self.N_samples)[None] * self.stepsize * self.voxel_size
            self.register_buffer('Zval', zs_world)
            self.norm_func = lambda xyz: (xyz - self.xyz_min.to(xyz)) / (self.xyz_max.to(xyz) - self.xyz_min.to(xyz)) * 2.0 - 1.0

        length_pose_encoding = 3

        self.pos_embedding = None
        self.pos_embedding1 = None
        self.input_channel = input_channel

        scene_centroid_x = 0.0
        scene_centroid_y = 0.0
        scene_centroid_z = 0.0

        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])

        self.register_buffer('scene_centroid', torch.from_numpy(scene_centroid).float())

        self.bounds = (self.real_size[0], self.real_size[3],
                       self.real_size[1], self.real_size[4],
                       self.real_size[2], self.real_size[5])
        #  bounds = (-40, 40, -40, 40, -1, 5.4)

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds, position = self.position, length_pose_encoding = length_pose_encoding, real_size=self.real_size,
            gs_sample=gs_sample, assert_cube=False)


        activate_fun = nn.ReLU(inplace=True)
        
        self.aggregation = '3dcnn'
        if self.aggregation == '3dcnn':
            self._3DCNN = S3DCNN(input_planes=input_channel, out_planes=1, planes=16,
                                 activate_fun=activate_fun)

        else:
            print('please define the aggregation')
            exit()


        if 'gs' in render_type:
            self.gs_vox_util = vox.Vox_util(
                self.Z_final, self.Y_final, self.X_final,
                scene_centroid = self.scene_centroid,
                bounds=self.bounds, position = self.position, 
                length_pose_encoding = length_pose_encoding, 
                real_size=self.real_size, 
                gs_sample=gs_sample,
                assert_cube=False)


    def feature2vox_simple(self, features, pix_T_cams, cam0_T_camXs, __p, __u):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)

        _, C, Hf, Wf = features.shape

        sy = Hf / float(self.opt.height)
        sx = Wf / float(self.opt.width)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_

        feat_mems_ = self.vox_util.unproject_image_to_mem(
            features,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, self.Z, self.Y, self.X)

        # feat_mems_ shape： torch.Size([6, 64, 24, 300, 300])
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X # torch.Size([1, 6, 128, 200, 8, 200])

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        feat_mem = feat_mem.permute(0, 1, 4, 3, 2) # [0, ...].unsqueeze(0) # ZYX -> XYZ

        return feat_mem


    def grid_sampler(self, xyz, *grids, align_corners=True, avail_mask=None, vis=False):
        '''Wrapper for the interp operation'''
        
        # pdb.set_trace()
        
        if self.semantic_sample_ratio < 1.0 and self.use_semantic and not vis:
            group_size = int(1.0 / self.semantic_sample_ratio)
            group_num = xyz.shape[1] // group_size
            xyz_sem = xyz[:, :group_size * group_num].reshape(xyz.shape[0], group_num, group_size, 3).mean(dim=2)
        else:
            xyz_sem = None


        # pdb.set_trace()

        if avail_mask is not None:
            if self.contracted_coord:
                ind_norm = self.norm_func(xyz)
                avail_mask = self.effective_points_mask(ind_norm)
                ind_norm = ind_norm[avail_mask]
                if xyz_sem is not None:
                    avail_mask_sem = avail_mask[:, :group_size * group_num].reshape(avail_mask.shape[0], group_num, group_size).any(dim=-1)
                    ind_norm_sem = self.norm_func(xyz_sem[avail_mask_sem])
            else:
                xyz_masked = xyz[avail_mask]
                ind_norm = self.norm_func(xyz_masked)
                if xyz_sem is not None:
                    avail_mask_sem = avail_mask[:, :group_size * group_num].reshape(avail_mask.shape[0], group_num, group_size).any(dim=-1)
                    ind_norm_sem = self.norm_func(xyz_sem[avail_mask_sem])

        else:

            ind_norm = self.norm_func(xyz)
            
            if xyz_sem is not None:
                ind_norm_sem = self.norm_func(xyz_sem)
                avail_mask_sem = None
        
        ind_norm = ind_norm.flip((-1,)) # 1, ZYX, 3(xyz) --> 1, ZYX, 3(zyx)?
        shape = ind_norm.shape[:-1]
        ind_norm = ind_norm.reshape(1, 1, 1, -1, 3)
        
        if xyz_sem is None:
            grid = grids[0] # BCXYZ # torch.Size([1, C, 256, 256, 16])
            ret_lst = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1])
            if self.use_semantic:
                semantic, feats = ret_lst[..., :self.semantic_classes], ret_lst[..., -1]
                return feats, avail_mask, semantic
            else:
                return ret_lst.squeeze(), avail_mask

        else:

            ind_norm_sem = ind_norm_sem.flip((-1,))
            shape_sem = ind_norm_sem.shape[:-1]
            ind_norm_sem = ind_norm_sem.reshape(1, 1, 1, -1, 3)
            grid_sem = grids[0][:, :self.semantic_classes] # BCXYZ # torch.Size([1, semantic_classes, H, W, Z])
            grid_geo = grids[0][:, -1:] # BCXYZ # torch.Size([1, 1, H, W, Z])
            ret_sem = F.grid_sample(grid_sem, ind_norm_sem, mode='bilinear', align_corners=align_corners).reshape(grid_sem.shape[1], -1).T.reshape(*shape_sem, grid_sem.shape[1])
            ret_geo = F.grid_sample(grid_geo, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid_geo.shape[1], -1).T.reshape(*shape, grid_geo.shape[1])

            # pdb.set_trace()

            return ret_geo.squeeze(), avail_mask, ret_sem, avail_mask_sem, group_num, group_size


    def sample_ray(self, rays_o, rays_d, is_train):
        '''Sample query points on rays'''
        Zval = self.Zval.to(rays_o)
        if is_train:
            Zval = Zval.repeat(rays_d.shape[-2], 1)
            Zval += (torch.rand_like(Zval[:, [0]]) * 0.2 - 0.1) * self.stepsize_log * self.voxel_size
            Zval = Zval.clamp(min=0.0)

        Zval = Zval + self.near
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * Zval[..., None]
        rays_pts_depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)

        if self.opt.contracted_coord:
            # contracted coordiante has infinite perception range
            mask_outbbox = torch.zeros_like(rays_pts[..., 0]).bool()
        else:
            mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)

        return rays_pts, mask_outbbox, Zval, rays_pts_depth
    

    def effective_points_mask(self, points):
        '''Mask out points that are too close to each other in the contracted coordinate'''
        dist = torch.diff(points, dim=-2, prepend=torch.zeros_like(points[..., :1, :])).abs()
        xyz_thresh = 0.4 / torch.tensor([self.X, self.Y, self.Z]).to(points)
        mask = (dist > xyz_thresh).bool().any(dim=-1)
        return mask

    def activate_density(self, density, dists):
        return 1 - torch.exp(-F.relu(density) * dists)


    def get_density(self, Voxel_feat, is_train, inputs, cam_num):

        dtype = torch.float16

        if 'gs' in self.render_type:
            semantic = None
            reg_loss = {}
            
            K, C2W, pc = self.prepare_gs_attribute(Voxel_feat, inputs)

            # pdb.set_trace()
            depth, rgb_marched = self.get_splatting_rendering(K, C2W, pc, inputs)

            if self.use_semantic:
                semantic = torch.cat(rgb_marched, dim=0).permute(0, 2, 3, 1).contiguous()

            # if self.opt.infinite_range:
            #     depth = depth.clamp(min=self.near, max=200)
            # else:
            depth = depth.clamp(min=self.near, max=self.far)

            if self.weight_distortion > 0:
                loss_distortion = total_variation(Voxel_feat)
                reg_loss['loss_distortion'] = self.weight_distortion * loss_distortion
                
            return depth.float(), rgb_marched, semantic, reg_loss


        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

         # rendering
        rays_o = __u(inputs['rays_o', 0])
        rays_d = __u(inputs['rays_d', 0])

        device = rays_o.device
        
        rays_o, rays_d, Voxel_feat = rays_o.to(dtype), rays_d.to(dtype), Voxel_feat.to(dtype)

        reg_loss = {}
        eps_time = time.time()

        with torch.no_grad():
            rays_o_i = rays_o[0, ...].flatten(0, 2)  # HXWX3
            rays_d_i = rays_d[0, ...].flatten(0, 2)  # HXWX3
            rays_pts, mask_outbbox, z_vals, rays_pts_depth = self.sample_ray(rays_o_i, rays_d_i, is_train=is_train)

        dists = rays_pts_depth[..., 1:] - rays_pts_depth[..., :-1]  # [num pixels, num points - 1]
        dists = torch.cat([dists, 1e4 * torch.ones_like(dists[..., :1])], dim=-1)  # [num pixels, num points]

        sample_ret = self.grid_sampler(rays_pts, Voxel_feat, avail_mask=~mask_outbbox)


        # false
        if self.use_semantic:
            if self.semantic_sample_ratio < 1.0:
                geo_feats, mask, semantic, mask_sem, group_num, group_size = sample_ret

            else:
                geo_feats, mask, semantic = sample_ret

        else:
            geo_feats, mask = sample_ret


        if self.render_type == 'prob':
            weights = torch.zeros_like(rays_pts[..., 0])
            weights[:, -1] = 1
            geo_feats = torch.sigmoid(geo_feats)


            if self.last_free:
                geo_feats = 1.0 - geo_feats  
                # the last channel is the probability of being free
            
            weights[mask] = geo_feats

            # accumulate
            weights = weights.cumsum(dim=1).clamp(max=1)
            alphainv_fin = weights[..., -1]
            weights = weights.diff(dim=1, prepend=torch.zeros((rays_pts.shape[:1])).unsqueeze(1).to(device=device, dtype=dtype))
            depth = (weights * z_vals).sum(-1)
            rgb_marched = 0


        elif self.render_type == 'density':

            alpha = torch.zeros_like(rays_pts[..., 0])  # [num pixels, num points]
            alpha[mask] = self.activate_density(geo_feats, dists[mask])

            weights, alphainv_cum = render.get_ray_marching_ray(alpha)
            alphainv_fin = alphainv_cum[..., -1]
            depth = (weights * z_vals).sum(-1)
            rgb_marched = 0

        else:
            raise NotImplementedError
        
        # pdb.set_trace()
        if self.use_semantic:
            if self.semantic_sample_ratio < 1.0:
                semantic_out = torch.zeros(mask_sem.shape + (self.semantic_classes, )).to(device=device, dtype=dtype)
                semantic_out[mask_sem] = semantic
                weights_sem = weights[:, :group_num * group_size].reshape(weights.shape[0], group_num, group_size).sum(dim=-1)
                semantic_out = (semantic_out * weights_sem[..., None]).sum(dim=-2)
                
            else:
                semantic_out = torch.ones(rays_pts.shape[:-1] + (self.semantic_classes, )).to(device=device, dtype=dtype)
                semantic_out[mask] = semantic
                semantic_out = (semantic_out * weights[..., None]).sum(dim=-2)

            semantic_out = semantic_out.reshape(cam_num, self.render_h, self.render_w, self.semantic_classes)
        
        else:
            semantic_out = None

        if is_train:
            if self.weight_entropy_last > 0:
                pout = alphainv_fin.float().clamp(1e-6, 1-1e-6)
                entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
                reg_loss["loss_entropy_last"] = self.weight_entropy_last * entropy_last_loss

            if self.weight_distortion > 0:
                loss_distortion = eff_distloss(weights.float(), z_vals.float(), dists.float())
                reg_loss['loss_distortion'] =  self.weight_distortion * loss_distortion
            
            if self.weight_sparse_reg > 0:
                geo_f = Voxel_feat[..., -1].float().flatten()
                if self.last_free:
                    geo_f = - geo_f
                loss_sparse_reg = F.binary_cross_entropy_with_logits(geo_f, torch.zeros_like(geo_f), reduction='mean')
                reg_loss['loss_sparse_reg'] = self.weight_sparse_reg * loss_sparse_reg


        depth = depth.reshape(cam_num, self.render_h, self.render_w).unsqueeze(1) 
            
        if self.infinite_range:
            depth = depth.clamp(min=self.near, max=200)
        else:
            depth = depth.clamp(min=self.near, max=self.far)


        return depth.float(), rgb_marched, semantic_out, reg_loss


    def get_ego2pix_rt(self, features, pix_T_cams, cam0_T_camXs):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)
        Hf, Wf = features.shape[-2], features.shape[-1]

        if self.opt.view_trans == 'query' or self.opt.view_trans == 'query1':
            featpix_T_cams_ = pix_T_cams_
        else:

            sy = Hf / float(self.opt.height)
            sx = Wf / float(self.opt.width)
            # unproject image feature to 3d grid
            featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)


        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_
        ego2featpix = basic.matmul2(featpix_T_cams_, camXs_T_cam0_)


        return ego2featpix, camXs_T_cam0_, Hf, Wf

    def get_voxel(self, features, inputs):

        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

        meta_similarity = None
        meta_feature = None
        curcar2precar = None
        nextcam2curego = None

        # input_channel=64
        feature_size = self.input_channel
        
        Extrix_RT = inputs['pose_spatial'][:6]
        Intrix_K = inputs[('K', 0, 0)][:6]
 #
        Voxel_feat = self.feature2vox_simple(features[0], Intrix_K, Extrix_RT, __p, __u) # torch.Size([1, 64, 300, 300, 24])
            

        return Voxel_feat, meta_similarity, meta_feature, nextcam2curego, feature_size


    def prepare_gs_attribute(self, Voxel_feat_list, inputs, is_train = None, index = 0):
        """_summary_

        Args:
            Voxel_feat_list (multi_scale voxel_feats): B, C, W(X), H(Y), D(Z)
            inputs (_type_): _description_
            is_train (_type_, optional): _description_. Defaults to None.
            index (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        # prepare gaussian
        pc = {}

        vox_grid, Z, Y, X = self.gs_vox_util.get_voxel_grid(device=Voxel_feat_list[0].device) # B, ZYX, 3(xyz)

        # pdb.set_trace()
        if self.gs_sample != 0:
           
            sample_ret = self.grid_sampler(vox_grid, Voxel_feat_list)

            if self.use_semantic:
                geo_feats, mask, semantic = sample_ret
                gs_attribute = semantic.squeeze(0)

            else:
                geo_feats, mask = sample_ret
                gs_attribute = geo_feats.unsqueeze(1)
            
            vox_grid = vox_grid.squeeze(0)
            pc['get_xyz_grid'] = vox_grid

        else:
            vox_grid = vox_grid.squeeze(0) # N,3 
            pc['get_xyz_grid'] = vox_grid # B, ZYX, 3(xyz)
            out_channel = Voxel_feat_list.shape[1]
            gs_attribute = Voxel_feat_list.permute(0, 4, 3, 2, 1).reshape(-1, out_channel)  # 3XYZ -> ZYX3 -> (ZYX)3

        if self.last_free:
            geo_feats = torch.sigmoid(gs_attribute[:, -1:])
            geo_feats = 1 - geo_feats

        else:
            geo_feats = torch.sigmoid(gs_attribute[:, -1:])
            geo_feats = 1 - geo_feats

        pc['get_opacity'] = geo_feats


        if self.weight_entropy_last > 0:
            loss_entropy = -pc['get_opacity'] * torch.log(pc['get_opacity'] + 1e-8)
            loss_entropy = self.weight_entropy_last * loss_entropy.mean()
            self.outputs["loss_entropy_last"] = loss_entropy

        pc['flow']  = 0
        pc['get_scaling'] =  torch.zeros_like(pc['get_opacity'], device="cuda").repeat(1, 3)
        pc['get_rotation'] = torch.zeros_like(pc['get_opacity'], device="cuda").repeat(1, 4)

        # fix initialization
        pc['get_scaling'][...] = -self.gs_scale

        if 1: # 根据点到原点的距离调整缩放值
            point_distance = torch.linalg.norm(vox_grid, dim =1) # # 计算每个点到相机原点的欧几里得距离 越远的点就需要加大他的缩放倍率
            out_depth_mask_1 = point_distance > self.ZMAX * 1.5
            # out_depth_mask_2 = point_distance > self.max_depth
            out_depth_mask_2 = point_distance > self.far
            pc['get_scaling'][out_depth_mask_1]  =  -self.gs_scale * 4
            pc['get_scaling'][out_depth_mask_2]  =  -self.gs_scale * 8

        pc['get_rotation'][:, 0] = 1
        pc['active_sh_degree'] = 0
        pc['confidence'] = torch.ones_like(pc['get_opacity'])

        # pdb.set_trace()
        if self.use_semantic:
            pc['semantic'] = gs_attribute

        pc['get_features'] = torch.ones_like(pc['get_opacity']).repeat(1, 3)

        # render depth map of each view
        K = np.stack(inputs['K_render']) # (cam_N, 4, 4)

        # if self.opt.surround_view:
        #     C2W = inputs['surround_pose'].to('cpu').numpy()
        # else:
        C2W = np.stack(inputs['T_cam_2_lidar']) # (cam_N, 4, 4)

        return K, C2W, pc


    def get_splatting_rendering(self, K, C2W, pc, inputs, flow_index = (0, 0)):
        # if self.opt.flow != 'No':
        rgb_spaltting = []
        depth = []

        pc['get_xyz'] = pc['get_xyz_grid']  # this is the normal one
           
        # R_only = inputs['pose_spatial'].clone()
        # R_only = geom.safe_inverse(R_only)
        # R_only[:, :3, 3] = 0

        rgb_marched = None

        # depth
        for j in range (C2W.shape[0]):

            if self.use_semantic:
                # # ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
                pc_list = [ 'get_opacity', 'get_scaling', 'get_rotation', 'confidence', 'semantic', 'get_features', 'get_xyz']

                pc_i = {}
                pc_i['active_sh_degree'] = pc['active_sh_degree']

                # pdb.set_trace()

                if j == 0:
                    fov_mask = pc['get_xyz_grid'][:,0] > C2W[j][0, 3]
                elif j == 1 or j == 2:
                    fov_mask = pc['get_xyz_grid'][:,1] > C2W[j][1, 3]
                elif j == 3:
                    fov_mask = pc['get_xyz_grid'][:,0] < C2W[j][0, 3]
                elif j == 4 or j == 5:
                    fov_mask = pc['get_xyz_grid'][:,1] < C2W[j][1, 3]
                else:
                    print('out of index!')
                    exit()

                for key in pc_list:
                        pc_i[key] = pc[key][fov_mask]

            else:
                pc_i = pc
            
            # all_cam_center = inputs['all_cam_center']
            viewpoint_camera = geom.setup_opengl_proj(w = self.render_w, h = self.render_h, k = K[j], c2w = C2W[j],near=self.near, far=100)
            
            render_pkg = splatting_render(viewpoint_camera, pc_i, use_semantic=self.use_semantic, render_type=self.render_type)
            
            depth_i = render_pkg['depth']
            rgb_marched_i = render_pkg['render'].unsqueeze(0)
            
            rgb_spaltting.append(rgb_marched_i)
            depth.append(depth_i) 

        depth = torch.cat(depth, dim=0).unsqueeze(1)
        
        return depth, rgb_spaltting


    def get_semantic_gt_loss(self, voxel_semantics, pred, mask_camera):

        # pdb.set_trace()

        preds = pred[0, ...].permute(1, 2, 3, 0) # 200, 200, 16, 18

        if mask_camera is not None:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.opt.semantic_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()

            loss_occ=self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor = num_total_samples)

        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)

            # ce loss
            loss_occ = self.loss_occ(preds, voxel_semantics)


        loss_voxel_sem_scal = sem_scal_loss(preds, voxel_semantics)
        loss_voxel_geo_scal = geo_scal_loss(preds, voxel_semantics, non_empty_idx=17)
        loss_voxel_lovasz = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

        loss_geometry = loss_voxel_sem_scal + loss_voxel_geo_scal + loss_voxel_lovasz

        self.outputs[("loss_geometry", 0)] = loss_geometry

        self.outputs[("loss_gt_occ", 0)] = loss_occ * 100 + loss_geometry

        self.outputs["pred_occ_logits"] = pred

        self.outputs[('disp', 0)] = torch.ones(6, 1, self.render_h, self.render_w).to('cuda')

        return 

    def forward(self, 
                representation, 
                metas=None, 
                **kwargs):
        is_train = kwargs.get('is_train', True)
        
        self.outputs = {}
        self.scales = [0]
        # B, H+1，W+1, D+1, C --> B, C, W, H, D to fit gsocc code 
        representation = F.interpolate(representation.permute(0, 4, 2, 1, 3), [self.X, self.Y, self.Z], mode='trilinear', align_corners=True)
        # representation = F.interpolate(representation.permute(0, 4, 2, 1, 3), [256, 256, 32], mode='trilinear', align_corners=True)
        inputs = metas[0] # semantickitti 
        Voxel_feat_list = self._3DCNN(representation) # B, C, X, Y, Z
        # Voxel_feat_list = [representation] 
        
        # if Voxel_feat_list is None: 
        #     # 2D to 3D
        #     Voxel_feat, meta_similarity, meta_feature, nextcam2curego, feature_size = self.get_voxel(features, inputs)# B, C, X, Y, Z
        #     # 3D aggregation
        #     Voxel_feat_list = self._3DCNN(Voxel_feat) # B, C, X, Y, Z
    
        # pdb.set_trace()
        # if self.opt.render_type == 'gt':
        #     preds = Voxel_feat_list[0]
        #     voxel_semantics = inputs['semantics_3d']
        #     mask_camera = inputs['mask_camera_3d']
        #     self.get_semantic_gt_loss(voxel_semantics, preds, mask_camera)
        #     return self.outputs
        
        # # rendering
        rendering_eps_time = time.time()
        cam_num = 1
        
        for scale in self.scales:

            eps_time = time.time()

            depth, rgb_marched, semantic, reg_loss = self.get_density(Voxel_feat_list[scale], is_train, inputs, cam_num)

            eps_time = time.time() - eps_time

            # print('single rendering {} :(eps time:'.format(self.opt.render_type), eps_time, 'secs)')

            # self.outputs[("disp", scale)] = depth
            self.outputs["disp"] = depth

            if semantic is not None:
                self.outputs[("semantic", scale)] = semantic    

            if reg_loss != None:
                for k, v in reg_loss.items():
                    self.outputs[k] = v


        self.outputs['render_time'] = time.time() - rendering_eps_time


        if not is_train:
            if self.dataset == 'nusc':
                H, W, Z = 200, 200, 16
                xyz_min = [-40, -40, -1]
                xyz_max = [40, 40, 5.4]

            elif self.dataset == 'ddad':
                H, W, Z = 200, 200, 16
                xyz_min = [-40, -40, -1]
                xyz_max = [40, 40, 5.4]

            elif self.dataset == 'kitti':
                H, W, Z = 256, 256, 32
                xyz_min = [-25.6, 0, -2]
                xyz_max = [25.6, 51.2, 4.4]
            else:
                raise NotImplementedError

            # generate the occupancy grid for test
            # xyz = basic.gridcloud3d(1, Z, W, H, device='cuda').to(Voxel_feat_list[0]) # 1， ZYX， 3(xyz)
            # xyz_min = torch.tensor(xyz_min).to(xyz)
            # xyz_max = torch.tensor(xyz_max).to(xyz)
            # occ_size = torch.tensor([H, W, Z]).to(xyz)
            # xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * self.voxel_size
            # ret = self.grid_sampler(xyz, Voxel_feat_list[0], vis=True)

            # if self.use_semantic:
            #     pred_occ_logits = ret[2]
            # else:
            #     pred_occ_logits = ret[0] # ZYX

            # self.outputs["pred_occ_logits"] = pred_occ_logits.reshape(Z, W, H, -1).permute(3, 2, 1, 0).unsqueeze(0) # 1,1,H,W,Z
            self.outputs["pred_occ_logits"] = F.interpolate(Voxel_feat_list[0], [H, W, Z], mode='trilinear', align_corners=True)
        # pdb.set_trace()
        return self.outputs


def total_variation(v, mask=None):

    # pdb.set_trace()
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    # if mask is not None:
    #     tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
    #     tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
    #     tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

