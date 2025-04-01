from mmseg.registry import MODELS
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmengine.model import ModuleList
import torch.nn as nn, torch, copy
from torch.nn.init import normal_
from torch.nn import functional as F
from mmengine.logging import MMLogger
logger = MMLogger.get_instance('selfocc')

from ..base_encoder import BaseEncoder
from ..bevformer.utils import point_sampling
from .utils import get_cross_view_ref_points, grid2meter
from ..bevformer.mappings import GridMeterMapping
from ..bevformer.attention import BEVCrossAttention, BEVDeformableAttention
from .attention import TPVCrossAttention, CrossViewHybridAttention
from .modules import CameraAwareSE
import numpy as np

@MODELS.register_module()
class TPVFormerEncoder(BaseEncoder):

    def __init__(
        self,
        mapping_args: dict,
        # bev_inner=128,
        # bev_outer=32,
        # range_inner=51.2,
        # range_outer=51.2,
        # nonlinear_mode='linear_upscale',
        # z_inner=20,
        # z_outer=10,
        # z_ranges=[-5.0, 3.0, 11.0],

        embed_dims=128,
        num_cams=6,
        num_feature_levels=4,
        positional_encoding=None,
        num_points_cross=[64, 64, 8],
        num_points_self=[16, 16, 16],
        transformerlayers=None, 
        num_layers=None,
        camera_aware=False,
        camera_aware_mid_channels=None,
        init_cfg=None,
        with_prev=False,
        align_after_tpv_encode=False,
        temporal_fuse_mode='conv',
        point_cloud_range=None,
        ):

        super().__init__(init_cfg)

        # self.bev_inner = bev_inner
        # self.bev_outer = bev_outer
        # self.range_inner = range_inner
        # self.range_outer = range_outer
        # assert nonlinear_mode == 'linear_upscale' # TODO
        # self.nonlinear_mode = nonlinear_mode
        # self.z_inner = z_inner
        # self.z_outer = z_outer
        # self.z_ranges = z_ranges
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.camera_aware = camera_aware
        self.with_prev = with_prev
        self.align_after_tpv_encode = align_after_tpv_encode
        self.temporal_fuse_mode = temporal_fuse_mode
        self.point_cloud_range=point_cloud_range
        self.return_mode = 'combine' if self.with_prev else '3p'
        
        if camera_aware:
            if camera_aware_mid_channels is None:
                camera_aware_mid_channels = embed_dims
            self.camera_se_net = CameraAwareSE(
                embed_dims,
                camera_aware_mid_channels,
                embed_dims)

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
        
        size_h = self.mapping.size_h # y
        size_w = self.mapping.size_w # x
        size_d = self.mapping.size_d # z
        
        hw_grid = torch.stack(
            [torch.arange(size_h, dtype=torch.float).unsqueeze(-1).expand(-1, size_w),
             torch.arange(size_w, dtype=torch.float).unsqueeze(0).expand(size_h, -1),
             torch.zeros(size_h, size_w)],
             dim=-1) # (H, W, 3(hw0)) 坐标顺序在后面的先增加
        hw_meter = self.mapping.grid2meter(hw_grid)[..., [0, 1]] # (H, W, 3(hw0)) --> (H, W, 2(wh))
        zh_grid = torch.stack(
            [torch.arange(size_h, dtype=torch.float).unsqueeze(0).expand(size_d, -1),
             torch.zeros(size_d, size_h),
             torch.arange(size_d, dtype=torch.float).unsqueeze(-1).expand(-1, size_h)],
             dim=-1) # (Z, H, 3(h0z)) 
        zh_meter = self.mapping.grid2meter(zh_grid)[..., [1, 2]] # (Z, H, 3(0hz)) --> (Z, H, 2(hz))
        wz_grid = torch.stack(
            [torch.zeros(size_w, size_d),
             torch.arange(size_w, dtype=torch.float).unsqueeze(-1).expand(-1, size_d),
             torch.arange(size_d, dtype=torch.float).unsqueeze(0).expand(size_w, -1)],
             dim=-1) # (W, Z, 3(0wz))
        wz_meter = self.mapping.grid2meter(wz_grid)[..., [0, 2]] # (W, Z, 3(w0z)) --> (W, Z, 2(wz))

        positional_encoding.update({'tpv_meters': [hw_meter, zh_meter, wz_meter]})
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.tpv_size = [size_h, size_w, size_d]

        # transformer layers
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.pre_norm = self.layers[0].pre_norm
        logger.info('use pre_norm: ' + str(self.pre_norm))
        
        # other learnable embeddings
        self.level_embeds = nn.Parameter(
            torch.randn(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.randn(self.num_cams, self.embed_dims))

        # prepare reference points used in image cross-attention and cross-view hybrid-attention
        self.num_points_cross = num_points_cross
        self.num_points_self = num_points_self

        uniform_d = torch.linspace(0, size_d - 1, num_points_cross[2])
        hw_3d_grid = torch.cat([
            hw_grid[..., [0, 1]].unsqueeze(2).expand(-1, -1, num_points_cross[2], -1),
            uniform_d.reshape(1, 1, -1, 1).expand(size_h, size_w, -1, -1)], dim=-1) # (H, W, P0, 3(hwz))
        ref_3d_hw = self.mapping.grid2meter(hw_3d_grid) # H, W, P0, 3(whz), whz = xyz

        uniform_w = torch.linspace(0, size_w - 1, num_points_cross[1])
        zh_3d_grid = torch.cat([
            zh_grid[..., :1].unsqueeze(2).expand(-1, -1, num_points_cross[1], -1),
            uniform_w.reshape(1, 1, -1, 1).expand(size_d, size_h, -1, -1),
            zh_grid[..., 2:].unsqueeze(2).expand(-1, -1, num_points_cross[1], -1)
        ], dim=-1) # (Z, H, P1, 3(hwz))
        ref_3d_zh = self.mapping.grid2meter(zh_3d_grid) # Z, H, P1, 3(whz)

        uniform_h = torch.linspace(0, size_h - 1, num_points_cross[0])
        wz_3d_grid = torch.cat([
            uniform_h.reshape(1, 1, -1, 1).expand(size_w, size_d, -1, -1),
            wz_grid[..., [1, 2]].unsqueeze(2).expand(-1, -1, num_points_cross[0], -1)
        ], dim=-1) # (W, Z, P2, 3(hwz)
        ref_3d_wz = self.mapping.grid2meter(wz_3d_grid) # W, Z, P2, 3(whz)

        # H, W, P0, 3(whz) -> H*W, P0, 3(whz) -> P0, H*W, 3(whz) permute和transpose操作并不会改变最后一维坐标的顺序
        self.register_buffer('ref_3d_hw', ref_3d_hw.flatten(0, 1).transpose(0, 1), False) 
        self.register_buffer('ref_3d_zh', ref_3d_zh.flatten(0, 1).transpose(0, 1), False)
        self.register_buffer('ref_3d_wz', ref_3d_wz.flatten(0, 1).transpose(0, 1), False)
        
        cross_view_ref_points = get_cross_view_ref_points(size_h, size_w, size_d, num_points_self)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points, False)
        # hw_grid_normed = hw_grid[..., [0, 1]].clone()
        # hw_grid_normed[..., 0] = hw_grid_normed[..., 0] / (size_h - 1)
        # hw_grid_normed[..., 1] = hw_grid_normed[..., 1] / (size_w - 1)

        # zh_grid_normed = zh_grid[..., [2, 0]].clone()
        # zh_grid_normed[..., 0] = zh_grid_normed[..., 0] / (size_d - 1)
        # zh_grid_normed[..., 1] = zh_grid_normed[..., 1] / (size_h - 1)

        # wz_grid_normed = wz_grid[..., [1, 2]].clone()
        # wz_grid_normed[..., 0] = wz_grid_normed[..., 0] / (size_w - 1)
        # wz_grid_normed[..., 1] = wz_grid_normed[..., 1] / (size_d - 1)

        # self.register_buffer('ref_2d_hw', hw_grid_normed, False) # H, W, 2
        # self.register_buffer('ref_2d_zh', zh_grid_normed, False) # H, W, 2
        # self.register_buffer('ref_2d_wz', wz_grid_normed, False) # H, W, 2
        
        # prepare grid points used in temporal tpv feature align
        if self.with_prev:
            # 1. 正确的写法
            hwz_grid = torch.stack(
                [torch.arange(size_h, dtype=torch.float).reshape(-1, 1, 1).expand(-1, size_w, size_d),
                 torch.arange(size_w, dtype=torch.float).reshape(1, -1, 1).expand(size_h, -1, size_d),
                 torch.arange(size_d, dtype=torch.float).reshape(1, 1, -1).expand(size_h, size_w, -1)],
                dim=-1) # H, W, D，3(hwz)
            grid_xyz_meter = self.mapping.grid2meter(hwz_grid) # H, W, D, 3(whz) 
            self.register_buffer('grid_xyz', grid_xyz_meter.permute(1, 0, 2, 3), False) # W, H, D, 3(whz)
            
            # 2. 
            # TODO 效果不好， 暂不清楚原因。
            # 因为在align时还需要进行坐标变换，变换矩阵要求的坐标顺序为xyz，这里给出的坐标顺序也应该是xyz
            # hwz_grid = torch.stack(
            #     [torch.arange(size_h, dtype=torch.float).reshape(-1, 1, 1).expand(-1, size_w, size_d),
            #      torch.arange(size_w, dtype=torch.float).reshape(1, -1, 1).expand(size_h, -1, size_d),
            #      torch.arange(size_d, dtype=torch.float).reshape(1, 1, -1).expand(size_h, size_w, -1)],
            #     dim=-1) # H, W, D，3(hwz)
            # grid_xyz_meter2 = grid2meter(hwz_grid) # H, W, D，3(hwz)
            # # self.register_buffer('grid_xyz', grid_xyz_meter, False) # H, W, D，3(hwz)
            # print((grid_xyz_meter2 == grid_xyz_meter).all())
            # print((grid_xyz_meter2-grid_xyz_meter)<1e-3)
            # print(((grid_xyz_meter2-grid_xyz_meter)<1e-3).all())
            
            
            
            # temporal feature fusion net： 1x1 conv
            if self.temporal_fuse_mode == 'conv':
                self.fusion_net = nn.Sequential(
                    nn.Conv3d(in_channels=2*self.embed_dims, out_channels=self.embed_dims, kernel_size=1),
                    nn.BatchNorm3d(self.embed_dims),
                    nn.ReLU(True)
                )
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, BEVCrossAttention) or \
                isinstance(m, MultiScaleDeformableAttention) or \
                    isinstance(m, BEVDeformableAttention) or \
                        isinstance(m, TPVCrossAttention) or \
                            isinstance(m, CrossViewHybridAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
    
    def forward_layers(
        self,
        tpv_query, # b, c, h, w
        key, # (num_cam, H*W++, bs, embed_dims)
        value,
        tpv_pos=None, # b, h, w, c
        spatial_shapes=None,
        level_start_index=None,
        img_metas=None,
        return_mode=None ,
        **kwargs
    ):
        bs = tpv_query[0].shape[0]

        reference_points_cams, tpv_masks = [], []
        for ref_3d in [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]:
            reference_points_cam, tpv_mask = point_sampling(
                ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1), img_metas)
            reference_points_cams.append(reference_points_cam) # num_cam, bs, hw++, #p, 2
            tpv_masks.append(tpv_mask)
        
        # ref_2d = self.ref_2d.unsqueeze(0).repeat(bs, 1, 1, 1) # bs, H, W, 2
        # ref_2d = ref_2d.reshape(bs, -1, 1, 2) # bs, HW, 1, 2
        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(
            0).expand(bs, -1, -1, -1, -1) # bs, hw++, 3, #p, 2

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks,
                tpv_size=self.tpv_size,
                **kwargs)
            tpv_query = output

        if return_mode == 'combine':
            # 融合三个平面的特征
            size_h, size_w, size_z = self.tpv_size[0], self.tpv_size[1], self.tpv_size[2]
            tpv_hw, tpv_zh, tpv_wz = tpv_query
            tpv_hw = tpv_hw.reshape(-1, size_h, size_w, 1, self.embed_dims)
            tpv_hw = tpv_hw.expand(-1, -1, -1, size_z, -1)

            tpv_zh = tpv_zh.reshape(-1, size_z, size_h, 1, self.embed_dims).permute(0, 2, 3, 1, 4)
            tpv_zh = tpv_zh.expand(-1, -1, size_w, -1, -1)

            tpv_wz = tpv_wz.reshape(-1, size_w, size_z, 1, self.embed_dims).permute(0, 3, 1, 2, 4)
            tpv_wz = tpv_wz.expand(-1, size_h, -1, -1, -1)

            tpv = tpv_hw + tpv_zh + tpv_wz # (B, H, W, Z, C)

            return tpv
        return tpv_query
        
    def forward(
        self,         
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        """Forward function.
        Args:
            img_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
                
        return:
            representation (Tensor): The output of the encoder.
                Has shape (B, H, W, Z, C).
        """
        
        # 拆分当前帧和前一帧的特征
        if self.with_prev:
            N = ms_img_feats[0].shape[1]
            # 确保 N 是3的倍数
            assert N % 3 == 0, "N must be an even number to split into two equal parts."
            # 拆分张量
            split_size = N // 3
            img_feats, prev_img_feats, next_img_feats = zip(*[torch.split(feat, split_size, dim=1) for feat in ms_img_feats])
            img_feats = list(img_feats)
            prev_img_feats = list(prev_img_feats)
            next_img_feats = list(next_img_feats)
        else:
            img_feats = ms_img_feats
            
        bs = img_feats[0].shape[0]
        dtype = img_feats[0].dtype
        device = img_feats[0].device

        # bev queries and pos embeds 
        tpv_queries = representation # bs, HW, C
        tpv_pos = self.positional_encoding()
        tpv_pos = [pos.unsqueeze(0).repeat(bs, 1, 1) for pos in tpv_pos] # 这里重复
        
        # add camera awareness if required
        if self.camera_aware:
            img_feats = self.camera_se_net(img_feats, metas)
        
        def get_tpv_feat(img_feats):
            # flatten image features of different scales
            feat_flatten = []
            spatial_shapes = []
            for lvl, feat in enumerate(img_feats):
                bs, num_cam, c, h, w = feat.shape
                spatial_shape = (h, w)
                feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
                feat = feat + self.cams_embeds[:, None, None, :]#.to(dtype)
                feat = feat + self.level_embeds[None, None, lvl:lvl+1, :]#.to(dtype)
                spatial_shapes.append(spatial_shape)
                feat_flatten.append(feat)
                    
            feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=device)
            level_start_index = torch.cat((spatial_shapes.new_zeros(
                (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

            # forward layers
            tpv_embed = self.forward_layers(
                tpv_queries,
                feat_flatten,
                feat_flatten,
                tpv_pos=tpv_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                img_metas=metas,
                return_mode=self.return_mode
            ) # tuple (tpv_hw, tpv_zh, tpv_wz)
            return tpv_embed
        
        tpv_embed = get_tpv_feat(img_feats)
        if self.with_prev:
            prev_tpv_embed = get_tpv_feat(prev_img_feats)
            next_tpv_embed = get_tpv_feat(next_img_feats)  
                      
            # 特征对齐+时序融合
            if self.align_after_tpv_encode:
                prev_tpv_embed = self.align_feature(input=prev_tpv_embed, img_metas=metas) # B, H, W, Z, C
            
            if self.temporal_fuse_mode == 'conv':
                fused_tpv_feature = torch.cat([tpv_embed, prev_tpv_embed], dim=-1).permute(0, 4, 1, 2, 3) # B, 2C, H, W, Z
                fused_tpv_feature = self.fusion_net(fused_tpv_feature).permute(0, 2, 3, 4, 1) # B, H, W, Z, 2C
            elif self.temporal_fuse_mode == 'add':
                fused_tpv_feature = tpv_embed + prev_tpv_embed
            elif self.temporal_fuse_mode == 'None':
                fused_tpv_feature = tpv_embed
            else:
                raise NotImplementedError
            
            # fused_tpv_feature = F.interpolate(
            #     fused_tpv_feature,
            #     size=(257, 257, 33),
            #     mode='trilinear',
            #     align_corners=False
            # ).permute(0, 2, 3, 4, 1)
            # tpv_embed = F.interpolate(
            #     tpv_embed.permute(0, 4, 1, 2, 3),
            #     size=(257, 257, 33),
            #     mode='trilinear',
            #     align_corners=False
            # ).permute(0, 2, 3, 4, 1)
            # prev_tpv_embed = F.interpolate(
            #     prev_tpv_embed.permute(0, 4, 1, 2, 3),
            #     size=(257, 257, 33),
            #     mode='trilinear',
            #     align_corners=False
            # ).permute(0, 2, 3, 4, 1)
            # next_tpv_embed = F.interpolate(
            #     next_tpv_embed.permute(0, 4, 1, 2, 3),
            #     size=(257, 257, 33),
            #     mode='trilinear',
            #     align_corners=False
            # ).permute(0, 2, 3, 4, 1)
            return {'representation': fused_tpv_feature, # fused_feature (B, H, W, Z, C)
                    'curr_rep': tpv_embed,
                    'prev_rep': prev_tpv_embed,
                    'next_rep': next_tpv_embed,
                    }
        
        return {
            'representation': tpv_embed,
            # 'prev_rep': prev_tpv_embed,
            # 'next_rep': next_tpv_embed,
            }
    @torch.cuda.amp.autocast(enabled=False)
    def align_feature(self, input, img_metas):
        """
        Converts 3D tpv points from current coordinates to previous coordinates.

        Parameters:
        - input: previous tpv feature. # (B, H, W, Z, C)
        - tpv_grid: 3D tpv points in current lidar coordinates.  # B, W, H, D, 3(whz/xyz)
        - tpv_range: Range of tpv points in lidar coordinates. [X,Y,Z]
        - img_metas: List of image metadata, containing calibration information.

        Returns:
        - aligned_tpv_grid: 3D tpv points in previous image coordinates. # (B, H, W, Z, C)
        - tpv_mask: Validity mask for the reference points.
        """
        B, H, W, Z, C = input.shape
        
        tpv_grid = self.grid_xyz.unsqueeze(0).repeat(B, 1, 1, 1, 1).float() # [B, W, H, D, 3(whz/xyz)]

        lidar2prevLidar = []
        for img_meta in img_metas:
            lidar2prevLidar.append(img_meta['lidar2prevLidar'])
        if isinstance(lidar2prevLidar[0], (np.ndarray, list)):
            lidar2prevLidar = np.asarray(lidar2prevLidar) # (1, 1, 4, 4)
            lidar2prevLidar = tpv_grid.new_tensor(lidar2prevLidar)  # (1, 1, 4, 4) .new_tensor继承device和type
        else:
            lidar2prevLidar = torch.stack(lidar2prevLidar, dim=0)
        
        # Homogeneous coordinate transformation for reference_points
        tpv_grid = torch.cat(
            (tpv_grid, torch.ones_like(tpv_grid[..., :1])), -1) # (B,W,H,D,4) H, W, D, 3(hwz)

        tpv_grid = tpv_grid.unsqueeze(-1) # (B, X, Y, Z, 4, 1)

        lidar2prevLidar = lidar2prevLidar.view(
            1, 1, 1, 1, 4, 4)

        tpv_grid_prev = torch.matmul(
            lidar2prevLidar.to(torch.float32),
            tpv_grid.to(torch.float32)).squeeze(-1) # 进行坐标变换时坐标顺序是否得是xyz1? (B,X,Y,Z,4)
               
        tpv_grid_prev = tpv_grid_prev[..., 0:3] # (B,X,Y,Z,3)

        # [-25.6, 0.0, -2.0, 25.6, 51.2, 4.4]
        tpv_range = self.point_cloud_range # (XMIN, YMIN, ZMIN，XMAX, YMAX, ZMAX)
        # normalize tpv_grid_prev to [-1, 1] for grid_sample operation
        tpv_grid_prev[..., 0] = (tpv_grid_prev[..., 0] - tpv_range[0]) / (tpv_range[3] - tpv_range[0]) * 2.0 - 1.0
        tpv_grid_prev[..., 1] = (tpv_grid_prev[..., 1] - tpv_range[1]) / (tpv_range[4] - tpv_range[1]) * 2.0 - 1.0
        tpv_grid_prev[..., 2] = (tpv_grid_prev[..., 2] - tpv_range[2]) / (tpv_range[5] - tpv_range[2]) * 2.0 - 1.0

        tpv_mask = ((tpv_grid_prev[..., 0] > -1.0) & (tpv_grid_prev[..., 0] < 1.0 ) &
                    (tpv_grid_prev[..., 1] > -1.0) & (tpv_grid_prev[..., 1] < 1.0 ) &
                    (tpv_grid_prev[..., 2] > -1.0) & (tpv_grid_prev[..., 2] < 1.0 )) 
        
        # tpv_mask = torch.nan_to_num(tpv_mask)
        
        # input_shape: 
        #   1. (B, H, W, Z, C) --> (B, C, W, H, Z) .permute(0, 4, 2, 1, 3)
        #   2. (B, H, W, Z, C) --> (B, C, H, W, Z) .permute(0, 4, 1, 2, 3)
        # grid_shape: 
        #   1. B, W, H, Z, 3(whz) --> B, W, H, Z, 3(zhw)
        #   2. B, H, W, Z, 3(hwz) --> B, H, W, Z, 3(zwh)
        # output_shape: 
        #   1. B, C, W, H, Z --> B, H, W, Z, C .permute(0, 3, 2, 4, 1)
        #   2. B, C, H, W, Z --> B, H, W, Z, C .permute(0, 2, 3, 4, 1)
        output = F.grid_sample(input.permute(0, 4, 2, 1, 3), tpv_grid_prev.to(input.dtype).flip(-1), align_corners=True) * tpv_mask
        return output.permute(0, 3, 2, 4, 1) # B, C, W, H, Z --> B, H, W, Z, C
        # output = F.grid_sample(input.permute(0, 4, 1, 2, 3), tpv_grid_prev.to(input.dtype).flip(-1), align_corners=True) * tpv_mask
        # return output.permute(0, 2, 3, 4, 1) 