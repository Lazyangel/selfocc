_base_ = [
    '../_base_/dataset_v1.py',
    '../_base_/optimizer.py',
    '../_base_/schedule.py',
]

img_size = [352, 1216]
# selfocc
num_rays = [55, 190] 
# gsocc
# num_rays = [96, 320]
amp = False
max_epochs = 24
warmup_iters = 1000
num_cams = 1
render_type='3dgs'
two_stage_opt = False

optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
        # eps=1e-4
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),}
    ),
)


data_path = 'data/kitti/'

train_dataset_config = dict(
    _delete_=True,
    type='Kitti_One_Frame',
    split = 'train',
    root = data_path,
    preprocess_root = data_path + 'preprocess',
    frames_interval=0.4,
    sequence_distance=[10, 40],
    cur_prob = 0.333,
    crop_size = img_size,
    strict = True,
    prev_prob = 0.2,
    choose_nearest = True,
    render_type = render_type,
    render_h = num_rays[0],
    render_w = num_rays[1],
)
    
val_dataset_config = dict(
    _delete_=True,
    type='Kitti_One_Frame',
    split = 'val',
    root = data_path,
    preprocess_root = data_path + 'preprocess',
    frames_interval=0.4,
    sequence_distance=[10, 40],
    cur_prob = 1.0,
    crop_size = img_size,
    strict = False,
    prev_prob = 0.2,
    choose_nearest = True,
    return_depth = True,
    render_type = render_type,
    render_h = num_rays[0],
    render_w = num_rays[1],
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes_temporal',
    phase='train', 
    scale_rate=1.0,
    photometric_aug=dict(
        use_swap_channel=False,
    ),
    img_norm_cfg=dict(
        mean=[124.16, 116.74, 103.94], 
        std=[58.624, 57.344, 57.6], to_rgb=True)
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes_temporal',
    phase='val', 
    scale_rate=1.0,
    photometric_aug=dict(
        use_swap_channel=False,
    ),
    img_norm_cfg=dict(
        mean=[124.16, 116.74, 103.94], 
        std=[58.624, 57.344, 57.6], to_rgb=True)
)

train_loader = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)
    
val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReprojLoss',
            weight=1.0,
            no_ssim=False,
            img_size=img_size,
            ray_resize=num_rays,
            input_dict={
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'metas': 'metas',
                'disp': 'disp',
                # 'deltas': 'deltas'
                }),
        dict(
            type='EntropyLastLoss',
            weight=1.0,
            input_dict={
                'loss_entropy_last': 'loss_entropy_last',
                }),
        dict(
            type='DistortionLoss',
            weight=1.0,
            input_dict={
                'loss_distortion': 'loss_distortion',
                }),
        # dict(
        #     type='SparsityLoss',
        #     weight=0.001,
        #     scale=0.1,
        #     input_dict={
        #         'density': 'uniform_sdf'}),
        ])

loss_input_convertion = dict(
    disp="disp",
    loss_entropy_last = "loss_entropy_last",
    loss_distortion = "loss_distortion",
    # deltas='deltas'
)

load_from = ''

_dim_ = 96
_ffn_dim_ = 2 * _dim_
num_heads = 6
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
# bev_inner = 160
# bev_outer = 1
# range_inner = 80.0
# range_outer = 1.0
# nonlinear_mode = 'linear_upscale'
# z_inner = 20
# z_outer = 10
# z_ranges = [-4.0, 4.0, 12.0]
tpv_h = 1 + 256
tpv_w = 1 + 2 * 128
tpv_z = 1 + 32 + 0
point_cloud_range = [-25.6, 0.0, -2.0, 25.6, 51.2, 4.4]

num_points_cross = [48, 48, 8]
num_points_self = 12

# gs
# gs_voxel_size = [16, 200, 200]
# gs_point_cloud_range = [-40, 0, -2, 40, 80, 4.4]
gs_voxel_size = [16, 128, 128]
gs_point_cloud_range = [-25.6, 0.0, -2.0, 25.6, 51.2, 4.4]

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='CrossViewHybridAttention',
            embed_dims=_dim_,
            num_heads=num_heads,
            num_levels=3,
            num_points=num_points_self,
            dropout=0.1,
            batch_first=True),
        dict(
            type='TPVCrossAttention',
            embed_dims=_dim_,
            num_cams=num_cams,
            dropout=0.1,
            batch_first=True,
            num_heads=num_heads,
            num_levels=4,
            num_points=num_points_cross)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)


model = dict(
    type='TPVSegmentor',
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_eval=False,
        style='pytorch',
        pretrained='./ckpts/resnet50-0676ba61.pth'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    lifter=dict(
        type='TPVQueryLifter',
        tpv_h=tpv_h,
        tpv_w=tpv_w,
        tpv_z=tpv_z, 
        dim=_dim_),
    encoder=dict(
        type='TPVFormerEncoder',
        # bev_inner=bev_inner,
        # bev_outer=bev_outer,
        # range_inner=range_inner,
        # range_outer=range_outer,
        # nonlinear_mode=nonlinear_mode,
        # z_inner=z_inner,
        # z_outer=z_outer,
        # z_ranges=z_ranges,
        mapping_args=mapping_args,

        embed_dims=_dim_,
        num_cams=num_cams,
        num_feature_levels=4,
        positional_encoding=dict(
            type='TPVPositionalEncoding',
            num_freqs=[12] * 3, 
            embed_dims=_dim_, 
            tot_range=point_cloud_range),
        num_points_cross=num_points_cross,
        num_points_self=[num_points_self] * 3,
        transformerlayers=[
            self_cross_layer,
            self_cross_layer,
            self_cross_layer,
            self_cross_layer], 
        num_layers=4),
    head=dict(
        type='GSHead',
        min_depth=0.1,
        max_depth=80,
        real_size=gs_point_cloud_range,
        voxels_size=gs_voxel_size,
        stepsize=0.5,
        input_channel=96,
        position='embedding',
        render_type='3dgs',
        gs_sample=0,
        render_h=num_rays[0],
        render_w=num_rays[1],
        ))