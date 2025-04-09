import torch, torch.nn as nn
import numpy as np
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone
from mmengine.registry import MODELS

from .base_segmentor import CustomBaseSegmentor

@SEGMENTORS.register_module()
class TPVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def extract_img_feat(self, imgs, metas, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if metas[0]['flip']:
                # img_feat = torch.fliplr(img_feat)
                img_feat = torch.flip(img_feat, [-1])
            if self.use_post_fusion:
                img_feats_reshaped.append(img_feat.unsqueeze(1))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            if metas[0]['flip']:
                img_feat_backbone = torch.flip(img_feat_backbone, [-1])
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return {
            'ms_img_feats_backbone': img_feats_backbone_reshaped,
            'ms_img_feats': img_feats_reshaped}
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped
    
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                img_feat_only=False,
                extra_backbone=False,
                occ_only=False,
                prepare=False,
                is_train=True,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
                
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'is_train': is_train,
        }
        results.update(kwargs) # kwargs 更新到 results
        outs = self.extract_img_feat(**results)
        # outs['ms_img_feats'] = [feat.float() for feat in outs['ms_img_feats']]
        results.update(outs)
        if img_feat_only:
            return results['ms_img_feats_backbone']
        # with torch.cuda.amp.autocast(enabled=False):
        outs = self.lifter(**results)# TPV query lifter 
        results.update(outs)
        outs = self.encoder(**results) 
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        elif prepare and hasattr(self.head, "prepare"):
            outs = self.head.prepare(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results
    
@SEGMENTORS.register_module()
class TPVSegmentor4D(TPVSegmentor):
    def __init__(
        self,
        align_after_view_transfromation=False,
        num_adj=1,
        with_prev=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        
        
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                img_feat_only=False,
                extra_backbone=False,
                occ_only=False,
                prepare=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
                
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
        }
        results.update(kwargs) # kwargs 更新到 results
        outs = self.extract_img_feat(**results)
        # outs['ms_img_feats'] = [feat.float() for feat in outs['ms_img_feats']]
        results.update(outs)
        if img_feat_only:
            return results['ms_img_feats_backbone']
        # with torch.cuda.amp.autocast(enabled=False):
        outs = self.lifter(**results)# TPV query lifter 
        results.update(outs)
        outs = self.encoder(**results) 
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        elif prepare and hasattr(self.head, "prepare"):
            outs = self.head.prepare(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results