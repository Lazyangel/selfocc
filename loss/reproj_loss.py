import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F
from model.encoder.bevformer.mappings import GridMeterMapping
from utils.layers import *

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
class ReprojLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'metas': 'metas',}
        else:
            self.input_dict = input_dict
        self.no_ssim = kwargs.get('no_ssim', False)
        self.img_size = kwargs.get('img_size', [352, 1216])
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
        
        self.disparity_smoothness = kwargs.get('disparity_smoothness', 0.001)
        if not self.no_ssim:
            self.ssim = SSIM()
        
        self.frame_ids = [0, -1, 1]
        self.loss_func = self.reproj_loss
        self.iter_counter = 0

        self.backproject_depth = {}
        self.project_3d = {}
        
        self.scales = [0]
        for scale in self.scales:
            h = self.img_size[0] // (2 ** scale)
            w = self.img_size[1] // (2 ** scale)

            num_cam = 1
            self.backproject_depth[scale] = BackprojectDepth(num_cam, h, w)
            # self.backproject_depth[scale].to(self.device)
            self.backproject_depth[scale].cuda()

            self.project_3d[scale] = Project3D(num_cam, h, w)
            # self.project_3d[scale].to(self.device)
            self.project_3d[scale].cuda()
        
    def reproj_loss(
            self, 
            curr_imgs, 
            prev_imgs, 
            next_imgs, 
            metas, 
            disp,
            deltas=None,
            curr2prev_flow=None, # 在当前lidar坐标系中的移动
            curr2next_flow=None,):
        # curr_imgs: B, N, C, H, W
        # depth: B, N, R
        # rays: R, 2
        # curr2prev_flow: B, 3, H, W, Z
        # prev_warp_list: list[num_cam] B, N, R, S, 3 lidar coord
        # import pdb; pdb.set_trace()
        device = curr_imgs.device
        bs, num_cams = curr_imgs.shape[:2]

        assert bs == 1
        # prepare transformation matrices
        cam2prevcam, cam2nextcam = [], []
        K, inv_K = [], []
        for meta in metas:
            cam2prevcam.append(meta[("cam_T_cam", -1)])
            cam2nextcam.append(meta[("cam_T_cam", 1)])
            K.append(meta[("K", 0, 0)])
            inv_K.append(meta[("inv_K", 0, 0)])

        def list2tensor(trans):
            if isinstance(trans[0], (np.ndarray, list)):
                trans = np.asarray(trans)
                trans = curr_imgs.new_tensor(trans) # B, 36(6tem * 6cur), 4, 4
            else:
                trans = torch.stack(trans, dim=0)
            # trans = trans.reshape(bs, num_cams, num_cams, 1, 4, 4)
            trans = trans.reshape(num_cams, 4, 4)
            # trans = trans.transpose(1, 2)
            return trans

        cam2prevcam = list2tensor(cam2prevcam)
        cam2nextcam = list2tensor(cam2nextcam)
        K = list2tensor(K)
        inv_K = list2tensor(inv_K)
        
        outputs = {
            ("disp", 0): disp, # B,N,H,W
        }
        inputs = {
            ("cam_T_cam", -1): cam2prevcam, # (1, 4, 4)
            ("cam_T_cam", 1): cam2nextcam, # (1, 4, 4)
            ("K", 0, 0): K, # (1, 4, 4)     
            ("inv_K", 0, 0): inv_K, # (1, 4, 4)    
            ("color", 0, 0): curr_imgs.squeeze(0), # (bs, num_cams, 3, H, W) --> (bs, num_cams, 3, H, W)
            ("color", 1, 0): next_imgs.squeeze(0), 
            ("color", -1, 0): prev_imgs.squeeze(0),
        }
        
        def generate_images_pred(inputs, outputs):
            """Generate the warped (reprojected) color images for a minibatch.
            Generated images are saved into the `outputs` dictionary.
            """
            for scale in self.scales:
                disp = outputs[("disp", scale)]
                disp = F.interpolate(
                    disp, [self.img_size[0], self.img_size[1]], mode="bilinear", align_corners=False)
                if scale == 0:
                    outputs[("disp", scale)] = disp
                source_scale = 0

                # if self.opt.volume_depth:
                #     depth = disp
                # else:
                #     depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth, abs=False)
                depth = disp
                
                outputs[("depth", 0, scale)] = depth


                for i, frame_id in enumerate(self.frame_ids[1:]):

                    T = inputs[("cam_T_cam", frame_id)] # curr_2_prev/next(cam)

                    cam_points = self.backproject_depth[source_scale]( # 1, 4, H*W
                        depth, inputs[("inv_K", 0, source_scale)])
                    
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", 0, source_scale)], T)

                    
                    outputs[("sample", frame_id, scale)] = pix_coords # 1, H, W, 2

                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True) #  1, 3, H, W

                    
                    if not self.no_automask:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]
        
        def compute_reprojection_loss(pred, target):
            """Computes reprojection loss between a batch of predicted and target images
            """
            abs_diff = torch.abs(target - pred)
            l1_loss = abs_diff.mean(1, True)

            if self.no_ssim:
                reprojection_loss = l1_loss
            else:
                ssim_loss = self.ssim(pred, target).mean(1, True)
                reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

            return reprojection_loss


        def compute_self_supervised_losses(inputs, outputs,  losses = {}):
            """Compute the reprojection and smoothness losses for a minibatch
            """
            # losses = {}
            total_loss = 0

            for scale in self.scales:
                loss = 0
                reprojection_losses = []
                if self.mot_mask:
                    output_mask = []

                source_scale = 0

                # pdb.set_trace()
                disp = outputs[("disp", scale)] # bs, num_cams, H, W

                # print('Scale {},  disp min {}, max {}'.format(scale, disp[0].min(), disp[0].max()))
                min_depth = disp[0].min()
                max_depth = disp[0].max()


                # if self.opt.volume_depth:  # in fact, it is depth
                disp = 1.0 / (disp + 1e-7)

                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]  
                # 多帧监督的时候的target frame 是 【-1， 0， 1】
                
                for frame_id in self.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.no_automask:
                    identity_reprojection_losses = []
                    for frame_id in self.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            compute_reprojection_loss(pred, target))

                    identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)


                elif self.mot_mask:
                    # use the mot mask
                    # TODO
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        mask = F.interpolate(
                            mask, [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)

                    reprojection_losses *= mask

                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

                if self.mot_mask:
                    reprojection_losses *= inputs["mask"] #* output_mask


                reprojection_loss = reprojection_losses

                if not self.no_automask:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).cuda() * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                # pdb.set_trace()
                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if not self.no_automask:
                    outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

                loss += to_optimise.mean()

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)

                smooth_loss = get_smooth_loss(norm_disp, color)

                loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
                
                losses[f"loss_pe/{scale}"] = loss

                semantic_loss = 0.0

                # if self.use_semantic and scale == 0:
                #     # pdb.set_trace()
                #     pred_semantic = outputs[("semantic", 0)].float()
                #     target_semantic = inputs["semantic"]
                    

                #     # target_semantic[target_semantic > 0] = target_semantic[target_semantic > 0] - 1
                #     target_semantic[target_semantic > 0] = target_semantic[target_semantic > 0]
                    
                #     target_semantic = F.interpolate(target_semantic.unsqueeze(1).float(), size=pred_semantic.shape[1:3], mode="nearest").squeeze(1)
                    
                #     # pdb.set_trace()
                #     if self.use_fix_mask:
                #         ddad_mask = F.interpolate(inputs["mask"][:,:1, :, :], size=pred_semantic.shape[1:3], mode="nearest").squeeze(1)
                #         ddad_mask = ddad_mask.bool()
                #         semantic_loss += self.sem_criterion(pred_semantic[ddad_mask], target_semantic[ddad_mask].long())
                    
                #     else:
                #         # pdb.set_trace()
                #         semantic_loss += self.sem_criterion(pred_semantic.view(-1, self.opt.semantic_classes), target_semantic.view(-1).long())
                    
                #     semantic_loss = self.opt.semantic_loss_weight * semantic_loss
                #     losses[f"loss_semantic/{scale}"] = semantic_loss
                

                # loss_reg = 0
                # for k, v in outputs.items():
                #     if isinstance(k, tuple) and k[0].startswith("loss") and k[1] == scale:
                #         losses[f"{k[0]}/{k[1]}"] = v
                #         loss_reg += v
                
                total_loss += loss + semantic_loss
                # losses["loss/{}".format(scale)] = loss + loss_reg + semantic_loss


            losses["self_loss"] = total_loss

            losses["min_d"] = min_depth
            losses["max_d"] = max_depth

            return losses   
         
        losses = {}
        tot_loss = 0.
        generate_images_pred(inputs, outputs)
        compute_self_supervised_losses(inputs, outputs, losses)

        tot_loss += losses["self_loss"]
        self.iter_counter += 1
        return tot_loss / num_cams