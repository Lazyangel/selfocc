from .base_loss import BaseLoss
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class DistortionLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'loss_distortion': 'loss_distortion',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.distortion_loss
    
    def distortion_loss(self, loss_distortion):
        # only use to return loss compute in gshead
        return loss_distortion