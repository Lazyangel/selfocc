from .base_loss import BaseLoss
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class EntropyLastLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'loss_entropy_last': 'loss_entropy_last',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.entropy_last_loss
    
    def entropy_last_loss(self, loss_entropy_last):
        # only use to return loss compute in gshead
        return loss_entropy_last