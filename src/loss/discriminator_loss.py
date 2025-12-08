from torch import nn

from src.loss.auxiliary_losses.adversarial_loss import DiscriminatorAdversarialLoss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adv_loss = DiscriminatorAdversarialLoss()

    def forward(self, gen_preds, gt_preds, **batch):
        return {"disc_loss": self.adv_loss(gen_preds, gt_preds)}
