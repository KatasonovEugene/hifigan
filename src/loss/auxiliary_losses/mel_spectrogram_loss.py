from torch import nn


class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_melspec, gt_melspec):
        return self.l1_loss(gen_melspec, gt_melspec)
