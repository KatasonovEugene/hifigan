from torch import nn


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_feats, gt_feats):
        loss = 0
        for gen_feat, gt_feat in zip(gen_feats, gt_feats):
            loss += self.l1_loss(gen_feat, gt_feat) 
        return loss
