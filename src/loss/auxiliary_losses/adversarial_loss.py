import torch
from torch import nn


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, gen_preds):
        loss = 0.0
        for gen_pred in gen_preds:
            gen_true_pred = torch.ones_like(gen_pred)
            loss = self.l2_loss(gen_pred, gen_true_pred)
        return loss


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, gen_preds, gt_preds):
        loss = 0.0
        for gen_pred, gt_pred in zip(gen_preds, gt_preds):
            gt_true_pred = torch.ones_like(gt_pred)
            gen_true_pred = torch.zeros_like(gen_pred)
            loss += self.l2_loss(gt_pred, gt_true_pred) + self.l2_loss(gen_pred, gen_true_pred)
        return loss
