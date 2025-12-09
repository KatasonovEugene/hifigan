import torch
from torch import nn


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_preds):
        loss = 0.0
        for gen_pred in gen_preds:
            loss += ((gen_pred - 1) ** 2).mean()
        return loss


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_preds, gt_preds):
        loss = 0.0
        for gen_pred, gt_pred in zip(gen_preds, gt_preds):
            loss += ((gt_pred - 1) ** 2 + gen_pred ** 2).mean()
        return loss
