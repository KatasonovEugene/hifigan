from torch import nn

from src.loss.auxiliary_losses.adversarial_loss import GeneratorAdversarialLoss
from src.loss.auxiliary_losses.feature_matching_loss import FeatureMatchingLoss
from src.loss.auxiliary_losses.mel_spectrogram_loss import MelSpectrogramLoss


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        lambda_fm,
        lambda_mel,
    ):
        super().__init__()
        self.adv_loss = GeneratorAdversarialLoss()
        self.fm_loss = FeatureMatchingLoss()
        self.mel_spec_loss = MelSpectrogramLoss()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def forward(self, gen_preds, gen_feats, gt_feats, gen_melspec, gt_melspec, **batch):
        adv_loss = self.adv_loss(gen_preds)
        fm_loss = self.fm_loss(gen_feats, gt_feats)
        mel_spec_loss = self.mel_spec_loss(gen_melspec, gt_melspec)
        gen_loss = adv_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_spec_loss

        return {
            "adv_loss": adv_loss,
            "fm_loss": fm_loss,
            "mel_spec_loss": mel_spec_loss,
            "gen_loss": gen_loss,
        }
