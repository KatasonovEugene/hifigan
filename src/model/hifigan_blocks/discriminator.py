import torch.nn as nn
import torch.nn.functional as F

from src.model.hifigan_blocks.mpdiscriminator import MPDiscriminator
from src.model.hifigan_blocks.msdiscriminator import MSDiscriminator


class Discriminator(nn.Module):
    def __init__(
        self,
        mpdiscriminator: MPDiscriminator,
        msdiscriminator: MSDiscriminator,
    ):
        super().__init__()
        self.mpdiscriminator = mpdiscriminator
        self.msdiscriminator = msdiscriminator

    def forward(self, gt_audio, gen_audio):
        gt_audio, gen_audio = self._pad_audio(gt_audio, gen_audio)

        gt_mpd_preds, gt_mpd_feats = self.mpdiscriminator(gt_audio)
        gt_msd_preds, gt_msd_feats = self.msdiscriminator(gt_audio)
    
        gen_mpd_preds, gen_mpd_feats = self.mpdiscriminator(gen_audio)
        gen_msd_preds, gen_msd_feats = self.msdiscriminator(gen_audio)

        return {
            'gt_preds': gt_mpd_preds + gt_msd_preds,
            'gt_feats': gt_mpd_feats + gt_msd_feats,
            'gen_preds': gen_mpd_preds + gen_msd_preds,
            'gen_feats': gen_mpd_feats + gen_msd_feats,
        }

    def _pad_audio(self, gt_audio, gen_audio):
        if gt_audio.shape[-1] > gen_audio.shape[-1]:
            shift = gt_audio.shape[-1] - gen_audio.shape[-1]
            gen_audio = F.pad(gen_audio, (0, shift))
        return gt_audio, gen_audio
             