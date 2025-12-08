import torch.nn as nn

from src.model.hifigan_blocks.generator import Generator
from src.model.hifigan_blocks.discriminator import Discriminator


class HiFiGAN(nn.Module):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, gt_melspec, **batch):
        return self.generator(gt_melspec)

    def discriminate(self, gt_audio, gen_audio, detach=False, **batch):
        if detach:
            gen_audio = gen_audio.detach()
        return self.discriminator(gt_audio, gen_audio)
