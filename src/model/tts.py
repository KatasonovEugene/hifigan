import torch.nn as nn

from src.model.hifigan_blocks.tacotron2 import Tacotron2MelGenerator
from src.model.hifigan import HiFiGAN


class TTS(nn.Module):
    def __init__(
        self,
        acoustic_model: Tacotron2MelGenerator,
        vocoder: HiFiGAN,
    ):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder

    def forward(self, text, **batch):
        melspec = self.acoustic_model(text)
        gen_audio = self.vocoder(melspec)
        return gen_audio
