# code was taken from https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.Tacotron2TTSBundle.html#torchaudio.pipelines.Tacotron2TTSBundle

import torch
import torchaudio
import torch.nn as nn


class Tacotron2MelGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.text_processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2()

    @torch.inference_mode()
    def forward(self, text):
        encoded_text, lengths = self.text_processor(text)
        device = next(self.parameters()).device
        encoded_text, lengths = encoded_text.to(device), lengths.to(device)
        melspec, _, _ = self.tacotron2.infer(encoded_text, lengths)
        return melspec
 