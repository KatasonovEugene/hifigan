import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import (
    weight_norm,
    spectral_norm,
)

class MSSubDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_sizes,
        strides,
        groups,
        paddings,
        relu_alpha,
        out_kernel_size,
        normalization,
    ):
        super().__init__()
        params_iterator = zip(in_channels, hidden_channels, kernel_sizes, strides, groups, paddings)
        self.layers = nn.ModuleList([
            nn.Sequential(
                normalization(
                    nn.Conv1d(
                        in_channels=in_chans,
                        out_channels=hidden_chans,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=group,
                    )
                ),
                nn.LeakyReLU(relu_alpha),
            )
            for (in_chans, hidden_chans, kernel_size, stride, group, padding) in params_iterator
        ])
        out_channels = 1
        self.proj_out = normalization(
            nn.Conv1d(
                in_channels=hidden_channels[-1],
                out_channels=out_channels,
                kernel_size=out_kernel_size,
                padding='same',
            )
        )

    def forward(self, audio):
        feat = audio
        feats = []
        for layer in self.layers:
            feat = layer(feat)
            feats.append(feat)
        pred = self.proj_out(feat)
        return pred, feats


class MSDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_sizes,
        strides,
        groups,
        paddings,
        relu_alpha,
        out_kernel_size,
    ):
        super().__init__()
        params = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            groups=groups,
            paddings=paddings,
            relu_alpha=relu_alpha,
            out_kernel_size=out_kernel_size,
        )
        self.sub_discriminators = nn.ModuleList([
            MSSubDiscriminator(**params, normalization=spectral_norm),
            MSSubDiscriminator(**params, normalization=weight_norm),
            MSSubDiscriminator(**params, normalization=weight_norm),
        ])

    def forward(self, audio):
        preds, feats = [], []
        for sub_discriminator in self.sub_discriminators:
            pred, feat = sub_discriminator(audio)
            preds.append(pred)
            feats += feat
            audio = F.adaptive_avg_pool1d(audio, audio.shape[-1] // 2)
        return preds, feats