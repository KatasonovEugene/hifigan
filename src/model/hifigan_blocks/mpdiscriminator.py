import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class MPSubDiscriminator(nn.Module):
    def __init__(
        self,
        period,
        num_layers,
        kernel_size,
        stride,
        padding,
        out_kernel_size,
        relu_alpha,
    ):
        super().__init__()
        self.period = period

        def get_mp_conv(i, stride):
            return nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=2 ** ((5 + i) * bool(i)),
                        out_channels=2 ** (6 + i),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ),
                nn.LeakyReLU(relu_alpha),
            )
        strides = [stride] * num_layers + [(1, 1)]
        self.layers = nn.ModuleList([
            get_mp_conv(i, stride)
            for i, stride in enumerate(strides)
        ])

        max_deg = 6 + num_layers
        out_channels = 1
        self.proj_out = nn.Conv2d(
            in_channels=2 ** max_deg,
            out_channels=out_channels,
            kernel_size=out_kernel_size,
            padding='same',
        )

    def forward(self, audio):
        batch_size, _, time = audio.shape
        if time % self.period != 0:
            shift = self.period - time % self.period
            audio = F.pad(audio, (0, shift))
            time += shift
        feat = audio.view(batch_size, 1, time // self.period, self.period)
        feats = []
        for layer in self.layers:
            feat = layer(feat)
            feats.append(feat)
        pred = self.proj_out(feat)
        return pred, feats


class MPDiscriminator(nn.Module):
    def __init__(
        self,
        periods,
        num_layers,
        kernel_size,
        stride,
        padding,
        out_kernel_size,
        relu_alpha,
    ):
        super().__init__()
        params = dict(
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            out_kernel_size=out_kernel_size,
            relu_alpha=relu_alpha,
        )
        self.sub_discriminators = nn.ModuleList([
            MPSubDiscriminator(**params, period=period)
            for period in periods
        ])

    def forward(self, audio):
        preds, feats = [], []
        for sub_discriminator in self.sub_discriminators:
            pred, feat = sub_discriminator(audio)
            preds.append(pred)
            feats += feat
        return preds, feats
