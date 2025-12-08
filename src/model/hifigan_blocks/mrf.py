import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        dilation_lists,
        relu_alpha,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(*[
                nn.Sequential(
                    nn.LeakyReLU(relu_alpha),
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding='same',
                    ),
                )
                for dilation in dilation_list
            ])
            for dilation_list in dilation_lists
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) + out
        return out


class MRF(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_sizes,
        dilations,
        relu_alpha,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels=in_channels,
                kernel_size=kernel_size,
                dilation_lists=dilation_lists,
                relu_alpha=relu_alpha,
            )
            for kernel_size, dilation_lists in zip(kernel_sizes, dilations)
        ])

    def forward(self, x):
        out = 0
        for res_block in self.res_blocks:
            out = res_block(x) + out
        return out
