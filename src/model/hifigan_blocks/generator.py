import torch.nn as nn

from src.model.hifigan_blocks.mrf import MRF

class Generator(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        proj_in_kernel_size,
        proj_out_kernel_size,
        upsample_kernel_sizes,
        mrf_kernel_sizes,
        mrf_dilations,
        relu_alpha,
    ):
        super().__init__()
        self.proj_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=proj_in_kernel_size,
            padding='same',
        )
        self.generator_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.LeakyReLU(relu_alpha),
                nn.ConvTranspose1d(
                    in_channels=hidden_channels // 2 ** i,
                    out_channels=hidden_channels // 2 ** (i + 1),
                    kernel_size=upsample_kernel_size,
                    stride=upsample_kernel_size // 2,
                    padding=upsample_kernel_size // 4,
                ),
                MRF(
                    in_channels=hidden_channels // 2 ** (i + 1),
                    kernel_sizes=mrf_kernel_sizes,
                    dilations=mrf_dilations,
                    relu_alpha=relu_alpha,
                ),
            )
            for i, upsample_kernel_size in enumerate(upsample_kernel_sizes)
        ])
        self.proj_out = nn.Sequential(
            nn.LeakyReLU(relu_alpha),
            nn.Conv1d(
                in_channels=hidden_channels // 2 ** len(upsample_kernel_sizes),
                out_channels=1,
                kernel_size=proj_out_kernel_size,
                padding='same',
            ),
            nn.Tanh(),
        )
        self.generator = nn.Sequential(
            self.proj_in,
            self.generator_blocks,
            self.proj_out,
        )

    def forward(self, melspec):
        return {"gen_audio": self.generator(melspec)}
