"""
Plain DRT baseline.

该实现来自旧实验中的 baseline：
- 去掉 rectangular transformer；
- 去掉 multiresolution paths；
- 去掉 contrastive learning；
- 保留较深的 ResBlock 融合主干和 spectral_refine。

为了接入当前 RADI-Net 工程，PlainDRTBaseline 默认 forward 只返回预测张量；
文件末尾的 baseline / Baseline / DRTBaseline 保留旧代码的双输出兼容形式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class PlainDRTBaseline(nn.Module):
    """
    当前工程使用的单输出版本。

    input:
        lr_hsi: B x n_bands x h x w
        hr_msi: B x n_select_bands x H x W
    output:
        pred: B x n_bands x H x W
    """

    uses_contrastive_learning = False
    uses_rectangular_transformer = False
    uses_multiresolution_features = False

    def __init__(
        self,
        arch: str = "plain_drt_baseline",
        scale_ratio: int = 4,
        n_select_bands: int = 5,
        n_bands: int = 103,
        dataset=None,
        n_colors=None,
        channels: int = 64,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.arch = arch
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.dataset = dataset

        self.lr_head = nn.Sequential(
            nn.Conv2d(n_bands, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
        )
        self.hr_head = nn.Sequential(
            nn.Conv2d(n_select_bands, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(channels) for _ in range(num_blocks)],
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_bands, kernel_size=3, padding=1),
        )
        self.spectral_refine = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
        )

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        target_size = hr_msi.shape[-2:]
        lr_up = F.interpolate(
            lr_hsi,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        lr_feat = self.lr_head(lr_up)
        hr_feat = self.hr_head(hr_msi)
        fused = self.fusion(torch.cat((lr_feat, hr_feat), dim=1))

        pred = lr_up + self.reconstruction(fused)
        pred = pred + self.spectral_refine(pred)
        return pred


class baseline(PlainDRTBaseline):
    """旧实验代码兼容版本：forward 返回 (pred, pred)。"""

    def __init__(
        self,
        arch: str = "baseline",
        scale_ratio: int = 4,
        n_select_bands: int = 5,
        n_bands: int = 103,
        dataset=None,
        n_colors=None,
        channels: int = 64,
        num_blocks: int = 8,
    ):
        super().__init__(
            arch=arch,
            scale_ratio=scale_ratio,
            n_select_bands=n_select_bands,
            n_bands=n_bands,
            dataset=dataset,
            n_colors=n_colors,
            channels=channels,
            num_blocks=num_blocks,
        )

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor):
        pred = super().forward(x_lr, x_hr)
        return pred, pred


Baseline = baseline
DRTBaseline = baseline
