"""
Plain DRT baseline and ASGF ablation variants.

PlainDRTBaseline 来自旧实验 baseline：
- 去掉 rectangular transformer；
- 去掉 multiresolution paths；
- 去掉 contrastive learning；
- 保留较深的 ResBlock 融合主干和 spectral_refine。

DRTASGFBaseline 在 PlainDRTBaseline 的融合主干中，把部分 ResBlock 替换为
HSI 上下文引导的 ASGF block，用于验证 ASGF 在强 ResBlock baseline 上是否仍然有效。
"""

from typing import Iterable, Tuple

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


class HSIContextASGFBlock(nn.Module):
    """
    用于替换融合主干中 ResBlock 的 ASGF block。

    输入是已经融合过的特征 x，光谱上下文来自 lr_feat：
    - small branch: 3x3 conv，保留局部细节；
    - mid branch: 5x5 conv，读取 x 与 HSI 上下文拼接后的稳定结构；
    - global gate / spatial gate: 由 HSI 特征生成，控制两个分支强度；
    - residual output: x + fused_asgf。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.small_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.mid_branch = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, hsi_context: torch.Tensor) -> torch.Tensor:
        small = self.small_branch(x) * self.global_gate(hsi_context)
        mid = self.mid_branch(torch.cat([x, hsi_context], dim=1)) * self.spatial_gate(hsi_context)
        return x + self.fuse(torch.cat([small, mid], dim=1))


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
        self.channels = channels
        self.num_blocks = num_blocks

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


class DRTASGFBaseline(nn.Module):
    """
    DRT ResBlock baseline + ASGF replacement.

    默认把融合主干中的第 2、4、6 个 ResBlock 替换为 HSIContextASGFBlock。
    其他设置与 PlainDRTBaseline 保持一致，方便公平比较。
    """

    uses_contrastive_learning = False
    uses_rectangular_transformer = False
    uses_multiresolution_features = False
    uses_asgf_blocks = True

    def __init__(
        self,
        arch: str = "drt_asgf_baseline",
        scale_ratio: int = 4,
        n_select_bands: int = 5,
        n_bands: int = 103,
        dataset=None,
        n_colors=None,
        channels: int = 64,
        num_blocks: int = 8,
        asgf_positions: Iterable[int] = (1, 3, 5),
    ):
        super().__init__()
        self.arch = arch
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.dataset = dataset
        self.channels = channels
        self.num_blocks = num_blocks
        self.asgf_positions: Tuple[int, ...] = tuple(asgf_positions)

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
        self.fusion_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion_blocks = nn.ModuleList(
            [
                HSIContextASGFBlock(channels) if i in self.asgf_positions else ResidualBlock(channels)
                for i in range(num_blocks)
            ]
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
        fused = self.fusion_in(torch.cat((lr_feat, hr_feat), dim=1))

        for block in self.fusion_blocks:
            if isinstance(block, HSIContextASGFBlock):
                fused = block(fused, lr_feat)
            else:
                fused = block(fused)

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


class DRTASGFBaselineCompat(DRTASGFBaseline):
    """双输出兼容版本。"""

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor):
        pred = super().forward(x_lr, x_hr)
        return pred, pred


Baseline = baseline
DRTBaseline = baseline
ASGFBaseline = DRTASGFBaselineCompat
DRTASGF = DRTASGFBaselineCompat
