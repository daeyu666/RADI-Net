"""
RADI-Net 第一版与 no-ASGF 消融版。

RADI-Net v1：
1. LR-HSI bicubic 上采样后提取 HSI 光谱主体特征；
2. 使用光谱上下文引导的 ASGF 提取 HR-MSI 空间特征；
3. 拼接 HSI/MSI 特征后预测一个小残差，叠加到上采样 HSI 上。

RADI-Net no-ASGF：
保持主干、融合头、重建头不变，只把 ASGF 替换为普通 ResBlock MSI 编码器，
用于验证 SpectralContextASGF 是否有效。
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(channels: int) -> nn.GroupNorm:
    """选择能整除通道数的 GroupNorm，避免 BatchNorm 对 batch size 敏感。"""
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvAct(nn.Module):
    """Conv + GroupNorm + GELU，作为基础卷积单元。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """轻量残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _make_group_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class PlainMSIResEncoder(nn.Module):
    """
    普通 MSI ResBlock 编码器。

    这是 no-ASGF 消融使用的替代模块：
    - 只从 HR-MSI 提取空间特征；
    - 不使用 HSI 光谱上下文 gate；
    - 不做小/中尺度非对称门控。
    """

    def __init__(self, msi_bands: int, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(msi_bands, channels, kernel_size=3),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

    def forward(self, hr_msi: torch.Tensor, hsi_feat: torch.Tensor = None) -> torch.Tensor:
        return self.body(hr_msi)


class SpectralContextASGF(nn.Module):
    """
    光谱上下文引导的 ASGF。

    简化后的设计：
    - 小尺度分支：3x3 conv，从 HR-MSI 中提取边缘和细节；
    - 中尺度分支：5x5 conv，联合 HR-MSI 与 HSI 主体特征提取稳定结构；
    - 光谱上下文门控：由 HSI 主体特征生成全局 gate 和空间 gate；
    - 输出：被 HSI 光谱上下文调制后的 MSI 空间特征。

    这里保留 ASGF 的“大尺度上下文调制小尺度细节”思想，但上下文来自 HSI，
    避免 MSI 自己无约束地放大伪纹理。
    """

    def __init__(self, msi_bands: int, channels: int):
        super().__init__()
        self.msi_small = nn.Sequential(
            ConvAct(msi_bands, channels, kernel_size=3),
            ConvAct(channels, channels, kernel_size=3),
        )
        self.msi_mid = nn.Sequential(
            ConvAct(msi_bands + channels, channels, kernel_size=5),
            ConvAct(channels, channels, kernel_size=3),
        )

        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            _make_group_norm(channels),
            nn.GELU(),
            ResidualBlock(channels),
        )

    def forward(self, hr_msi: torch.Tensor, hsi_feat: torch.Tensor) -> torch.Tensor:
        small = self.msi_small(hr_msi)
        mid = self.msi_mid(torch.cat([hr_msi, hsi_feat], dim=1))

        global_gate = self.global_gate(hsi_feat)
        spatial_gate = self.spatial_gate(hsi_feat)

        small = small * global_gate
        mid = mid * spatial_gate

        return self.fuse(torch.cat([small, mid], dim=1))


class RADINet(nn.Module):
    """
    RADI-Net 第一版主网络。

    参数：
        hsi_bands: HSI 波段数；
        msi_bands: MSI 波段数；
        channels: 中间特征通道数；
        residual_scale: 限制残差幅度，避免第一版训练初期破坏 LR-HSI 光谱主体；
        use_asgf: True 使用 SpectralContextASGF，False 使用普通 MSI ResBlock 编码器。
    """

    def __init__(
        self,
        hsi_bands: int,
        msi_bands: int,
        channels: int = 64,
        residual_scale: float = 0.2,
        upsample_mode: str = "bicubic",
        use_asgf: bool = True,
    ):
        super().__init__()
        self.hsi_bands = hsi_bands
        self.msi_bands = msi_bands
        self.channels = channels
        self.residual_scale = residual_scale
        self.upsample_mode = upsample_mode
        self.use_asgf = use_asgf

        self.hsi_encoder = nn.Sequential(
            ConvAct(hsi_bands, channels, kernel_size=3),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

        if use_asgf:
            self.msi_encoder = SpectralContextASGF(msi_bands=msi_bands, channels=channels)
        else:
            self.msi_encoder = PlainMSIResEncoder(msi_bands=msi_bands, channels=channels)

        self.fusion = nn.Sequential(
            ConvAct(channels * 2, channels, kernel_size=3),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

        self.reconstruction = nn.Sequential(
            ConvAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, hsi_bands, kernel_size=3, padding=1),
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        target_size = hr_msi.shape[-2:]
        hsi_up = F.interpolate(
            lr_hsi,
            size=target_size,
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ("bilinear", "bicubic") else None,
        )
        hsi_up = torch.clamp(hsi_up, 0.0, 1.0)

        hsi_feat = self.hsi_encoder(hsi_up)
        msi_feat = self.msi_encoder(hr_msi, hsi_feat)

        fused = self.fusion(torch.cat([hsi_feat, msi_feat], dim=1))
        delta = torch.tanh(self.reconstruction(fused)) * self.residual_scale
        pred = torch.clamp(hsi_up + delta, 0.0, 1.0)

        if return_aux:
            return {
                "pred": pred,
                "hsi_up": hsi_up,
                "hsi_feat": hsi_feat,
                "msi_feat": msi_feat,
                "delta": delta,
            }
        return pred


class RADINetNoASGF(RADINet):
    """把 ASGF 替换为普通 MSI ResBlock 编码器的消融版本。"""

    def __init__(
        self,
        hsi_bands: int,
        msi_bands: int,
        channels: int = 64,
        residual_scale: float = 0.2,
        upsample_mode: str = "bicubic",
    ):
        super().__init__(
            hsi_bands=hsi_bands,
            msi_bands=msi_bands,
            channels=channels,
            residual_scale=residual_scale,
            upsample_mode=upsample_mode,
            use_asgf=False,
        )


# 兼容不同命名习惯。
RADI_Net = RADINet
RADI_Net_No_ASGF = RADINetNoASGF


def build_radi_net(
    hsi_bands: int,
    msi_bands: int,
    channels: int = 64,
    residual_scale: float = 0.2,
) -> RADINet:
    return RADINet(
        hsi_bands=hsi_bands,
        msi_bands=msi_bands,
        channels=channels,
        residual_scale=residual_scale,
        use_asgf=True,
    )


def build_radi_net_no_asgf(
    hsi_bands: int,
    msi_bands: int,
    channels: int = 64,
    residual_scale: float = 0.2,
) -> RADINetNoASGF:
    return RADINetNoASGF(
        hsi_bands=hsi_bands,
        msi_bands=msi_bands,
        channels=channels,
        residual_scale=residual_scale,
    )
