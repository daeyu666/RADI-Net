"""
RADI-Net 第一版训练脚本。

示例：
python train_radi.py --dataset PaviaU --msi_mode srf --srf_band_set wv2_visible6 --epochs 300 --batch_size 4
"""

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from config import parse_args, print_config
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.radi_net import RADINet
from utils import (
    CSVLogger,
    count_parameters,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def get_run_name(cfg) -> str:
    if cfg.save_name:
        return cfg.save_name
    msi_tag = cfg.srf_band_set if getattr(cfg, "msi_mode", "uniform") == "srf" else f"uniform{cfg.n_select_bands}"
    return f"radi_net_v1_{cfg.dataset}_{msi_tag}_x{cfg.scale_ratio}"


def make_checkpoint_paths(cfg, run_name: str) -> Dict[str, str]:
    save_dir = os.path.join(cfg.checkpoint_root, "radi_net", run_name)
    os.makedirs(save_dir, exist_ok=True)
    return {
        "best": os.path.join(save_dir, "best.pth"),
        "last": os.path.join(save_dir, "last.pth"),
    }


def compact_info(info: dict) -> dict:
    """避免把 SRF 权重矩阵、波长数组完整写进日志。"""
    out = {k: v for k, v in info.items() if k not in ("srf_weights", "hsi_wavelengths")}
    if info.get("srf_weights") is not None:
        out["srf_weights_shape"] = tuple(info["srf_weights"].shape)
    if info.get("hsi_wavelengths") is not None:
        out["hsi_wavelengths_shape"] = tuple(info["hsi_wavelengths"].shape)
    return out


def hsi_to_msi_torch(
    hsi: torch.Tensor,
    srf_weights: Optional[torch.Tensor],
    n_msi_bands: int,
) -> torch.Tensor:
    """把 BxCxHxW 的 HSI 投影成 BxMxHxW 的 MSI。"""
    if srf_weights is not None:
        weights = srf_weights.to(device=hsi.device, dtype=hsi.dtype)
        return torch.einsum("bchw,mc->bmhw", hsi, weights)

    # uniform 模式下，和 data_loader.py 的均匀抽波段逻辑保持一致。
    indices = torch.linspace(
        0,
        hsi.shape[1] - 1,
        steps=n_msi_bands,
        device=hsi.device,
    ).round().long()
    return hsi.index_select(1, indices)


def compute_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    cfg,
    sam_loss: nn.Module,
    srf_weights: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """第一版使用简单稳定的重建损失组合。"""
    l1 = F.l1_loss(pred, gt)
    mse = F.mse_loss(pred, gt)
    sam = sam_loss(pred, gt)

    pred_lr = F.interpolate(pred, size=lr_hsi.shape[-2:], mode="bicubic", align_corners=False)
    lr_consistency = F.l1_loss(pred_lr, lr_hsi)

    pred_msi = hsi_to_msi_torch(pred, srf_weights=srf_weights, n_msi_bands=hr_msi.shape[1])
    msi_consistency = F.l1_loss(pred_msi, hr_msi)

    total = (
        getattr(cfg, "lambda_l1", 1.0) * l1
        + getattr(cfg, "lambda_mse", 1.0) * mse
        + getattr(cfg, "lambda_sam", 0.1) * sam
        + getattr(cfg, "lambda_dc", 0.1) * lr_consistency
        + getattr(cfg, "lambda_srf_region", 0.3) * msi_consistency
    )
    return {
        "total": total,
        "l1": l1.detach(),
        "mse": mse.detach(),
        "sam": sam.detach(),
        "lr_consistency": lr_consistency.detach(),
        "msi_consistency": msi_consistency.detach(),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg,
    sam_loss: nn.Module,
    srf_weights: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.train()
    meters = {key: 0.0 for key in ["total", "l1", "mse", "sam", "lr_consistency", "msi_consistency"]}
    count = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        pred = model(lr_hsi, hr_msi)
        losses = compute_loss(pred, gt, lr_hsi, hr_msi, cfg, sam_loss, srf_weights)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = gt.shape[0]
        count += batch_size
        for key in meters:
            meters[key] += float(losses[key].item()) * batch_size

    return {key: value / max(count, 1) for key, value in meters.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, cfg) -> Dict[str, float]:
    model.eval()
    averager = MetricAverager()
    for batch in loader:
        batch = move_to_device(batch, device)
        pred = torch.clamp(model(batch["lr_hsi"], batch["hr_msi"]), 0.0, 1.0)
        averager.update(calc_metrics(pred=pred, target=batch["gt"], scale_ratio=cfg.scale_ratio))
    return averager.average()


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    model = RADINet(
        hsi_bands=info["n_bands"],
        msi_bands=info["n_select_bands"],
        channels=64,
        residual_scale=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sam_loss = SAMLoss().to(device)

    srf_weights = info.get("srf_weights", None)
    if srf_weights is not None:
        srf_weights = torch.from_numpy(srf_weights).float()

    run_name = get_run_name(cfg)
    ckpt_paths = make_checkpoint_paths(cfg, run_name)
    log_path = os.path.join(cfg.log_root, f"{run_name}.log")
    csv_path = os.path.join(cfg.log_root, f"{run_name}.csv")
    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "train_total", "train_l1", "train_mse", "train_sam",
            "train_lr_consistency", "train_msi_consistency",
            "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC", "best_psnr",
        ],
    )

    write_log(log_path, f"Run name: {run_name}")
    write_log(log_path, f"Model parameters: {count_parameters(model):.3f} M")
    write_log(log_path, f"Dataset info: {compact_info(info)}")

    start_epoch = 1
    best_psnr = -1.0
    if cfg.resume:
        loaded_epoch, loaded_best = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=False,
            map_location=device,
        )
        start_epoch = int(loaded_epoch) + 1
        best_psnr = float(loaded_best)
        write_log(log_path, f"Resume from {cfg.resume}, start_epoch={start_epoch}, best_psnr={best_psnr:.4f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, cfg, sam_loss, srf_weights)

        if epoch % cfg.eval_interval == 0:
            val_metrics = evaluate(model, test_loader, device, cfg)
        else:
            val_metrics = {key: 0.0 for key in ["PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC"]}

        current_psnr = val_metrics.get("PSNR", 0.0)
        is_best = current_psnr > best_psnr
        if is_best:
            best_psnr = current_psnr

        extra = {
            "cfg": cfg.__dict__,
            "info": compact_info(info),
            "run_name": run_name,
            "model_name": "RADINet-v1",
        }
        save_checkpoint(model, optimizer, epoch, best_psnr, ckpt_paths["last"], extra=extra)
        if is_best:
            save_checkpoint(model, optimizer, epoch, best_psnr, ckpt_paths["best"], extra=extra)

        row = {
            "epoch": epoch,
            "train_total": train_stats["total"],
            "train_l1": train_stats["l1"],
            "train_mse": train_stats["mse"],
            "train_sam": train_stats["sam"],
            "train_lr_consistency": train_stats["lr_consistency"],
            "train_msi_consistency": train_stats["msi_consistency"],
            **val_metrics,
            "best_psnr": best_psnr,
        }
        csv_logger.write(row)

        write_log(
            log_path,
            (
                f"Epoch [{epoch:03d}/{cfg.epochs:03d}] "
                f"loss={train_stats['total']:.6f} "
                f"l1={train_stats['l1']:.6f} "
                f"sam_loss={train_stats['sam']:.6f} "
                f"PSNR={val_metrics['PSNR']:.4f} "
                f"SAM={val_metrics['SAM']:.4f} "
                f"best={best_psnr:.4f}"
            ),
        )

    write_log(log_path, f"Training finished. Best checkpoint: {ckpt_paths['best']}")


if __name__ == "__main__":
    main()
