"""
DRT baseline + all sparse self-attention blocks 训练脚本。

该脚本仍然使用 DRTSparseBaseline，但把 8 个 fusion blocks 全部设为
SparseSelfAttentionBlock，比 train_drt_sparse_baseline.py 的 3-block 版本更接近
ASSANet 中“以 adaptive sparse self-attention block 为基本单元”的设置。

示例：
python train_drt_sparse_all_baseline.py --dataset PaviaU --msi_mode srf --srf_band_set wv2_visible6 --epochs 300 --batch_size 4
"""

import os

import torch

from config import parse_args, print_config
from data_loader import build_loaders
from losses import SAMLoss
from models.plain_drt_baseline import DRTSparseBaseline
from train_radi import compact_info, evaluate, train_one_epoch
from utils import (
    CSVLogger,
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_log,
)


SPARSE_POSITIONS = tuple(range(8))


def get_run_name(cfg) -> str:
    if cfg.save_name:
        return cfg.save_name
    msi_tag = cfg.srf_band_set if getattr(cfg, "msi_mode", "uniform") == "srf" else f"uniform{cfg.n_select_bands}"
    return f"drt_sparse_all_baseline_{cfg.dataset}_{msi_tag}_x{cfg.scale_ratio}"


def make_checkpoint_paths(cfg, run_name: str):
    save_dir = os.path.join(cfg.checkpoint_root, "baselines", run_name)
    os.makedirs(save_dir, exist_ok=True)
    return {
        "best": os.path.join(save_dir, "best.pth"),
        "last": os.path.join(save_dir, "last.pth"),
    }


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    model = DRTSparseBaseline(
        arch="drt_sparse_all_baseline",
        scale_ratio=cfg.scale_ratio,
        n_select_bands=info["n_select_bands"],
        n_bands=info["n_bands"],
        dataset=cfg.dataset,
        channels=64,
        num_blocks=8,
        sparse_positions=SPARSE_POSITIONS,
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
    write_log(log_path, "Model variant: DRTSparseAllBaseline")
    write_log(log_path, f"Sparse positions: {SPARSE_POSITIONS}")
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
            "model_name": "DRTSparseAllBaseline",
            "sparse_positions": SPARSE_POSITIONS,
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
