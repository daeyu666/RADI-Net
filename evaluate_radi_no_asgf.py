"""
RADI-Net no-ASGF 消融评估脚本。

示例：
python evaluate_radi_no_asgf.py --dataset PaviaU --msi_mode srf --srf_band_set wv2_visible6 --resume checkpoints/radi_net/radi_net_no_asgf_PaviaU_wv2_visible6_x4/best.pth
"""

import os
from typing import Dict

import torch

from config import parse_args, print_config
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models.radi_net import RADINetNoASGF
from utils import get_device, load_checkpoint, move_to_device, save_mat, set_seed, tensor_to_numpy


def get_run_name(cfg) -> str:
    if cfg.save_name:
        return cfg.save_name
    msi_tag = cfg.srf_band_set if getattr(cfg, "msi_mode", "uniform") == "srf" else f"uniform{cfg.n_select_bands}"
    return f"radi_net_no_asgf_{cfg.dataset}_{msi_tag}_x{cfg.scale_ratio}"


@torch.no_grad()
def evaluate_and_save(model, loader, device, cfg, save_prediction: bool = True) -> Dict[str, float]:
    model.eval()
    averager = MetricAverager()

    save_dir = os.path.join(cfg.output_root, "predictions", cfg.dataset)
    os.makedirs(save_dir, exist_ok=True)

    for idx, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        pred = torch.clamp(model(batch["lr_hsi"], batch["hr_msi"]), 0.0, 1.0)

        metrics = calc_metrics(
            pred=pred,
            target=batch["gt"],
            scale_ratio=cfg.scale_ratio,
        )
        averager.update(metrics)

        if save_prediction:
            save_path = os.path.join(save_dir, f"radi_net_no_asgf_sample{idx}.mat")
            save_mat(
                save_path,
                {
                    "pred": tensor_to_numpy(pred),
                    "gt": tensor_to_numpy(batch["gt"]),
                    "lr_hsi": tensor_to_numpy(batch["lr_hsi"]),
                    "hr_msi": tensor_to_numpy(batch["hr_msi"]),
                },
            )

    return averager.average()


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    model = RADINetNoASGF(
        hsi_bands=info["n_bands"],
        msi_bands=info["n_select_bands"],
        channels=64,
        residual_scale=0.2,
    ).to(device)

    if not cfg.resume:
        run_name = get_run_name(cfg)
        cfg.resume = os.path.join(
            cfg.checkpoint_root,
            "radi_net",
            run_name,
            "best.pth",
        )

    load_checkpoint(
        model,
        cfg.resume,
        optimizer=None,
        strict=False,
        map_location=device,
    )

    metrics = evaluate_and_save(model, test_loader, device, cfg, save_prediction=True)

    print("=" * 80)
    print("RADI-Net no-ASGF Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {cfg.resume}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print("=" * 80)

    metrics_dir = os.path.join(cfg.output_root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    run_name = get_run_name(cfg)
    metrics_path = os.path.join(metrics_dir, f"{run_name}_eval.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {cfg.resume}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
