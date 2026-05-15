# -*- coding: utf-8 -*-
"""
PyTorch training script for DTCWT desnow model.

Changes compared with the first converted version:
    1. Add direct pixel reconstruction loss between final output and GT.
    2. Keep DWT refine loss, VGG perceptual loss, and CCP loss optional.
    3. Remove random 90-degree rotation augmentation because it swaps 480x640 to 640x480.
    4. Keep synchronized horizontal/vertical flip augmentation only.
    5. Save best checkpoint and periodic checkpoints.

Expected data format:
    data.npy: [N, 480, 640, 3], BGR, uint8/int, range 0~255
    gt.npy:   [N, 480, 640, 3], BGR, uint8/int, range 0~255

Network input:
    x_pad: [B, 3, 512, 672], YCrCb, float32, range 0~1

Final output:
    pred: [B, 3, 480, 640], YCrCb, float32

Recommended first run:
    python train_torch_fixed.py --dataPath C:\\Users\\jason\\Desktop\\FYP\\code\\CSD\\Train\\data.npy --gtPath C:\\Users\\jason\\Desktop\\FYP\\code\\CSD\\Train\\gt.npy --logPath C:\\Users\\jason\\Desktop\\FYP\\code\\CSD\\logs_fix --batchsize 1 --epochs 50 --steps_per_epoch 100 --use_vgg_loss 0 --ccp_weight 0 --dwt_weight 0.1
"""

import argparse
import csv
import os
import random
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from torchvision import models
except Exception:
    models = None

# Your model file should be: HDCWNet/model/torch_model.py
from model.torch_model import build_DTCWT_model


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Losses
# -----------------------------

def l1_charbonnier_loss(y_true, y_pred, eps=1e-6):
    return torch.mean(torch.sqrt((y_true - y_pred) ** 2 + eps))


def l2_charbonnier_loss(y_true, y_pred, eps=1e-6):
    # Original Keras implementation effectively equals diff^2 + eps.
    return torch.mean((y_true - y_pred) ** 2 + eps)


def ccp_loss(y_true, y_pred, pool_size=35):
    # Original Keras version:
    # min over channels -> max pooling -> sigmoid(mean(abs(diff)))
    pred_min = torch.min(y_pred, dim=1, keepdim=True).values
    true_min = torch.min(y_true, dim=1, keepdim=True).values
    pad = pool_size // 2
    pred_pool = F.max_pool2d(pred_min, kernel_size=pool_size, stride=1, padding=pad)
    true_pool = F.max_pool2d(true_min, kernel_size=pool_size, stride=1, padding=pad)
    return torch.sigmoid(torch.mean(torch.abs(pred_pool - true_pool)))


class VGGPerceptualLoss(nn.Module):
    """VGG16 perceptual loss.

    Note:
        VGG is trained on RGB ImageNet images. Your network output is YCrCb.
        So this loss is optional and should be added only after pixel loss is stable.
    """

    def __init__(self, device):
        super().__init__()
        if models is None:
            raise ImportError("torchvision is required for VGG loss. Use --use_vgg_loss 0 if unavailable.")

        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights).features
        except Exception:
            vgg = models.vgg16(pretrained=True).features

        self.features = vgg.eval().to(device)
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, y_true, y_pred):
        y_true = torch.clamp(y_true, 0.0, 1.0)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)
        y_true = (y_true - self.mean) / self.std
        y_pred = (y_pred - self.mean) / self.std
        return F.mse_loss(self.features(y_pred), self.features(y_true))


# -----------------------------
# Dataset
# -----------------------------

class DesnowNpyDataset(Dataset):
    def __init__(self, data, label, pad_edge=16, augment=False):
        self.data = data
        self.label = label
        self.pad_edge = pad_edge
        self.augment = augment

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def _to_ycrcb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    def _augment_pair(self, x, y):
        # Synchronized flips only. Do NOT use rot90 because it changes 480x640 to 640x480.
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=0).copy()
            y = np.flip(y, axis=0).copy()
        return x, y

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]

        x = self._to_ycrcb(x.astype(np.uint8))
        y = self._to_ycrcb(y.astype(np.uint8))

        if self.augment:
            x, y = self._augment_pair(x, y)

        x = x.astype(np.float32) / 255.0
        y = y.astype(np.float32) / 255.0

        x_pad = np.pad(
            x,
            ((self.pad_edge, self.pad_edge), (self.pad_edge, self.pad_edge), (0, 0)),
            mode="constant",
        )
        y_pad = np.pad(
            y,
            ((self.pad_edge, self.pad_edge), (self.pad_edge, self.pad_edge), (0, 0)),
            mode="constant",
        )

        # NHWC -> NCHW
        x_pad = torch.from_numpy(x_pad.transpose(2, 0, 1)).float()
        y_pad = torch.from_numpy(y_pad.transpose(2, 0, 1)).float()
        y = torch.from_numpy(y.transpose(2, 0, 1)).float()

        return x_pad, y_pad, y


def split_train_val(data, label, validation_num, seed=1):
    n = data.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    validation_num = int(validation_num)
    val_idx = indices[:validation_num]
    train_idx = indices[validation_num:]
    return data[train_idx], data[val_idx], label[train_idx], label[val_idx]


# -----------------------------
# DWT loss target construction
# -----------------------------

@torch.no_grad()
def build_gt_dwt_list(model, gt_pad):
    """Build GT DWT list in the same order as model refine outputs.

    RefineDwtList order from model:
        [x1H0R, x1H1R, x2H0R, x2H1R, x1LR, x2LR]

    Therefore GT order should be:
        [gt0_h0, gt0_h1, gt1_h0, gt1_h1, gt0_low, gt1_low]
    """
    core = model.core
    gt0_low, gt0_h0, gt0_h1 = core.dtcwt0(gt_pad)
    gt1_low, gt1_h0, gt1_h1 = core.dtcwt1(gt0_low)
    return [gt0_h0, gt0_h1, gt1_h0, gt1_h1, gt0_low, gt1_low]


def dwt_refine_loss(gt_dwt_list, refine_list):
    if len(gt_dwt_list) != len(refine_list):
        raise ValueError(f"DWT list length mismatch: gt={len(gt_dwt_list)}, pred={len(refine_list)}")

    loss = torch.zeros((), device=refine_list[0].device)
    for gt_item, pred_item in zip(gt_dwt_list, refine_list):
        loss = loss + l2_charbonnier_loss(gt_item, pred_item)
    return loss


# -----------------------------
# Train / Validate
# -----------------------------

def run_one_epoch(model, loader, optimizer, device, vgg_loss_fn, args, train=True, max_batches=None):
    model.train(train)

    total = 0.0
    total_pixel = 0.0
    total_dwt = 0.0
    total_vgg = 0.0
    total_ccp = 0.0
    count = 0

    for batch_idx, (x_pad, y_pad, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_pad = x_pad.to(device, non_blocking=True)
        y_pad = y_pad.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred, refine_list = model(x_pad, return_refine=True)

            # Safety alignment. Expected pred/y: [B, 3, 480, 640]
            if pred.shape[-2:] != y.shape[-2:]:
                raise RuntimeError(f"Output/GT size mismatch: pred={pred.shape}, gt={y.shape}")

            # 1. Main image reconstruction loss. This is essential.
            loss_pixel = l1_charbonnier_loss(y, pred)

            # 2. DWT refine supervision.
            if args.dwt_weight > 0:
                gt_dwt_list = build_gt_dwt_list(model, y_pad)
                loss_dwt = dwt_refine_loss(gt_dwt_list, refine_list)
            else:
                loss_dwt = torch.zeros((), device=device)

            # 3. Optional VGG loss.
            if args.use_vgg_loss:
                loss_vgg = vgg_loss_fn(y, pred)
            else:
                loss_vgg = torch.zeros((), device=device)

            # 4. Optional CCP loss.
            if args.ccp_weight > 0:
                loss_ccp = ccp_loss(y, pred)
            else:
                loss_ccp = torch.zeros((), device=device)

            loss = (
                args.pixel_weight * loss_pixel
                + args.dwt_weight * loss_dwt
                + args.vgg_weight * loss_vgg
                + args.ccp_weight * loss_ccp
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        bs = x_pad.size(0)
        total += loss.item() * bs
        total_pixel += loss_pixel.item() * bs
        total_dwt += loss_dwt.item() * bs
        total_vgg += loss_vgg.item() * bs
        total_ccp += loss_ccp.item() * bs
        count += bs

    return {
        "loss": total / max(count, 1),
        "pixel_loss": total_pixel / max(count, 1),
        "dwt_loss": total_dwt / max(count, 1),
        "vgg_loss": total_vgg / max(count, 1),
        "ccp_loss": total_ccp / max(count, 1),
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args, save_time=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "save_time": save_time,
            "loss_weights": {
                "pixel_weight": args.pixel_weight,
                "dwt_weight": args.dwt_weight,
                "vgg_weight": args.vgg_weight,
                "ccp_weight": args.ccp_weight,
                "use_vgg_loss": args.use_vgg_loss,
            },
            "args": vars(args),
        },
        str(path),
    )


def make_time_string():
    """Return a Windows-safe timestamp string for checkpoint filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch DTCWT desnow model.")

    parser.add_argument("--logPath", type=str, required=True)
    parser.add_argument("--dataPath", type=str, required=True)
    parser.add_argument("--gtPath", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--modelPath", type=str, default="", help="Checkpoint path for continuing training")
    parser.add_argument("--validation_num", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Loss switches and weights.
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--dwt_weight", type=float, default=0.1)
    parser.add_argument("--use_vgg_loss", type=int, default=0)
    parser.add_argument("--vgg_weight", type=float, default=0.0)
    parser.add_argument("--ccp_weight", type=float, default=0.0)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--note", type=str, default="")

    args = parser.parse_args()

    set_seed(args.seed)

    log_dir = Path(args.logPath)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_start_time = make_time_string()

    print("Parameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    with open(log_dir / "inputParam.txt", "w", encoding="utf-8") as f:
        f.write(f"run_start_time: {run_start_time}\n")
        f.write("loss_weights:\n")
        f.write(f"  pixel_weight: {args.pixel_weight}\n")
        f.write(f"  dwt_weight: {args.dwt_weight}\n")
        f.write(f"  use_vgg_loss: {args.use_vgg_loss}\n")
        f.write(f"  vgg_weight: {args.vgg_weight}\n")
        f.write(f"  ccp_weight: {args.ccp_weight}\n")
        f.write("\nall_args:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print("load Data")
    data = np.load(args.dataPath)
    print("load Gt")
    label = np.load(args.gtPath)

    print("raw data:", data.shape, data.dtype, data.min(), data.max())
    print("raw label:", label.shape, label.dtype, label.min(), label.max())

    if data.shape != label.shape:
        raise RuntimeError(f"data/label shape mismatch: data={data.shape}, label={label.shape}")
    if data.ndim != 4 or data.shape[-1] != 3:
        raise RuntimeError(f"Expected data shape [N,H,W,3], got {data.shape}")

    train_data, val_data, train_label, val_label = split_train_val(
        data, label, validation_num=args.validation_num, seed=args.seed
    )

    print("train data:", train_data.shape)
    print("train label:", train_label.shape)
    print("val data:", val_data.shape)
    print("val label:", val_label.shape)

    train_dataset = DesnowNpyDataset(train_data, train_label, pad_edge=16, augment=True)
    val_dataset = DesnowNpyDataset(val_data, val_label, pad_edge=16, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device(args.device)

    # Input shape after padding: 480+32, 640+32 = 512, 672.
    model = build_DTCWT_model((3, data.shape[1] + 32, data.shape[2] + 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    start_epoch = 1

    if args.modelPath:
        print("Continue train!")
        ckpt = torch.load(args.modelPath, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        else:
            model.load_state_dict(ckpt)
        print("Load Weights Success!")

    if args.use_vgg_loss:
        vgg_loss_fn = VGGPerceptualLoss(device=device).to(device)
    else:
        vgg_loss_fn = None

    writer = SummaryWriter(str(log_dir)) if SummaryWriter is not None else None
    csv_path = log_dir / "log.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "epoch",
            "time",
            "train_loss",
            "train_pixel_loss",
            "train_dwt_loss",
            "train_vgg_loss",
            "train_ccp_loss",
            "val_loss",
            "val_pixel_loss",
            "val_dwt_loss",
            "val_vgg_loss",
            "val_ccp_loss",
            "pixel_weight",
            "dwt_weight",
            "use_vgg_loss",
            "vgg_weight",
            "ccp_weight",
            "lr",
            "is_best",
            "best_save_time",
            "best_checkpoint_path",
        ]
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer_csv.writeheader()

        max_train_batches = args.steps_per_epoch if args.steps_per_epoch > 0 else None

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                vgg_loss_fn,
                args,
                train=True,
                max_batches=max_train_batches,
            )

            val_metrics = run_one_epoch(
                model,
                val_loader,
                optimizer,
                device,
                vgg_loss_fn,
                args,
                train=False,
                max_batches=None,
            )

            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_pixel={train_metrics['pixel_loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_pixel={val_metrics['pixel_loss']:.6f} "
                f"dwt={val_metrics['dwt_loss']:.6f} "
                f"vgg={val_metrics['vgg_loss']:.6f} "
                f"ccp={val_metrics['ccp_loss']:.6f}"
            )

            epoch_time = make_time_string()
            is_best = val_metrics["loss"] < best_val_loss
            best_save_time = ""
            best_checkpoint_path = ""

            row = {
                "epoch": epoch,
                "time": epoch_time,
                "train_loss": train_metrics["loss"],
                "train_pixel_loss": train_metrics["pixel_loss"],
                "train_dwt_loss": train_metrics["dwt_loss"],
                "train_vgg_loss": train_metrics["vgg_loss"],
                "train_ccp_loss": train_metrics["ccp_loss"],
                "val_loss": val_metrics["loss"],
                "val_pixel_loss": val_metrics["pixel_loss"],
                "val_dwt_loss": val_metrics["dwt_loss"],
                "val_vgg_loss": val_metrics["vgg_loss"],
                "val_ccp_loss": val_metrics["ccp_loss"],
                "pixel_weight": args.pixel_weight,
                "dwt_weight": args.dwt_weight,
                "use_vgg_loss": args.use_vgg_loss,
                "vgg_weight": args.vgg_weight,
                "ccp_weight": args.ccp_weight,
                "lr": lr,
                "is_best": int(is_best),
                "best_save_time": best_save_time,
                "best_checkpoint_path": best_checkpoint_path,
            }

            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"val/{name}", value, epoch)
                writer.add_scalar("lr", lr, epoch)

            # Periodic checkpoint.
            if epoch % 100 == 0:
                periodic_time = make_time_string()
                save_checkpoint(
                    log_dir / f"model.{epoch:04d}-{val_metrics['loss']:.4f}-{periodic_time}.pth",
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    args,
                    save_time=periodic_time,
                )

            # Best checkpoint. Save with current time in filename and keep a latest alias.
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_save_time = make_time_string()
                best_path = log_dir / f"modelBest_epoch{epoch:04d}_valloss{best_val_loss:.6f}_{best_save_time}.pth"
                save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, args, save_time=best_save_time)
                # Also overwrite a stable alias so predict_torch.py can still use logs/modelBest.pth.
                save_checkpoint(log_dir / "modelBest.pth", model, optimizer, epoch, best_val_loss, args, save_time=best_save_time)
                best_checkpoint_path = str(best_path)
                row["best_save_time"] = best_save_time
                row["best_checkpoint_path"] = best_checkpoint_path
                print(f"Saved best checkpoint: val_loss={best_val_loss:.6f}, path={best_path}")

            writer_csv.writerow(row)
            csv_file.flush()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
