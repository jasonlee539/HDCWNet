# -*- coding: utf-8 -*-
"""
PyTorch training script for dtcwt_model_torch.py.

This script corresponds to the original Keras training code, but replaces:
    - Keras ImageDataGenerator -> PyTorch Dataset/DataLoader
    - model.compile / fit_generator -> manual train/validation loops
    - Keras callbacks -> CSV logging + TensorBoard + torch checkpoint saving

Expected data format:
    data.npy:  [N, H, W, 3], BGR uint8/float, same as the original code
    gt.npy:    [N, H, W, 3], BGR uint8/float

Network input:
    PyTorch NCHW float tensor, normalized to [0, 1]

Important:
    The model output is cropped by 16 pixels on each side, so the input and GT
    fed into the network are padded by 16 pixels, while the reconstruction losses
    are computed against the original unpadded GT.
"""

import argparse
import csv
import os
import random
from pathlib import Path

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

# from .model.torch_model import build_DTCWT_model
from model.torch_model import build_DTCWT_model


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Losses
# -----------------------------

def l1_charbonnier_loss(y_true, y_pred, eps=1e-6):
    return torch.mean(torch.sqrt((y_true - y_pred) ** 2 + eps))


def l2_charbonnier_loss(y_true, y_pred, eps=1e-6):
    # Equivalent to the original Keras code:
    # sqrt(diff^2 + eps) * sqrt(diff^2 + eps) = diff^2 + eps
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
    """VGG16 feature loss similar to the original VGGloss.

    Set --use_vgg_loss 0 if torchvision / pretrained weights are unavailable.
    """

    def __init__(self, device):
        super().__init__()
        if models is None:
            raise ImportError("torchvision is required for VGG loss. Install torchvision or use --use_vgg_loss 0.")

        # Compatible with both newer and older torchvision APIs.
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights).features
        except Exception:
            vgg = models.vgg16(pretrained=True).features

        # Original Keras VGG16(include_top=False) uses the whole convolutional feature extractor.
        self.features = vgg.eval().to(device)
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, y_true, y_pred):
        # y_true / y_pred are expected in [0, 1].
        y_true = (y_true - self.mean) / self.std
        y_pred = (y_pred - self.mean) / self.std
        return F.mse_loss(self.features(y_pred), self.features(y_true))


# -----------------------------
# Data
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
        # Original code uses cv2.COLOR_BGR2YCR_CB.
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    def _augment_pair(self, x, y):
        # Lightweight synchronized augmentation corresponding to the original ImageDataGenerator idea.
        # It uses flips and 90-degree rotations to avoid changing target dimensions.
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=0).copy()
            y = np.flip(y, axis=0).copy()
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(0, 1)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()
        return x, y

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]

        # Keep behavior close to the original code: BGR -> YCrCb, then /255.
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
    # Same order as the original code:
    # [output_gt_0_h0, output_gt_0_h1, output_gt_1_h0, output_gt_1_h1, output_gt_0_low, output_gt_1_low]
    # The uploaded original had a likely typo: it used output_1_h1 instead of output_gt_1_h1.
    core = model.core
    gt0_low, gt0_h0, gt0_h1 = core.dtcwt0(gt_pad)
    gt1_low, gt1_h0, gt1_h1 = core.dtcwt1(gt0_low)
    return [gt0_h0, gt0_h1, gt1_h0, gt1_h1, gt0_low, gt1_low]


def dwt_refine_loss(gt_dwt_list, refine_list):
    if len(gt_dwt_list) != len(refine_list):
        raise ValueError(f"DWT list length mismatch: gt={len(gt_dwt_list)}, pred={len(refine_list)}")
    loss = 0.0
    for gt_item, pred_item in zip(gt_dwt_list, refine_list):
        loss = loss + l2_charbonnier_loss(gt_item, pred_item)
    return loss


# -----------------------------
# Train / validate
# -----------------------------

def run_one_epoch(model, loader, optimizer, device, vgg_loss_fn, args, train=True):
    model.train(train)
    total = 0.0
    total_dwt = 0.0
    total_vgg = 0.0
    total_ccp = 0.0
    count = 0

    for x_pad, y_pad, y in loader:
        x_pad = x_pad.to(device, non_blocking=True)
        y_pad = y_pad.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred, refine_list = model(x_pad, return_refine=True)
            gt_dwt_list = build_gt_dwt_list(model, y_pad)

            loss_dwt = dwt_refine_loss(gt_dwt_list, refine_list)
            if args.use_vgg_loss:
                loss_vgg = vgg_loss_fn(y, pred)
            else:
                loss_vgg = torch.zeros((), device=device)
            loss_ccp = ccp_loss(y, pred)

            loss = args.dwt_weight * loss_dwt + args.vgg_weight * loss_vgg + args.ccp_weight * loss_ccp

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        bs = x_pad.size(0)
        total += loss.item() * bs
        total_dwt += loss_dwt.item() * bs
        total_vgg += loss_vgg.item() * bs
        total_ccp += loss_ccp.item() * bs
        count += bs

    return {
        "loss": total / max(count, 1),
        "dwt_loss": total_dwt / max(count, 1),
        "vgg_loss": total_vgg / max(count, 1),
        "ccp_loss": total_ccp / max(count, 1),
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch DTCWT desnow model.")
    parser.add_argument("--logPath", type=str, required=True)
    parser.add_argument("--dataPath", type=str, default="/path_to_data/data.npy")
    parser.add_argument("--gtPath", type=str, default="/path_to_gt/gt.npy")
    parser.add_argument("--batchsize", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--modelPath", type=str, default="", help="Checkpoint path for continuing training")
    parser.add_argument("--validation_num", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_vgg_loss", type=int, default=1)
    parser.add_argument("--dwt_weight", type=float, default=2.0)
    parser.add_argument("--vgg_weight", type=float, default=0.1)
    parser.add_argument("--ccp_weight", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    log_dir = Path(args.logPath)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Parameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    with open(log_dir / "inputParam.txt", "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print("load Data")
    data = np.load(args.dataPath)
    print("load Gt")
    label = np.load(args.gtPath)

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
    model = build_DTCWT_model((3, train_data.shape[1] + 32, train_data.shape[2] + 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    start_epoch = 1

    if args.modelPath:
        print("Continue train!")
        ckpt = torch.load(args.modelPath, map_location=device)
        if "model_state_dict" in ckpt:
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

    max_train_batches = args.steps_per_epoch if args.steps_per_epoch > 0 else None

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_dwt_loss",
            "train_vgg_loss",
            "train_ccp_loss",
            "val_loss",
            "val_dwt_loss",
            "val_vgg_loss",
            "val_ccp_loss",
            "lr",
        ]
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer_csv.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            if max_train_batches is not None:
                # Rebuild a limited view over the train loader, equivalent to Keras steps_per_epoch.
                limited_batches = []
                for i, batch in enumerate(train_loader):
                    if i >= max_train_batches:
                        break
                    limited_batches.append(batch)
                train_iterable = limited_batches
            else:
                train_iterable = train_loader

            train_metrics = run_one_epoch(
                model, train_iterable, optimizer, device, vgg_loss_fn, args, train=True
            )
            val_metrics = run_one_epoch(
                model, val_loader, optimizer, device, vgg_loss_fn, args, train=False
            )

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"dwt={val_metrics['dwt_loss']:.6f} "
                f"vgg={val_metrics['vgg_loss']:.6f} "
                f"ccp={val_metrics['ccp_loss']:.6f}"
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dwt_loss": train_metrics["dwt_loss"],
                "train_vgg_loss": train_metrics["vgg_loss"],
                "train_ccp_loss": train_metrics["ccp_loss"],
                "val_loss": val_metrics["loss"],
                "val_dwt_loss": val_metrics["dwt_loss"],
                "val_vgg_loss": val_metrics["vgg_loss"],
                "val_ccp_loss": val_metrics["ccp_loss"],
                "lr": lr,
            }
            writer_csv.writerow(row)
            csv_file.flush()

            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"val/{name}", value, epoch)
                writer.add_scalar("lr", lr, epoch)

            # Save every 100 epochs, matching the original ModelCheckpoint(period=100).
            if epoch % 100 == 0:
                save_checkpoint(log_dir / f"model.{epoch:04d}-{val_metrics['loss']:.4f}.pth", model, optimizer, epoch, best_val_loss, args)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(log_dir / "modelBest.pth", model, optimizer, epoch, best_val_loss, args)
                print(f"Saved best checkpoint: val_loss={best_val_loss:.6f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
