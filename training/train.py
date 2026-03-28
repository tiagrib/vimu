"""
VIMU Phase 3: Training

Usage:
    python train.py --data-dir ./data --num-joints 6 --epochs 100
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import VimuModel
from dataset import VimuDataset, get_train_transforms, get_val_transforms


def masked_mse(pred, target, mask=None):
    diff = (pred - target) ** 2
    if mask is not None and mask.any():
        diff = diff * mask.float()
        return diff.sum() / mask.float().sum().clamp(min=1)
    return diff.mean()


def train_epoch(model, loader, optimizer, device, joint_w=1.0, base_w=0.5):
    model.train()
    total, joint_total, base_total, n = 0, 0, 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        joints_t = batch["joints"].to(device)
        base_t = batch["base"].to(device)
        base_m = batch["base_mask"].to(device)

        optimizer.zero_grad()
        out = model(images)

        num_j = joints_t.shape[1]
        j_loss = nn.functional.mse_loss(out[:, :num_j], joints_t)

        if base_m.any():
            b_loss = masked_mse(out[:, num_j:], base_t, base_m)
        else:
            b_loss = torch.tensor(0.0, device=device)

        loss = joint_w * j_loss + base_w * b_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += loss.item()
        joint_total += j_loss.item()
        base_total += b_loss.item()
        n += 1

    return total / n, joint_total / n, base_total / n


@torch.no_grad()
def validate(model, loader, device, num_joints):
    model.eval()
    joint_mae_sum, base_mae_sum, n, n_base = 0, 0, 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        joints_t = batch["joints"].to(device)
        base_t = batch["base"].to(device)
        base_m = batch["base_mask"].to(device)

        out = model(images)
        joint_mae_sum += (out[:, :num_joints] - joints_t).abs().mean().item()

        if base_m.any():
            masked = (out[:, num_joints:] - base_t).abs() * base_m.float()
            base_mae_sum += (masked.sum() / base_m.float().sum().clamp(1)).item()
            n_base += 1
        n += 1

    return joint_mae_sum / n, base_mae_sum / max(n_base, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-joints", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--base-weight", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    full = VimuDataset(args.data_dir, args.num_joints, get_train_transforms())
    val_n = int(len(full) * args.val_split)
    train_n = len(full) - val_n
    train_set, val_set = random_split(full, [train_n, val_n])

    val_ds = VimuDataset(args.data_dir, args.num_joints, get_val_transforms())
    val_set = torch.utils.data.Subset(val_ds, val_set.indices)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=2, pin_memory=True)
    print(f"Train: {train_n} | Val: {val_n}")

    # Model
    model = VimuModel(args.num_joints).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging
    log_path = os.path.join(args.output_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "j_loss", "b_loss", "val_j_mae", "val_b_mae", "lr"])

    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        loss, j_loss, b_loss = train_epoch(
            model, train_loader, optimizer, device,
            args.joint_weight, args.base_weight,
        )
        val_j_mae, val_b_mae = validate(model, val_loader, device, args.num_joints)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
              f"Loss {loss:.4f} | "
              f"Joint MAE {val_j_mae:.4f} ({val_j_mae * 57.3:.1f}°) | "
              f"Base MAE {val_b_mae:.4f} | "
              f"LR {lr:.2e}")

        if val_j_mae < best_mae:
            best_mae = val_j_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_joints": args.num_joints,
                "output_dim": model.output_dim,
                "val_joint_mae": val_j_mae,
            }, os.path.join(args.output_dir, "best.pt"))
            print(f"  → Saved best ({val_j_mae * 57.3:.1f}°)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss, j_loss, b_loss, val_j_mae, val_b_mae, lr])

    print(f"\nDone. Best joint MAE: {best_mae * 57.3:.1f}°")
    print(f"Export: python export_onnx.py --checkpoint {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
