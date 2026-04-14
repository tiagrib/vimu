"""
VIMU v2 Phase 2: Train YOLO11n-seg segmentor from SAM2 masks.

Converts binary masks to YOLO polygon format, then fine-tunes YOLO11n-seg.

Usage:
    python train_segmentor.py --data seg_data/ --output vimu_seg.pt --epochs 50

Input:
    seg_data/
        frames/*.jpg
        masks/*.png     # binary masks (255 = robot)

Output:
    vimu_seg.pt         # YOLO checkpoint for real-time segmentation
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_polygon(mask: np.ndarray) -> str | None:
    """Convert a binary mask to YOLO polygon annotation (class 0).

    Returns a line like: "0 x1 y1 x2 y2 ... xN yN" with normalized coords,
    or None if no contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    # Simplify to reduce point count
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    h, w = mask.shape[:2]
    points = contour.reshape(-1, 2)
    normalized = []
    for x, y in points:
        normalized.append(f"{x / w:.6f}")
        normalized.append(f"{y / h:.6f}")

    return "0 " + " ".join(normalized)


def prepare_yolo_dataset(data_dir: Path, yolo_dir: Path, val_split: float = 0.15):
    """Convert seg_data/ to YOLO segmentation format."""
    frames_dir = data_dir / "frames"
    masks_dir = data_dir / "masks"

    # Collect valid pairs
    pairs = []
    for mask_path in sorted(masks_dir.glob("*.png")):
        frame_path = frames_dir / f"{mask_path.stem}.jpg"
        if frame_path.exists():
            pairs.append((frame_path, mask_path))

    if not pairs:
        raise ValueError(f"No frame/mask pairs found in {data_dir}")

    print(f"Found {len(pairs)} frame/mask pairs")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    val_n = int(len(pairs) * val_split)
    val_indices = set(indices[:val_n])

    # Create YOLO directory structure
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    for i, (frame_path, mask_path) in enumerate(pairs):
        split = "val" if i in val_indices else "train"
        name = frame_path.stem

        # Copy image
        shutil.copy2(frame_path, yolo_dir / "images" / split / f"{name}.jpg")

        # Convert mask to polygon label
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        label = mask_to_yolo_polygon(mask)
        if label is None:
            skipped += 1
            continue

        label_path = yolo_dir / "labels" / split / f"{name}.txt"
        label_path.write_text(label + "\n")

    if skipped:
        print(f"  Skipped {skipped} frames with no valid contour")

    # Create dataset YAML
    yaml_path = yolo_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {yolo_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: robot\n"
    )

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="VIMU v2: Train YOLO segmentor")
    parser.add_argument("--data", required=True, help="Path to seg_data/ directory")
    parser.add_argument("--output", default="vimu_seg.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data)
    yolo_dir = data_dir / "_yolo_format"

    print("Converting masks to YOLO polygon format...")
    yaml_path = prepare_yolo_dataset(data_dir, yolo_dir, args.val_split)

    print(f"\nTraining YOLO11n-seg for {args.epochs} epochs...")
    from ultralytics import YOLO

    model = YOLO("yolo11n-seg.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        task="segment",
        verbose=True,
    )

    # Copy best checkpoint
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        shutil.copy2(best_path, args.output)
        print(f"\nSaved best checkpoint to {args.output}")
    else:
        last_path = Path(results.save_dir) / "weights" / "last.pt"
        if last_path.exists():
            shutil.copy2(last_path, args.output)
            print(f"\nSaved last checkpoint to {args.output}")

    print(f"Next step: python collect_pose.py --seg-model {args.output} ...")


if __name__ == "__main__":
    main()
