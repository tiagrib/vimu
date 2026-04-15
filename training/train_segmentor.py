"""
VIMU v2 Phase 2: Train YOLO11n-seg segmentor from SAM2 masks.

Collects frame/mask pairs from the per-video seg_data structure,
converts binary masks to YOLO polygon format, then fine-tunes YOLO11n-seg.

Usage:
    # Uses large model masks by default
    python train_segmentor.py --data seg_data/

    # Use a specific model's masks
    python train_segmentor.py --data seg_data/ --model tiny

Input:
    seg_data/
        <video_name>/
            frames/*.jpg
            masks/<model>/*.png

Output:
    vimu_seg.pt         # YOLO checkpoint for real-time segmentation
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SAM2_MODEL_PRIORITY = ["large", "base_plus", "small", "tiny"]


def load_dotenv():
    """Load .env file from the script's directory if it exists."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def detect_best_model(data_dir: Path) -> str | None:
    """Find the best available model by priority (large > base_plus > small > tiny)."""
    available = set()
    for video_dir in data_dir.iterdir():
        masks_dir = video_dir / "masks"
        if not masks_dir.exists():
            continue
        for model_dir in masks_dir.iterdir():
            if model_dir.is_dir():
                available.add(model_dir.name)
    for model in SAM2_MODEL_PRIORITY:
        if model in available:
            return model
    return None


def preflight_check(data_dir: Path, model: str) -> list[tuple[Path, Path]]:
    """Verify all videos have matching frames and masks for the given model.

    Returns list of (frames_dir, masks_dir) pairs. Aborts if any video is incomplete.
    """
    video_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name != "_yolo_format" and d.name != "comparison"
    )

    if not video_dirs:
        print(f"ERROR: No video folders found in {data_dir}")
        sys.exit(1)

    valid = []
    errors = []
    total_frames = 0
    total_masks = 0

    print(f"Pre-flight check (model: {model})")
    print(f"{'Video':25s}  {'Frames':>8s}  {'Masks':>8s}  {'Status'}")
    print("-" * 65)

    for video_dir in video_dirs:
        frames_dir = video_dir / "frames"
        masks_dir = video_dir / "masks" / model
        ann_path = video_dir / "annotations.json"

        if not ann_path.exists():
            # Not annotated — skip silently
            continue

        n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        n_masks = len(list(masks_dir.glob("*.png"))) if masks_dir.exists() else 0

        total_frames += n_frames
        total_masks += n_masks

        if n_frames == 0:
            status = "MISSING FRAMES"
            errors.append(f"  {video_dir.name}: no frames extracted")
        elif n_masks == 0:
            status = "MISSING MASKS"
            errors.append(f"  {video_dir.name}: no {model} masks (run --process-only --model {model})")
        elif n_frames != n_masks:
            status = f"MISMATCH ({n_frames} vs {n_masks})"
            errors.append(f"  {video_dir.name}: {n_frames} frames but {n_masks} masks")
        else:
            status = "OK"
            valid.append((frames_dir, masks_dir))

        print(f"{video_dir.name:25s}  {n_frames:>8d}  {n_masks:>8d}  {status}")

    print("-" * 65)
    print(f"{'Total':25s}  {total_frames:>8d}  {total_masks:>8d}  {len(valid)} videos OK")

    if errors:
        print(f"\nERROR: {len(errors)} video(s) have issues:")
        for e in errors:
            print(e)
        print("\nFix the issues above and re-run. Aborting.")
        sys.exit(1)

    if not valid:
        print("\nERROR: No valid video folders with frames and masks found.")
        sys.exit(1)

    print(f"\nReady to train on {total_frames} frames from {len(valid)} videos.\n")
    return valid


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


def prepare_yolo_dataset(
    video_pairs: list[tuple[Path, Path]],
    yolo_dir: Path,
    val_split: float = 0.15,
) -> Path:
    """Convert per-video frame/mask pairs to YOLO segmentation format."""
    # Collect all pairs across videos
    pairs = []
    for frames_dir, masks_dir in video_pairs:
        video_name = frames_dir.parent.name
        for mask_path in sorted(masks_dir.glob("*.png")):
            frame_path = frames_dir / f"{mask_path.stem}.jpg"
            if frame_path.exists():
                pairs.append((frame_path, mask_path, video_name))

    if not pairs:
        print("ERROR: No valid frame/mask pairs found.")
        sys.exit(1)

    print(f"Preparing {len(pairs)} frame/mask pairs for YOLO training...")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    val_n = int(len(pairs) * val_split)
    val_indices = set(indices[:val_n])

    # Create YOLO directory structure
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    for i, (frame_path, mask_path, video_name) in enumerate(pairs):
        split = "val" if i in val_indices else "train"
        # Use video_name prefix to avoid filename collisions across videos
        name = f"{video_name}_{frame_path.stem}"

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

    train_n = len(pairs) - val_n
    print(f"  Train: {train_n}, Val: {val_n}")
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
    load_dotenv()

    parser = argparse.ArgumentParser(description="VIMU v2: Train YOLO segmentor")
    parser.add_argument("--data", default=os.environ.get("OUTPUT_DIR", "./seg_data"),
                        help="Path to seg_data/ directory")
    parser.add_argument("--model", default=os.environ.get("MODEL"),
                        help="Which SAM2 model's masks to use (default: best available)")
    parser.add_argument("--output", default="vimu_seg.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data)

    # Determine which model's masks to use
    model = args.model
    if not model:
        model = detect_best_model(data_dir)
        if not model:
            print("ERROR: No masks found in any video folder.")
            sys.exit(1)
        print(f"Auto-detected best model: {model}\n")

    # Pre-flight check
    video_pairs = preflight_check(data_dir, model)

    # Prepare YOLO dataset
    yolo_dir = data_dir / "_yolo_format"
    yaml_path = prepare_yolo_dataset(video_pairs, yolo_dir, args.val_split)

    print(f"Training YOLO11n-seg for {args.epochs} epochs...")
    from ultralytics import YOLO

    yolo = YOLO("yolo11n-seg.pt")
    results = yolo.train(
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
