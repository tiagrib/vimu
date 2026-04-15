"""
Compare SAM2 mask quality across models.

For each video/model, stacks all mask frames into a single image using a
yellow-to-blue color gradient (first frame = yellow, last = blue). Then
assembles a grid of all videos per model, so comparing two models is as
easy as placing two images side by side.

Usage:
    # Compare all models found in seg_data/
    python compare_masks.py --data seg_data/

    # Compare specific models
    python compare_masks.py --data seg_data/ --models large tiny

    # Custom output
    python compare_masks.py --data seg_data/ --output comparison/
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np


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


def stack_masks(masks_dir: Path, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Stack all masks into one image with yellow-to-blue color gradient.

    First mask = yellow (255, 255, 0), last mask = blue (0, 100, 255).
    Each mask is blended additively with low opacity so overlapping regions
    show where tracking was consistent.
    """
    mask_paths = sorted(masks_dir.glob("*.png"))
    if not mask_paths:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    canvas = np.zeros((size[1], size[0], 3), dtype=np.float32)
    n = len(mask_paths)

    for i, path in enumerate(mask_paths):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, size)
        binary = (mask > 127).astype(np.float32)

        # Interpolate yellow (0,1) -> blue (1,0) in normalized t
        t = i / max(n - 1, 1)
        # BGR: yellow=(0,255,255), blue=(255,100,0)
        b = t * 255
        g = (1 - t) * 255 + t * 100
        r = (1 - t) * 255

        alpha = 1.0 / max(n * 0.3, 1)  # scale opacity by frame count
        canvas[:, :, 0] += binary * b * alpha
        canvas[:, :, 1] += binary * g * alpha
        canvas[:, :, 2] += binary * r * alpha

    # Normalize luminance so all stacks look equally bright
    result = np.clip(canvas, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_max = hsv[:, :, 2].max()
    if v_max > 0:
        hsv[:, :, 2] = hsv[:, :, 2] * (255.0 / v_max)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def make_grid(
    images: list[np.ndarray],
    labels: list[str],
    cols: int = 0,
    cell_size: tuple[int, int] = (320, 240),
    title: str = "",
) -> np.ndarray:
    """Arrange images into a labeled grid."""
    n = len(images)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    if cols <= 0:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    label_h = 30
    total_cell_h = cell_size[1] + label_h
    title_h = 40 if title else 0

    grid_w = cols * cell_size[0]
    grid_h = title_h + rows * total_cell_h

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    if title:
        cv2.putText(grid, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        x = c * cell_size[0]
        y = title_h + r * total_cell_h

        resized = cv2.resize(img, cell_size)
        grid[y:y + cell_size[1], x:x + cell_size[0]] = resized

        # Label below the image
        cv2.putText(grid, label, (x + 5, y + cell_size[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return grid


def find_models(data_dir: Path) -> list[str]:
    """Find all model names that have masks in any video folder."""
    models = set()
    for video_dir in sorted(data_dir.iterdir()):
        masks_dir = video_dir / "masks"
        if not masks_dir.exists():
            continue
        for model_dir in masks_dir.iterdir():
            if model_dir.is_dir() and list(model_dir.glob("*.png")):
                models.add(model_dir.name)
    return sorted(models)


def find_videos_with_masks(data_dir: Path, model: str) -> list[tuple[str, Path]]:
    """Find all videos that have masks for a given model."""
    results = []
    for video_dir in sorted(data_dir.iterdir()):
        masks_dir = video_dir / "masks" / model
        if masks_dir.exists() and list(masks_dir.glob("*.png")):
            results.append((video_dir.name, masks_dir))
    return results


def get_frame_size(data_dir: Path) -> tuple[int, int]:
    """Get frame dimensions from the first available frame."""
    for video_dir in sorted(data_dir.iterdir()):
        frames_dir = video_dir / "frames"
        if not frames_dir.exists():
            continue
        for frame_path in sorted(frames_dir.glob("*.jpg")):
            img = cv2.imread(str(frame_path))
            if img is not None:
                return (img.shape[1], img.shape[0])
    return (640, 480)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Compare SAM2 mask quality across models")
    parser.add_argument("--data", default=os.environ.get("OUTPUT_DIR", "./seg_data"),
                        help="Segmentation data directory")
    parser.add_argument("--models", nargs="*", help="Models to compare (default: all found)")
    parser.add_argument("--cell-size", default="320x240",
                        help="Grid cell size as WxH (default: 320x240)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = data_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    cw, ch = (int(x) for x in args.cell_size.split("x"))
    cell_size = (cw, ch)

    frame_size = get_frame_size(data_dir)

    available = find_models(data_dir)
    if not available:
        print(f"No masks found in {data_dir}")
        return

    models = args.models if args.models else available
    models = [m for m in models if m in available]
    if not models:
        print(f"None of the requested models found. Available: {available}")
        return

    print(f"Models: {models}")
    print(f"Frame size: {frame_size}")

    for model in models:
        videos = find_videos_with_masks(data_dir, model)
        print(f"\n{model}: {len(videos)} videos")

        stacks = []
        labels = []
        for video_name, masks_dir in videos:
            n_masks = len(list(masks_dir.glob("*.png")))
            print(f"  {video_name}: {n_masks} masks")
            stack = stack_masks(masks_dir, size=frame_size)
            stacks.append(stack)
            labels.append(f"{video_name} ({n_masks}f)")

            # Save individual stack
            ind_path = output_dir / f"{video_name}_{model}.png"
            cv2.imwrite(str(ind_path), stack)

        grid = make_grid(stacks, labels, cell_size=cell_size, title=f"Model: {model}")
        grid_path = output_dir / f"grid_{model}.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"  Grid saved: {grid_path}")

    print(f"\nDone! Compare by opening grid images side by side:")
    for model in models:
        print(f"  {output_dir / f'grid_{model}.png'}")


if __name__ == "__main__":
    main()
