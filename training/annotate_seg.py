"""
VIMU v2 Phase 1: Segmentation Data Annotation with SAM2

Two-step workflow:
  1. Annotate: click foreground/background points on each video's first frame
  2. Process: SAM2 propagates masks through all annotated videos in batch

Usage:
    # Annotate only (no GPU needed)
    python annotate_seg.py --video-dir ./videos/ --output seg_data/ --annotate-only

    # Process with default model (large)
    python annotate_seg.py --video-dir ./videos/ --output seg_data/ --process-only

    # Process with a specific model
    python annotate_seg.py --video-dir ./videos/ --output seg_data/ --process-only --model tiny

    # Both (default): annotate then process
    python annotate_seg.py --video-dir ./videos/ --output seg_data/

Controls (annotation):
    L-click     = add foreground point (green)
    R-click     = add background point (red)
    Ctrl+Z      = undo last point
    Enter       = accept annotation
    n           = skip video
    q           = quit annotation

Models:
    tiny        = sam2.1_hiera_t  (fastest, ~156MB)
    small       = sam2.1_hiera_s  (~~350MB)
    base_plus   = sam2.1_hiera_b+ (~350MB)
    large       = sam2.1_hiera_l  (best quality, ~900MB, default)

Output:
    seg_data/
        <video_name>/
            annotations.json        # saved click points (shared across models)
            frames/*.jpg            # extracted video frames (shared across models)
            masks/
                large/*.png         # masks from large model
                tiny/*.png          # masks from tiny model
                ...
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

SAM2_MODELS = {
    "tiny": {
        "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "repo": "facebook/sam2.1-hiera-tiny",
        "file": "sam2.1_hiera_tiny.pt",
    },
    "small": {
        "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "repo": "facebook/sam2.1-hiera-small",
        "file": "sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "repo": "facebook/sam2.1-hiera-base-plus",
        "file": "sam2.1_hiera_base_plus.pt",
    },
    "large": {
        "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "repo": "facebook/sam2.1-hiera-large",
        "file": "sam2.1_hiera_large.pt",
    },
}


def load_sam2(model_name: str = "large", device: str = "cuda"):
    """Load SAM2 model and predictor."""
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    model = SAM2_MODELS[model_name]
    checkpoint = hf_hub_download(model["repo"], filename=model["file"])

    predictor = build_sam2_video_predictor(
        model["cfg"], checkpoint, device=torch.device(device)
    )
    return predictor


def extract_first_frame(video_path: str) -> np.ndarray | None:
    """Extract just the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_frames(video_path: str, every_n: int = 2) -> list[np.ndarray]:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    print(f"  Extracted {len(frames)} frames (every {every_n})")
    return frames


def save_frames(frames: list[np.ndarray], frames_dir: Path):
    """Save extracted frames to disk."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), frame)


def load_frames(frames_dir: Path) -> list[np.ndarray]:
    """Load previously saved frames from disk."""
    paths = sorted(frames_dir.glob("*.jpg"))
    return [cv2.imread(str(p)) for p in paths]


# --- Annotation persistence ---

def save_annotations(points: list[tuple[int, int, int]], path: Path):
    """Save annotation points to JSON."""
    data = [{"x": x, "y": y, "label": l} for x, y, l in points]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_annotations(path: Path) -> list[tuple[int, int, int]] | None:
    """Load annotation points from JSON. Returns None if not found."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [(p["x"], p["y"], p["label"]) for p in data]


# --- Interactive UI ---

def get_click_points(
    frame: np.ndarray,
    existing: list[tuple[int, int, int]] | None = None,
    window_name: str = "Click on the robot",
) -> list[tuple[int, int, int]] | None:
    """Show a frame and collect click points. Pre-loads existing points if provided."""
    points = list(existing) if existing else []
    redraw = [True]

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y, 1))  # foreground
            redraw[0] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y, 0))  # background
            redraw[0] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_click)

    redraw[0] = True

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return None
        if key == ord("n"):
            cv2.destroyWindow(window_name)
            return None
        if key == 26 and points:  # Ctrl+Z
            points.pop()
            redraw[0] = True
        if key == 13 and points:  # Enter
            cv2.destroyWindow(window_name)
            return points
        if redraw[0]:
            display = frame.copy()
            for x, y, label in points:
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(display, (x, y), 5, color, -1)
            fg = sum(1 for _, _, l in points if l == 1)
            bg = len(points) - fg
            cv2.putText(display, f"{fg} fg / {bg} bg | L=fg, R=bg, Ctrl+Z=undo, Enter=accept",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            redraw[0] = False


# --- SAM2 processing ---

def prepare_video_dir(frames: list[np.ndarray], tmp_dir: Path) -> Path:
    """Save frames to a temporary directory for SAM2 video predictor."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        Image.fromarray(rgb).save(tmp_dir / f"{i:06d}.jpg")
    return tmp_dir


def propagate_masks(
    predictor,
    frames: list[np.ndarray],
    click_points: list[tuple[int, int, int]],
    tmp_dir: Path,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Use SAM2 video predictor to propagate mask through all frames."""
    video_dir = prepare_video_dir(frames, tmp_dir)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video_dir))

        points = np.array([[x, y] for x, y, _ in click_points], dtype=np.float32)
        labels = np.array([l for _, _, l in click_points], dtype=np.int32)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        masks = [None] * len(frames)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
            masks[frame_idx] = mask

        predictor.reset_state(state)

    return masks


def save_masks(masks: list[np.ndarray], masks_dir: Path) -> int:
    """Save masks to a directory. Returns count saved."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for mask in masks:
        if mask is None:
            continue
        cv2.imwrite(str(masks_dir / f"{count:06d}.png"), mask)
        count += 1
    return count


def show_preview(frame: np.ndarray, mask: np.ndarray, window: str = "Mask Preview"):
    """Show frame with mask overlay."""
    overlay = frame.copy()
    colored = np.zeros_like(frame)
    colored[:, :, 1] = 180  # green tint
    overlay[mask > 0] = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)[mask > 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    cv2.imshow(window, overlay)


# --- Workflow steps ---

def annotate_videos(videos: list[str], output_dir: Path):
    """Step 1: Annotate all videos interactively. Saves points to JSON."""
    annotated = 0
    for video_path in videos:
        video_name = Path(video_path).stem
        video_out = output_dir / video_name
        ann_path = video_out / "annotations.json"

        existing = load_annotations(ann_path)
        status = f"({len(existing)} pts saved)" if existing else "(new)"
        print(f"\nAnnotating: {video_name} {status}")

        frame = extract_first_frame(video_path)
        if frame is None:
            print("  Cannot read video, skipping")
            continue

        points = get_click_points(frame, existing=existing, window_name=f"Annotate: {video_name}")
        if points is None:
            print("  Skipped")
            continue

        save_annotations(points, ann_path)
        fg = sum(1 for _, _, l in points if l == 1)
        bg = len(points) - fg
        print(f"  Saved {fg} fg + {bg} bg points")
        annotated += 1

    cv2.destroyAllWindows()
    print(f"\nAnnotation done: {annotated}/{len(videos)} videos annotated")


def process_videos(videos: list[str], output_dir: Path, every_n: int, device: str, model_name: str):
    """Step 2: Run SAM2 on all annotated videos."""
    to_process = []
    for video_path in videos:
        video_name = Path(video_path).stem
        video_out = output_dir / video_name
        ann_path = video_out / "annotations.json"
        model_masks_dir = video_out / "masks" / model_name

        if not ann_path.exists():
            continue
        if model_masks_dir.exists() and list(model_masks_dir.glob("*.png")):
            print(f"  Skipping {video_name} ({model_name} masks exist, delete to redo)")
            continue
        to_process.append((video_path, video_out, ann_path))

    if not to_process:
        print("No videos to process (all done or none annotated)")
        return

    print(f"\n{len(to_process)} videos to process with model '{model_name}'. Loading...")
    predictor = load_sam2(model_name, device)

    for video_path, video_out, ann_path in to_process:
        video_name = Path(video_path).stem
        print(f"\nProcessing: {video_name}")

        points = load_annotations(ann_path)
        frames_dir = video_out / "frames"

        # Reuse saved frames if available
        if frames_dir.exists() and list(frames_dir.glob("*.jpg")):
            print("  Loading saved frames...")
            frames = load_frames(frames_dir)
        else:
            frames = extract_frames(video_path, every_n)
            if not frames:
                print("  No frames extracted, skipping")
                continue
            print("  Saving frames...")
            save_frames(frames, frames_dir)

        print(f"  Propagating mask from {len(points)} points ({model_name})...")
        tmp_dir = output_dir / "_tmp_sam2_frames"
        masks = propagate_masks(predictor, frames, points, tmp_dir, device=device)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        # Preview
        print("  Showing preview (press any key to accept, 'q' to discard)...")
        discard = False
        for i in range(0, len(frames), max(1, len(frames) // 10)):
            if masks[i] is not None:
                show_preview(frames[i], masks[i])
                key = cv2.waitKey(500) & 0xFF
                if key == ord("q"):
                    print("  Discarded by user")
                    discard = True
                    break
        cv2.destroyAllWindows()

        if discard:
            continue

        model_masks_dir = video_out / "masks" / model_name
        count = save_masks(masks, model_masks_dir)
        print(f"  Saved {count} masks → masks/{model_name}/")

    print("\nProcessing complete!")


def show_status(videos: list[str], output_dir: Path, video_dir: str):
    """Show annotation and processing status for all videos."""
    print(f"  Video dir:  {Path(video_dir).resolve()}")
    print(f"  Output dir: {output_dir.resolve()}")
    print()

    annotated = 0
    total_models = {}

    for video_path in videos:
        video_name = Path(video_path).stem
        video_out = output_dir / video_name
        ann_path = video_out / "annotations.json"
        frames_dir = video_out / "frames"
        masks_dir = video_out / "masks"

        # Annotation status
        points = load_annotations(ann_path)
        if points:
            annotated += 1
            fg = sum(1 for _, _, l in points if l == 1)
            bg = len(points) - fg
            ann_str = f"{fg} fg + {bg} bg pts"
        else:
            ann_str = "not annotated"

        # Frames status
        n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        frames_str = f"{n_frames} frames" if n_frames else "no frames"

        # Model masks status
        models_done = []
        if masks_dir.exists():
            for model_dir in sorted(masks_dir.iterdir()):
                if model_dir.is_dir():
                    n_masks = len(list(model_dir.glob("*.png")))
                    if n_masks:
                        models_done.append(f"{model_dir.name}({n_masks})")
                        total_models[model_dir.name] = total_models.get(model_dir.name, 0) + 1
        masks_str = ", ".join(models_done) if models_done else "none"

        print(f"  {video_name:20s}  ann: {ann_str:20s}  {frames_str:12s}  masks: {masks_str}")

    # Summary
    print(f"\n  Total: {len(videos)} videos, {annotated} annotated")
    if total_models:
        model_summary = ", ".join(f"{m}: {c}" for m, c in sorted(total_models.items()))
        print(f"  Masks by model: {model_summary}")


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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="VIMU v2: SAM2 segmentation annotation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", help="Path to a single video file")
    group.add_argument("--video-dir", default=os.environ.get("VIDEO_DIR"),
                       help="Directory containing video files")
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "./seg_data"),
                        help="Output directory")
    parser.add_argument("--every-n", type=int,
                        default=int(os.environ.get("EVERY_N", "2")),
                        help="Extract every Nth frame from video (default: 2)")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"),
                        help="Device for SAM2 (default: cuda)")
    parser.add_argument("--model", default=os.environ.get("MODEL", "large"),
                        choices=list(SAM2_MODELS.keys()),
                        help="SAM2 model size (default: large)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--annotate-only", action="store_true",
                      help="Only annotate, don't run SAM2")
    mode.add_argument("--process-only", action="store_true",
                      help="Only run SAM2 on already-annotated videos")
    mode.add_argument("--status", action="store_true",
                      help="Show annotation and processing status")
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        parser.error("--video or --video-dir is required (or set VIDEO_DIR in .env)")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        videos = [args.video]
    else:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = sorted(
            str(p) for p in Path(args.video_dir).iterdir()
            if p.suffix.lower() in video_exts
        )
    print(f"Found {len(videos)} video files")

    if args.status:
        show_status(videos, output_dir, args.video or args.video_dir)
    elif args.process_only:
        process_videos(videos, output_dir, args.every_n, args.device, args.model)
    elif args.annotate_only:
        annotate_videos(videos, output_dir)
    else:
        annotate_videos(videos, output_dir)
        process_videos(videos, output_dir, args.every_n, args.device, args.model)


if __name__ == "__main__":
    main()
