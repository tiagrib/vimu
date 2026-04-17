"""
VIMU v2 Phase 1: Segmentation Data Annotation with SAM2

Unified collection model:
  Every subdirectory of seg_data/ with a frames/ folder is a "collection".
  Each collection is either:
    - sequence mode (default): annotate FIRST frame, SAM2 video-propagates through all frames
    - nonseq mode: annotate EVERY frame individually, SAM2 image-segments each

  To mark a collection as nonseq, drop an empty `nonseq` file in its folder.

Workflow:
  1. Populate seg_data/<name>/frames/ either by:
     - Running with --video-dir: videos are extracted into seg_data/<stem>/frames/
     - Manually copying frames (e.g. from pose_data/.../raw/) into seg_data/<name>/frames/
  2. Run --annotate-only to annotate all un-annotated collections
  3. Run --process-only to generate masks for the chosen SAM2 model
  4. Train: train_segmentor.py picks up all collections with matching masks

Usage:
    # Extract frames from videos + annotate + process (full Phase 1)
    python annotate_seg.py --video-dir ./videos/ --output seg_data/

    # Just annotate existing collections
    python annotate_seg.py --annotate-only

    # Just process (generate masks)
    python annotate_seg.py --process-only

    # Specify SAM2 model variant for processing
    python annotate_seg.py --process-only --model tiny

    # Refinement: manually populate seg_data/refinement_v1/frames/ then:
    python annotate_seg.py            # picks it up automatically

    # Mark a collection as per-frame annotation mode:
    touch seg_data/refinement_v1/nonseq
    python annotate_seg.py

Controls (annotation):
    L-click     = add foreground point (green)
    R-click     = add background point (red)
    Ctrl+Z      = undo last point
    Enter       = accept annotation
    n / q       = skip / quit

Models:
    tiny        = sam2.1_hiera_t  (fastest, ~156MB)
    small       = sam2.1_hiera_s  (~350MB)
    base_plus   = sam2.1_hiera_b+ (~350MB)
    large       = sam2.1_hiera_l  (best quality, ~900MB, default)

Output:
    seg_data/
        <collection>/
            frames/*.jpg            # source frames (from video or manual)
            annotations.json        # click points (list for seq, dict for nonseq)
            nonseq                  # optional marker file — presence = per-frame mode
            masks/
                large/*.png
                tiny/*.png
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
    """Load SAM2 model and video predictor (for video mask propagation)."""
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    model = SAM2_MODELS[model_name]
    checkpoint = hf_hub_download(model["repo"], filename=model["file"])

    predictor = build_sam2_video_predictor(
        model["cfg"], checkpoint, device=torch.device(device)
    )
    return predictor


def load_sam2_image(model_name: str = "large", device: str = "cuda"):
    """Load SAM2 model and image predictor (for single-frame segmentation)."""
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = SAM2_MODELS[model_name]
    checkpoint = hf_hub_download(model["repo"], filename=model["file"])

    sam_model = build_sam2(model["cfg"], checkpoint, device=torch.device(device))
    return SAM2ImagePredictor(sam_model)


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
    """Save annotation points to JSON (video mode: single list of points)."""
    data = [{"x": x, "y": y, "label": l} for x, y, l in points]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_annotations(path: Path) -> list[tuple[int, int, int]] | None:
    """Load annotation points from JSON (video mode). Returns None if not found."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # Only return if it's the old format (list); images mode uses a dict
    if not isinstance(data, list):
        return None
    return [(p["x"], p["y"], p["label"]) for p in data]


def save_image_annotations(ann_by_image: dict, path: Path):
    """Save per-image annotations (images mode: dict keyed by filename)."""
    data = {
        name: [{"x": x, "y": y, "label": l} for x, y, l in pts]
        for name, pts in ann_by_image.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_image_annotations(path: Path) -> dict:
    """Load per-image annotations. Returns dict keyed by filename (or empty dict)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return {}
    return {
        name: [(p["x"], p["y"], p["label"]) for p in pts]
        for name, pts in data.items()
    }


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


# --- Unified collection workflow ---

SKIP_DIRS = {"_yolo_format", "comparison", "_tmp_sam2_frames"}


def _find_images(d: Path) -> list[Path]:
    """List image files in a directory (jpg/jpeg/png)."""
    exts = {".jpg", ".jpeg", ".png"}
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def discover_collections(output_dir: Path) -> list[Path]:
    """Return all collection folders (subdirs with a frames/ populated)."""
    if not output_dir.exists():
        return []
    return sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and _find_images(d / "frames")
    )


def is_nonseq(collection_dir: Path) -> bool:
    """A collection is nonseq if a 'nonseq' marker file is present."""
    return (collection_dir / "nonseq").exists()


def extract_video_frames(videos: list[str], output_dir: Path, every_n: int):
    """Populate seg_data/<stem>/frames/ for each video (skip if already extracted)."""
    for video_path in videos:
        name = Path(video_path).stem
        frames_dir = output_dir / name / "frames"
        if _find_images(frames_dir):
            continue
        frames = extract_frames(video_path, every_n)
        if not frames:
            continue
        save_frames(frames, frames_dir)
        print(f"  Extracted {name}: {len(frames)} frames")


def annotate_collections(output_dir: Path):
    """Annotate each collection (first frame for seq, every frame for nonseq)."""
    collections = discover_collections(output_dir)
    if not collections:
        print(f"No collections found in {output_dir}")
        return
    print(f"\n{len(collections)} collections discovered:")
    for c in collections:
        mode = "nonseq" if is_nonseq(c) else "seq"
        print(f"  {c.name:30s}  [{mode}]  {len(_find_images(c / 'frames'))} frames")
    print()

    for coll in collections:
        if is_nonseq(coll):
            _annotate_nonseq(coll)
        else:
            _annotate_seq(coll)
    cv2.destroyAllWindows()


def _annotate_seq(coll: Path):
    """Annotate the first frame of a sequence collection."""
    frames_dir = coll / "frames"
    ann_path = coll / "annotations.json"
    frame_paths = _find_images(frames_dir)
    if not frame_paths:
        return

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print(f"[seq] {coll.name}: cannot read {frame_paths[0].name}")
        return

    existing = load_annotations(ann_path)
    status = f"({len(existing)} pts saved)" if existing else "(new)"
    print(f"[seq] {coll.name} {status}  {len(frame_paths)} frames")

    points = get_click_points(first_frame, existing=existing,
                              window_name=f"Annotate (seq): {coll.name}")
    if points is None:
        print("  Skipped")
        return

    save_annotations(points, ann_path)
    fg = sum(1 for _, _, l in points if l == 1)
    bg = len(points) - fg
    print(f"  Saved {fg} fg + {bg} bg points")


def _annotate_nonseq(coll: Path):
    """Annotate every frame in a non-sequence collection individually."""
    frames_dir = coll / "frames"
    ann_path = coll / "annotations.json"
    ann_by_image = load_image_annotations(ann_path)

    frame_paths = _find_images(frames_dir)
    n_done = sum(1 for p in frame_paths if p.name in ann_by_image)
    print(f"[nonseq] {coll.name}: {n_done}/{len(frame_paths)} already annotated")

    for img_path in frame_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        existing = ann_by_image.get(img_path.name)
        status = f"({len(existing)} pts)" if existing else "(new)"
        print(f"  {img_path.name} {status}")
        points = get_click_points(frame, existing=existing,
                                  window_name=f"Annotate (nonseq): {coll.name}/{img_path.name}")
        if points is None:
            print("    Skipped")
            continue
        ann_by_image[img_path.name] = points
        save_image_annotations(ann_by_image, ann_path)


def process_collections(output_dir: Path, device: str, model_name: str):
    """Generate masks for all annotated collections using the chosen SAM2 variant."""
    collections = discover_collections(output_dir)
    seq_todo: list[Path] = []
    nonseq_todo: list[Path] = []

    for coll in collections:
        ann_path = coll / "annotations.json"
        if not ann_path.exists():
            continue
        model_masks_dir = coll / "masks" / model_name
        if model_masks_dir.exists() and list(model_masks_dir.glob("*.png")):
            print(f"  Skipping {coll.name} ({model_name} masks exist, delete to redo)")
            continue
        if is_nonseq(coll):
            nonseq_todo.append(coll)
        else:
            seq_todo.append(coll)

    if not seq_todo and not nonseq_todo:
        print("Nothing to process (all done or none annotated)")
        return

    # Sequence collections: video predictor
    if seq_todo:
        print(f"\n{len(seq_todo)} seq collections to process with SAM2-{model_name}. Loading...")
        predictor = load_sam2(model_name, device)
        for coll in seq_todo:
            _process_seq(coll, predictor, output_dir, device, model_name)

    # Non-sequence collections: image predictor
    if nonseq_todo:
        print(f"\n{len(nonseq_todo)} nonseq collections to process with SAM2-{model_name}. Loading...")
        predictor = load_sam2_image(model_name, device)
        for coll in nonseq_todo:
            _process_nonseq(coll, predictor, device, model_name)

    print("\nProcessing complete!")


def _process_seq(coll: Path, predictor, output_dir: Path, device: str, model_name: str):
    name = coll.name
    frames_dir = coll / "frames"
    ann_path = coll / "annotations.json"

    points = load_annotations(ann_path)
    if not points:
        print(f"[seq] {name}: no valid annotations, skipping")
        return
    frames = load_frames(frames_dir)
    if not frames:
        print(f"[seq] {name}: no frames loaded, skipping")
        return

    print(f"\n[seq] {name}: propagating {len(points)} points over {len(frames)} frames")
    tmp_dir = output_dir / "_tmp_sam2_frames"
    masks = propagate_masks(predictor, frames, points, tmp_dir, device=device)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    model_masks_dir = coll / "masks" / model_name
    count = save_masks(masks, model_masks_dir)
    print(f"  Saved {count} masks -> masks/{model_name}/")


def _process_nonseq(coll: Path, predictor, device: str, model_name: str):
    name = coll.name
    frames_dir = coll / "frames"
    ann_path = coll / "annotations.json"
    model_masks_dir = coll / "masks" / model_name
    model_masks_dir.mkdir(parents=True, exist_ok=True)

    ann_by_image = load_image_annotations(ann_path)
    if not ann_by_image:
        print(f"[nonseq] {name}: no annotations, skipping")
        return

    print(f"\n[nonseq] {name}: {len(ann_by_image)} annotated frames")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        saved = 0
        for fname, pts in ann_by_image.items():
            mask_path = model_masks_dir / f"{Path(fname).stem}.png"
            if mask_path.exists():
                continue
            img_path = frames_dir / fname
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)
            point_coords = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
            point_labels = np.array([l for _, _, l in pts], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = (masks[0] > 0).astype(np.uint8) * 255
            cv2.imwrite(str(mask_path), mask)
            saved += 1
        print(f"  Saved {saved} masks -> masks/{model_name}/")


def show_status(output_dir: Path, video_dir: str | None):
    """Show annotation and processing status for all collections in seg_data/."""
    print(f"  Output dir: {output_dir.resolve()}")
    if video_dir:
        print(f"  Video dir:  {Path(video_dir).resolve()}")
    print()

    collections = discover_collections(output_dir)
    if not collections:
        print("  No collections found.")
        return

    total_models: dict[str, int] = {}
    annotated = 0

    for coll in collections:
        frames_dir = coll / "frames"
        ann_path = coll / "annotations.json"
        masks_dir = coll / "masks"
        mode = "nonseq" if is_nonseq(coll) else "seq"

        n_frames = len(_find_images(frames_dir))
        frames_str = f"{n_frames} frames"

        if mode == "seq":
            points = load_annotations(ann_path)
            if points:
                annotated += 1
                fg = sum(1 for _, _, l in points if l == 1)
                bg = len(points) - fg
                ann_str = f"{fg} fg + {bg} bg"
            else:
                ann_str = "not annotated"
        else:
            ann = load_image_annotations(ann_path)
            if ann:
                annotated += 1
                ann_str = f"{len(ann)}/{n_frames} annotated"
            else:
                ann_str = "not annotated"

        models_done = []
        if masks_dir.exists():
            for model_dir in sorted(masks_dir.iterdir()):
                if model_dir.is_dir():
                    n_masks = len(list(model_dir.glob("*.png")))
                    if n_masks:
                        models_done.append(f"{model_dir.name}({n_masks})")
                        total_models[model_dir.name] = total_models.get(model_dir.name, 0) + 1
        masks_str = ", ".join(models_done) if models_done else "none"

        print(f"  {coll.name:30s}  [{mode:6s}]  {frames_str:12s}  ann: {ann_str:22s}  masks: {masks_str}")

    print(f"\n  Total: {len(collections)} collections, {annotated} annotated")
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
    group.add_argument("--video", help="Path to a single video file (extracts to seg_data/<stem>/frames/)")
    group.add_argument("--video-dir", default=os.environ.get("VIDEO_DIR"),
                       help="Directory of videos (each extracted to seg_data/<stem>/frames/)")
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "./seg_data"),
                        help="Output directory (contains collection folders)")
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
                      help="Only run SAM2 on already-annotated collections")
    mode.add_argument("--status", action="store_true",
                      help="Show annotation and processing status")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: extract frames from videos into collection folders
    if args.video or args.video_dir:
        if args.video:
            videos = [args.video]
        else:
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            videos = sorted(
                str(p) for p in Path(args.video_dir).iterdir()
                if p.suffix.lower() in video_exts
            )
        if videos:
            print(f"Found {len(videos)} video files in {args.video_dir or args.video}")
            extract_video_frames(videos, output_dir, args.every_n)

    # Dispatch action
    if args.status:
        show_status(output_dir, args.video or args.video_dir)
    elif args.process_only:
        process_collections(output_dir, args.device, args.model)
    elif args.annotate_only:
        annotate_collections(output_dir)
    else:
        annotate_collections(output_dir)
        process_collections(output_dir, args.device, args.model)


if __name__ == "__main__":
    main()
