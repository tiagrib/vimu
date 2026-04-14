"""
VIMU v2 Phase 1: Segmentation Data Annotation with SAM2

Interactive tool that uses SAM2-tiny to generate binary masks from video
clips of the robot. You click once on the robot in the first frame, and
SAM2 propagates the mask through the entire clip.

Usage:
    # Single video
    python annotate_seg.py --video robot_orbit.mp4 --output seg_data/

    # Directory of videos
    python annotate_seg.py --video-dir ./videos/ --output seg_data/

    # Adjust frame extraction rate
    python annotate_seg.py --video robot.mp4 --output seg_data/ --every-n 3

Controls:
    Click       = add point prompt on the robot (first frame or correction)
    Enter       = accept and propagate mask through video
    n           = skip to next video
    q           = quit

Output:
    seg_data/
        <video_name>/
            frames/*.jpg    # extracted video frames
            masks/*.png     # binary masks (255 = robot, 0 = background)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def load_sam2(device: str = "cuda"):
    """Load SAM2-tiny model and predictor."""
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint = hf_hub_download(
        "facebook/sam2.1-hiera-tiny", filename="sam2.1_hiera_tiny.pt"
    )

    predictor = build_sam2_video_predictor(
        model_cfg, checkpoint, device=torch.device(device)
    )
    return predictor


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
    print(f"  Extracted {len(frames)} frames from {video_path} (every {every_n})")
    return frames


def get_click_points(frame: np.ndarray, window_name: str = "Click on the robot") -> list[tuple[int, int, int]]:
    """Show a frame and collect click points. Returns list of (x, y, label) where label=1 foreground, 0 background."""
    points = []  # (x, y, label)
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

    display = frame.copy()
    cv2.putText(display, "L=foreground, R=background, Ctrl+Z=undo, Enter=accept",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
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


def prepare_video_dir(frames: list[np.ndarray], tmp_dir: Path) -> Path:
    """Save frames to a temporary directory for SAM2 video predictor."""
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
) -> list[np.ndarray]:
    """Use SAM2 video predictor to propagate mask through all frames."""
    video_dir = prepare_video_dir(frames, tmp_dir)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video_dir))

        # Add all clicks as point prompts on frame 0
        points = np.array([[x, y] for x, y, _ in click_points], dtype=np.float32)
        labels = np.array([l for _, _, l in click_points], dtype=np.int32)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate through all frames
        masks = [None] * len(frames)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
            masks[frame_idx] = mask

        predictor.reset_state(state)

    return masks


def save_results(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    video_dir: Path,
) -> int:
    """Save frames and masks to a per-video output directory. Returns count saved."""
    frames_dir = video_dir / "frames"
    masks_dir = video_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame, mask in zip(frames, masks):
        if mask is None:
            continue
        fname = f"{count:06d}"
        cv2.imwrite(str(frames_dir / f"{fname}.jpg"), frame)
        cv2.imwrite(str(masks_dir / f"{fname}.png"), mask)
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


def process_video(predictor, video_path: str, output_dir: Path, every_n: int) -> bool:
    """Process a single video: extract, prompt, propagate, save. Returns True if saved."""
    video_name = Path(video_path).stem
    video_out = output_dir / video_name

    if video_out.exists():
        print(f"\nSkipping (already done): {video_path}")
        return True

    print(f"\nProcessing: {video_path}")
    frames = extract_frames(video_path, every_n)
    if not frames:
        print("  No frames extracted, skipping")
        return False

    clicks = get_click_points(frames[0])
    if clicks is None:
        print("  Skipped by user")
        return False

    print(f"  Propagating mask from {len(clicks)} points...")
    tmp_dir = output_dir / "_tmp_sam2_frames"
    masks = propagate_masks(predictor, frames, clicks, tmp_dir)

    # Clean up temp dir
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Preview
    print("  Showing preview (press any key to accept, 'q' to discard)...")
    for i in range(0, len(frames), max(1, len(frames) // 10)):
        if masks[i] is not None:
            show_preview(frames[i], masks[i])
            key = cv2.waitKey(500) & 0xFF
            if key == ord("q"):
                print("  Discarded by user")
                cv2.destroyAllWindows()
                return False

    cv2.destroyAllWindows()

    count = save_results(frames, masks, video_out)
    print(f"  Saved {count} frame/mask pairs → {video_out}")
    return True


def main():
    parser = argparse.ArgumentParser(description="VIMU v2: SAM2 segmentation annotation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single video file")
    group.add_argument("--video-dir", help="Directory containing video files")
    parser.add_argument("--output", default="./seg_data", help="Output directory for frames + masks")
    parser.add_argument("--every-n", type=int, default=2,
                        help="Extract every Nth frame from video (default: 2)")
    parser.add_argument("--device", default="cuda", help="Device for SAM2 (default: cuda)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM2-tiny...")
    predictor = load_sam2(args.device)

    if args.video:
        videos = [args.video]
    else:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = sorted(
            str(p) for p in Path(args.video_dir).iterdir()
            if p.suffix.lower() in video_exts
        )
        print(f"Found {len(videos)} video files")

    done = 0
    for video in videos:
        if process_video(predictor, video, output_dir, args.every_n):
            done += 1

    print(f"\nDone! {done}/{len(videos)} videos processed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
