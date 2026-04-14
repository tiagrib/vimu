"""
VIMU v2 Phase 5: Export segmentor to ONNX.

Thin wrapper around Ultralytics' built-in ONNX export.

Usage:
    python export_seg.py --model vimu_seg.pt --output vimu_seg.onnx
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO segmentor to ONNX")
    parser.add_argument("--model", required=True, help="YOLO checkpoint (vimu_seg.pt)")
    parser.add_argument("--output", default="vimu_seg.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    export_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)

    # Move to requested output path
    if export_path and Path(export_path).exists() and str(export_path) != args.output:
        shutil.copy2(export_path, args.output)

    print(f"Segmentor exported to {args.output}")
    print(f"Size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
