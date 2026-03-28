"""
VIMU Phase 4a: Export trained model to ONNX for Rust inference.

Usage:
    python export_onnx.py --checkpoint ./checkpoints/best.pt --output ./vimu.onnx
"""

import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import VimuModel


def export(checkpoint_path: str, output_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_joints = ckpt["num_joints"]
    output_dim = ckpt["output_dim"]

    model = VimuModel(num_joints=num_joints)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {num_joints} joints, {output_dim} outputs")
    print(f"  Trained to epoch {ckpt['epoch']}, val MAE {ckpt['val_joint_mae']*57.3:.1f}°")

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["state"],
        dynamic_axes={"image": {0: "batch"}, "state": {0: "batch"}},
        opset_version=17,
    )

    # Validate
    onnx.checker.check_model(onnx.load(output_path))

    session = ort.InferenceSession(output_path)
    test_in = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = session.run(None, {"image": test_in})
    assert result[0].shape == (1, output_dim), f"Unexpected shape: {result[0].shape}"
    print(f"ONNX validated: output shape {result[0].shape}")

    # Save metadata for Rust
    meta = {
        "num_joints": num_joints,
        "output_dim": output_dim,
        "input_size": 224,
        "outputs": [f"joint_{i+1}" for i in range(num_joints)] + ["base_roll", "base_pitch"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    meta_path = output_path.replace(".onnx", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported → {output_path}")
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./vimu.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
