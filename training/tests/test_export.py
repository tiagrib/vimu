import json
import os
import tempfile

import onnx
import torch
import pytest

from model import VimuModel
from export_onnx import export


@pytest.fixture
def fake_checkpoint():
    """Create a temporary checkpoint file and output path."""
    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, "test_best.pt")
    onnx_path = os.path.join(tmpdir, "test_model.onnx")

    num_joints = 6
    model = VimuModel(num_joints=num_joints)

    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "num_joints": num_joints,
        "output_dim": model.output_dim,
        "val_joint_mae": 0.05,
    }, ckpt_path)

    yield ckpt_path, onnx_path, num_joints, model.output_dim

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


class TestExportOnnx:
    def test_export_creates_file(self, fake_checkpoint):
        ckpt_path, onnx_path, num_joints, output_dim = fake_checkpoint
        export(ckpt_path, onnx_path)
        assert os.path.isfile(onnx_path), "ONNX file should exist after export"

    def test_onnx_is_valid(self, fake_checkpoint):
        ckpt_path, onnx_path, num_joints, output_dim = fake_checkpoint
        export(ckpt_path, onnx_path)
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

    def test_metadata_json(self, fake_checkpoint):
        ckpt_path, onnx_path, num_joints, output_dim = fake_checkpoint
        export(ckpt_path, onnx_path)

        meta_path = onnx_path.replace(".onnx", ".json")
        assert os.path.isfile(meta_path), "Metadata JSON should exist"

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["num_joints"] == num_joints
        assert meta["output_dim"] == output_dim
        assert meta["input_size"] == 224
        assert len(meta["outputs"]) == output_dim
        assert "imagenet_mean" in meta
        assert "imagenet_std" in meta

    def test_output_shape(self, fake_checkpoint):
        ckpt_path, onnx_path, num_joints, output_dim = fake_checkpoint
        export(ckpt_path, onnx_path)

        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = session.run(None, {"image": test_input})
        assert result[0].shape == (1, output_dim)
