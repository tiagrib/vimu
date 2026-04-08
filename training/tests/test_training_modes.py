import os
import subprocess
import sys

import torch
import pytest
from PIL import Image

from dataset import get_synthetic_transforms
from model import VimuModel

TRAINING_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


class TestSyntheticTransforms:
    def test_output_shape(self):
        """Verify synthetic transforms produce correct tensor shape."""
        transform = get_synthetic_transforms(224)
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_is_normalized(self):
        """Verify output is ImageNet-normalized (not 0-255)."""
        transform = get_synthetic_transforms(224)
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.min() >= -3.0 and tensor.max() <= 3.0  # ImageNet range

    def test_custom_size(self):
        """Verify synthetic transforms respect custom size parameter."""
        transform = get_synthetic_transforms(128)
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 128, 128)


class TestPretrainMode:
    def test_unfreezes_model(self):
        """Verify pretrain mode unfreezes all parameters (freeze_backbone=False)."""
        model = VimuModel(num_joints=6, freeze_backbone=False)
        for name, param in model.named_parameters():
            assert param.requires_grad, (
                f"With freeze_backbone=False (pretrain), {name} should require grad"
            )


class TestFinetuneMode:
    def test_requires_checkpoint(self):
        """Verify finetune mode errors without --pretrained-checkpoint."""
        result = subprocess.run(
            [sys.executable, "train.py", "--mode", "finetune", "--data-dir", "/nonexistent"],
            capture_output=True, text=True, cwd=TRAINING_DIR
        )
        assert result.returncode != 0
        stderr_lower = result.stderr.lower()
        assert "pretrained-checkpoint" in stderr_lower or "required" in stderr_lower

    def test_loads_pretrained_weights(self, tmp_path):
        """Verify finetune mode can load a pretrained checkpoint."""
        # Create a model and save a checkpoint
        model = VimuModel(num_joints=6, freeze_backbone=False)
        checkpoint_path = tmp_path / "pretrained.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_joints": 6,
            "output_dim": 8,
            "mode": "pretrain",
        }, checkpoint_path)

        # Load into a new model (simulating finetune)
        new_model = VimuModel(num_joints=6, freeze_backbone=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weights mismatch at {n1}"

    def test_loads_raw_state_dict(self, tmp_path):
        """Verify finetune mode can load a raw state_dict checkpoint."""
        model = VimuModel(num_joints=6, freeze_backbone=False)
        checkpoint_path = tmp_path / "raw.pt"
        torch.save(model.state_dict(), checkpoint_path)

        new_model = VimuModel(num_joints=6, freeze_backbone=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            new_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            new_model.load_state_dict(checkpoint)

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weights mismatch at {n1}"
