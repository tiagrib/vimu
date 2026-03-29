import os
import csv
import tempfile

import numpy as np
import torch
import pytest
from PIL import Image

from dataset import VimuDataset, get_train_transforms, get_val_transforms


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock images and labels.csv."""
    tmpdir = tempfile.mkdtemp()
    frames_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frames_dir)

    num_samples = 5
    num_joints = 6

    # Create dummy JPEG images
    for i in range(num_samples):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(os.path.join(frames_dir, f"{i:06d}.jpg"))

    # Create labels.csv
    joint_cols = [f"joint_{j+1}" for j in range(num_joints)]
    header = ["frame"] + joint_cols + ["base_roll", "base_pitch"]

    labels_path = os.path.join(tmpdir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(num_samples):
            joints = [f"{np.random.uniform(-1, 1):.6f}" for _ in range(num_joints)]
            # First 3 rows have base orientation, last 2 have NaN (missing)
            if i < 3:
                base = [f"{np.random.uniform(-0.5, 0.5):.6f}" for _ in range(2)]
            else:
                base = ["", ""]
            writer.writerow([f"{i:06d}.jpg"] + joints + base)

    yield tmpdir, num_samples, num_joints

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


class TestVimuDataset:
    def test_loads_correctly(self, mock_data_dir):
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints)
        assert ds is not None

    def test_len(self, mock_data_dir):
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints)
        assert len(ds) == num_samples

    def test_getitem_shapes(self, mock_data_dir):
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints)
        sample = ds[0]

        assert sample["image"].shape == (3, 224, 224)
        assert sample["joints"].shape == (num_joints,)
        assert sample["base"].shape == (2,)
        assert sample["base_mask"].shape == (2,)

    def test_base_mask_present(self, mock_data_dir):
        """Rows 0-2 have base orientation values, mask should be True."""
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints)
        sample = ds[0]
        assert sample["base_mask"].all(), "First rows should have base mask=True"

    def test_base_mask_missing(self, mock_data_dir):
        """Rows 3-4 have empty base values, mask should be False."""
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints)
        sample = ds[3]
        assert not sample["base_mask"].any(), "Missing base rows should have mask=False"

    def test_train_transforms(self, mock_data_dir):
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints, transform=get_train_transforms())
        sample = ds[0]
        assert sample["image"].shape == (3, 224, 224)

    def test_val_transforms(self, mock_data_dir):
        tmpdir, num_samples, num_joints = mock_data_dir
        ds = VimuDataset(tmpdir, num_joints=num_joints, transform=get_val_transforms())
        sample = ds[0]
        assert sample["image"].shape == (3, 224, 224)
