import torch
import pytest

from model import VimuModel


class TestVimuModelInstantiation:
    def test_default_args(self):
        model = VimuModel(num_joints=6)
        assert model.num_joints == 6
        assert model.output_dim == 8  # 6 joints + 2 base

    def test_custom_joints(self):
        for n in [3, 4, 7]:
            model = VimuModel(num_joints=n)
            assert model.num_joints == n
            assert model.output_dim == n + 2


class TestVimuModelForward:
    def test_output_shape_default(self):
        model = VimuModel(num_joints=6)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 8)

    def test_output_shape_different_joints(self):
        for n in [3, 5]:
            model = VimuModel(num_joints=n)
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
            assert out.shape == (1, n + 2)


class TestVimuModelFreezing:
    def test_frozen_params_require_grad_false(self):
        model = VimuModel(num_joints=6, freeze_backbone=True)
        for name, param in model.features.named_parameters():
            if "6" not in name and "7" not in name:
                assert not param.requires_grad, (
                    f"Parameter {name} should be frozen but requires_grad=True"
                )

    def test_unfrozen_blocks_require_grad_true(self):
        model = VimuModel(num_joints=6, freeze_backbone=True)
        # Blocks 6 and 7 should be unfrozen
        for name, param in model.features.named_parameters():
            if "6" in name or "7" in name:
                assert param.requires_grad, (
                    f"Parameter {name} should be unfrozen but requires_grad=False"
                )

    def test_head_requires_grad(self):
        model = VimuModel(num_joints=6, freeze_backbone=True)
        for name, param in model.head.named_parameters():
            assert param.requires_grad, (
                f"Head parameter {name} should require grad"
            )

    def test_no_freeze(self):
        model = VimuModel(num_joints=6, freeze_backbone=False)
        for name, param in model.named_parameters():
            assert param.requires_grad, (
                f"With freeze_backbone=False, {name} should require grad"
            )
