"""
VIMU model: ResNet-18 backbone → regression head.

Output: [joint_1, ..., joint_N, base_roll, base_pitch]

The base_x, base_y outputs from before are dropped since we're
focused on proprioception (internal state), not external tracking.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VimuModel(nn.Module):
    def __init__(self, num_joints: int = 6, freeze_backbone: bool = True):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = num_joints + 2  # + base_roll, base_pitch

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        if freeze_backbone:
            for name, param in self.features.named_parameters():
                # Unfreeze only the last two residual blocks (children 6, 7)
                if "6" not in name and "7" not in name:
                    param.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.output_dim),
        )

        # Small init for stable early training
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalized RGB tensor.
        Returns:
            (B, num_joints + 2) — joint angles then [base_roll, base_pitch]
        """
        return self.head(self.features(x))


if __name__ == "__main__":
    model = VimuModel(num_joints=6)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")
