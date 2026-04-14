"""
VIMU v2 model: DINOv2-small backbone + LoRA adapters + regression head.

Output: [joint_1, ..., joint_N, base_roll, base_pitch]

DINOv2's self-supervised features encode part-level spatial structure,
which is ideal for understanding articulated robot joints. LoRA keeps
the trainable parameter count low (~300K) to avoid overfitting on small
datasets.
"""

import torch
import torch.nn as nn


class VimuModel(nn.Module):
    def __init__(self, num_joints: int = 5, lora_rank: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = num_joints + 2  # + base_roll, base_pitch

        # Load DINOv2-small (ViT-S/14, 384-dim CLS token)
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA to attention projection layers
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Regression head on top of [CLS] token
        self.head = nn.Sequential(
            nn.Linear(384, 256),
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
            (B, num_joints + 2) -- joint angles then [base_roll, base_pitch]
        """
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0]  # (B, 384)
        return self.head(cls_token)

    def merge_lora(self):
        """Merge LoRA weights into the base model for export."""
        self.backbone = self.backbone.merge_and_unload()

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = VimuModel(num_joints=5)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 7)
    print(f"Params: {model.trainable_params():,} trainable / {model.total_params():,} total")
