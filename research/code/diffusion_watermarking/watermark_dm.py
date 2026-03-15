"""
WatermarkDM-style pipelines for diffusion models.
Based on: Zhao et al., arXiv:2303.10137 (A Recipe for Watermarking Diffusion Models)

Two pipelines:
1. Unconditional/class-conditional: embed binary watermark in training data via encoder E_phi,
   train DM on watermarked data; decode with D_phi from generated images.
2. Text-to-image: finetune pretrained DM with (trigger prompt, watermark image) pair and
   weight-constrained regularization (L1 on theta - theta_hat).
"""

import numpy as np
from typing import Optional, Tuple

# Optional PyTorch for encoder/decoder and finetuning
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pipeline 1: Binary watermark encoder / decoder (for unconditional/class-conditional)
# Paper: Eq. (2) – train E_phi, D_phi with L_BCE(w, D(E(x,w))) + gamma * ||x - E(x,w)||^2
# ---------------------------------------------------------------------------


if TORCH_AVAILABLE:

    class WatermarkEncoder(nn.Module):
        """
        Encoder E_phi: (image, binary_string) -> watermarked_image.
        Architecture: conv layers with residual connections (following Yu et al. / paper Appendix A.1).
        """

        def __init__(
            self,
            in_channels: int = 3,
            bit_length: int = 64,
            base_channels: int = 64,
            num_blocks: int = 4,
        ):
            super().__init__()
            self.bit_length = bit_length
            # Encode bit string to spatial embedding
            self.bit_embed = nn.Sequential(
                nn.Linear(bit_length, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, base_channels * 4 * 4),
            )
            self.conv_in = nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1)
            layers = []
            for _ in range(num_blocks):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(base_channels, base_channels, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_channels, base_channels, 3, padding=1),
                    )
                )
            self.blocks = nn.ModuleList(layers)
            self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # w: (B, bit_length)
            B, C, H, W = x.shape
            b_emb = self.bit_embed(w)  # (B, base_channels*4*4)
            b_emb = b_emb.view(B, -1, 4, 4)
            b_emb = nn.functional.interpolate(b_emb, size=(H, W), mode="bilinear", align_corners=False)
            h = torch.cat([x, b_emb], dim=1)
            h = self.conv_in(h)
            for block in self.blocks:
                h = h + block(h)
                h = torch.relu(h)
            return x + self.conv_out(h)

    class WatermarkDecoder(nn.Module):
        """
        Decoder D_phi: image -> predicted binary string (logits per bit).
        Conv + linear classifier (Appendix A.1).
        """

        def __init__(
            self,
            in_channels: int = 3,
            bit_length: int = 64,
            base_channels: int = 64,
        ):
            super().__init__()
            self.bit_length = bit_length
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(base_channels * 4, bit_length)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.features(x)
            h = h.flatten(1)
            return self.fc(h)

    def train_encoder_decoder(
        encoder: WatermarkEncoder,
        decoder: WatermarkDecoder,
        dataloader,
        device: torch.device,
        num_epochs: int = 100,
        lr: float = 1e-3,
        gamma: float = 1.0,
    ):
        """
        Train E and D with Eq. (2): L_BCE(w, D(E(x,w))) + gamma * ||x - E(x,w)||^2.
        """
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
        )
        encoder.train()
        decoder.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch["image"].to(device)
                B = x.size(0)
                w = torch.randint(0, 2, (B, encoder.bit_length), device=device, dtype=torch.float32)
                xw = encoder(x, w)
                logits = decoder(xw)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, w)
                recon = nn.functional.mse_loss(xw, x)
                loss = bce + gamma * recon
                opt.zero_grad()
                loss.backward()
                opt.step()
        return encoder, decoder

else:
    WatermarkEncoder = None
    WatermarkDecoder = None

    def train_encoder_decoder(*args, **kwargs):
        raise RuntimeError("PyTorch is required for WatermarkDM encoder/decoder. Install torch.")


# ---------------------------------------------------------------------------
# Pipeline 2: Text-to-image trigger watermark (conceptual + loss)
# Finetuning objective: Eq. (5) – reconstruction + lambda * ||theta - theta_hat||_1
# ---------------------------------------------------------------------------


def text_to_image_watermark_loss(
    pred_noise: "torch.Tensor",
    target_noise: "torch.Tensor",
    weight_penalty: "torch.Tensor",
    lambda_reg: float = 1e-3,
) -> "torch.Tensor":
    """
    Loss for one step of weight-constrained finetuning:
    eta_t * ||x_theta(noisy_watermark, trigger_c) - watermark||^2 + lambda * ||theta - theta_hat||_1
    Here we assume caller passes already-computed reconstruction loss and weight penalty.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    recon = ((pred_noise - target_noise) ** 2).mean()
    return recon + lambda_reg * weight_penalty


def get_weight_penalty_l1(model: "nn.Module", ref_state: dict) -> "torch.Tensor":
    """L1 penalty between current params and reference (pretrained) state."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if name in ref_state:
            penalty = penalty + (p - ref_state[name]).abs().sum()
    return penalty
