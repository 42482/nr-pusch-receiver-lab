"""Grouped code-aware error prediction with explicit NR erasure handling."""

import torch
from torch import nn


class GroupedECCT(nn.Module):
    def __init__(self, codec):
        super().__init__()
        # Codec is provided to forward, not duplicated inside trainable checkpoints.
        self.codec_layout_id = codec.layout_id
        self.n = codec.n
        groups = (codec.n + len(codec.h)) // 2
        self.embedding = nn.Linear(2, 128)
        self.position = nn.Parameter(torch.randn(1, groups, 128) * .02)
        self.role_embedding = nn.Embedding(4, 128)
        roles = torch.full((groups,), 3, dtype=torch.long, device=codec.h.device)
        roles[:codec.sent // 2] = 0
        roles[codec.sent // 2:(codec.sent + codec.punctured) // 2] = 1
        roles[(codec.sent + codec.punctured) // 2:codec.n // 2] = 2
        self.register_buffer("roles", roles)
        self.register_buffer("blocked", codec.grouped_mask())
        layer = nn.TransformerEncoderLayer(128, 8, 512, dropout=0., activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.readout = nn.Linear(128, 2)

    def forward(self, code_llr, codec):
        if codec.layout_id != self.codec_layout_id:
            raise ValueError("G-ECCT parity/mapping layout mismatch")
        hard = (code_llr > 0).float()  # zero erasures deterministically start at bit zero
        syndrome = codec.syndrome(hard)
        features = torch.cat([code_llr.abs().clamp_max(20), syndrome], dim=-1)
        x = self.embedding(features.reshape(len(features), -1, 2))
        x = x + self.position + self.role_embedding(self.roles)
        x = self.transformer(x, mask=self.blocked)
        error_logits = self.readout(x[:, :self.n // 2]).flatten(1)
        decoded = (1 - 2 * hard) * error_logits
        decoded = decoded.masked_fill(codec.known_zero, -20.)
        return decoded, error_logits
