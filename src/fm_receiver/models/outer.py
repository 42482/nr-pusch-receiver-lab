"""Small FM-style outer receiver. Logits use log(P(1)/P(0))."""

import torch
from torch import nn


class OuterTransformer(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        rg = adapter.resource_grid
        self.layout_id = adapter.layout_id
        self.embedding = nn.Sequential(nn.Conv2d(5, 64, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.symbol_position = nn.Parameter(torch.randn(1, rg.num_ofdm_symbols, 1, 128) * .02)
        self.frequency_position = nn.Parameter(torch.randn(1, 1, rg.fft_size, 128) * .02)
        self.resource_embedding = nn.Embedding(4, 128)
        self.register_buffer("types", adapter.type_grid[0, 0].long())
        self.register_buffer("data_indices", (adapter.data_re_indices[:, 0] * rg.fft_size +
                                             adapter.data_re_indices[:, 1]).long())
        layer = nn.TransformerEncoderLayer(128, 8, 512, dropout=0., activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.readout = nn.Linear(128, 2)

    def forward(self, observation, h_hat):
        if observation.layout_id != self.layout_id:
            raise ValueError("Outer receiver layout mismatch")
        y, h = observation.y[:, 0, 0], h_hat[:, 0, 0, 0, 0]
        scale2 = y.abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-8)
        scale = scale2.sqrt()
        y, h = y / scale, h / scale
        no = observation.no.reshape(-1, 1, 1)
        if no.numel() == 1:
            no = no.expand(len(y), 1, 1)
        noise = (no / scale2).clamp_min(1e-8).log10().expand_as(y.real)
        features = torch.stack([y.real, y.imag, h.real, h.imag, noise], dim=1)
        x = self.embedding(features).permute(0, 2, 3, 1)
        x = x + self.symbol_position + self.frequency_position + self.resource_embedding(self.types)
        x = self.transformer(x.flatten(1, 2))
        return self.readout(x[:, self.data_indices]).reshape(len(y), 1, -1)
