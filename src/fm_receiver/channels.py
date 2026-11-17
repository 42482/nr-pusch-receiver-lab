"""Frequency-domain channel application; noise power is per complex sample."""

import math
import torch
from sionna.phy.channel import AWGN


def esn0_to_no(snr_db: float) -> float:
    if not math.isfinite(snr_db):
        raise ValueError("Es/N0 must be finite")
    no = 10.0 ** (-snr_db / 10.0)
    if not math.isfinite(no) or no <= 0:
        raise ValueError("Es/N0 produces an invalid noise variance")
    return no


class SoftwareChannel:
    def __init__(self, device: str, resource_grid=None, config=None):
        self.device = device
        self.awgn = AWGN(precision="single", device=device)
        self.resource_grid, self.config = resource_grid, config
        self.tdl = None
        self.last_h = None
        self.diagnostics = {}

    def __call__(self, x, scenario: str, snr_db: float | None, receiver_noise_floor: float):
        self.last_h = None
        if scenario == "passthrough":
            return x.clone(), torch.tensor(receiver_noise_floor, device=self.device), torch.zeros_like(x)
        if scenario not in ("awgn", "tdl-a") or snr_db is None:
            raise ValueError("Expected passthrough, awgn or tdl-a with an Es/N0 value")
        no = torch.tensor(esn0_to_no(snr_db), dtype=torch.float32, device=self.device)
        clean = x
        if scenario == "tdl-a":
            if self.tdl is None:
                from sionna.phy.channel import OFDMChannel
                from sionna.phy.channel.tr38901 import TDL
                if self.resource_grid is None or self.config is None:
                    raise ValueError("TDL requires a resource grid and channel configuration")
                c, rg = self.config, self.resource_grid
                model = TDL("A", c.delay_spread_s, c.carrier_frequency_hz,
                            min_speed=c.speed_mps, max_speed=c.speed_mps,
                            precision="single", device=self.device, spec_version=c.spec_version)
                max_delay = float(model.delays.max())
                cp_s = rg.cyclic_prefix_length / (rg.fft_size * rg.subcarrier_spacing)
                if max_delay >= cp_s:
                    raise ValueError("TDL maximum delay exceeds CP; frequency-domain model is invalid")
                self.diagnostics = {"max_delay_s": max_delay, "cp_duration_s": cp_s,
                                    "symbol_constant_channel": True, "models_ici_isi": False}
                self.tdl = OFDMChannel(model, rg, normalize_channel=c.normalize,
                                       return_channel=True, precision="single", device=self.device)
            clean, self.last_h = self.tdl(x)
        y = self.awgn(clean, no)
        return y, no, y - clean
