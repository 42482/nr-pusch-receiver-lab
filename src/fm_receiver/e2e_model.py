"""Transmit/channel orchestration; receivers can reuse the same observations."""

import torch
from .builders import build_pusch
from .channels import SoftwareChannel
from .data.schema import ObservationBatch, SimulatedBatch, TxLabels


class E2EModel:
    def __init__(self, config, device: str, seed: int):
        self.config, self.device = config, device
        self.components = build_pusch(config, device)
        self.channel = SoftwareChannel(device, self.components.adapter.resource_grid, config.channel)
        # Payload generation is independent of channel randomness and device.
        self.payload_rng = torch.Generator(device="cpu").manual_seed(seed)
        self.next_frame_id = 0

    @property
    def receiver(self):
        return self.components.receiver

    def generate(self, batch_size: int, scenario: str, snr_db: float | None = None):
        adapter = self.components.adapter
        bits = torch.randint(0, 2, (batch_size, 1, adapter.encoder.k),
                             generator=self.payload_rng, dtype=torch.int64).to(
                                 device=self.device, dtype=torch.float32)
        x, mapper_bits = adapter.transmit(bits)
        y, no, noise = self.channel(x, scenario, snr_db,
                                     self.config.evaluation.receiver_noise_floor)
        frame_ids = torch.arange(self.next_frame_id, self.next_frame_id + batch_size)
        self.next_frame_id += batch_size
        observation = ObservationBatch(frame_ids, y, no, adapter.layout_id)
        return SimulatedBatch(observation, TxLabels(bits, mapper_bits), x, noise, self.channel.last_h)
