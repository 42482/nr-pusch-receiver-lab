"""Receiver observations are deliberately separate from supervision labels."""

from dataclasses import dataclass
from typing import Protocol
from torch import Tensor


@dataclass
class ObservationBatch:
    frame_ids: Tensor
    y: Tensor
    no: Tensor
    layout_id: str


@dataclass
class TxLabels:
    tb_bits: Tensor
    mapper_bits: Tensor


@dataclass
class SimulatedBatch:
    observation: ObservationBatch
    labels: TxLabels
    tx_grid: Tensor
    noise: Tensor
    h_true: Tensor | None = None


@dataclass
class ReceiverResult:
    tb_hat: Tensor
    tb_crc: Tensor
    llr: Tensor | None = None
    h_hat: Tensor | None = None
    err_var: Tensor | None = None


class Receiver(Protocol):
    def __call__(self, observation: ObservationBatch) -> ReceiverResult: ...
