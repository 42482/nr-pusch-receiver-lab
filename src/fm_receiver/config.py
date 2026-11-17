"""Typed experiment configuration, independent of Sionna runtime objects."""

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SystemConfig:
    num_tx: int = 1
    num_layers: int = 1
    num_tx_ant: int = 1
    num_rx_ant: int = 1
    n_size_bwp: int = 4
    num_ofdm_symbols: int = 14
    subcarrier_spacing_khz: int = 15
    cyclic_prefix: str = "normal"
    precoding: str = "non-codebook"
    mcs_table: int = 1
    mcs_index: int = 9
    n_cell_id: int = 1
    n_rnti: int = 1
    slot_number: int = 0
    frame_number: int = 0


@dataclass(frozen=True)
class DMRSConfig:
    mapping_type: str = "A"
    config_type: int = 1
    type_a_position: int = 2
    length: int = 1
    additional_position: int = 0
    port: int = 0
    num_cdm_groups_without_data: int = 2
    n_scid: int = 0


@dataclass(frozen=True)
class ReceiverConfig:
    interpolation: str = "lin"
    equalizer: str = "lmmse"
    demapping: str = "maxlog"
    num_bp_iter: int = 20
    cn_update: str = "boxplus-phi"
    precision: str = "single"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    seed: int = 42
    cpu_threads: int = 4


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 16
    passthrough_tbs: int = 256
    receiver_noise_floor: float = 1e-6
    snr_db: tuple[float, ...] = (-4., -2., 0., 2., 4., 6., 8.)
    min_tbs: int = 256
    max_tbs: int = 4096
    target_block_errors: int = 100


@dataclass(frozen=True)
class SamplesConfig:
    num_frames: int = 64
    snr_db: float = 2.
    seed: int = 43


@dataclass(frozen=True)
class ChannelConfig:
    kind: str = "awgn"
    delay_spread_s: float = 100e-9
    carrier_frequency_hz: float = 3.5e9
    speed_mps: float = 3.0
    normalize: bool = True
    spec_version: str = "19.2"


@dataclass(frozen=True)
class ExperimentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    dmrs: DMRSConfig = field(default_factory=DMRSConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    samples: SamplesConfig = field(default_factory=SamplesConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def validate(self):
        s, d, r, e = self.system, self.dmrs, self.receiver, self.evaluation
        if (s.num_tx, s.num_layers, s.num_tx_ant, s.num_rx_ant) != (1, 1, 1, 1):
            raise ValueError("This milestone supports one user, one layer and 1Tx/1Rx only")
        if s.cyclic_prefix != "normal" or s.num_ofdm_symbols != 14:
            raise ValueError("The first baseline requires normal CP and 14 symbols")
        if s.precoding != "non-codebook":
            raise ValueError("The first baseline requires non-codebook precoding")
        if not 1 <= s.n_size_bwp <= 275 or s.subcarrier_spacing_khz not in (15, 30, 60, 120):
            raise ValueError("Invalid PRB allocation or subcarrier spacing")
        if (s.mcs_table, s.mcs_index) != (1, 9):
            raise ValueError("The first baseline requires MCS table 1, index 9")
        if not 0 <= s.n_cell_id <= 1007 or not 0 <= s.n_rnti <= 65535:
            raise ValueError("Invalid cell ID or RNTI")
        if s.frame_number < 0 or s.slot_number < 0:
            raise ValueError("Frame and slot numbers must be non-negative")
        if (d.mapping_type, d.config_type, d.type_a_position, d.length,
            d.additional_position, d.port, d.num_cdm_groups_without_data, d.n_scid) != (
                "A", 1, 2, 1, 0, 0, 2, 0):
            raise ValueError("The first baseline uses the frozen Hello World DMRS layout")
        if (r.interpolation, r.equalizer, r.demapping, r.precision) != (
                "lin", "lmmse", "maxlog", "single"):
            raise ValueError("Unsupported first-round receiver settings")
        if r.num_bp_iter <= 0 or r.cn_update not in ("boxplus-phi", "boxplus", "minsum"):
            raise ValueError("Invalid BP decoder settings")
        for name in ("batch_size", "passthrough_tbs", "min_tbs", "max_tbs", "target_block_errors"):
            if getattr(e, name) <= 0:
                raise ValueError(f"evaluation.{name} must be positive")
        if e.min_tbs > e.max_tbs:
            raise ValueError("min_tbs cannot exceed max_tbs")
        if not math.isfinite(e.receiver_noise_floor) or e.receiver_noise_floor <= 0:
            raise ValueError("receiver_noise_floor must be finite and positive")
        if not e.snr_db or any(not math.isfinite(v) for v in e.snr_db):
            raise ValueError("snr_db must contain finite values")
        if any(a >= b for a, b in zip(e.snr_db, e.snr_db[1:])):
            raise ValueError("snr_db must be strictly increasing")
        if self.samples.num_frames <= 0 or not math.isfinite(self.samples.snr_db):
            raise ValueError("Invalid sample export settings")
        if self.runtime.cpu_threads <= 0 or min(self.runtime.seed, self.samples.seed) < 0:
            raise ValueError("Invalid thread count or random seed")
        ch = self.channel
        if ch.kind not in ("awgn", "tdl-a"):
            raise ValueError("channel.kind must be awgn or tdl-a")
        if any(not math.isfinite(v) or v <= 0 for v in
               (ch.delay_spread_s, ch.carrier_frequency_hz)):
            raise ValueError("Channel delay spread and frequency must be finite and positive")
        if not math.isfinite(ch.speed_mps) or ch.speed_mps < 0 or ch.spec_version != "19.2":
            raise ValueError("Invalid channel speed or unsupported TDL specification")
        return self

    def to_dict(self):
        return asdict(self)


def _section(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a TOML table")
    default = cls()
    known = {f.name for f in fields(cls)}
    unknown = values.keys() - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    result = {}
    for key, value in values.items():
        expected = getattr(default, key)
        if isinstance(expected, tuple):
            if not isinstance(value, list) or any(type(v) not in (int, float) for v in value):
                raise ValueError(f"{key} must be an array of numbers")
            value = tuple(float(v) for v in value)
        elif type(expected) is float:
            if type(value) not in (int, float):
                raise ValueError(f"{key} must be numeric")
            value = float(value)
        elif type(value) is not type(expected):
            raise ValueError(f"{key} must have type {type(expected).__name__}")
        result[key] = value
    return cls(**result)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)
    classes = dict(system=SystemConfig, dmrs=DMRSConfig, receiver=ReceiverConfig,
                   runtime=RuntimeConfig, evaluation=EvaluationConfig, samples=SamplesConfig,
                   channel=ChannelConfig)
    if raw.keys() - classes.keys():
        raise ValueError(f"Unknown configuration sections: {sorted(raw.keys() - classes.keys())}")
    return ExperimentConfig(**{key: _section(cls, raw.get(key, {}))
                               for key, cls in classes.items()}).validate()
