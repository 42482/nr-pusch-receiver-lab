"""Versioned IQ capture contract and offline, explicitly synchronized ingress."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import h5py
import numpy as np
import torch
from scipy.signal import resample_poly
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator

from ..data.schema import ObservationBatch
from ..paths import data_path


@dataclass(frozen=True)
class WaveformSpec:
    layout_id: str
    logical_rate_hz: float
    fft_size: int
    cp_samples: tuple[int, ...]
    subcarrier_spacing_hz: float

    @classmethod
    def from_adapter(cls, adapter):
        rg = adapter.resource_grid
        return cls(adapter.layout_id, float(rg.fft_size * rg.subcarrier_spacing),
                   int(rg.fft_size), (int(rg.cyclic_prefix_length),) * int(rg.num_ofdm_symbols),
                   float(rg.subcarrier_spacing))

    @property
    def frame_samples(self):
        return len(self.cp_samples) * self.fft_size + sum(self.cp_samples)

    def validate(self):
        if self.fft_size <= 0 or len(self.cp_samples) != 14 or not self.layout_id:
            raise ValueError("Invalid waveform layout")
        if any(type(cp) is not int or not 0 <= cp <= self.fft_size for cp in self.cp_samples):
            raise ValueError("Invalid CP samples")
        if not math.isfinite(self.logical_rate_hz) or self.logical_rate_hz <= 0 or not math.isclose(
                self.logical_rate_hz, self.fft_size * self.subcarrier_spacing_hz):
            raise ValueError("Logical sample rate must match FFT/subcarrier spacing")
        return self


def _frame_indices(frame_ids, frame_starts):
    ids, starts = np.asarray(frame_ids), np.asarray(frame_starts)
    if ids.ndim != 1 or starts.shape != ids.shape:
        raise ValueError("Frame IDs/starts must be matching one-dimensional arrays")
    if any(a.size and (a.dtype.kind not in "iu" or (a < 0).any() or
                      (a > np.iinfo(np.int64).max).any()) for a in (ids, starts)):
        raise ValueError("Frame IDs/starts require nonnegative int64 values")
    ids, starts = ids.astype(np.int64), starts.astype(np.int64)
    if len(np.unique(ids)) != len(ids) or (len(starts) > 1 and (np.diff(starts) <= 0).any()):
        raise ValueError("Frame IDs must be unique and starts strictly increasing")
    return ids, starts


def _capture_metadata(metadata, spec):
    for name in ("actual_rx_rate_hz", "noise_variance_grid"):
        if name not in metadata or not math.isfinite(metadata[name]) or metadata[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if metadata.get("sync_source") not in ("provided", "estimated", "synthetic"):
        raise ValueError("Capture must identify how synchronization was obtained")
    rate = metadata["actual_rx_rate_hz"]
    if rate < spec.logical_rate_hz and not math.isclose(rate, spec.logical_rate_hz, rel_tol=1e-9):
        raise ValueError("Native rate cannot undersample the current full-bandwidth logical waveform")


def _capture_events(events):
    events = list(events)
    for event in events:
        for name in ("start_sample", "num_samples"):
            value = event.get(name)
            if type(value) is not int or value < (1 if name == "num_samples" else 0):
                raise ValueError("Capture events require integer sample coordinates and positive lengths")
        if not isinstance(event.get("kind"), str) or not event["kind"]:
            raise ValueError("Capture events require a kind")
    return events


def export_waveform(model, batch, directory):
    directory = data_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    adapter = model.components.adapter
    spec = WaveformSpec.from_adapter(adapter).validate()
    if len(set(spec.cp_samples)) != 1:
        raise ValueError("Current software modulator uses a constant CP")
    mod = OFDMModulator(spec.cp_samples[0], precision="single", device=model.device)
    iq = mod(batch.tx_grid).detach().cpu().numpy()[:, 0, 0]
    samples = np.ascontiguousarray(iq.reshape(-1, 1), dtype="<c8")
    with h5py.File(directory / "tx.h5", "x") as file:
        file.attrs["schema_version"] = 1
        file.attrs["waveform_spec"] = json.dumps(asdict(spec))
        file.attrs["iq_sha256"] = hashlib.sha256(samples.tobytes()).hexdigest()
        file.create_dataset("iq", data=samples, chunks=True)
        file.create_dataset("frame_ids", data=batch.observation.frame_ids.cpu().numpy())
        file.create_dataset("frame_starts", data=np.arange(len(iq)) * spec.frame_samples)
    # Inference/capture reader never opens this separate supervision file.
    np.savez_compressed(directory / "labels.npz",
                        frame_ids=batch.observation.frame_ids.cpu().numpy(),
                        tb_bits=batch.labels.tb_bits.cpu().numpy().astype(np.uint8),
                        mapper_bits=batch.labels.mapper_bits.cpu().numpy().astype(np.uint8))
    (directory / "waveform.json").write_text(json.dumps({"schema_version": 1,
        "waveform": asdict(spec), "iq_axes": ["sample", "channel"],
        "label_file": "labels.npz", "hardware_parameters_frozen": False}, indent=2) + "\n")
    return samples, spec


def write_capture(path, iq, spec, metadata, frame_ids, frame_starts, events=()):
    path = data_path(path)
    spec.validate()
    iq = np.asarray(iq, dtype="<c8")
    if iq.ndim != 2 or iq.shape[1] != 1 or not np.isfinite(iq).all():
        raise ValueError("First-round IQ must have finite [sample,1] complex samples")
    ids, starts = _frame_indices(frame_ids, frame_starts)
    _capture_metadata(metadata, spec)
    events = _capture_events(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as file:
        file.attrs["schema_version"] = 1
        file.attrs["waveform_spec"] = json.dumps(asdict(spec))
        file.attrs["metadata"] = json.dumps(metadata, allow_nan=False)
        file.attrs["events"] = json.dumps(list(events), allow_nan=False)
        file.create_dataset("iq", data=iq, chunks=True)
        file.create_dataset("frame_ids", data=ids)
        file.create_dataset("frame_starts", data=starts)


def load_observations(path, expected_spec, device="cpu"):
    """Read IQ and explicit frame timing only; no truth/label access.

    Frame starts use native capture sample coordinates. This is not a blind
    synchronizer. noise_variance_grid is calibrated after all frontend scaling.
    """
    expected_spec.validate()
    with h5py.File(path, "r") as file:
        if file.attrs["schema_version"] != 1:
            raise ValueError("Unsupported capture schema")
        stored = json.loads(file.attrs["waveform_spec"])
        stored["cp_samples"] = tuple(stored["cp_samples"])
        if WaveformSpec(**stored) != expected_spec:
            raise ValueError("Capture waveform/layout differs from receiver")
        metadata = json.loads(file.attrs["metadata"])
        events = json.loads(file.attrs["events"])
        iq = file["iq"][:]
        ids, starts = file["frame_ids"][:], file["frame_starts"][:]
    if iq.ndim != 2 or iq.shape[1] != 1 or not np.isfinite(iq).all():
        raise ValueError("Invalid capture IQ")
    ids, starts = _frame_indices(ids, starts)
    _capture_metadata(metadata, expected_spec)
    events = _capture_events(events)
    native_rate = float(metadata["actual_rx_rate_hz"])
    no = float(metadata["noise_variance_grid"])
    cfo = float(metadata.get("frequency_offset_hz", 0.))
    scale = float(metadata.get("receive_amplitude_scale", 1.))
    if not all(math.isfinite(x) for x in (native_rate, no, cfo, scale)) or min(native_rate, no, scale) <= 0:
        raise ValueError("Invalid receiver rate/noise/frontend calibration")
    ratio = Fraction(expected_spec.logical_rate_hz / native_rate).limit_denominator(10000)
    if not math.isclose(float(ratio), expected_spec.logical_rate_hz / native_rate, rel_tol=1e-9):
        raise ValueError("Sample rate ratio needs a more accurate resampling specification")
    corrected = iq[:, 0] * np.exp(-2j * np.pi * cfo * np.arange(len(iq)) / native_rate) * scale
    logical = (resample_poly(corrected, ratio.numerator, ratio.denominator)
               if ratio != 1 else corrected).astype(np.complex64)
    frames, kept, rejected = [], [], []
    n = expected_spec.frame_samples
    for frame_id, start in zip(ids, starts):
        native_stop = start + math.ceil(n * native_rate / expected_spec.logical_rate_hz)
        invalid = [e for e in events if e["start_sample"] < native_stop and
                   e["start_sample"] + e["num_samples"] > start]
        logical_start = round(float(start) * float(ratio))
        if invalid or logical_start + n > len(logical):
            rejected.append({"frame_id": int(frame_id),
                             "reason": "capture_event" if invalid else "truncated",
                             "events": invalid})
            continue
        frames.append(logical[logical_start:logical_start+n])
        kept.append(int(frame_id))
    if not frames:
        raise ValueError("No complete event-free frames in capture")
    grids = []
    for frame in frames:
        offset, symbols = 0, []
        for cp in expected_spec.cp_samples:
            useful = torch.from_numpy(frame[offset+cp:offset+cp+expected_spec.fft_size]).to(device)
            # Sionna OFDM uses centered carriers and a unitary FFT.
            symbols.append(torch.fft.fftshift(torch.fft.fft(useful, norm="ortho")))
            offset += cp + expected_spec.fft_size
        grids.append(torch.stack(symbols))
    y = torch.stack(grids)[:, None, None]
    observation = ObservationBatch(torch.tensor(kept), y,
                                   torch.full((len(kept), 1, 1), no, device=device),
                                   expected_spec.layout_id)
    quality = {"accepted_frame_ids": kept, "rejected": rejected,
               "sync_source": metadata.get("sync_source"), "blind_synchronization": False,
               "resampling": [ratio.numerator, ratio.denominator], "metadata": metadata}
    return observation, quality


def reference_demodulate(iq, spec, device="cpu"):
    """Reference used to check custom per-symbol-CP demodulation."""
    if len(set(spec.cp_samples)) != 1:
        raise ValueError("Reference uses constant CP")
    demod = OFDMDemodulator(spec.fft_size, 0, spec.cp_samples[0], device=device)
    return demod(torch.as_tensor(iq, device=device))
