"""NRX-style component construction, separate from configuration parsing."""

from dataclasses import dataclass
import numpy as np
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver, PUSCHLSChannelEstimator
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import LinearDetector

from .config import ExperimentConfig
from .decoders.tb import build_tb_decoder
from .receivers.baseline import BaselineReceiver
from .sionna_adapter import PuschAdapter, require_sionna_version


@dataclass
class PuschComponents:
    adapter: PuschAdapter
    receiver: BaselineReceiver
    default_receiver: PUSCHReceiver


def build_pusch(config: ExperimentConfig, device: str) -> PuschComponents:
    config.validate()
    require_sionna_version()
    s, d, r = config.system, config.dmrs, config.receiver
    pc = PUSCHConfig()
    pc.carrier.n_size_grid = s.n_size_bwp
    pc.carrier.subcarrier_spacing = s.subcarrier_spacing_khz
    pc.carrier.cyclic_prefix = s.cyclic_prefix
    pc.carrier.n_cell_id = s.n_cell_id
    pc.carrier.slot_number = s.slot_number
    pc.carrier.frame_number = s.frame_number
    pc.n_size_bwp = s.n_size_bwp
    pc.n_start_bwp = 0
    pc.num_antenna_ports = s.num_tx_ant
    pc.num_layers = s.num_layers
    pc.precoding = s.precoding
    pc.symbol_allocation = [0, s.num_ofdm_symbols]
    pc.mapping_type = d.mapping_type
    pc.n_rnti = s.n_rnti
    pc.tb.mcs_table = s.mcs_table
    pc.tb.mcs_index = s.mcs_index
    pc.tb.n_id = s.n_cell_id
    for name in ("config_type", "type_a_position", "length", "additional_position",
                 "num_cdm_groups_without_data", "n_scid"):
        setattr(pc.dmrs, name, getattr(d, name))
    pc.dmrs.dmrs_port_set = [d.port]
    pc.dmrs.n_id = [s.n_cell_id, s.n_cell_id]
    pc.check_config()
    tx = PUSCHTransmitter(pc, return_bits=False, output_domain="freq",
                          precision=r.precision, device=device)
    adapter = PuschAdapter(tx, pc)
    if int(pc.tb.num_bits_per_symbol) != 2:
        raise RuntimeError("Resolved MCS is not QPSK")
    if tx.resource_grid.fft_size != 12 * s.n_size_bwp:
        raise RuntimeError("Resolved allocation differs from the requested PRBs")
    est = PUSCHLSChannelEstimator(tx.resource_grid, d.length, d.additional_position,
                                 d.num_cdm_groups_without_data, interpolation_type=r.interpolation,
                                 precision=r.precision, device=device)
    streams = StreamManagement(np.ones((1, 1), dtype=bool), 1)
    detector = LinearDetector(r.equalizer, "bit", r.demapping, tx.resource_grid, streams,
                              "qam", 2, precision=r.precision, device=device)
    decoder = build_tb_decoder(adapter.encoder, r, device)
    rx = PUSCHReceiver(tx, channel_estimator=est, mimo_detector=detector, tb_decoder=decoder,
                       return_tb_crc_status=True, precision=r.precision, device=device)
    default = PUSCHReceiver(tx, return_tb_crc_status=True, precision=r.precision, device=device)
    return PuschComponents(adapter, BaselineReceiver(rx, est, decoder, adapter.layout_id), default)
