"""The only module that accesses version-specific Sionna internal attributes.

This adapter captures the *actual* Mapper input, including rate matching,
interleaving and scrambling. It does not infer labels from received symbols.
"""

import hashlib
import importlib.metadata
import json
import numpy as np
import torch


def require_sionna_version():
    for distribution in ("sionna-no-rt", "sionna"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if version != "2.2.0":
            raise RuntimeError(f"Expected Sionna 2.2.0, found {distribution} {version}")
        return
    raise RuntimeError("Install sionna-no-rt==2.2.0 in the project environment")


def scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class PuschAdapter:
    def __init__(self, transmitter, pusch_config):
        require_sionna_version()
        self.transmitter = transmitter
        self.pusch_config = pusch_config
        self.encoder = transmitter._tb_encoder
        self.mapper = transmitter._mapper
        self.layer_mapper = transmitter._layer_mapper
        self.resource_grid = transmitter.resource_grid
        self.type_grid = self.resource_grid.build_type_grid().detach().cpu()
        if self.type_grid.shape[:2] != (1, 1):
            raise ValueError("The first-round mapping adapter supports one user and one layer")
        self.data_re_indices = torch.nonzero(self.type_grid[0, 0] == 0, as_tuple=False)
        self.data_mask = self.type_grid[0, 0] == 0
        identity = {"tb_size": int(self.encoder.k), "coded_bits": int(self.encoder.n),
                    "rnti": pusch_config.n_rnti, "scrambling_id": pusch_config.tb.n_id,
                    "subcarrier_spacing": self.resource_grid.subcarrier_spacing,
                    "slot_number": pusch_config.carrier.slot_number}
        digest = hashlib.sha256(self.type_grid.numpy().tobytes())
        digest.update(self.resource_grid.pilot_pattern.pilots.detach().cpu().numpy().tobytes())
        digest.update(json.dumps(identity, sort_keys=True, default=scalar).encode())
        self.layout_id = digest.hexdigest()[:16]

    def transmit(self, tb_bits):
        captured = []

        def capture_mapper_input(module, args):
            captured.append(args[0].detach().clone())

        handle = self.mapper.register_forward_pre_hook(capture_mapper_input)
        try:
            x = self.transmitter(tb_bits)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("The Sionna Mapper capture contract changed")
        c = captured[0]
        expected = 2 * len(self.data_re_indices)
        if c.shape[-1] != expected:
            raise RuntimeError(f"Expected {expected} QPSK mapping bits, got {c.shape[-1]}")
        return x, c

    def validate_mapping(self, mapper_bits, tx_grid):
        symbols = self.mapper(mapper_bits)
        flat_indices = (self.data_re_indices[:, 0] * self.resource_grid.fft_size
                        + self.data_re_indices[:, 1]).to(tx_grid.device)
        grid_symbols = tx_grid[:, 0, 0].flatten(-2).index_select(-1, flat_indices)
        torch.testing.assert_close(symbols[:, 0], grid_symbols, rtol=1e-5, atol=1e-6)
        return True

    def metadata(self):
        pc, rg, enc = self.pusch_config, self.resource_grid, self.encoder
        ldpc = enc.ldpc_encoder
        return {
            "tb_size": int(enc.k), "num_coded_bits": int(enc.n),
            "target_coderate": scalar(pc.tb.target_coderate),
            "effective_information_coderate": int(enc.k) / int(enc.n),
            "num_bits_per_symbol": int(scalar(pc.tb.num_bits_per_symbol)),
            "num_code_blocks": int(enc.num_cbs),
            "ldpc": {key: scalar(getattr(ldpc, key, None))
                     for key in ("k", "n", "k_ldpc", "n_ldpc", "z")},
            "ldpc_base_graph": getattr(ldpc, "_bg", None),
            "layout_id": self.layout_id, "fft_size": int(rg.fft_size),
            "num_ofdm_symbols": int(rg.num_ofdm_symbols),
            "num_data_re": len(self.data_re_indices),
            "cyclic_prefix_samples": int(rg.cyclic_prefix_length),
            "sample_rate_hz": float(rg.fft_size * rg.subcarrier_spacing),
            "dmrs_symbol_indices": list(pc.dmrs_symbol_indices),
            "pilot_normalization": bool(rg.pilot_pattern.normalize),
            "data_re_order": "symbol-major, subcarrier-minor, zero-based; QPSK bit 0 then bit 1",
            "llr_convention": "log(P(bit=1)/P(bit=0)); positive means 1",
            "scrambler_enabled": True,
            "hardware_parameters_frozen": False,
        }


def encoder_from_transmitter(transmitter):
    return transmitter._tb_encoder


def soft_detect(components, observation, h_true=None):
    """Soft detector without BP; perfect CSI is diagnostic-only."""
    rx = components.receiver
    if h_true is None:
        h, err = rx.channel_estimator(observation.y, observation.no)
    else:
        h, err = h_true, torch.zeros((), device=observation.y.device)
    llr = rx.receiver._mimo_detector(observation.y, h, err, observation.no)
    return rx.receiver._layer_demapper(llr), h, err


class CodingAdapter(torch.nn.Module):
    """Single-CB/RV0 NR mapping; all private Sionna codec access stays here.

    Code coordinates put transmitted bits in Mapper order first, followed by
    punctured and known-zero bits. Scrambling is removed before applying H.
    """

    def __init__(self, encoder, tb_decoder):
        super().__init__()
        require_sionna_version()
        self.encoder = encoder
        self.tb_decoder = tb_decoder
        ld = encoder.ldpc_encoder
        decoder = tb_decoder._decoder
        if encoder.num_cbs != 1 or encoder.k_padding != 0 or encoder.n != ld.n:
            raise ValueError("Neural codec supports the frozen single-CB, unpadded RV0 profile")
        self.n = int(decoder._n_pruned)
        self.k = int(ld.k)
        self.sent = int(encoder.n)
        self.punctured = 2 * int(ld.z)
        self.fillers = int(ld.k_filler)
        if self.n != self.sent + self.punctured + self.fillers:
            raise ValueError("Unsupported NR rate matching: repetition/additional puncturing")
        dev = encoder.device
        columns = torch.cat([torch.arange(self.punctured, self.k),
                             torch.arange(ld.k_ldpc, self.n)]).to(dev)
        if ld.num_bits_per_symbol is not None:
            columns = columns[ld.out_int.long()]
        columns = columns[torch.argsort(encoder.output_perm_inv)]
        order = torch.cat([columns, torch.arange(self.punctured, device=dev),
                           torch.arange(self.k, ld.k_ldpc, device=dev)])
        if len(torch.unique(order)) != self.n:
            raise RuntimeError("Codec coordinates are not bijective")
        self.register_buffer("order", order)
        self.register_buffer("inverse_order", torch.argsort(order))
        h = torch.as_tensor(decoder.pcm.toarray(), dtype=torch.float32, device=dev)[:, order]
        self.register_buffer("h", h)
        known = torch.arange(self.n, device=dev) >= self.n - self.fillers
        self.register_buffer("known_zero", known)
        self.layout_id = hashlib.sha256(h.cpu().numpy().tobytes() +
                                         order.cpu().numpy().tobytes()).hexdigest()[:16]

    def to_code_llr(self, mapper_llr):
        if mapper_llr.shape[-1] != self.sent:
            raise ValueError("Mapper LLR length differs from codec")
        llr = mapper_llr.reshape(-1, 1, self.sent)
        if self.tb_decoder._descrambler is not None:
            llr = self.tb_decoder._descrambler(llr)
        llr = llr.reshape(-1, self.sent)
        return torch.cat([llr, llr.new_zeros(len(llr), self.punctured),
                          llr.new_full((len(llr), self.fillers), -20.)], dim=-1)

    def code_labels(self, tb_bits):
        """Legal codeword labels; never called by receiver inference."""
        ld = self.encoder.ldpc_encoder
        info = self.encoder.tb_crc_encoder(tb_bits).reshape(-1, self.k)
        padded = torch.cat([info, info.new_zeros(len(info), self.fillers)], dim=-1)
        full = ld._encode_fast(padded).reshape(-1, ld.n_ldpc)
        return full[:, :self.n][:, self.order]

    def syndrome(self, bits):
        return torch.remainder(bits.float() @ self.h.T, 2.)

    def to_tb(self, decoded_llr):
        code = (decoded_llr > 0).float()[:, self.inverse_order]
        info = code[:, :self.k].unsqueeze(1)
        bits, crc = self.tb_decoder._tb_crc_decoder(info)
        return bits, crc.squeeze(-1).bool()

    def grouped_mask(self):
        h = self.h
        n, checks = self.n, h.shape[0]
        if n % 2 or checks % 2:
            raise ValueError("QPSK grouping requires even code/check lengths")
        size = n + checks
        connected = torch.eye(size, dtype=torch.bool, device=h.device)
        connected[:n, :n] |= h.T @ h > 0
        connected[:n, n:] = h.T.bool()
        connected[n:, :n] = h.bool()
        allowed = connected.reshape(size // 2, 2, size // 2, 2).any(3).any(1)
        return ~allowed  # nn.TransformerEncoder: True means blocked.

    def metadata(self):
        return {"codec_layout_id": self.layout_id, "transmitted_bits": self.sent,
                "code_bits": self.n, "checks": len(self.h), "punctured": self.punctured,
                "known_zero": self.fillers, "rv": 0, "group_size": 2}
