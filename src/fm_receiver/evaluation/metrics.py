from dataclasses import asdict, dataclass
import math


@dataclass
class ErrorCounts:
    bit_errors: int = 0
    total_bits: int = 0
    block_errors: int = 0
    total_blocks: int = 0
    crc_failures: int = 0
    noise_energy: float = 0.
    noise_real_energy: float = 0.
    noise_imag_energy: float = 0.
    noise_samples: int = 0

    def update(self, truth, result, noise=None):
        if truth.shape != result.tb_hat.shape:
            raise ValueError("Transmitted and decoded TB shapes differ")
        if tuple(result.tb_crc.shape) != tuple(truth.shape[:-1]):
            raise ValueError("TB CRC shape does not match the block axes")
        if not bool(((truth == 0) | (truth == 1)).all()):
            raise ValueError("TB labels must be binary")
        if not bool(((result.tb_hat == 0) | (result.tb_hat == 1)).all()):
            raise ValueError("Decoded TB values must be binary")
        errors = truth != result.tb_hat
        self.bit_errors += int(errors.sum().item())
        self.total_bits += errors.numel()
        self.block_errors += int(errors.any(dim=-1).sum().item())
        self.total_blocks += errors.numel() // errors.shape[-1]
        self.crc_failures += int((~result.tb_crc.bool()).sum().item())
        if noise is not None:
            self.noise_energy += float(noise.abs().square().double().sum().item())
            self.noise_real_energy += float(noise.real.square().double().sum().item())
            self.noise_imag_energy += float(noise.imag.square().double().sum().item())
            self.noise_samples += noise.numel()

    def to_dict(self):
        result = asdict(self)
        result.update(ber=self.bit_errors / self.total_bits if self.total_bits else None,
                      bler=self.block_errors / self.total_blocks if self.total_blocks else None,
                      measured_noise_variance=self.noise_energy / self.noise_samples
                      if self.noise_samples else None)
        return result

    def validate_noise(self, expected_no):
        if not self.noise_samples or expected_no <= 0:
            raise ValueError("Noise validation needs nonzero expected variance and samples")
        # Eight standard errors, with a 3% floor for floating point subtraction.
        tolerance = max(.03, 8 / math.sqrt(self.noise_samples))
        for energy, expected in ((self.noise_energy, expected_no),
                                 (self.noise_real_energy, expected_no / 2),
                                 (self.noise_imag_energy, expected_no / 2)):
            actual = energy / self.noise_samples
            if abs(actual / expected - 1) > tolerance:
                raise AssertionError(f"Measured AWGN power {actual} differs from expected {expected}")
        return True
