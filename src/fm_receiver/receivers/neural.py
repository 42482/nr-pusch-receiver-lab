"""Adapters keep the common observation/result contract for B1 and B2."""

from ..data.schema import ReceiverResult
from .baseline import require_finite


class NeuralReceiver:
    def __init__(self, components, outer, codec=None, inner=None):
        self.components, self.outer, self.codec, self.inner = components, outer, codec, inner

    def __call__(self, observation):
        if observation.layout_id != self.components.adapter.layout_id:
            raise ValueError("Neural receiver layout mismatch")
        require_finite("Y", observation.y)
        require_finite("noise", observation.no)
        if not bool((observation.no > 0).all()):
            raise ValueError("Noise variance must be positive")
        h, err = self.components.receiver.channel_estimator(observation.y, observation.no)
        llr = self.outer(observation, h)
        require_finite("neural LLR", llr)
        if self.inner is None:
            bits, crc = self.components.receiver.decoder(llr)
        else:
            decoded, _ = self.inner(self.codec.to_code_llr(llr), self.codec)
            bits, crc = self.codec.to_tb(decoded)
        return ReceiverResult(bits, crc, llr, h, err)
