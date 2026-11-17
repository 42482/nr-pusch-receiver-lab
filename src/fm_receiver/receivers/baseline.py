import torch
from ..data.schema import ObservationBatch, ReceiverResult


def require_finite(name, tensor):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Non-finite values in {name}")


class BaselineReceiver:
    """Official PUSCHReceiver with observable, explicitly injected components."""

    def __init__(self, receiver, channel_estimator, decoder, layout_id):
        self.receiver = receiver
        self.channel_estimator = channel_estimator
        self.decoder = decoder
        self.layout_id = layout_id

    def __call__(self, observation: ObservationBatch) -> ReceiverResult:
        if observation.layout_id != self.layout_id:
            raise ValueError("Observation and receiver resource layouts differ")
        require_finite("Y", observation.y)
        require_finite("noise variance", observation.no)
        if not bool((observation.no > 0).all()):
            raise ValueError("Receiver noise variance must be positive")
        intermediate = {}

        def capture_estimate(module, args, output):
            intermediate["h_hat"], intermediate["err_var"] = output

        def capture_decoder_input(module, args):
            intermediate["llr"] = args[0]

        handles = [self.channel_estimator.register_forward_hook(capture_estimate),
                   self.decoder.register_forward_pre_hook(capture_decoder_input)]
        try:
            bits, crc = self.receiver(observation.y, observation.no)
        finally:
            for handle in handles:
                handle.remove()
        for name, value in intermediate.items():
            require_finite(name, value)
        require_finite("decoded bits", bits)
        return ReceiverResult(bits, crc, **intermediate)
