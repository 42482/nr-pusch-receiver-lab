from sionna.phy.nr import TBDecoder


def build_tb_decoder(encoder, config, device):
    return TBDecoder(encoder, num_bp_iter=config.num_bp_iter,
                     cn_update=config.cn_update, precision=config.precision, device=device)
