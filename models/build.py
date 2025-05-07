from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .mfm import build_mfm
from .mfm_mask import build_mfm_mask
from .mfm_mask_eval import build_mfm_mask_eval
from .mfm_fag import build_mfm_FAG
from .mfm_fag_latent import build_mfm_FAG_latent



def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model


def build_model_mask(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm_mask(config)
    else:
        raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model


def build_model_mask_eval(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm_mask_eval(config)
    else:
        raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model


def build_model_FAG(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm_FAG(config)
    else:
        raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model


def build_model_FAG_latent(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm_FAG_latent(config)
    else:
        raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
