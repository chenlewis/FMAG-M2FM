from .vision_transformer import build_vit
from .mfm_mask import build_mfm_mask


def build_model_mask(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm_mask(config)
    else:
        raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model

