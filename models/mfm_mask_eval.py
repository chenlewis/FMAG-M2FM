from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_
from timm.models.resnet import Bottleneck, ResNet

# from frequency_loss import FrequencyLoss
# from swin_transformer import SwinTransformer
# from utils import get_2d_sincos_pos_embed
# from vision_transformer import VisionTransformer
# from transformer_model import ViTB16
# import matplotlib.pyplot as plt
# from torchvision.transforms.functional import to_pil_image

from .frequency_loss import FrequencyLoss
from .swin_transformer import SwinTransformer
from .utils import get_2d_sincos_pos_embed
from .vision_transformer import VisionTransformer
from .transformer_model import ViTB16
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image





class VisionTransformerForMFM(VisionTransformer):
    def __init__(self, config, use_fixed_pos_emb=False, **kwargs):
        super().__init__(**kwargs)
        self.decoder_depth = config.MODEL.VIT.DECODER.DEPTH
        self.filter_type = config.DATA.FILTER_TYPE
        assert self.num_classes == 0

        if use_fixed_pos_emb:
            assert self.pos_embed is None
            self.pos_embed = nn.Parameter(torch.zeros(
                1, self.patch_embed.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        assert self.pos_embed is not None

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, x_fft):
        if self.filter_type == 'mfm':
            x = x_fft
        x = self.patch_embed(x)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        if self.decoder_depth == 0:
            # remove cls token
            x = x[:, 1:]
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x





class VisionTransformerDecoderForMFM(VisionTransformer):
    def __init__(self, config, use_fixed_pos_emb=False, **kwargs):
        super().__init__(**kwargs)
        assert config.MODEL.VIT.DECODER.DEPTH > 0
        assert self.num_classes == 0
        self.embed = nn.Linear(config.MODEL.VIT.EMBED_DIM, self.embed_dim, bias=True)
        if use_fixed_pos_emb:
            assert self.pos_embed is None
            self.pos_embed = nn.Parameter(torch.zeros(
                1, self.patch_embed.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        assert self.pos_embed is not None
        self.pred = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.patch_size ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.patch_size),
        )
        self.patch_embed = None
        self.cls_token = None

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x):
        # embed tokens
        x = self.embed(x)

        # add pos embed
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        # remove cls token
        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # predictor projection
        x = self.pred(x)
        return x


class MFM_mask_eval(nn.Module):
    def __init__(self, encoder, encoder_stride, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        
        self.FAG = None

        self.normalize_img = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        if self.decoder is None:
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )
    
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def FAG_eval(self, x):
        x = self.normalize_img(x)
        z = self.encoder(x, x)
        x_rec = self.decoder(z)
        
        # x_rec = self.FAG_normalize(x_rec)
        result = self.FAG(x_rec)
        return result

    
    def forward(self, x, x_target, mask=None, label=None):
        return self.FAG_eval(x)

    
    

    
    
    

def build_mfm_mask_eval(config):
    model_type = config.MODEL.TYPE
    
    encoder = VisionTransformerForMFM(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.VIT.PATCH_SIZE,
        in_chans=config.MODEL.VIT.IN_CHANS,
        num_classes=0,
        embed_dim=config.MODEL.VIT.EMBED_DIM,
        depth=config.MODEL.VIT.DEPTH,
        num_heads=config.MODEL.VIT.NUM_HEADS,
        mlp_ratio=config.MODEL.VIT.MLP_RATIO,
        qkv_bias=config.MODEL.VIT.QKV_BIAS,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=config.MODEL.VIT.INIT_VALUES,
        use_abs_pos_emb=config.MODEL.VIT.USE_APE,
        use_fixed_pos_emb=config.MODEL.VIT.USE_FPE,
        use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
        use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
        use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING,
        config=config)
    encoder_stride = 16
    if config.MODEL.VIT.DECODER.DEPTH > 0:
        decoder = VisionTransformerDecoderForMFM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.DECODER.EMBED_DIM,
            depth=config.MODEL.VIT.DECODER.DEPTH,
            num_heads=config.MODEL.VIT.DECODER.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_fixed_pos_emb=config.MODEL.VIT.USE_FPE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING,
            config=config)
    else:
        print ('---------------------------------------------------------')
        print ('---------------------------------------------------------')
        print ('---------------------------------------------------------')
        print ('---------------------------------------------------------')
        print ('no decoder!!!')
        print ('---------------------------------------------------------')
        print ('---------------------------------------------------------')
        print ('---------------------------------------------------------')
        print ('---------------------------------------------------------')
        decoder = None

    model = MFM_mask_eval(encoder=encoder, encoder_stride=encoder_stride, decoder=decoder, config=config)

    return model
