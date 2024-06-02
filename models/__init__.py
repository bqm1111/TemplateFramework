from .backbone.dual_swin_semseg import (
    dual_swin_semseg_b,
    dual_swin_semseg_s,
    dual_swin_semseg_t,
)
from .decoders import MLPDecoderHead, DeepLabV3Plus, UPerHead, FCNHead
from .backbone.dual_dat import Dual_DAT
AVAI_BACKBONE = {
    "swin_s": dual_swin_semseg_s,
    "swin_b": dual_swin_semseg_b,
    "swin_t": dual_swin_semseg_t,
    "dat_s": Dual_DAT
}
AVAI_DECODER = {
    "mlp": MLPDecoderHead,
    "uper": UPerHead,
    "fcn": FCNHead,
    "deeplabv3": DeepLabV3Plus,
}


def get_decoder(model_name, **kwargs):
    if model_name not in AVAI_DECODER:
        print("not supported decoder name, please implement it first.")
    return AVAI_DECODER[model_name](**kwargs)


def get_backbone(model_name, **kwargs):
    if model_name not in AVAI_BACKBONE:
        print("not supported backbone name, please implement it first.")
    return AVAI_BACKBONE[model_name](**kwargs)

