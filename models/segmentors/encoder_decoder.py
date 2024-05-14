import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_root_logger
from utils.init_func import init_weight
from engine import get_backbone, get_decoder

logger = get_root_logger()


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="mean", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
    ):
        super(EncoderDecoder, self).__init__()
        self.backbone = get_backbone(cfg.backbone)
        self.decoder = get_decoder(cfg.decoder)
        self.aux_head = get_decoder(cfg.aux_head)
        self.norm_layer = norm_layer

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info("Loading pretrained model: {}".format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")
        init_weight(
            self.decoder,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.aux_head:
            init_weight(
                self.aux_head,
                nn.init.kaiming_normal_,
                self.norm_layer,
                cfg.bn_eps,
                cfg.bn_momentum,
                mode="fan_in",
                nonlinearity="relu",
            )

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decoder.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(
                aux_fm, size=orisize[2:], mode="bilinear", align_corners=False
            )
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out
