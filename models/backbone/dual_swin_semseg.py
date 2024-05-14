import torch
import torch.nn as nn

from models.backbone.swin import (
    BasicLayer,
    PatchEmbedding,
    PatchMerging,
)
from models.pos_embed import get_2d_sincos_pos_embed
from models.net_utils import FeatureFusionModule as FFM
from models.net_utils import FeatureRectifyModule as FRM
from utils.logger import get_root_logger
from utils.checkpoint import load_dual_branch_model_from_mae_pretrained

logger = get_root_logger()


class DualSwinSemSeg(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        in_chans: int = 3,
        depths: tuple = (2, 2, 6, 2),
        embed_dim: int = 96,
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        norm_fuse=nn.BatchNorm2d,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        ape=False,
        norm_pix_loss=False,
        target_type: str = "origin",
    ):

        super().__init__()
        logger.info(f"norm_pix_loss = {norm_pix_loss}")
        logger.info(f"target_type = {target_type}")

        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.norm_fuse = norm_fuse
        self.out_indices = out_indices
        self.target_type = target_type
        self.frozen_stages = frozen_stages
        self.ape = ape

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channel=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.patch_embed_d = PatchEmbedding(
            patch_size=patch_size,
            in_channel=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        if self.ape:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
            )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)

        self.layers = self.build_layers()
        self.layers_d = self.build_layers()
        self.downsample, self.downsample_d = self.build_downsample_layers()

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.FRMs = self.build_FRM()
        self.FFMs = self.build_FFM()
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
            layer_d = norm_layer(num_features[i_layer])
            layer_name_d = f"norm_d{i_layer}"
            self.add_module(layer_name_d, layer_d)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        if self.ape:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=False
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if isinstance(pretrained, str):
            load_dual_branch_model_from_mae_pretrained(self, pretrained)
        else:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_FRM(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = FRM(dim=int(self.embed_dim * 2**i), reduction=1)
            layers.append(layer)

        return layers

    def build_FFM(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = FFM(
                dim=int(self.embed_dim * 2**i),
                reduction=1,
                num_heads=self.num_heads[i],
                norm_layer=self.norm_fuse,
            )
            layers.append(layer)

        return layers

    def build_downsample_layers(self):
        layers = nn.ModuleList()
        layers_d = nn.ModuleList()
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                downsample = PatchMerging(
                    dim=int(self.embed_dim * 2**i), norm_layer=self.norm_layer
                )
                downsample_d = PatchMerging(
                    dim=int(self.embed_dim * 2**i), norm_layer=self.norm_layer
                )

                layers.append(downsample)
                layers_d.append(downsample_d)

        return layers, layers_d

    def build_layers(self):
        layers = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dim * 2**i,
                depth=self.depths[i],
                window_size=self.window_size,
                num_heads=self.num_heads[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i]) : sum(self.depths[: i + 1])],
                norm_layer=self.norm_layer,
                # downsample=PatchMerging if i < self.num_layers - 1 else None,
                downsample=None,
            )
            layers.append(layer)
        return layers

    def forward_encoder(self, x: torch.Tensor, x_d: torch.Tensor):
        B, H, W, C = x.shape
        x = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)

        x = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            layer_d = self.layers_d[i]
            x = layer(x)
            x_d = layer_d(x_d)

            x = x.permute(0, 3, 1, 2).contiguous()
            x_d = x_d.permute(0, 3, 1, 2).contiguous()
            x, x_d = self.FRMs[i](x, x_d)
            x = x.permute(0, 2, 3, 1).contiguous()
            x_d = x_d.permute(0, 2, 3, 1).contiguous()
            x_out, x_out_d = x, x_d
            if i < self.num_layers - 1:
                x = self.downsample[i](x)
                x_d = self.downsample_d[i](x_d)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                norm_layer_d = getattr(self, f"norm_d{i}")
                x_out_d = norm_layer_d(x_out_d)

                x_out = x_out.permute(0, 3, 1, 2).contiguous()
                x_out_d = x_out_d.permute(0, 3, 1, 2).contiguous()
                out = self.FFMs[i](x_out, x_out_d)

                outs.append(out)
        return tuple(outs)

    def forward(self, inputs):
        x = inputs["rgb"]
        x_d = inputs["depth"]
        outs = self.forward_encoder(x, x_d)
        return outs

class dual_swin_semseg_t(DualSwinSemSeg):
    def __init__(self, **kwargs):
        super(dual_swin_semseg_t, self).__init__(
            img_size=224,
            patch_size=4,
            in_chans=3,
            decoder_embed_dim=768,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.3,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
            **kwargs,
        )


class dual_swin_semseg_s(DualSwinSemSeg):
    def __init__(self, **kwargs):
        super(dual_swin_semseg_s, self).__init__(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.3,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
            **kwargs,
        )


class dual_swin_semseg_b(DualSwinSemSeg):
    def __init__(self, **kwargs):
        super(dual_swin_semseg_b, self).__init__(
            img_size=384,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.3,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
            **kwargs,
        )


if __name__ == "__main__":
    net = dual_swin_semseg_s()
    model_file = "output_dir/dual_swin_small_normalized/checkpoint-2880.pth"
    load_dual_branch_model_from_mae_pretrained(net, model_file)
    inputs = {"rgb": torch.ones(1, 3, 224, 224), "depth": torch.ones(1, 3, 224, 224)}
    y = net(inputs)
    for out in y:
        print(f"Output shape = {out.shape}")
    