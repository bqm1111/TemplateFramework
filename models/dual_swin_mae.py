from curses import window
import imp
import time
from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from models.swin_unet import PatchExpanding
from utils.logger import get_root_logger
from .swin import PatchEmbed, PatchMerging, BasicLayer, BasicLayerUp
from .net_utils import FeatureFusionModule as FFM
from .net_utils import FeatureRectifyModule as FRM

logger = get_root_logger()


class DualSwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        decoder_embed_dim=512,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        norm_fuse=nn.BatchNorm2d,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        norm_pix_loss=False,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_pix_loss = norm_pix_loss
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # split depth into non-overlapping patches
        self.patch_embed_d = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

            self.absolute_pos_embed_d = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed_d, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)

        # stochastic depth
        self.dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers and fusemodules
        self.layers = self.build_layers()
        self.layers_d = self.build_layers()
        self.downsamples = nn.ModuleList()
        self.downsamples_d = nn.ModuleList()
        self.FRMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()
        for i_layer in range(self.num_layers):
            fr = FRM(dim=int(embed_dim * 2**i_layer), reduction=1)
            self.FRMs.append(fr)
            # patch merging layer
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(
                    dim=int(embed_dim * 2**i_layer), norm_layer=norm_layer
                )
                self.downsamples.append(downsample)
                downsample_d = PatchMerging(
                    dim=int(embed_dim * 2**i_layer), norm_layer=norm_layer
                )
                self.downsamples_d.append(downsample_d)

            fuse = FFM(
                dim=int(embed_dim * 2**i_layer),
                reduction=1,
                num_heads=num_heads[i_layer],
                norm_layer=norm_fuse,
            )
            self.FFMs.append(fuse)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
            layer_d = norm_layer(num_features[i_layer])
            layer_name_d = f"norm_d{i_layer}"
            self.add_module(layer_name_d, layer_d)

        # Decoder and masking
        # RGB branch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #
        self.first_patch_expanding = PatchExpanding(
            dim=decoder_embed_dim, norm_layer=norm_layer
        )
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim // 8, patch_size**2 * in_chans, bias=True
        )

        # Depth branch
        self.mask_token_d = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.first_patch_expanding_d = PatchExpanding(
            dim=decoder_embed_dim, norm_layer=norm_layer
        )
        self.layers_up_d = self.build_layers_up()
        self.norm_up_d = norm_layer(embed_dim)
        self.decoder_pred_d = nn.Linear(
            decoder_embed_dim // 8, patch_size**2 * in_chans, bias=True
        )

        self._freeze_stages()

    def build_layers(self):
        layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                norm_layer=self.norm_layer,
                use_checkpoint=self.use_checkpoint,
            )
            layers.append(layer)

        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicLayerUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
            )
            layers_up.append(layer)
        return layers_up

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    def window_masking(
        self,
        x: torch.Tensor,
        r: int = 4,
        remove: bool = False,
        mask_len_sparse: bool = False,
    ):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, "B H W C -> B (H W) C")
        B, L, D = x.shape
        assert int(L**0.5 / r) == L**0.5 / r
        d = int(L**0.5 // r)

        noise = torch.rand(B, d**2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, : int(d**2 * (1 - self.mask_ratio))]

        index_keep_part = (
            torch.div(sparse_keep, d, rounding_mode="floor") * d * r**2
            + sparse_keep % d * r
        )
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat(
                    [index_keep, index_keep_part + int(L**0.5) * i + j], dim=1
                )

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(
                index_all[i], index_keep.cpu().numpy()[i], assume_unique=True
            )
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d**2], device=x.device)
            mask[:, : sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, : index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(
                x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D)
            )
            x_masked = rearrange(
                x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
            )
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked = rearrange(
                x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
            )
            return x_masked, mask

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_dualpath_model(self, pretrained)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_encoder(self, x, x_d):
        """Forward function."""
        x = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            absolute_pos_embed_d = F.interpolate(
                self.absolute_pos_embed_d, size=(Wh, Ww), mode="bicubic"
            )
            x_d = (x_d + absolute_pos_embed_d).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer_d = self.layers_d[i]

            x, H, W = layer(x, Wh, Ww)
            x_d, _, _ = layer_d(x_d, Wh, Ww)

            # Feature Rectify
            x = x.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            x_d = (
                x_d.view(-1, H, W, self.num_features[i])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x, x_d = self.FRMs[i](x, x_d)
            x = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)

            x_out, x_out_d = x, x_d

            if i < self.num_layers - 1:
                x = self.downsamples[i](x, H, W)
                x_d = self.downsamples_d[i](x_d, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                norm_layer_d = getattr(self, f"norm_d{i}")
                x_out_d = norm_layer_d(x_out_d)

                x_out = (
                    x_out.view(-1, H, W, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                x_out_d = (
                    x_out_d.view(-1, H, W, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out = self.FFMs[i](x_out, x_out_d)

                outs.append(out)

        return tuple(outs)
    # 
    def forward_decoder(self, x, x_d):
        x = self.first_patch_expanding(x)
        x_d = self.first_patch_expanding(x_d)
        for layer in self.layers_up:
            x = layer(x)

        x = self.norm_up(x)
        x = rearrange(x, "B H W C -> B (H W) C")
        x = self.decoder_pred(x)
        for layer in self.layers_up_d:
            x = layer(x)

        x_d = self.norm_up_d(x_d)
        x_d = rearrange(x_d, "B H W C -> B (H W) C")
        x_d = self.decoder_pred(x_d)
        return x, x_d

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x, x_d):
        outs = self.forward_encoder(x, x_d)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DualSwinTransformer, self).train(mode)
        self._freeze_stages()


class swin_s(DualSwinTransformer):
    def __init__(self, **kwargs):
        super(swin_s, self).__init__(
            pretrain_img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.3,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
        )


class swin_b(DualSwinTransformer):
    def __init__(self, **kwargs):
        super(swin_b, self).__init__(
            pretrain_img_size=384,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.3,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
        )


def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device("cpu"))
        # raw_state_dict = torch.load(model_file)
        if "model" in raw_state_dict.keys():
            raw_state_dict = raw_state_dict["model"]
    else:
        raw_state_dict = model_file
    # copy to  hha backbone
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find("downsample") >= 0 and k.find("layer") >= 0:
            name = k.replace("downsample.", "")
            name = name.replace("layers", "downsamples")
            state_dict[name] = v
            name = name.replace("downsamples", "downsamples_d")
            state_dict[name] = v
        elif k.find("patch_embed") >= 0:
            state_dict[k] = v
            state_dict[k.replace("patch_embed", "patch_embed_d")] = v
        elif k.find("layer") >= 0:
            state_dict[k] = v
            state_dict[k.replace("layers", "layers_d")] = v
        elif k.find("norm") >= 0:
            state_dict[k] = v
            state_dict[k.replace("norm", "norm_d")] = v

    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module." + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend
        )
    )

    return model
