import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from models.backbone.swin import (
    BasicLayer,
    PatchExpanding,
    BasicLayer_up,
    PatchEmbedding,
    PatchMerging,
)
from models.pos_embed import get_2d_sincos_pos_embed
from models.net_utils import FeatureFusionModule as FFM
from models.net_utils import FeatureRectifyModule as FRM
from utils.logger import get_root_logger

logger = get_root_logger()


class DualSwinMAE(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        in_chans: int = 3,
        decoder_embed_dim=768,
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
        self.decoder_embed_dim = decoder_embed_dim
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

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.layers = self.build_layers()
        self.layers_d = self.build_layers()
        self.downsample, self.downsample_d = self.build_downsample_layers()
        self.norm_up = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim // 8, patch_size**2 * in_chans, bias=True
        )

        self.layers_up = self.build_layers_up()
        self.layers_up_d = self.build_layers_up()
        self.norm_up_d = norm_layer(embed_dim)
        self.decoder_pred_d = nn.Linear(
            decoder_embed_dim // 8, patch_size**2 * in_chans, bias=True
        )

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

        self.initialize_weights()
        # self._freeze_stages()

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

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=0.02)
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

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        for i in range(self.num_layers):
            index = self.num_layers - 1 - i
            if i == 0:
                layer = PatchExpanding(
                    dim=self.decoder_embed_dim, norm_layer=self.norm_layer
                )
            else:
                layer = BasicLayer_up(
                    dim=self.embed_dim * 2**index,
                    depth=self.depths[index],
                    window_size=self.window_size,
                    num_heads=self.num_heads[index],
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    drop_path=dpr[
                        sum(self.depths[:index]) : sum(self.depths[: index + 1])
                    ],
                    attn_drop=self.attn_drop_rate,
                    upsample=PatchExpanding if i < self.num_layers - 1 else None,
                    norm_layer=self.norm_layer,
                )
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, x: torch.Tensor, x_d: torch.Tensor):
        B, H, W, C = x.shape
        x = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)

        x = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        x, mask = self.window_masking(x, remove=False, mask_len_sparse=False)
        x_d, mask_d = self.window_masking(x_d, remove=False, mask_len_sparse=False)
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
        return x, mask, x_d, mask_d

    def forward_decoder(self, x, x_d):
        for layer in self.layers_up:
            x = layer(x)
        x = self.norm_up(x)
        x = rearrange(x, "B H W C -> B (H W) C")
        x = self.decoder_pred(x)

        for layer in self.layers_up:
            x_d = layer(x_d)
        x_d = self.norm_up_d(x_d)
        x_d = rearrange(x_d, "B H W C -> B (H W) C")
        x_d = self.decoder_pred_d(x_d)

        return x, x_d

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, inputs: dict):
        print(inputs.keys())
        x = inputs["rgb"]
        x_d = inputs["depth"]
        depth_anything_target = inputs["depth_anything"]
        latent, mask, latent_d, mask_d = self.forward_encoder(x, x_d)
        pred, pred_d = self.forward_decoder(latent, latent_d)
        loss = self.forward_loss(x, pred, mask)
        if self.target_type == "origin":
            loss_d = self.forward_loss(x_d, pred_d, mask_d)
        elif self.target_type == "rawDepthAnything":
            loss_d = self.forward_loss(depth_anything_target, pred_d, mask_d)
        elif self.target_type == "mix":
            raise NotImplementedError

        total_loss = 0.5 * loss + 0.5 * loss_d
        return (
            total_loss,
            {"rgb": pred, "depth": pred_d},
            {"rgb": mask, "depth": mask_d},
        )


class dual_swinmae_t(DualSwinMAE):
    def __init__(self, **kwargs):
        super(dual_swinmae_t, self).__init__(
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


class dual_swinmae_s(DualSwinMAE):
    def __init__(self, **kwargs):
        super(dual_swinmae_s, self).__init__(
            img_size=224,
            patch_size=4,
            in_chans=3,
            decoder_embed_dim=768,
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


class dual_swinmae_b(DualSwinMAE):
    def __init__(self, **kwargs):
        super(dual_swinmae_b, self).__init__(
            img_size=384,
            patch_size=4,
            in_chans=3,
            decoder_embed_dim=1024,
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
    net = dual_swinmae_b()
    inputs = {"rgb": torch.ones(1, 3, 384, 384), "depth": torch.ones(1, 3, 384, 384)}
    y = net(inputs)
