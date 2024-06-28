from unittest.util import strclass
import torch
import torch.nn as nn

from utils.logger import get_root_logger
from models.dat_utils.dat_blocks import *
from utils.checkpoint import load_dat_pretrained_model
from models.backbone.dat import LayerNormProxy, TransformerStage
from models.net_utils import FeatureRectifyModule as FRM
from models.net_utils import MFA
from models.net_utils import MPG
logger = get_root_logger()


class VPT_DAT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=40, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 4, 18, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.5,
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[
                     ["N", "D"],
                     ["N", "D", "N", "D"],
                     ["N", "D", "N", "D", "N", "D", "N", "D", "N",
                      "D", "N", "D", "N", "D", "N", "D", "N", "D"],
                     ["D", "D"]],
                 groups=[1, 2, 3, 6],
                 use_pes=[True, True, True, True],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[7, 7, 7, 7],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[True, True, True, True],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 pretrained=None,
                 prompt_tuning_config=False,
                 **kwargs
                 ):
        super().__init__()
        self.dims = dims
        self.num_heads = heads
        self.out_indices = out_indices
        self.log_cpb = log_cpb[0]
        self.dwc_pe = dwc_pes[0]
        self.slide = stage_spec[0][0] == "E"

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
                                                 LayerNormProxy(dim_stem))
        self.patch_proj_d = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
                                                 LayerNormProxy(dim_stem))

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        if prompt_tuning_config:
            pt_config = {}
            pt_config["num_token"] = 32
            pt_config["token_dim"] = 30
            pt_config["reduction_ratio"] = 32
        else:
            pt_config = None

        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    local_orf[i], local_kv_sizes[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i],
                    ksize_qnas[i],
                    nqs[i],
                    qna_activation,
                    deform_groups[i],
                    layer_scale_values[i],
                    use_lpus[i],
                    use_cmt_mlps[i],
                    log_cpb[i],
                    i, use_checkpoint, prompt_tuning_config=pt_config
                )
            )
            if i in self.out_indices:
                self.norms.append(
                    LayerNormProxy(dim2)
                )
            else:
                self.norms.append(nn.Identity)
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        self.mpg = self.build_MPG()

        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.lower_lr_kvs = lower_lr_kvs
        self.pretrained = pretrained
        self.reset_parameters()

    def build_FRM(self):
        layers = nn.ModuleList()
        for i in range(4):
            layer = FRM(self.dims[i], reduction=1)
            layers.append(layer)

        return layers

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def init_weights(self, pretrained):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

        if isinstance(pretrained, str):
            load_dat_pretrained_model(self, pretrained)
            logger.info("DAT backbone has been loaded successfully!")
        for name, param in self.named_parameters():
            if "mfa" in name or "mpg" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def build_MPG(self):
        layers = nn.ModuleList()
        for i in range(4):
            if i == 0:
                layer = MPG(self.dims[i], self.dims[i],
                            None, None, None, 1, False)
            else:
                layer = MPG(self.dims[i-1], self.dims[i],
                            kernel_size=3, stride=2, padding=1)
            layers.append(layer)

        return layers

    def forward(self, x, x_d):
        x = self.patch_proj(x)
        x_d = self.patch_proj_d(x_d)

        outs = []
        for i in range(4):
            x, x_d = self.mpg[i](x, x_d)

            x = self.stages[i](x)
            y = self.norms[i](x)
            outs.append(y.contiguous())
            if i < 3:
                x = self.down_projs[i](x)

        return outs


if __name__ == '__main__':
    net = VPT_DAT(prompt_tuning_config=True)
    model_file = "pretrained/upn_dat_s_160k.pth"
    state_dict = torch.load(model_file)["state_dict"]
    net.init_weights(None)
    for name, param in net.named_parameters():
        if param.requires_grad is True:
            print(name)
    # load_dat_pretrained_model(net, model_file)
    # for _, name in net.parameters():
    #     print(param)
    # y = net(torch.ones(1, 3, 480, 640), torch.ones(1, 3, 480, 640))
