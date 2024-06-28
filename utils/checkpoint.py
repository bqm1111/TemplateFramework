from email.policy import strict
import os.path as osp

import torch
from torch.nn import functional as F
from utils.logger import get_root_logger

logger = get_root_logger()


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    if len(err_msg) > 0:
        err_msg.insert(
            0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    if not osp.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    checkpoint = torch.load(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"No state_dict found in checkpoint file {filename}")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # reshape absolute position embedding
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
                N2, H, W, C2
            ).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode="bicubic",
                )
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(
                    1, 0
                )

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_dual_branch_model_from_mae_pretrained(model, model_file: str):
    raw_state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    if "model" in raw_state_dict.keys():
        raw_state_dict = raw_state_dict["model"]

    state_dict = {}
    mae_keys_only = ["_up", "pos_emb", "mask_token", "decoder"]
    for k, v in raw_state_dict.items():
        if any(subkey in k for subkey in mae_keys_only):
            continue
        state_dict[k] = v
    model.load_state_dict(state_dict, strict=False)

    del state_dict
    logger.info("Successfully load dual branch model from mae pretrained")
    return model


def load_swin_pretrained_model(model, model_file):
    logger.info(f"Loading pretrained from {model_file}")
    pretrained = torch.load(model_file)
    state_dict = {}
    if "state_dict" in pretrained.keys():
        for key, value in pretrained["state_dict"].items():
            if "backbone." in key:
                # Hack to load weight from key "backbone.layers.1.downsample.norm.weight" to key "downsample.1.reduction.weight"
                if "downsample" in key:
                    all_key_parts = key.split(".")
                    new_key_part = [all_key_parts[3], all_key_parts[2],
                                    all_key_parts[4], all_key_parts[5]]
                    new_key = ".".join(new_key_part)
                    prefix_depth = new_key.split(".")[0]
                else:
                    new_key = key[9:]
                    prefix_depth = new_key.split(".")[0]

                depth_key = prefix_depth + "_d"
                depth_key = depth_key + new_key[len(prefix_depth):]
                state_dict[new_key] = value
                state_dict[depth_key] = value
    elif "model" in pretrained.keys():
        for key, value in pretrained["model"].items():
            # Hack to load weight from key "backbone.layers.1.downsample.norm.weight" to key "downsample.1.reduction.weight"
            if "downsample" in key:
                all_key_parts = key.split(".")
                new_key_part = [all_key_parts[2], all_key_parts[1],
                                all_key_parts[3], all_key_parts[4]]
                new_key = ".".join(new_key_part)
                prefix_depth = new_key.split(".")[0]
            else:
                new_key = key
                prefix_depth = new_key.split(".")[0]

            depth_key = prefix_depth + "_d"
            depth_key = depth_key + new_key[len(prefix_depth):]
            state_dict[new_key] = value
            state_dict[depth_key] = value
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    logger.info("Successfully load swin pretrained model")

    return model

def load_dual_dat_pretrained_model(model, model_file):
    logger.info(f"Loading pretrained from {model_file}")
    pretrained = torch.load(model_file)
    state_dict = {}
    for key, value in pretrained["state_dict"].items():
        if "backbone." in key:
            new_key = key[9:]
            prefix_depth = new_key.split(".")[0]

            depth_key = prefix_depth + "_d"
            depth_key = depth_key + new_key[len(prefix_depth):]
            state_dict[new_key] = value
            state_dict[depth_key] = value

    model.load_state_dict(state_dict, strict=True)
    del state_dict
    logger.info("Successfully load DAT++ pretrained model")

def load_dat_pretrained_model(model, model_file):
    logger.info(f"Loading pretrained from {model_file}")
    pretrained = torch.load(model_file)
    state_dict = {}
    for key, value in pretrained["state_dict"].items():
        if "backbone." in key:
            new_key = key[9:]
            state_dict[new_key] = value

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    logger.info("Successfully load DAT++ pretrained model")