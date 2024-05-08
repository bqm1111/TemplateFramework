import torch
import numpy as np
from einops import rearrange


def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
    return x


def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(x.shape[0], 3, h * p, h * p)
    return imgs


def window_masking(
    x: torch.Tensor,
    mask_token,
    r: int = 4,
    remove: bool = False,
    mask_len_sparse: bool = False,
    mask_ratio=0.75,
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
    sparse_keep = sparse_shuffle[:, : int(d**2 * (1 - mask_ratio))]

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
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
        x_masked = rearrange(
            x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
        )
        return x_masked, mask
