import torch
from torch import nn

from torchsparseplusplus import SparseTensor
from torchsparseplusplus.nn.utils import fapply

__all__ = ["BatchNorm", "GroupNorm", "InstanceNorm", "SyncBatchNorm"]


class InstanceNorm(nn.InstanceNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class GroupNorm(nn.GroupNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride

        batch_size = torch.max(coords[:, 0]).item() + 1
        num_channels = feats.shape[1]

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, 0] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = SparseTensor(
            coords=coords,
            feats=nfeats,
            stride=stride,
            spatial_range=input.spatial_range,
        )
        output._caches = input._caches
        return output


class SyncBatchNorm(nn.SyncBatchNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


def convert_sync_batchnorm(module, process_group=None, memo=None):
    """
    Recursively converts all BatchNorm layers to SyncBatchNorm layers.
    Args:
        module (torch.nn.Module): Module containing BatchNorm layers
        process_group (optional): Process group for synchronization
        memo (dict, optional): Memory dictionary for tracking converted modules
    Returns:
        torch.nn.Module: Converted module with SyncBatchNorm layers
    """
    if memo is None:
        memo = {}

    # Return already converted module from memo
    if module in memo:
        return memo[module]

    # Convert BatchNorm module to SyncBatchNorm
    if isinstance(module, BatchNorm):
        # Create new SyncBatchNorm module
        module_converted = SyncBatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            process_group,
        )

        # Copy parameters and buffers
        if module.affine:
            with torch.no_grad():
                if module.weight is not None:
                    module_converted.weight.copy_(module.weight)
                if module.bias is not None:
                    module_converted.bias.copy_(module.bias)

        with torch.no_grad():
            if module.running_mean is not None:
                module_converted.running_mean.copy_(module.running_mean)
            if module.running_var is not None:
                module_converted.running_var.copy_(module.running_var)
            if module.num_batches_tracked is not None:
                module_converted.num_batches_tracked.copy_(
                    module.num_batches_tracked)

        memo[module] = module_converted
        return module_converted

    # For non-BatchNorm modules
    memo[module] = module  # Memoize before recursion to handle shared modules

    # Recursively convert children
    for name, child in module.named_children():
        new_child = convert_sync_batchnorm(child, process_group, memo)
        if new_child is not child:
            setattr(module, name, new_child)

    return module
