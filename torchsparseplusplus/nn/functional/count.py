import torch

import torchsparseplusplus.backend

__all__ = ["spcount"]


def spcount(coords: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
    coords = coords.contiguous()
    if coords.device.type == "cuda":
        return torchsparseplusplus.backend.count_cuda(coords, num)
    elif coords.device.type == "cpu":
        return torchsparseplusplus.backend.count_cpu(coords, num)
    else:
        device = coords.device
        return torchsparseplusplus.backend.count_cpu(coords.cpu(), num).to(device)
