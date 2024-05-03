from torch import nn

from torchsparseplusplus import SparseTensor
from torchsparseplusplus.nn.utils import fapply

__all__ = ["ReLU", "LeakyReLU", "SiLU"]


class ReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class SiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
