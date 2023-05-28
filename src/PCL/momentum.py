from typing import Dict, Iterable
import torch
from torch import Tensor
from torch.nn import Module


class Momentum(Module):
    def __init__(
        self,
        src: Iterable | Module | Dict[str, Tensor],
        dst: Iterable | Module | Dict[str, Tensor],
        momentum: float,
    ):
        super().__init__()
        self.src = src
        self.dst = dst
        self.momentum = momentum

    @torch.no_grad()
    def step(self) -> None:
        step(self.src, self.dst, self.momentum)


@torch.no_grad()
def step(
    src: Iterable | Module | Dict[str, Tensor],
    dst: Iterable | Module | Dict[str, Tensor],
    momentum: float,
) -> None:
    r"""
    Update momentum module with current module

    Args:
        module (torch.nn.Module): current module
        momentum_module (torch.nn.Module): momentum module
        momentum (float): momentum value
    """
    if isinstance(src, Module):
        src = src.parameters()
    if isinstance(dst, Module):
        dst = dst.parameters()

    if isinstance(src, Dict) and not isinstance(dst, Dict):
        raise ValueError("src and dst must be both dict or both not dict")

    if not isinstance(src, Dict) and isinstance(dst, Dict):
        raise ValueError("src and dst must be both dict or both not dict")

    if isinstance(src, Dict) and isinstance(dst, Dict):
        src_keys = src.keys()
        dst_keys = dst.keys()
        if src_keys != dst_keys:
            raise ValueError(
                f"src and dst keys are not equal: {src_keys} != {dst_keys}"
            )
        src = [src[key] for key in src_keys]
        dst = [dst[key] for key in src_keys]

    for param, momentum_param in zip(src, dst):
        momentum_param.data = momentum_param.data * momentum + param.data.detach().to(
            momentum_param.data.device
        ) * (1 - momentum)
