r"""
Module to build pretrained models from torchvision.models
- Module level Variable ``MODELS``: list of available models
- Function ``build``: build pretrained models from torchvision.models
"""

from typing import Any, Callable
import torch
from torchvision.models import get_model_builder, get_model_weights, list_models
from torch import nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
MODELS = list_models()


class _PredefinedModels(nn.Module):
    pass


class _TorchBuiltinModels(_PredefinedModels):
    def __init__(
        self,
        model: str,
        pretrained: bool,
        feature_size: int | None = None,
        pre_transforms: Callable | nn.Module | None = None,
        post_transforms: Callable | nn.Module | None = None,
        **kwargs,
    ):
        super().__init__()
        builder = get_model_builder(model)
        weights_enum = get_model_weights(builder)
        self.ecnoder = builder(
            weights=weights_enum.DEFAULT if pretrained else None, **kwargs
        )
        if feature_size is not None:
            self.head = nn.LazyLinear(feature_size)
        if pre_transforms is not None:
            self.pre_transforms = pre_transforms
        if post_transforms is not None:
            self.post_transforms = post_transforms

    def forward(self, x: Any) -> torch.Tensor:
        if hasattr(self, "pre_transforms"):
            x = self.pre_transforms(x)
        logits = self.ecnoder(x)
        if hasattr(self, "head"):
            logits = self.head(logits)
        if hasattr(self, "post_transforms"):
            logits = self.post_transforms(logits)
        assert isinstance(logits, torch.Tensor)
        return logits


def build(
    model: str,
    pretrained: bool = False,
    feature_size: int | None = None,
    pre_transforms: Callable | nn.Module | None = None,
    post_transform: Callable | nn.Module | None = None,
    **kwargs,
) -> nn.Module:
    r"""
    Builtin models from torchvision.models

    Args:
        - model (str): model name
        - pretrained (bool): load pretrained weights
        - feature_size (int): output feature size
        - pre_transforms (Callable | nn.Module | None): transforms before model
        - post_transform (Callable | nn.Module | None): transforms after model
        - \*\*kwargs: other arguments for model

    """
    return _TorchBuiltinModels(
        model, pretrained, feature_size, pre_transforms, post_transform, **kwargs
    )
