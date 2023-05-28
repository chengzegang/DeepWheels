import os
from typing import Any, Tuple
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    Module,
)
import torch

__PRETRAINED_CKPT__ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "superpoint.py"
)


class VGGBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(F.relu(self.conv(x)))
        return x


class VGGBackbone(Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.conv1_1 = VGGBlock(1, 64, 3, padding=1)
        self.conv1_2 = VGGBlock(64, 64, 3, padding=1)
        self.pool1 = MaxPool2d(2, 2)

        self.conv2_1 = VGGBlock(64, 64, 3, padding=1)
        self.conv2_2 = VGGBlock(64, 64, 3, padding=1)
        self.pool2 = MaxPool2d(2, 2)

        self.conv3_1 = VGGBlock(64, 128, 3, padding=1)
        self.conv3_2 = VGGBlock(128, 128, 3, padding=1)
        self.pool3 = MaxPool2d(2, 2)

        self.conv4_1 = VGGBlock(128, 128, 3, padding=1)
        self.conv4_2 = VGGBlock(128, 128, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        return x


class Detector(Module):
    def __init__(self, *args, **kwargs: Any):
        super().__init__()
        self.conv1 = Conv2d(128, 256, 3, padding=1)
        self.bn1 = BatchNorm2d(256)
        self.conv2 = Conv2d(256, 8**2 + 1, 1)
        self.bn2 = BatchNorm2d(8**2 + 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.softmax(x, dim=1)
        x = x[:, :-1]
        x = F.pixel_shuffle(x, 8)
        x = x.nan_to_num(0)
        return x


class Descriptor(Module):
    def __init__(self, *args, **kwargs: Any):
        super().__init__()
        self.conv1 = Conv2d(128, 256, 3, padding=1)
        self.bn1 = BatchNorm2d(256)
        self.conv2 = Conv2d(256, 256, 1)
        self.bn2 = BatchNorm2d(256)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.interpolate(x, scale_factor=8, mode="bicubic", align_corners=False)
        x = F.normalize(x, dim=1, p=2)
        return x


class SuperPoint(Module):
    def __init__(self, pretrained: bool = True, *args, **kwargs: Any):
        super().__init__()
        self.vgg = VGGBackbone()
        self.detector = Detector()
        self.descriptor = Descriptor()
        if pretrained:
            self.load_state_dict(torch.load(__PRETRAINED_CKPT__))

    def _to_grayscale(self, x: Tensor) -> Tensor:
        x = x / 255.0
        x = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        x = x.unsqueeze(1)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self._to_grayscale(x)
        x = self.vgg(x)
        heatmaps = self.detector(x)
        descriptors = self.descriptor(x)

        return heatmaps, descriptors
