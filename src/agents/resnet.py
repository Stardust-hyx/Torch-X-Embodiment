import functools as ft
import torch
import torch.nn as nn
import numpy as np

from functools import partial
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple

ModuleDef = Any


class AddSpatialCoordinates(nn.Module):
        
    def __init__(self, shape=(224, 224), dtype=np.float32) -> None:
        super().__init__()
        grid = np.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in shape]),
                axis=0,
            ),
            dtype=dtype,
        )
        self.register_buffer('grid', torch.tensor(grid))

    def forward(self, x):
        grid = self.grid.unsqueeze(0).expand((x.shape[0], -1, -1, -1))
        # print(f'x.shape {x.shape}', flush=True)
        # print(f'grid.shape {grid.shape}', flush=True)
        return torch.concat((x, grid), dim=-3)


class ResNetBlock(nn.Module):
    """ResNet block."""

    def __init__(
            self,
            in_channels: int,
            filters: int,
            act: ModuleDef,
            stride: int = 1,
        ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=filters),
            act(inplace=True),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=filters),
        )
        self.stride = stride
        self.in_channels = in_channels
        self.filters = filters
        if stride != 1 or in_channels != filters:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=filters),
            )
        self.act = act(inplace=True)

    def forward(self, x):
        residual = x
        y = self.layers(x)

        if residual.shape != y.shape:
            residual = self.downsample(x)

        return self.act(residual + y)
    

class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    def __init__(
            self,
            in_channels: int,
            filters: int,
            act: ModuleDef,
            stride: int = 1,
        ) -> None:
        super().__init__()

        raise NotImplementedError()
        


class ResNetEncoder(nn.Module):
    """ResNetV1."""

    def __init__(
            self,
            stage_sizes: Sequence[int],
            block_cls: ModuleDef = ResNetBlock,
            num_filters: int = 64,
            act: str = "relu",
            norm: str = "group",
            add_spatial_coordinates: bool = False,
            pooling_method: str = "avg",
            num_spatial_blocks: int = 8,
            input_img_shape: Sequence[int] = (224, 224),
            input_channels: int = 6,
        ) -> None:
        super().__init__()

        self.add_spatial_coordinates = add_spatial_coordinates
        if add_spatial_coordinates:
            self.spatial_coordinates = AddSpatialCoordinates(shape=input_img_shape)

        assert norm == "group"
  
        act = getattr(nn, act)

        # conv_init, norm_init, pool_init
        layers = [
            nn.Conv2d(
                in_channels=input_channels+2 if add_spatial_coordinates else input_channels,
                out_channels=num_filters, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.GroupNorm(num_groups=4, num_channels=num_filters),
            act(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]

        in_channels = num_filters
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                stride = 2 if i > 0 and j == 0 else 1
                out_channels = num_filters * 2**i
                layers.append(
                    block_cls(
                        in_channels,
                        out_channels,
                        stride=stride,
                        act=act,
                    )
                )
                in_channels = out_channels

        self.layers = nn.Sequential(*layers)

        assert pooling_method == "avg"

    def forward(self, observations: torch.Tensor):
        # put inputs in [-1, 1]
        x = observations.to(torch.float32) / 127.5 - 1.0

        if self.add_spatial_coordinates:
            x = self.spatial_coordinates(x)

        x = self.layers(x)

        # gobal avg pooling
        x = torch.mean(x, dim=(-2, -1))

        return x


resnetv1_configs = {
    "resnetv1-18": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock
    ),
    "resnetv1-34": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock
    ),
    "resnetv1-50": ft.partial(
        ResNetEncoder, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
    ),
    "resnetv1-18-deeper": ft.partial(
        ResNetEncoder, stage_sizes=(3, 3, 3, 3), block_cls=ResNetBlock
    ),
    "resnetv1-18-deepest": ft.partial(
        ResNetEncoder, stage_sizes=(4, 4, 4, 4), block_cls=ResNetBlock
    ),
    "resnetv1-18-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(2, 2, 2, 2),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-34-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-50-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
    ),
}
