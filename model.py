# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch

__all__ = [
    "ArbSRRCAN",
    "arbsr_rcan",
]


class ArbSRRCAN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcab: int = 20,
            num_rg: int = 10,
            reduce_channels: int = 16,
            bias: bool = False,
            num_experts: int = 4,
    ) -> None:
        super(ArbSRRCAN, self).__init__()
        self.num_rg = num_rg

        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual Group
        trunk = []
        for _ in range(num_rg):
            trunk.append(_ResidualGroup(channels, reduce_channels, num_rcab))
        self.trunk = nn.Sequential(*trunk)

        # Scale-aware feature adaption block
        scale_aware_adaption = []
        for i in range(num_rg):
            scale_aware_adaption.append(_ScaleAwareFeatureAdaption(channels))
        self.scale_aware_adaption = nn.Sequential(*scale_aware_adaption)

        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Scale-aware upsampling layer
        self.scale_aware_upsample = _ScaleAwareUpsampling(channels, bias, num_experts)

        # Final output layer
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor, w_scale: Tensor, h_scale: Tensor = None) -> Tensor:
        return self._forward_impl(x, w_scale, h_scale)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor, w_scale: Tensor, h_scale: Tensor) -> Tensor:
        out1 = self.conv1(x)

        out = out1
        for i in range(self.num_rg):
            out = self.trunk[i](out)
            out = self.scale_aware_adaption[i](out, w_scale, h_scale)

        out = self.conv2(out)
        out = torch.add(out, out1)
        out = self.scale_aware_upsample(out, w_scale, h_scale)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out


# Copy from `https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR/blob/master/model/arbrcan.py`
def _grid_sample(x: Tensor, offset: Tensor, w_scale: Tensor, h_scale: Tensor) -> Tensor:
    # Generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(w_scale * w)), range(round(h_scale * h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / w_scale - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / h_scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F_torch.grid_sample(x, grid, padding_mode="zeros", align_corners=True)

    return output


class _ScaleAwareConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: str,
            bias: bool = False,
            num_experts: int = 4,
    ) -> None:
        super(_ScaleAwareConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.num_experts = num_experts

        # Use fc layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # Initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, out_channels))
            # Calculate fan_in
            dimensions = self.weight_pool.dim()
            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

            num_input_feature_maps = self.weight_pool.size(1)
            receptive_field_size = 1
            if self.weight_pool.dim() > 2:
                # math.prod is not always available, accumulate the product manually
                # we could use functools.reduce but that is not supported by TorchScript
                for s in self.weight_pool.shape[2:]:
                    receptive_field_size *= s
            fan_in = num_input_feature_maps * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x: Tensor, w_scale: Tensor, h_scale: Tensor) -> Tensor:
        device = x.device

        # Use fc layers to generate routing weights
        w_scale /= torch.ones(1, 1).to(device)
        h_scale /= torch.ones(1, 1).to(device)
        routing_weights = self.routing(torch.cat([h_scale, w_scale], 1)).view(self.num_experts, 1, 1)

        # Fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        out = F_torch.conv2d(x, fused_weight, fused_bias, self.stride, self.padding)

        return out


class _ScaleAwareFeatureAdaption(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ScaleAwareFeatureAdaption, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.AvgPool2d(2),

            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(16, 1, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adaption = _ScaleAwareConv(channels, channels, 3, 1, "same")

    def forward(self, x: Tensor, w_scale: Tensor, h_scale: Tensor) -> Tensor:
        identity = x

        mask = self.mask(x)
        adaption = self.adaption(x, h_scale, w_scale)

        out = torch.mul(adaption, mask)
        out = torch.add(out, identity)

        return out


class _ScaleAwareUpsampling(nn.Module):
    def __init__(
            self,
            channels: int,
            bias: bool = False,
            num_experts: int = 4,
    ) -> None:
        super(_ScaleAwareUpsampling, self).__init__()
        self.channels = channels
        self.bias = bias
        self.num_experts = num_experts

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels // 8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels // 8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # Feature layer
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
        )

        # Offset layer
        self.offset = nn.Conv2d(64, 2, (1, 1), (1, 1), (0, 0))

        # Routing layer
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, w_scale: Tensor, h_scale: Tensor) -> Tensor:
        device = x.device
        batch_size, channels, height, width = x.size()

        # HR coordinates space
        coord_hr = [torch.arange(0, round(height * h_scale), 1).unsqueeze(0).float().to(device),
                    torch.arange(0, round(width * w_scale), 1).unsqueeze(0).float().to(device)]

        # Accord HR coordinates space calculate LR coordinates space
        coord_height = ((coord_hr[0] + 0.5) / h_scale) - (torch.floor((coord_hr[0] + 0.5) / h_scale + 1e-3)) - 0.5
        coord_height = coord_height.permute(1, 0)
        coord_width = ((coord_hr[1] + 0.5) / w_scale) - (torch.floor((coord_hr[1] + 0.5) / w_scale + 1e-3)) - 0.5

        feature_coord = torch.cat((
            torch.ones_like(coord_height).expand([-1, round(w_scale * width)]).unsqueeze(0) / w_scale,
            torch.ones_like(coord_height).expand([-1, round(w_scale * width)]).unsqueeze(0) / h_scale,
            coord_height.expand([-1, round(w_scale * width)]).unsqueeze(0),
            coord_width.expand([round(h_scale * height), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)

        # Prediction filters
        embedding = self.features(feature_coord)
        routing_weights = self.routing(embedding)

        routing_weights = routing_weights.view(self.num_experts, round(h_scale * height) * round(w_scale * width))
        routing_weights = routing_weights.transpose(0, 1)

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1,
                                               round(h_scale * height),
                                               round(w_scale * width),
                                               self.channels // 8,
                                               self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1,
                                           round(h_scale * height),
                                           round(w_scale * width),
                                           self.channels,
                                           self.channels // 8)

        # Prediction offsets
        offset = self.offset(embedding)

        # A k×k neighborhood centered at (L(x) + δx, L(y) + δy) is
        # sampled using bilinear interpolation and convolved with the
        # predicted filters to produce the output features at (x, y).
        # Grid sample
        feature_grid = _grid_sample(x, offset, h_scale, w_scale)
        feature = feature_grid.unsqueeze(-1).permute(0, 2, 3, 1, 4)

        # Spatially-varying filtering
        out = torch.matmul(weight_compress.expand([batch_size, -1, -1, -1, -1]), feature)
        out = torch.matmul(weight_expand.expand([batch_size, -1, -1, -1, -1]), out)
        out = out.squeeze(-1)  # B*H*W*C*1 convert B*H*W*C

        out = out.permute(0, 3, 1, 2)
        out = torch.add(out, feature_grid)

        return out


class _ChannelAttentionLayer(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(_ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.channel_attention_layer(x)

        out = torch.mul(out, identity)

        return out


class _ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(_ResidualChannelAttentionBlock, self).__init__()
        self.residual_channel_attention_block = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            _ChannelAttentionLayer(channel, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_channel_attention_block(x)

        out = torch.add(out, identity)

        return out


class _ResidualGroup(nn.Module):
    def __init__(self, channel: int, reduction: int, num_rcab: int):
        super(_ResidualGroup, self).__init__()
        residual_group = []

        for _ in range(num_rcab):
            residual_group.append(_ResidualChannelAttentionBlock(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)))

        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_group(x)

        out = torch.add(out, identity)

        return out


def arbsr_rcan(**kwargs: Any) -> ArbSRRCAN:
    model = ArbSRRCAN(**kwargs)

    return model
