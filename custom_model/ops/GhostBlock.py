import torch.nn as nn
import torch
import math
from mmcv.cnn import ConvModule
from .build import CUSTOM_CONV_OP

@CUSTOM_CONV_OP.register_module()
class GhostBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(GhostBlock, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvModule(inp, init_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=kernel_size//2,
                                       norm_cfg=norm_cfg,
                                       bias=False,
                                       act_cfg=act_cfg)

        self.cheap_operation = ConvModule(init_channels, new_channels,
                                       kernel_size=dw_size,
                                       stride=1,
                                       padding=dw_size//2,
                                       norm_cfg=norm_cfg,
                                       groups=init_channels,
                                       bias=False,
                                       act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

