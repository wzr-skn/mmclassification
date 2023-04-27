import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from ..builder import BACKBONES
import os
from collections import OrderedDict
import custom_model.ops
from ..ops.build import build_op
from mmcls.models.backbones.base_backbone import BaseBackbone

class RepVGGStage(nn.Module):
    def __init__(self, in_ch, stage_ch, num_block, kernel_size=3, groups=1, act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'), conv_cfg=dict(type="DBBConv")):
        super(RepVGGStage, self).__init__()
        LayerDict = OrderedDict()

        for num in range(num_block):
            if num == 0:
                LayerDict["Block{}".format(num)] = ConvModule(in_ch, in_ch, groups=in_ch, kernel_size=kernel_size, stride=2,
                                                              act_cfg=act_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
                LayerDict["Block{}".format(num+1)] = ConvModule(in_ch, stage_ch,  kernel_size=1, stride=1,
                                                              act_cfg=act_cfg, norm_cfg=norm_cfg)
                continue
            LayerDict["Block{}".format(num+1)] = ConvModule(stage_ch, stage_ch, groups=stage_ch, kernel_size=kernel_size, stride=1,
                                                            act_cfg=act_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
        self.Block = nn.Sequential(LayerDict)

    def forward(self, x):
        return self.Block(x)



@BACKBONES.register_module()
class Sep_RepVGGNet(BaseBackbone):
    def __init__(self,
                 num_classes,
                 stem_channels,
                 stage_channels,
                 block_per_stage,
                 kernel_size=3,
                 num_out=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 conv_cfg=dict(type="DBBConv")
                 ):
        super(Sep_RepVGGNet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size for _ in range(len(stage_channels))]
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(stage_channels), \
            "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            kernel_sizes = kernel_size

        self.num_classes = num_classes
        self.norm_eval = norm_eval
        assert num_out <= len(stage_channels), 'num output should be less than stage channels!'

        self.stage_nums = len(stage_channels)
        self.stem = ConvModule(3, stem_channels, kernel_size=3, stride=2, padding=1,
                               norm_cfg=norm_cfg)
        '''defult end_stage is the last stage'''
        self.start_stage = len(stage_channels)-num_out+1

        self.stages = nn.ModuleList()
        self.last_stage = len(stage_channels)
        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = RepVGGStage(in_channel, stage_channels[num_stages],
                                            block_per_stage[num_stages],
                                            kernel_size=kernel_sizes[num_stages],
                                            groups=1,
                                            conv_cfg=conv_cfg)
            in_channel = stage_channels[num_stages]
            self.stages.append(stage)

        self.backbone_channel = in_channel
        if self.num_classes > 0:
            layers = []
            layers.extend([
                ConvModule(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=4,
                    stride=1),
                ConvModule(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=1,
                    stride=1),
                ConvModule(
                    in_channels=in_channel,
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    act_cfg=None)
            ])
            self.classifier = nn.Sequential(*layers)

        # if self.num_classes > 0:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(128 * 4 * 4, 1024),
        #         nn.ReLU(True),
        #         nn.Dropout(),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(True),
        #         nn.Dropout(),
        #         nn.Linear(1024, num_classes),
        #     )

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         import torch
    #         assert os.path.isfile(pretrained), "file {} not found.".format(pretrained)
    #         self.load_state_dict(torch.load(pretrained), strict=False)
    #     elif pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 kaiming_init(m)
    #             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #                 constant_init(m, 1)
    #     else:
    #         raise TypeError('pretrained must be a str or None')


    def init_weights(self):
        constant_init(self.classifier[2], 0, 0.5)
        x = nn.Linear(self.backbone_channel * 4 * 4, self.backbone_channel)
        weight = x.weight.reshape(self.backbone_channel, self.backbone_channel, 4, 4)
        bias = x.bias
        self.classifier[0].conv.weight = nn.Parameter(weight)
        self.classifier[0].conv.bias = nn.Parameter(bias)

        x = nn.Linear(self.backbone_channel, self.backbone_channel)
        weight = x.weight.reshape(self.backbone_channel, self.backbone_channel, 1, 1)
        bias = x.bias
        self.classifier[1].conv.weight = nn.Parameter(weight)
        self.classifier[1].conv.bias = nn.Parameter(bias)
        x = nn.Linear(self.backbone_channel, self.num_classes)
        weight = x.weight.reshape(self.num_classes, self.backbone_channel, 1, 1)
        bias = x.bias
        self.classifier[2].conv.weight = nn.Parameter(weight)
        self.classifier[2].conv.bias = nn.Parameter(bias)


    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
        if self.num_classes > 0:
            # x = x.view(x.size(0), -1)
            x = self.classifier(x)
        x = x.view(x.size(0),-1)
        outs.append(x)
        return  tuple(outs)

    def train(self, mode=True):
        super(Sep_RepVGGNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

