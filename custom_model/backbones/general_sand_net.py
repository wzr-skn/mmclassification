from mmcv.cnn import ConvModule
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
# from mmdet.ops.build import build_op
from collections import OrderedDict
from ..builder import BACKBONES
from mmcls.models.backbones.base_backbone import BaseBackbone


class SandBottleNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 stride=1,
                 kernel_size=3,
                 dilation=1,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 expansion=4,
                 conv_cfg=dict(type="Conv2d")):
        super().__init__()
        expansion_channels = expansion * mid_channels
        conv_dict = OrderedDict()

        if conv_cfg["type"] in ["RepVGGConv", "DBBConv"]:
            norm_cfg = None

        conv_dict["dw1"] = ConvModule(in_channels, in_channels, stride=stride, kernel_size=kernel_size, groups=in_channels,
                                      act_cfg=act_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
        conv_dict["pw1"] = ConvModule(in_channels, out_channels=mid_channels, kernel_size=1, stride=1,
                                      act_cfg=None, norm_cfg=norm_cfg)
        conv_dict["pw2"] = ConvModule(mid_channels, out_channels=expansion_channels, kernel_size=1, stride=1,
                                      act_cfg=act_cfg, norm_cfg=norm_cfg)
        conv_dict["dw2"] = ConvModule(expansion_channels, expansion_channels, stride=1, kernel_size=kernel_size, groups=expansion_channels,
                                      act_cfg=act_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg)

        self.block = nn.Sequential(conv_dict)
        if in_channels == mid_channels * expansion and stride == 1:
            self.short_cut = nn.Identity()
        else:
            shortcut_dict = OrderedDict()
            shortcut_dict["dw"] = ConvModule(in_channels, in_channels, stride=stride, kernel_size=kernel_size, groups=in_channels,
                                             act_cfg=act_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg)

            shortcut_dict["pw"] = ConvModule(in_channels, expansion_channels, kernel_size=1, stride=1,
                                             act_cfg=act_cfg, norm_cfg=norm_cfg)
            self.short_cut = nn.Sequential(shortcut_dict)

    def forward(self, x):
        return self.short_cut(x) + self.block(x)



class SandStage(nn.Module):
    def __init__(self, in_channel, stage_channel, num_block, kernel_size=3, expansion=4,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 conv_cfg=dict(type="Conv2d")):
        super().__init__()
        LayerDict = OrderedDict()
        for num in range(num_block):
            if num == 0:
                LayerDict["Block{}".format(num)] = SandBottleNeck(in_channel, stage_channel, stride=2,
                                                                  kernel_size=kernel_size, expansion=expansion,
                                                                  act_cfg=act_cfg, norm_cfg=norm_cfg,
                                                                  conv_cfg=conv_cfg
                                                                  )
                continue
            LayerDict["Block{}".format(num)] = SandBottleNeck(stage_channel*expansion, stage_channel, stride=1,
                                                              kernel_size=kernel_size, expansion=expansion,
                                                              act_cfg=act_cfg, norm_cfg=norm_cfg,
                                                              conv_cfg=conv_cfg
                                                              )
        self.stage = nn.Sequential(LayerDict)

    def forward(self, x):
        return self.stage(x)


@BACKBONES.register_module()
class GeneralSandNet(BaseBackbone):
    def __init__(self,
                 num_classes,
                 stem_channels,
                 stage_channels,
                 block_per_stage,
                 in_channels=3,
                 expansion=4,
                 kernel_size=3,
                 num_out=5,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 conv_cfg=dict(type="DBBConv")
                 ):
        super(GeneralSandNet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size for _ in range(len(stage_channels))]
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(stage_channels), \
            "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            kernel_sizes = kernel_size
        if isinstance(expansion, int):
            expansions = [expansion for _ in range(len(stage_channels))]
        if isinstance(expansion, list):
            assert len(expansion) == len(stage_channels), \
            "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            expansions = expansion
        assert num_out <= len(stage_channels), 'num output should be less than stage channels!'

        self.num_classes = num_classes
        self.norm_eval = norm_eval
        self.start_stage = len(stage_channels) - num_out + 1
        self.stage_nums = len(stage_channels)
        self.stages = nn.ModuleList()
        self.stem = ConvModule(in_channels, stem_channels, kernel_size=3, stride=2, padding=1,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)

        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = SandStage(in_channel, stage_channel=stage_channels[num_stages],
                              num_block=block_per_stage[num_stages], kernel_size=kernel_sizes[num_stages],
                              expansion=expansions[num_stages], act_cfg=act_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
            in_channel = stage_channels[num_stages] * expansions[num_stages]
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

    # def init_weights(self):
    #     constant_init(self.classifier[2], 0, 0.5)
    #     x = nn.Linear(self.backbone_channel * 4 * 4, self.backbone_channel)
    #     weight = x.weight.reshape(self.backbone_channel, self.backbone_channel, 4, 4)
    #     bias = x.bias
    #     self.classifier[0].conv.weight = nn.Parameter(weight)
    #     self.classifier[0].conv.bias = nn.Parameter(bias)
    #
    #     x = nn.Linear(self.backbone_channel, self.backbone_channel)
    #     weight = x.weight.reshape(self.backbone_channel, self.backbone_channel, 1, 1)
    #     bias = x.bias
    #     self.classifier[1].conv.weight = nn.Parameter(weight)
    #     self.classifier[1].conv.bias = nn.Parameter(bias)
    #     x = nn.Linear(self.backbone_channel, self.num_classes)
    #     weight = x.weight.reshape(self.num_classes, self.backbone_channel, 1, 1)
    #     bias = x.bias
    #     self.classifier[2].conv.weight = nn.Parameter(weight)
    #     self.classifier[2].conv.bias = nn.Parameter(bias)


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
        return tuple(outs)

    def forward_dummy(self, x):
        outs = []
        x = self.stem(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
        if self.num_classes > 0:
            # x = x.view(x.size(0), -1)
            x = self.classifier(x)
        outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(GeneralSandNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()



