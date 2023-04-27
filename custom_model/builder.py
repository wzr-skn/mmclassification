# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models import BACKBONES, NECKS, HEADS, LOSSES, CLASSIFIERS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)