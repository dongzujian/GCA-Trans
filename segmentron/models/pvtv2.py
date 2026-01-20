
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
import math
from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _FCNHead
from ..config import cfg
from .backbones.pvtv2_mix_transformer import Attention, Mlp


__all__ = ['PVTV2']


@MODEL_REGISTRY.register(name='PVTV2')
class PVTV2(SegBaseModel):

    def __init__(self):
        super().__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        elif self.backbone.startswith('resnet18'):
            c1_channels = 64
            c4_channels = 512
        elif self.backbone.startswith('mit'):
            c1_channels = 64
            c4_channels = 512
        elif self.backbone.startswith('resnet34'):
            c1_channels = 64
            c4_channels = 512
        elif self.backbone.startswith('hrnet_w18_small_v1'):
            c1_channels = 16
            c4_channels = 128
        else:
            c1_channels = 256
            c4_channels = 2048

        vit_params = cfg.MODEL.GCA_Trans
        c4_HxW = (cfg.TRAIN.BASE_SIZE // 32) ** 2

        vit_params['decoder_feat_HxW'] = c4_HxW
        vit_params['nclass'] = self.nclass
        vit_params['emb_chans'] = cfg.MODEL.EMB_CHANNELS

        self.fpn = FPN(vit_params)
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['fpn', 'auxlayer'] if self.aux else ['fpn'])


    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        outputs = list()
        x = self.fpn(c1, c2, c3, c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


#-------------------------------FPN-------------------------------#
class FPN(nn.Module):
    def __init__(self, vit_params):
        super().__init__()
        in_channels = [64, 128, 320, 512]
        out_channels = vit_params['emb_chans']
        num_classes = vit_params['nclass']

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.lateral_conv4=nn.Conv2d(in_channels=in_channels[3],out_channels=out_channels,kernel_size=1)
        self.lateral_conv3=nn.Conv2d(in_channels=in_channels[2],out_channels=out_channels,kernel_size=1)
        self.lateral_conv2=nn.Conv2d(in_channels=in_channels[1],out_channels=out_channels,kernel_size=1)
        self.lateral_conv1=nn.Conv2d(in_channels=in_channels[0],out_channels=out_channels,kernel_size=1)

        self.fpn_conv4=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.fpn_conv3=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.fpn_conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.fpn_conv1=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)

        self.fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        m4=self.lateral_conv4(c4)
        m3=self.lateral_conv3(c3)
        m2=self.lateral_conv2(c2)
        m1=self.lateral_conv1(c1)

        shape3=m3.size()[2:]
        m3+=F.interpolate(m4,size=shape3,mode='bilinear', align_corners=True)
        shape2=m2.size()[2:]
        m2+=F.interpolate(m3,size=shape2,mode='bilinear', align_corners=True)
        shape1=m1.size()[2:]
        m1+=F.interpolate(m2,size=shape1,mode='bilinear', align_corners=True)

        p4=self.fpn_conv4(m4)
        p3=self.fpn_conv3(m3)
        p2=self.fpn_conv2(m2)
        p1=self.fpn_conv1(m1)
        outs=[p1,p2,p3,p4]
        
        target_size = m1.size()[2:]
        
        resized_outs = []
        for feat in outs:
            resized_outs.append(
                F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            )
            
        out = torch.cat(resized_outs, dim=1)
        
        out = self.fusion_conv(out)
        
        out = self.pred(out)
        
        return out
#-------------------------------FPN-------------------------------#