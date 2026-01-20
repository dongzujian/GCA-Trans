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
import torch
import warnings

__all__ = ['PVTV2_MCM']


@MODEL_REGISTRY.register(name='PVTV2_MCM')
class PVTV2_MCM(SegBaseModel):

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

        self.MCM_head = MCMHead(vit_params)
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['MCM_head', 'auxlayer'] if self.aux else ['MCM_head'])


    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        outputs = list()
        x = self.MCM_head(c1, c2, c3, c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

#-------------------------------MCM-------------------------------#

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class MCMBlock(nn.Module):
    def __init__(self,in_chans,emb_chans):
        super().__init__()
        out_chans=in_chans // 4
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=1,dilation=1),   #for poolscale=input_size//8
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=2,dilation=2),   #for poolscale=input_size//6
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=4,dilation=4),   #for poolscale=input_size//4
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=8,dilation=8),   #for poolscale=input_size//2
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.pool_conv1=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),             #1
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.pool_conv2=nn.Sequential(
            nn.AdaptiveAvgPool2d(2),             #2
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.pool_conv3=nn.Sequential(
            nn.AdaptiveAvgPool2d(3),             #3
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.pool_conv4=nn.Sequential(
            nn.AdaptiveAvgPool2d(6),             #6
            nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.out_conv=nn.Sequential(
            nn.Conv2d(in_channels=in_chans*2,out_channels=emb_chans,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(emb_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,output_size):
        f1=self.conv1(x)
        f11=self.pool_conv1(x)
        c1_size=output_size
        x_size=x.size()[2:]
        f11_up=resize(f11,size=x_size,mode='bilinear',align_corners=False)
        f1=f1+f11_up

        f2=self.conv2(x)
        f22=self.pool_conv2(x)
        f22_up=resize(f22,size=x_size,mode='bilinear',align_corners=False)
        f2=f2+f22_up

        f3=self.conv3(x)
        f33=self.pool_conv3(x)
        f33_up=resize(f33,size=x_size,mode='bilinear',align_corners=False)
        f3=f3+f33_up

        f4=self.conv4(x)
        f44=self.pool_conv4(x)
        f44_up=resize(f44,size=x_size,mode='bilinear',align_corners=False)
        f4=f4+f44_up

        out = torch.cat([x,f1,f2,f3,f4],dim=1)
        
        out = resize(out,size=c1_size,mode='bilinear',align_corners=False)
        return self.out_conv(out)



#-------------------------------MCM-------------------------------#


class MCMHead(nn.Module):
    def __init__(self, vit_params):
        super().__init__()
        self.emb_chans = vit_params['emb_chans']

        self.head1=MCMBlock(in_chans=64,emb_chans=self.emb_chans)
        self.head2=MCMBlock(in_chans=128,emb_chans=self.emb_chans)
        self.head3=MCMBlock(in_chans=320,emb_chans=self.emb_chans)
        self.head4=MCMBlock(in_chans=512,emb_chans=self.emb_chans)

        self.pred = nn.Conv2d(self.emb_chans, vit_params['nclass'], 1)


    def forward(self, c1, c2, c3, c4):
        size = c1.size()[2:]

        out = self.head4(c4,output_size=size)

        out += self.head3(c3,output_size=size)

        out += self.head2(c2,output_size=size)

        out += self.head1(c1,output_size=size)
        out = self.pred(out)
        return out