import torch
from numpy.ma.bench import xs
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from cbam import *
__all__ = ['DEUNeXt']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim, self=None, W=None, H=None):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DEUNeXt(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        # ****************** Encoder 1 ******************
        # Conv layer
        self.encoder11 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder12 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder13 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        # BN layer
        self.ebn11 = nn.BatchNorm2d(16)
        self.ebn12 = nn.BatchNorm2d(32)
        self.ebn13 = nn.BatchNorm2d(128)

        # LN of Tok-MLP block
        self.norm13 = norm_layer(embed_dims[1])
        self.norm14 = norm_layer(embed_dims[2])

        # Tok-MLP Block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.block11 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block12 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # ****************** Encoder 1 ******************

        # ****************** Encoder 2 ******************
        # Conv layer
        self.encoder21 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder22 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder23 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        # BN layer
        self.ebn21 = nn.BatchNorm2d(16)
        self.ebn22 = nn.BatchNorm2d(32)
        self.ebn23 = nn.BatchNorm2d(128)

        # LN of Tok-MLP block
        self.norm23 = norm_layer(embed_dims[1])
        self.norm24 = norm_layer(embed_dims[2])

        # Tok-MLP Block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block21 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block22 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # ****************** Encoder 2 ******************

        # ****************** Skip Connection ******************
        # Conv1*1
        self.skip_conv1 = nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.skip_conv2 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.skip_conv3 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
        self.skip_conv4 = nn.Conv2d(320, 160, 1, stride=1, padding=0)
        self.skip_conv5 = nn.Conv2d(512, 256, 1, stride=1, padding=0)

        # CBAM 
        self.cbam1 = CBAM(16)
        self.cbam2 = CBAM(32)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(160)
        # ****************** Skip Connection ******************

        # ****************** Decoder ******************
        # LN of Tok-MLP block
        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        # Tok-MLP Block
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        # ****************** Decoder ******************
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        # ****************** Encoder1 ******************
        ### Conv Stage
        ### Stage 1
        out1 = F.relu(F.max_pool2d(self.ebn11(self.encoder11(x)),2,2))
        t11 = out1 
        ### Stage 2
        out1 = F.relu(F.max_pool2d(self.ebn12(self.encoder12(out1)),2,2))
        t12 = out1
        ### Stage 3
        out1 = F.relu(F.max_pool2d(self.ebn13(self.encoder13(out1)),2,2))
        t13 = out1
        ### Tokenized MLP Stage
        ### Stage 4
        out1,H,W = self.patch_embed3(out1)
        for i, blk in enumerate(self.block11):
            out1 = blk(out1, H, W)
        out1 = self.norm13(out1)
        out1 = out1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t14 = out1
        ### Bottleneck
        out1 ,H,W= self.patch_embed4(out1)
        for i, blk in enumerate(self.block12):
            out1 = blk(out1, H, W)
        out1 = self.norm14(out1)
        out1 = out1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ****************** Encoder1 ******************


        # ****************** Encoder2 ******************
        # Invert the input tensor by grayscale
        x_inverted = 255 - x
        ### Conv Stage
        ### Stage 1
        out2 = F.relu(F.max_pool2d(self.ebn21(self.encoder21(x_inverted)),2,2))
        t21 = out2
        ### Stage 2
        out2 = F.relu(F.max_pool2d(self.ebn22(self.encoder22(out2)),2,2))
        t22 = out2
        ### Stage 3
        out2 = F.relu(F.max_pool2d(self.ebn23(self.encoder23(out2)),2,2))
        t23 = out2
        ### Tokenized MLP Stage
        ### Stage 4
        out2,H,W = self.patch_embed3(out2)
        for i, blk in enumerate(self.block21):
            out2 = blk(out2, H, W)
        out2 = self.norm23(out2)
        out2 = out2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t24 = out2
        ### Bottleneck
        out2 ,H,W= self.patch_embed4(out2)
        for i, blk in enumerate(self.block22):
            out2 = blk(out2, H, W)
        out2 = self.norm24(out2)
        out2 = out2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ****************** Encoder2 ******************

        # ****************** Concat each stage of the two encoder  ******************
        skip_t1 = self.cbam1(self.skip_conv1(torch.cat((t11, t21), dim = 1)))
        skip_t2 = self.cbam2(self.skip_conv2(torch.cat((t12, t22), dim = 1)))
        skip_t3 = self.cbam3(self.skip_conv3(torch.cat((t13, t23), dim = 1)))
        skip_t4 = self.cbam4(self.skip_conv4(torch.cat((t14, t24), dim = 1)))
        # ****************** Concat each stage of the two encoder  ******************

        ### Decoder
        ### Stage 4
        out = self.skip_conv5(torch.cat((out1, out2), dim = 1))
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,skip_t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,skip_t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,skip_t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,skip_t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)



