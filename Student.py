import math

import numpy as np
import torch
import torchvision
import os
import torch.nn as nn
# from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
import torch.nn.functional as F
from backbone.Shunted.SSA import shunted_t
from config import cfg
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, padding, dilation):
        super(DepthWiseConv, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False,
                                    groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    groups=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class Student(nn.Module):
    def __init__(self, config=cfg):
        super(Student, self).__init__()
        self.backbone = shunted_t()
        load_state_dict = torch.load('/home/xug/PycharmProjects/TLD/pth/ckpt_T.pth')
        self.backbone.load_state_dict(load_state_dict)
        self.cfg = config
        self.csc0 = ASRW(64)
        self.csc1 = ASRW(128)
        self.csc2 = ASRW(256)
        self.csc3 = ASRW(512)
        self.conv1to3 = BasicConv2d(1,3,1,1,0)
        self.merge1 = merge_out(256,512)
        self.merge2 = merge_out(128,256)
        self.merge3 = merge_out(64,128)
        self.conv64_3 = BasicConv2d(64,1,3,1,1)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    def forward(self, x, t):
        t = self.conv1to3(t)
        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.backbone(x)
        x1_t, x2_t, x3_t, x4_t = self.backbone(t)
        csc0 = self.csc0(x1_rgb, x1_t)
        csc1 = self.csc1(x2_rgb, x2_t)
        csc2 = self.csc2(x3_rgb, x3_t)
        csc3 = self.csc3(x4_rgb, x4_t)
        merge_1 = self.merge1(csc2,csc3)
        merge_2 = self.merge2(csc1,merge_1)
        merge_3 = self.merge3(csc0,merge_2)
        out = self.conv64_3(self.up(merge_3))
        tensor_list = [csc3, csc2, csc1, csc0]
        merged_tensor = federated_averaging(tensor_list)
        return out, csc0, csc1, csc2, csc3, merged_tensor
def federated_averaging(tensors):
    shapes = [tensor.shape for tensor in tensors]
    max_shape = max(shapes, key=lambda x: x[2])
    resized_tensors = []
    for tensor in tensors:
        resized_tensor = F.interpolate(tensor, size=max_shape[2:], mode='bilinear', align_corners=True)
        resized_tensors.append(resized_tensor)
    fused_tensor = torch.cat(resized_tensors, dim=1)
    fused_tensor = torch.mean(fused_tensor,dim=1, keepdim=True)
    return fused_tensor
class merge_out(nn.Module):
    def __init__(self, in1,in2):
        super(merge_out, self).__init__()
        self.last_conv1 = BasicConv2d(in2, in1, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = BasicConv2d(in1*2, in1, kernel_size=1, stride=1, padding=0)
        self.last_conv3 = BasicConv2d(in1, in1, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x, y):
        y = self.last_conv1(y)
        y = self.up(y)
        x = torch.cat((x,y),dim=1)
        out = self.last_conv2(x)
        return out
class ASRW(nn.Module):
    def __init__(self, in_):
        super(ASRW, self).__init__()
        self.relu = nn.ReLU(True)
        self.rgb_1 = DepthWiseConv(in_, in_ // 4, 1, 1)
        self.rgb_2 = DepthWiseConv(in_, in_ // 4, 2, 2)
        self.rgb_3 = DepthWiseConv(in_, in_ // 4, 4, 4)
        self.rgb_4 = DepthWiseConv(in_, in_ // 4, 6, 6)
        self.t_1 = DepthWiseConv(in_, in_ // 4, 1, 1)
        self.t_2 = DepthWiseConv(in_, in_ // 4, 2, 2)
        self.t_3 = DepthWiseConv(in_, in_ // 4, 4, 4)
        self.t_4 = DepthWiseConv(in_, in_ // 4, 6, 6)
        self.rf = AdaptiveMatrix(in_ // 4)
        self.cdc = CDC(in_ // 4)
        self.conv = BasicConv2d(in_, in_, 3, 1, 1)
    def forward(self, x_rgb, x_t):
        x1_rgb = self.rgb_1(x_rgb)
        x2_rgb = self.rgb_2(x_rgb)
        x3_rgb = self.rgb_3(x_rgb)
        x4_rgb = self.rgb_4(x_rgb)
        x1_t = self.t_1(x_t)
        x2_t = self.t_2(x_t)
        x3_t = self.t_3(x_t)
        x4_t = self.t_4(x_t)
        rf = self.rf(x1_t, x2_t, x3_t, x4_t)
        out1 = self.cdc(x1_rgb, rf)
        out2 = self.cdc(x2_rgb, rf)
        out3 = self.cdc(x3_rgb, rf)
        out4 = self.cdc(x4_rgb, rf)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.conv(out)
        return out
class CDC(nn.Module):
    def __init__(self, _in):
        super(CDC, self).__init__()
        self.conv3x3 = BasicConv2d(_in, _in, 3, 1, 1)
        self.conv1x1 = BasicConv2d(_in * 2, _in, 1, 1,0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_1, x_2):
        x_2_forward = x_2
        x_2 = self.conv3x3(x_2)
        x_2= self.sigmoid(x_2)
        x = torch.mul(x_1, x_2)
        x = x + x_1
        x = torch.cat([x,x_2_forward], dim=1)
        x = self.conv1x1(x)
        return x



class AdaptiveMatrix(nn.Module):
    def __init__(self, in_chanel):
        super(AdaptiveMatrix, self).__init__()
        self.in_channel = in_chanel
        self.linearW = nn.Linear(in_chanel, in_chanel, bias=False)
    def forward(self, x1, x2, x3, x4):
        size = x1.size()[2:]
        all_dim = size[0] * size[1]
        x1 = x1.view(-1, self.in_channel, all_dim)
        x2 = x2.view(-1, self.in_channel, all_dim)
        x3 = x3.view(-1, self.in_channel, all_dim)
        x4 = x4.view(-1, self.in_channel, all_dim)
        x11 = torch.transpose(x1, 1, 2).contiguous()
        x22 = torch.transpose(x2, 1, 2).contiguous()
        x33 = torch.transpose(x3, 1, 2).contiguous()
        x44 = torch.transpose(x4, 1, 2).contiguous()
        x1_corr = self.linearW(x11)
        x2_corr = self.linearW(x22)
        x3_corr = self.linearW(x33)
        x4_corr = self.linearW(x44)
        x111 = torch.bmm(x1, x1_corr)
        x222 = torch.bmm(x2, x2_corr)
        x333 = torch.bmm(x3, x3_corr)
        x444 = torch.bmm(x4, x4_corr)
        a1 = F.softmax(x111.clone(), dim=2)
        a1 = F.softmax(a1, dim=1)
        x1_out = torch.bmm(a1, x1).contiguous()
        x1_out = x1_out + x1
        a2 = F.softmax(x222.clone(), dim=2)
        a2 = F.softmax(a2, dim=1)
        x2_out = torch.bmm(a2, x2).contiguous()
        x2_out = x2_out + x2
        a3 = F.softmax(x333.clone(), dim=2)
        a3 = F.softmax(a3, dim=1)
        x3_out = torch.bmm(a3, x3).contiguous()
        x3_out = x3_out + x3
        a4 = F.softmax(x444.clone(), dim=2)
        a4 = F.softmax(a4, dim=1)
        x4_out = torch.bmm(a4, x4).contiguous()
        x4_out = x4_out + x4
        x1 = x1_out.view(-1, self.in_channel, size[0], size[1])
        x2 = x2_out.view(-1, self.in_channel, size[0], size[1])
        x3 = x3_out.view(-1, self.in_channel, size[0], size[1])
        x4 = x4_out.view(-1, self.in_channel, size[0], size[1])
        out = x1 + x2 + x3 + x4
        return out
