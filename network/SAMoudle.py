import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.mynn import Norm2d, Upsample
from network.PosEmbedding import PosEmbedding1D, PosEncoding1D

scale_factor = 4

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            Norm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            Norm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
# class SqueezeAttentionBlock(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(SqueezeAttentionBlock, self).__init__()
#         print("##SAMoudle is used")
#         self.avg_pool = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)
#         self.conv = conv_block(ch_in, ch_out)
#         self.conv_atten = conv_block(ch_in, ch_out)
#         self.upsample = nn.Upsample(scale_factor=scale_factor)

#     def forward(self, x):
#         # print(x.shape)
#         # 第三层 bc 256 96 96
#         x_res = self.conv(x)
#         # print(x_res.shape)
#         y = self.avg_pool(x)
#         # print(y.shape)
#         y = self.conv_atten(y)
#         # print(y.shape)
#         y = self.upsample(y)
#         # print(y.shape, x_res.shape)
#         return (y * x_res) + y

# 带selfAtt的版本
class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        print("##SAMoudle is used")
        self.avg_pool = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.conv = conv_block(ch_in, ch_out)
        # self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.Att = SASelfAtt()

    def forward(self, x):
        # print(x.shape)
        # 第三层 bc 256 96 96
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.Att(y)
        # print(y.shape)
        y = self.upsample(y)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y

class SASelfAtt(nn.Module):
    def __init__(self):
        super(SASelfAtt, self).__init__()
        self.GetQ = nn.Linear(576, 576)
        self.GetK = nn.Linear(576, 576)
        self.GetV = nn.Linear(576, 576)
        self.softMax = nn.Softmax(2)


    def forward(self, x):
        b_c, c_nums, h, w = x.shape
        x = x.view((b_c * c_nums, -1))
        # print(x.shape)
        # exit()
        
        Q = self.GetQ(x)
        K = self.GetK(x)
        V = self.GetV(x)
        Q = Q.unsqueeze(2)
        V = V.unsqueeze(2)
        K = K.unsqueeze(1)
        W = torch.bmm(Q, K)
        W = self.softMax(W)
        Att = torch.bmm(W, V)
        Att = Att.view((b_c, c_nums, h, w))
        return Att
