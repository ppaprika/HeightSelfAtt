import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.mynn import Norm2d, Upsample
from network.PosEmbedding import PosEmbedding1D, PosEncoding1D


inputSize = 16
outputSize = 16

Debug = False

class HeightSelfAtt(nn.Module):
    def __init__(self, ):
        print("Using HAS moudle.")
        super(HeightSelfAtt, self).__init__()
        self.GetQLinear = nn.Linear(inputSize, outputSize)
        self.GetKLinear = nn.Linear(inputSize, outputSize)
        self.GetVLinear = nn.Linear(inputSize, outputSize)
        self.SoftmaxForW = nn.Softmax(2)

    def forward(self, input):
        tShape = input.shape
        if Debug:
            print("//////////////////////////")
            print(tShape)
        bc = tShape[0]
        ch = tShape[1]
        hcap = tShape[2]
        input = input.view([-1, hcap])
        if Debug:
            print(input.shape)
        Q = self.GetQLinear(input)
        K = self.GetKLinear(input)
        V = self.GetVLinear(input)
        Q = torch.unsqueeze(Q, 2)
        K = torch.unsqueeze(K, 1)
        V = torch.unsqueeze(V, 2)
        W = torch.bmm(Q, K)
        if Debug:
            print(W.shape)
        W = self.SoftmaxForW(W)
        output = torch.bmm(W, V)
        output = output.view([bc, ch, hcap])
        return output



class HeightChannelAtt(nn.Module):
    def __init__(self):
        super(HeightChannelAtt, self).__init__()
        self.LinearForQ = nn.Linear(16, 16)
        self.LinearForK = nn.Linear(16, 16)
        self.LinearForV = nn.Linear(16, 16)
        self.SoftmaxForW = nn.Softmax(2)

    def forward(self, input):
        tShape = input.shape
        # print(tShape)
        bc = tShape[0]
        ch = tShape[1]
        hcap = tShape[2]
        # ori = input
        Q = self.LinearForQ(input)
        K = self.LinearForK(input)
        K = K.transpose(1, 2)
        V = self.LinearForV(input)
        W = torch.bmm(Q, K)
        W = self.SoftmaxForW(W)
        att = torch.bmm(W, V)
        att = torch.mul(att, input)
        return att
        # 可以再来个与原始矩阵的element wise multiply