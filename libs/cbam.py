# 这里引入的是 CBAM 模块
# 2023-09-18 这个模块是直接从 CBAM 那个文章中移植过来的

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, bs, c, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),  # 张量维度先缩减再增加
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw # maxpool 和 avgpool并行处理，因此结果加和

        # 这里计算出的scale是表示channel attention的一维向量
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

        # 返回的是进行了通道注意以后的结果
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        # 这里我不太清楚他为什么是对于维度1进行max操作，按理说是维度2？
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, bs, c):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        # 这里是两个平面输入，实际上就是maxpool和avgpool的平面，输出一个层，进行激活后得到空间注意(spatial attention)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

        self.bs = bs
        self.c = c

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting

        # 这里则是进行了空间注意的结果
        return x * scale

class CBAM(nn.Module):
    def __init__(self, bs, c, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):

        """

        Args:
            gate_channels: 时间域的通道数量
            reduction_ratio: 指的是对时间域进行注意力机制的时候要经过一个MLP的结构，
                                这个结构是先减空间维度再增加空间维度，这里就是减多少空间维度
            pool_types: 池化种类，指的是在空间注意力机制中的部分
            no_spatial: 是否不加入空间注意力机制

        但是这里需要改一下，因为实际上输入进来的数据 x 是5维的数据，即z_vol:  (float tensor, (bs, c, d, h, w)): feature volumes.
        bs是batch size，c是color，剩下三个才是我们需要关心的东西

        那如果是这样的话，直接把多余的维度先合成起来就行了
        """
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(bs, c,gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        self.bs = bs    # batch size
        self.c = c      # color
        if not no_spatial:
            self.SpatialGate = SpatialGate(bs, c)
    def forward(self, x):
        assert 5 == x.dim() #这里是针对我们的数据进行的修正
        x = x.view(-1,x.shape[-3],x.shape[-2],x.shape[-1])
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        x_out = x_out.view(self.bs, self.c, x_out.shape[-3], x_out.shape[-2], x_out.shape[-1])
        return x_out
