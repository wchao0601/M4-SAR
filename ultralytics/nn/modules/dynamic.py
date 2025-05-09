import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from einops.layers.torch import Rearrange
from einops import rearrange
from matplotlib import pyplot as plt
from timm.models.layers import trunc_normal_
from torch.nn import Parameter
# from models.dynamic_conv import DynamicConv, DynamicConv1
# from models.spatial_transformer import FFM, CEM

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        res= self.act(self.bn(self.conv(x)))
        return res

    def fuseforward(self, x):
        res = self.act(self.conv(x))

        return res
    
class MAM2(nn.Module):
    def __init__(self, in_channel):
        super(MAM2, self).__init__()
        self.channel264 = nn.Sequential(
            Conv(in_channel*2, 128, 3, 2, 1),
            Conv(128, 128, 1, 1, 1),
            convblock(128, 64, 3, 2, 1),
            convblock(64, 64, 1, 1, 0),
            convblock(64, 32, 3, 2, 1),
            convblock(32, 32, 1, 1, 0),
        )
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 2, 1, 1, 0)
        )
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 1, 1, 1, 0)
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 1, 1, 1, 0)
        )
        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.scale1[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale1[-1].bias.data.zero_()
        self.scale2[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale2[-1].bias.data.zero_()
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        in_ = torch.cat([gr, gt], dim=1)
        # in_ = gt -gr
        n1 = self.channel264(in_)
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        if in_.is_cuda:
            identity_theta = identity_theta.cuda().detach()
        shift_xy = self.xy(n1)
        shift_s1 = self.scale1(n1)
        shift_s2 = self.scale2(n1)
        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1)
        identity_theta[:, :, 2] += shift_xy.squeeze()
        identity_theta[:, :1, :1] += shift_s1.squeeze(2)
        identity_theta[:, 1, 1] += shift_s2.squeeze()
        # identity_theta = identity_theta.half()
        identity_theta = identity_theta
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1,2)
        dtype = gt.dtype
        if gr.dtype != wrap_grid.dtype:
            gr = gr.type(torch.float)
        wrap_gr = F.grid_sample(gr, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',align_corners=True)
        wrap_gr = wrap_gr.type(dtype)
        fuse = self.fus1(torch.cat([wrap_gr, gt], dim=1))
        return fuse

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        # inner_dim = 682
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # b, 65, 1024, heads = 8
        b, n, _, h = *x.shape, self.heads

        # self.to_qkv(x): b, 65, 64*8*3
        # qkv: b, 65, 64*8
        # x = x.unsqueeze(dim=2)       682

        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)

        # b, 65, 64, 8
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q_s, q_t = torch.chunk(q, 2, 2)
        k_s, k_t = torch.chunk(k, 2, 2)
        v_s, v_t = torch.chunk(v, 2, 2)
        #
        # dots:b, 65, 64, 64
        # dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # mask_value = -torch.finfo(dots.dtype).max
        # dots_t = torch.einsum('bhid,bhjd->bhij', q_t, k_s) * self.scale
        dots_t = torch.einsum('bhid,bhjd->bhij', q_t, k_t) * self.scale
        dots_s = torch.einsum('bhid,bhjd->bhij', q_s, k_t) * self.scale
        # dots_s = torch.einsum('bhid,bhjd->bhij', q_s, k_s) * self.scale
        # mask_value = -torch.finfo(dots.dtype).max
        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value=True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, mask_value)
        #     del mask
        #
        # attn:b, 65, 64, 64
        # attn = dots.softmax(dim=-1)
        attn_s = dots_s.softmax(dim=-1)
        attn_t = dots_t.softmax(dim=-1)

        # 使用einsum表示矩阵乘法：
        # out:b, 65, 64, 8
        # out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out_s = torch.einsum('bhij,bhjd->bhid', attn_s, v_s)
        out_t = torch.einsum('bhij,bhjd->bhid', attn_t, v_t)
        out = torch.cat([out_s, out_t], dim=2)
        # out:b, 64, 65*8
        out = rearrange(out, 'b h n d -> b n (h d)')

        # out:b, 64, 1024
        out = self.to_out(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout): #1,1,1,32,16,0.1
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
    
class CEM1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.transformer = Transformer(dim=1, depth=1, heads=1, dim_head=16, mlp_dim=8, dropout=0.1)
        self.attention_weight = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.pos_embedding = nn.Parameter(torch.randn(1, channels*2, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, rgb,t):
        x = torch.cat([rgb,t], dim=1)
        b, c, h, w = rgb.size()  # 32, 256, 72, 36
        input = self.gap(x).squeeze(-1)  # 32， 256， 72*36=2592
        _, c, _ = input.shape
        input = input + self.pos_embedding[:, :(c)]
        input = self.dropout(input)
        output = self.transformer(input)  # 32, 256, 1
        output = torch.unsqueeze(output, dim=3)  # 32, 256, 1, 1
        weight = torch.sigmoid(output)  # 32, 256, 1, 1
        final = (weight * x).view(b,2,c//2,h,w)
        rgb_ = final[:,0,:,:,:]
        t_ = final[:,1,:,:,:]
        # fuse = rgb_+t_
        return rgb_, t_
            
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# Channel-wise Correlation
# class CCorrM(nn.Module):
#     def __init__(self, all_channel):
#         super(CCorrM, self).__init__()
#         self.linear_e = nn.Linear(all_channel, all_channel, bias=False) #weight
#         self.channel = all_channel
#         self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
#         self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)
#
#     def forward(self,x):  # exemplar: f1, query: f2
#         query = x[0]
#         exemplar = x[1]
#         fea_size = query.size()[2:]
#         exemplar = F.interpolate(exemplar, size=fea_size, mode="bilinear", align_corners=True)
#         all_dim = fea_size[0] * fea_size[1]
#         exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C1,H,W -> N,C1,H*W
#         query_flat = query.view(-1, self.channel, all_dim)  # N,C2,H,W -> N,C2,H*W
#         exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batchsize x dim x num, N,H*W,C1
#         exemplar_corr = self.linear_e(exemplar_t)  # batchsize x dim x num, N,H*W,C1
#         A = torch.bmm(query_flat, exemplar_corr)  # ChannelCorrelation: N,C2,H*W x N,H*W,C1 = N,C2,C1
#
#         A1 = F.softmax(A.clone(), dim=2)  # N,C2,C1. dim=2 is row-wise norm. Sr
#         # B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc
#         query_att = torch.bmm(A1, exemplar_flat).contiguous()  # N,C2,C1 X N,C1,H*W = N,C2,H*W
#         # exemplar_att = torch.bmm(B, query_flat).contiguous()  # N,C1,C2 X N,C2,H*W = N,C1,H*W
#
#         # exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C1,H*W -> N,C1,H,W
#         # exemplar_out = self.conv1(exemplar_att + exemplar)
#
#         query_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C2,H*W -> N,C2,H,W
#         query_out = self.conv1(query_att + query)
#         out = query_out + exemplar
#         return out


# Edge-based Enhancement Unit (EEU)
class EEU(nn.Module):
    def __init__(self, in_channel):
        super(EEU, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # edge = self.PReLU(edge)
        out = weight * x + x
        return out


# Edge Self-Alignment Module (ESAM)
class ESAM(nn.Module):
    def __init__(self, in_channel):
        super(ESAM, self).__init__()
        self.eeu = EEU(in_channel)
    def forward(self, t):  # x1 16*144*14; x2 24*72*72
        t_2 = self.eeu(t)
        return t_2  # (24*2)*144*144


# class Fuse(nn.Module):
#     def __init__(self, in_channel):
#         super(Fuse, self).__init__()
#         self.esam = ESAM(in_channel)
#         self.DSMM = DSMM(in_channel)
#         self.mam  = MAM2(in_channel)
#     def forward(self,x):
#         rgb = x[0]
#         t = x[1]
#         t1 = self.esam(t)
#         rgb1 = self.DSMM(rgb)
#         x = [rgb1,t1]
#         final = self.mam(x)
#         return final


class CMA(nn.Module):
    def __init__(self, in_channel):
        super(CMA, self).__init__()
        # self.esam = ESAM(in_channel)
        # self.DSMM = DSMM1(in_channel)
        self.mam = MAM2(in_channel)
        self.fuse = FRM(in_channel)
        # self.dy = DynamicConv1(in_channel, in_channel, 3, 1, 1)
        # self.fus1 = Conv(in_channel * 2, in_channel, 1, 1, 0)
    def forward(self,x):
        rgb = x[0]
        t = x[1]
        # t1 = self.esam(t)
        # rgb1 = self.DSMM(rgb,t)
        x = [rgb, t]
        # x = [t, rgb]
        # x = [t, rgb]
        gr = self.mam(x)
        # final = gr+t
        # map_rgb = torch.unsqueeze(torch.mean(final, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(128, 128), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("2.png")
        # gt = self.mam(x)
        # dy_rgb = self.dy(gr,t)
        final = self.fuse(gr, t)
        # final = gr + t
        # final = self.fuse(rgb, gt)
        # fuse = self.fus1(torch.cat([gr, t], dim=1))
        # final = gr+t
        # map_rgb = torch.unsqueeze(torch.mean(final, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(80, 80), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("1.png")
        return final


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        # x = torch.cat((x1, x2), dim=1)
        # avg1 = self.avg_pool(x1).view(B, self.dim)
        avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
        avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
        # avg2 = self.avg_pool(x2).view(B, self.dim)
        max1 = self.max_pool(x1).view(B, self.dim)
        max2 = self.max_pool(x2).view(B, self.dim)
        avg = avg1+avg2
        max = max1+max2
        y = torch.cat((max, avg), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FRM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

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

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        x1 = x1 + self.lambda_c * channel_weights[0] * x1
        # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        x2 = x2 + self.lambda_c * channel_weights[1] * x2
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_s * spatial_weights[0] * x1
        out_x2 = x2 + self.lambda_s * spatial_weights[1] * x2
        out = out_x1 + out_x2
        return out


if __name__ == "__main__":
    x = torch.rand(4,64,80,80).half().cuda()
    y = torch.rand(4,64,80,80).half().cuda()
    m = [x,y]
    fuse = Fuse1(64).half().cuda()
    out = fuse(m)
    print(out.shape)