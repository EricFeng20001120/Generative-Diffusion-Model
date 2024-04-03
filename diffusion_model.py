import torch
from torch import nn
import math
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, mid_channels=None, use_groupnorm=True, num_groups=1, use_relu=False):
        super(Block, self).__init__()


        if not mid_channels:
          mid_channels = out_ch

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1, bias = False),
            nn.ReLU() if use_relu else nn.GELU(),
            nn.GroupNorm(num_groups, mid_channels) if use_groupnorm else nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_ch, kernel_size=3, padding=1, bias = False),
            nn.ReLU() if use_relu else nn.GELU(),
            nn.GroupNorm(num_groups, out_ch) if use_groupnorm else nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=256, use_groupnorm = False, use_relu = False):
        super().__init__()
        self.down_conv_block = nn.Sequential(
            Block(in_ch, out_ch,use_groupnorm = use_groupnorm,use_relu = use_relu),
            nn.MaxPool2d(2),
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t):
      x = self.down_conv_block(x)
      time_emb = self.act(self.time_mlp(t))
      time_emb = time_emb.view(-1, time_emb.size(1), 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
      return x + time_emb

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=256,use_groupnorm = False, use_relu = False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv_block = nn.Sequential(
            Block(in_ch, out_ch, in_ch//2,use_groupnorm = use_groupnorm,use_relu = use_relu),
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, skip, t):
      #print(x.shape, skip.shape)
      x = self.up(x)
      x = torch.cat([skip,x], dim=1)
      x = self.up_conv_block(x)
      #print(x.size(),skip.size())
      time_emb = self.act(self.time_mlp(t))
      time_emb = time_emb.view(-1, time_emb.size(1), 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
      #print(x.size(),time_emb.size())
      return x + time_emb

class SelfAttention(nn.Module):
    def __init__(self, channel, size):
        super(SelfAttention, self).__init__()
        self.channel = channel
        self.size = size
        self.mha = nn.MultiheadAttention(channel, 4, batch_first=True)
        self.ln = nn.LayerNorm([channel])
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.LayerNorm([channel]),
            nn.Linear(channel, channel),
        )

    def forward(self, x):
      x = x.view(-1, self.channel, self.size * self.size).swapaxes(1, 2)
      x_ln = self.ln(x)
      x_attn, _ = self.mha(x_ln, x_ln, x_ln)
      x_attn = x_attn + x
      x_attn = self.mlp(x_attn)+ x_attn
      return x_attn.swapaxes(2, 1).view(-1, self.channel, self.size, self.size)



class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels = 3 , time_emb_dim=256, use_groupnorm=False, use_relu = False, device = "cuda"):
        super().__init__()
        self.device = device
        self.time_emb_dim = time_emb_dim
        self.conv0 = Block(in_channels, 64, use_groupnorm=use_groupnorm, use_relu = use_relu)
        self.down1 = Down(64,128, use_groupnorm=use_groupnorm, use_relu = use_relu)
        self.att1 = SelfAttention(128,32)
        self.down2 = Down(128,256, use_groupnorm=use_groupnorm, use_relu = use_relu)
        self.att2 = SelfAttention(256,16)
        self.down3 = Down(256,256, use_groupnorm=use_groupnorm, use_relu = use_relu)
        self.att3 = SelfAttention(256,8)

        self.mid1 = Block(256, 512, use_groupnorm=use_groupnorm, use_relu = use_relu)
        self.mid2 = Block(512, 512, use_groupnorm=use_groupnorm, use_relu = use_relu)
        self.mid3 = Block(512, 256, use_groupnorm=use_groupnorm, use_relu = use_relu)

        self.up1 = Up(512,128, use_groupnorm=use_groupnorm,use_relu = use_relu)
        self.att4 = SelfAttention(128,16)
        self.up2 = Up(256,64, use_groupnorm=use_groupnorm,use_relu = use_relu)
        self.att5 = SelfAttention(64,32)
        self.up3 = Up(128,64, use_groupnorm=use_groupnorm,use_relu = use_relu)
        self.att6 = SelfAttention(64,64)

        self.final_conv = nn.Conv2d(64,3, kernel_size=1)

    def SinusoidalPositionEmbeddings(self,time,channels):
      device = time.device
      half_dim = channels // 2
      emb = math.log(10000) / (half_dim - 1)
      emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
      emb = time[:, None] * emb[None, :]
      return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, x, timestep):
        #print('timestep:',timestep)
        t = self.SinusoidalPositionEmbeddings(timestep,self.time_emb_dim)
        #print('t:',t)
        x1 = self.conv0(x)
        #print(x1.shape)
        x2 = self.down1(x1, t)
        x2 = self.att1(x2)
        x3 = self.down2(x2, t)
        x3 = self.att2(x3)
        x4 = self.down3(x3, t)
        x4 = self.att3(x4)

        x4 = self.mid1(x4)
        x4 = self.mid2(x4)
        x4 = self.mid3(x4)

        x = self.up1(x4, x3, t)
        x = self.att4(x)
        x = self.up2(x, x2, t)
        x = self.att5(x)
        x = self.up3(x, x1, t)
        x = self.att6(x)

        return self.final_conv(x)
