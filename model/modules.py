import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class MPFF(nn.Module):
    def __init__(self, size=8):
        super(MPFF, self).__init__()
        # fa: b * c * h1 * w1
        # fb: b * c * h2 * h2
        self.size = size

    def forward(self, fa, fb):
        b1, c1, h1, w1 = fa.size()
        b2, c2, h2, w2 = fb.size()
        assert b1 == b2 and c1 == c2 and self.size == h2
        padding = abs(h1 % self.size - self.size) % self.size
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(fa.device)
        fa = pad(fa)
        b1, c1, h1, w1 = fa.size()
        assert h1 % self.size == 0

        window = h1 // self.size
        unfold = nn.Unfold(kernel_size=window, dilation=1, padding=0, stride=window)
        fb = fb.repeat_interleave(window, dim=2)
        fb = fb.repeat_interleave(window, dim=3)
        ff = torch.tanh(fa * fb)
        ff = torch.sum(ff, dim=1, keepdim=True)
        ff = unfold(ff).view(b1, -1, self.size, self.size)

        return ff.to(fa.device)


class CMCE(nn.Module):
    def __init__(self, in_channel=3):
        super(CMCE, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, fa, fb):
        (b1, c1, h1, w1), (b2, c2, h2, w2) = fa.size(), fb.size()
        assert c1 == c2
        cos_sim = F.cosine_similarity(fa, fb, dim=1)
        cos_sim = cos_sim.unsqueeze(1)
        fa = fa + fb * cos_sim
        fb = fb + fa * cos_sim
        fa = self.relu(fa)
        fb = self.relu(fb)

        return fa, fb


class LFGA(nn.Module):
    def __init__(self, in_channel=3, ratio=4):
        super(LFGA, self).__init__()
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.chanel_in = in_channel
        
        self.query_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.chanel_in)


    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        proj_query = self.query_conv(fb).view(
            B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(fb).view(
            B, -1, H*W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        # attention = F.normalize(energy, dim=-1)

        proj_value = self.value_conv(fa).view(
                B, -1, H*W)  # B , C , HW
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + fa

        return self.relu(out)
    


