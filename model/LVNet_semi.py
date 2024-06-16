import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
from components.attention import ChannelAttention, SpatialAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from components.linear_fusion import HdmProdBilinearFusion
from model.xception import TransferModel
from model.modules import *


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 4096
        self.mid_channel = 512
        self.seg_size = 19
        self.cls_size = 10
        self.channels = [64, 128, 256, 728, 728, 728]
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.relu = nn.ReLU(inplace=True)

        # self.msff = FULL_MSFF(dim=self.seg_size)
        self.score0 = BasicConv2d(self.channels[0], self.mid_channel, kernel_size=1)
        self.score1 = BasicConv2d(self.channels[1], self.mid_channel, kernel_size=1)
        self.score2 = BasicConv2d(self.channels[2], self.mid_channel, kernel_size=1)
        self.score3 = BasicConv2d(self.channels[3], self.mid_channel, kernel_size=1)
        self.score4 = BasicConv2d(self.channels[4], self.mid_channel, kernel_size=1)
        self.score5 = BasicConv2d(self.channels[5], self.mid_channel, kernel_size=1)

        self.msff = MPFF(size=self.seg_size)
        self.HBFusion = HdmProdBilinearFusion(dim1=(64+128+256+728+728), dim2=2048, hidden_dim=2048, output_dim=self.output_dim)

        self.cmc0 = CMCE(in_channel=64)
        self.cmc1 = CMCE(in_channel=128)
        self.cmc2 = CMCE(in_channel=256)

        self.lfe0 = LFGA(in_channel=728)
        self.lfe1 = LFGA(in_channel=728)
        self.lfe2 = LFGA(in_channel=728)

        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.output_dim, 2),
        )
        self.seg_header = nn.Sequential(
            nn.BatchNorm2d(728+64+16+4+1+1),
            nn.ReLU(inplace=True),
            nn.Conv2d(728+64+16+4+1+1, 2, kernel_size=1, bias=False),
        )
        self.pro_header = nn.Sequential(
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            nn.Conv2d(728, 256, kernel_size=1, bias=False),
        )

    def pad_max_pool(self, x):
        b, c, h, w = x.size()
        padding = abs(h % self.cls_size - self.cls_size) % self.cls_size 
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(x.device)
        x = pad(x)
        b, c, h, w = x.size()
        
        max_pool = nn.MaxPool2d(kernel_size=h // self.cls_size, stride=h // self.cls_size, padding=0)
        return max_pool(x)
    
    def get_mask(self, mask):
        b, c, h, w = mask.size() 
        padding = abs(h % self.seg_size - self.seg_size) % self.seg_size 
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(mask.device)
        max_pool = nn.MaxPool2d(kernel_size=h // self.seg_size, stride=h // self.seg_size, padding=0)

        return max_pool(mask)

    def features(self, x):
        srm = self.srm_conv0(x)

        # 64 * 150 * 150
        x0 = self.xception_rgb.model.fea_part1_0(x)
        y0 = self.xception_srm.model.fea_part1_0(srm)
        x0, y0 = self.cmc0(x0, y0)

        # 128 * 75 * 75
        x1 = self.xception_rgb.model.fea_part1_1(x0)
        y1 = self.xception_srm.model.fea_part1_1(y0)
        x1, y1 = self.cmc1(x1, y1)

        # 256 * 38 * 38
        x2 = self.xception_rgb.model.fea_part1_2(x1)
        y2 = self.xception_srm.model.fea_part1_2(y1)
        x2, y2 = self.cmc2(x2, y2)

        # 728 * 19 * 19
        x3 = self.xception_rgb.model.fea_part1_3(x2+y2)
        y3 = self.xception_srm.model.fea_part1_3(x2+y2)
        y3 = self.lfe0(y3, x3)
        # y3 = y3 + self.lsa0(x3)

        # 728 * 19 * 19
        x4 = self.xception_rgb.model.fea_part2_0(x3)
        y4 = self.xception_srm.model.fea_part2_0(y3)
        y4 = self.lfe1(y4, x4)
        # y4 = y4 + self.lsa1(x4)

        # 728 * 19 * 19
        x5 = self.xception_rgb.model.fea_part2_1(x4)
        y5 = self.xception_srm.model.fea_part2_1(y4)
        y5 = self.lfe2(y5, x5)
        # y5 = y5 + self.lsa2(x5)
        
        # 2048 * 10 * 10
        x6 = self.xception_rgb.model.fea_part3(x5)
        y6 = self.xception_srm.model.fea_part3(y5)

        x0u, x1u, x2u, x3u, x4u, x5u = self.score0(x0), self.score1(x1), self.score2(x2), self.score3(x3), self.score4(x4), self.score5(x5)
        x4m = self.msff(x4u, x5u)
        x3m = self.msff(x3u, x5u)
        x2m = self.msff(x2u, x5u)
        x1m = self.msff(x1u, x5u)
        x0m = self.msff(x0u, x5u)
        seg_feas = torch.cat((x0m, x1m, x2m, x3m, x4m, x5), dim=1)


        y0m = self.pad_max_pool(y0)
        y1m = self.pad_max_pool(y1)
        y2m = self.pad_max_pool(y2)
        y3m = self.pad_max_pool(y3)
        y5m = self.pad_max_pool(y5)
        mul_feas = torch.cat((y0m, y1m, y2m, y3m, y5m), dim=1)
        cls_feas = self.HBFusion(mul_feas, y6)

        return cls_feas, seg_feas, x5

    def forward(self, x, mask=None):
        feas = self.features(x)
        cls_preds = self.cls_header(feas[0])
        seg_preds = self.seg_header(feas[1])
        pro_feas = self.pro_header(feas[2])

        if mask is not None:
            if isinstance(mask, list):
                for i in range(len(mask)):
                    mask[i] = self.get_mask(mask[i])
                    mask[i][mask[i] > 0] = 1.0
            else:
                mask = self.get_mask(mask)
                mask[mask > 0] = 1.0

        return cls_preds, seg_preds, pro_feas, mask


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    model = Two_Stream_Net()
    dummy = torch.rand((1, 3, 299, 299))
    mask = torch.rand((1, 1, 299, 299))
    out = model(dummy, mask)
    print(out[0].size(), out[1].size(), out[2].size())
