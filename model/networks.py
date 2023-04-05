# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import LAB2RGB, RGB2BGR

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#raw pic in lab color transfer by predic matrix(sum by parm_six and parm_mt1), pixel color transfer one by one
class color_trans(object):
    def __init__(self, channel = 3):
        self.split_dev = channel
    def trans(self, parm_mt2, pic_ct, mean, std):
        Lab2Rgb = LAB2RGB()
        batch_size, channel, _, _ = pic_ct.size()
        for m in range(batch_size):
            for n in range(channel):
                sd = 3 + n
                t = pic_ct[m, n, :, :]
                #color transfer: t = (t-mean[m, n, :, :])*(parm_mt2[m, sd, :, :]/std[m, n,:,:]) + parm_mt2[m, n, :, :]
                mean1 = torch.sub(t, mean[m, n, :, :])
                std_st = torch.div(parm_mt2[m, sd, :, :], std[m, n, :, :])
                out1 = torch.mul(mean1, std_st)
                out2 = torch.add(out1, parm_mt2[m, n, :, :])
                pic_ct[m, n, :, :] = out2
        pic_ct2 = Lab2Rgb.myPSlab2rgb(pic_ct)
        pic_ct3 = (RGB2BGR(pic_ct2)) / (1. / 255)

        #get pic:b*c*H*W
        return pic_ct3

    def cal_mt(self, parm_six, parm_dev_mt):
        mt1 = torch.unsqueeze(torch.unsqueeze(parm_six, -1), -1)
        parm_final_mt = mt1 + parm_dev_mt
        return parm_final_mt

    def cal_dev_mean(self, parm_six, parm_dev_mt):
        parm_six_mid = torch.unsqueeze(torch.unsqueeze(parm_six, -1), -1)
        parm_six_reshape = parm_six_mid.expand([-1,-1, 256, 256])
        mean, std = torch.split(parm_six_reshape, self.split_dev, dim=1)
        parm_final_mean_mt = mean + parm_dev_mt
        parm_final_mt = torch.cat((parm_final_mean_mt, std), 1)
        return parm_final_mt

    '''
        parm_six_reshape = torch.unsqueeze(torch.unsqueeze(parm_six, -1), -1)
        mean, std = torch.split(parm_six_reshape, self.split_dev, dim=1)
        parm_final_mean_mt = mean + parm_dev_mean_mt
        parm_final_mt = torch.cat((parm_final_mean_mt, std), 1)
        return parm_final_mt
        def cal_mt(self, parm_six, parm_mt1):
        mt1 = torch.unsqueeze(torch.unsqueeze(parm_six, -1), -1)
        parm_mt2 = mt1 + parm_mt1
        count_zero_mt = torch.zeros_like(parm_mt2[:,3:,:,:])
        mid_mt = torch.where(parm_mt2[:,3:,:,:]>0, parm_mt2[:,3:,:,:], count_zero_mt)
        parm_mt2[:, 3:, :, :] = mid_mt
        return parm_mt2
    '''














