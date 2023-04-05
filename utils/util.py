import torch
import numpy as np


class LAB2RGB():
    def __init__(self):
        self.M = np.array([[0.412453, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                           [0.019334, 0.119193, 0.950227]])
        self.Mt = np.linalg.inv(self.M)

    # def BGR2RGB(self, img):  # img为[b,3,H,W]的张量
    #     return torch.stack([img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]], dim=1)
    #
    # def RGB2BGR(self, img):  # img为[b,3,H,W]的张量
    #     return (torch.stack([img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]], dim=1)).cuda()
    #
    # def gamma(self, r):
    #     r2 = r / 12.92
    #     index = r > 0.04045  # pow:0.0031308072830676845,/12.92:0.0031308049535603713
    #     r2[index] = torch.pow((r[index] + 0.055) / 1.055, 2.4)
    #     return r2

    def anti_F(self, X):  # 逆操作。
        tFX = (X - 0.137931) / 7.787
        index = X > 0.206893
        tFX[index] = torch.pow(X[index], 3)
        return tFX

    def anti_g(self, r):
        r2 = r * 12.92
        index = r > 0.0031308072830676845
        r2[index] = torch.pow(r[index], 1.0 / 2.4) * 1.055 - 0.055
        return r2

    def myPSlab2rgb(self, Lab):
        fY = (Lab[:, 0, :, :] + 16.0) / 116.0
        fX = Lab[:, 1, :, :] / 500.0 + fY
        fZ = fY - Lab[:, 2, :, :] / 200.0

        x = self.anti_F(fX)
        y = self.anti_F(fY)
        z = self.anti_F(fZ)
        x = x * 0.964221
        z = z * 0.825211

        r = 3.13405134 * x - 1.61702771 * y - 0.49065221 * z
        g = -0.97876273 * x + 1.91614223 * y + 0.03344963 * z
        b = 0.07194258 * x - 0.22897118 * y + 1.40521831 * z

        r = self.anti_g(r)
        g = self.anti_g(g)
        b = self.anti_g(b)
        return (torch.stack([r, g, b], dim=1).clamp(0.0, 1.0))


def BGR2RGB(img):  # img为[b,3,H,W]的张量
    return torch.stack([img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]], dim=1)


def RGB2BGR(img):  # img为[b,3,H,W]的张量
    return (torch.stack([img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]], dim=1)).cuda()


def get_gray(pic_lab):
    pic_L = pic_lab[:, :, 0]
    pic_A = pic_lab[:, :, 1]
    pic_B = pic_lab[:, :, 2]
    pic_L_reshape = torch.Tensor(pic_L).float().unsqueeze(0).unsqueeze(0)
    pic_A_reshape = torch.Tensor(pic_A).float().unsqueeze(0).unsqueeze(0)
    pic_B_reshape = torch.Tensor(pic_B).float().unsqueeze(0).unsqueeze(0)

    count_128 = torch.full_like(pic_A_reshape, 128.0)
    count_100 = torch.full_like(pic_A_reshape, 100.0)
    count_255 = torch.full_like(pic_A_reshape, 255.0)
    count_1 = torch.full_like(pic_A_reshape, 1.0)

    gray_A = torch.add(pic_A_reshape, count_128)
    gray_B = torch.add(pic_B_reshape, count_128)

    gray_L_mid = torch.div(pic_L_reshape, count_100)
    gray_A_mid = torch.div(gray_A, count_255)
    gray_B_mid = torch.div(gray_B, count_255)

    gray_L_f = torch.sub(count_1, gray_L_mid)
    gray_A_f = torch.sub(count_1, gray_A_mid)
    gray_B_f = torch.sub(count_1, gray_B_mid)

    gray_gsL = gray_L_f.detach().clone().squeeze(0).squeeze(0)
    gray_gsA = gray_A_f.detach().clone().squeeze(0).squeeze(0)
    gray_gsB = gray_B_f.detach().clone().squeeze(0).squeeze(0)

    return gray_gsL, gray_gsA, gray_gsB


def get_newpic(src, dst_size, align_corners=False):
    src_n, src_c, src_h, src_w = src.shape
    dst_n, dst_c, (dst_w, dst_h) = src_n, src_c, dst_size

    """将dst的H和W坐标映射到src的H和W坐标"""
    hd = torch.arange(0, dst_h)
    wd = torch.arange(0, dst_w)
    if align_corners:
        h = float(src_h - 1) / (dst_h - 1) * hd
        w = float(src_w - 1) / (dst_w - 1) * wd
    else:
        h = (float(src_h) / dst_h * (hd + 0.5) - 0.5).cuda()
        w = (float(src_w) / dst_w * (wd + 0.5) - 0.5).cuda()

    h = torch.clamp(h, 0, src_h - 1)  # 防止越界，0相当于上边界padding
    w = torch.clamp(w, 0, src_w - 1)  # 防止越界，0相当于左边界padding

    h = h.view(dst_h, 1)  # 1维dst_h个，变2维dst_h*1个
    w = w.view(1, dst_w)  # 1维dst_w个，变2维1*dst_w个
    h = h.repeat(1, dst_w)  # H方向重复1次，W方向重复dst_w次
    w = w.repeat(dst_h, 1)  # H方向重复dsth次，W方向重复1次

    """求出四点坐标"""
    h0 = torch.clamp(torch.floor(h), 0, src_h - 2).cuda()  # -2相当于下边界padding
    w0 = torch.clamp(torch.floor(w), 0, src_w - 2).cuda()  # -2相当于右边界padding
    h0 = h0.long()  # torch坐标必须是long
    w0 = w0.long()  # torch坐标必须是long

    h1 = h0 + 1
    w1 = w0 + 1

    """求出四点值"""
    q00 = src[..., h0, w0]
    q01 = src[..., h0, w1]
    q10 = src[..., h1, w0]
    q11 = src[..., h1, w1]

    """公式计算"""
    r0 = (w1 - w) * q00 + (w - w0) * q01  # 双线性插值的r0
    r1 = (w1 - w) * q10 + (w - w0) * q11  # 双线性差值的r1
    dst = (h1 - h) * r0 + (h - h0) * r1  # 双线性差值的q

    return dst


def trans_np(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    avg = np.array(avg)
    std = np.array(std)
    return (avg, std)
