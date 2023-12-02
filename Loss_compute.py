import math

import torch
import numpy as np
from math import exp

from torch.autograd import Variable
import torch.nn.functional as F

#计算SSIM
def compute_SSIM(img1, img2, window_size = 11, channel = 1, size_average = True):
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1, 1, shape_, shape_)
        img2 = img2.view(1, 1, shape_, shape_)
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu12

    value = ((2.0 * sigma12 + C2) * (2.0 * mu12 + C1)) / ((sigma1_sq + sigma2_sq + C2) * (mu1_sq + mu2_sq + C1))
    if size_average:
        return value.mean().item()
    else:
        return value.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size.contiguous()))
    return window


#计算PSNR
def compute_PSNR(img1, img2):
    mse = compute_MSE(img1, img2)
    if type(img1) == torch.Tensor:
        return 20 * torch.log10(255 / torch.sqrt(mse)).item()
    else:
        return 20 * np.log10(255 / np.sqrt(mse))

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def cosine_loss(image1, image2):
    # 将图像展平为向量
    image1 = image1.view(image1.size(0), -1)
    image2 = image2.view(image2.size(0), -1)

    # 计算余弦相似度
    similarity = F.cosine_similarity(image1, image2, dim=1)

    # 计算余弦距离（损失为1减去余弦相似度）
    loss = 1 - similarity.mean()

    return loss

def edge_loss(target, synthesized):
    # 使用Sobel滤波器提取目标图像的边缘特征
    target_edge = torch.abs(F.conv2d(target, torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(target.device)))

    # 使用Sobel滤波器提取合成图像的边缘特征
    synthesized_edge = torch.abs(F.conv2d(synthesized, torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(synthesized.device)))

    # 计算均方差损失
    mse_loss = F.mse_loss(target_edge, synthesized_edge)

    return mse_loss