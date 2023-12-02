import netron
import torch
import torch.nn as nn
import numpy as np



class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        self.level = level
        self.conv0 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(192, 64, (1, 1), (1, 1), (0, 0))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.block = block()
    def forward(self, input):
        out0 = self.lu(self.bn(self.conv0(input)))
        out1 = self.lu(self.bn(self.conv1(out0)))
        out2 = self.lu(self.bn(self.conv2(out1)))
        out3 = torch.cat([out0, out1, out2], 1)
        out = self.lu(self.bn(self.conv3(out3)))

        tem0 = self.block(out)
        tem1 = self.block(tem0)
        tem2 = self.block(tem1)
        # tem2 = self.block(tem2)

        tem3 = out + tem2

        return tem3


class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU()
        self.norm = nn.BatchNorm2d(64)
        self.block1 = block1()
        self.block2 = block2()
        self.block3 = block3()
        self.bag = bag()
        self.ppm = ppm()
    def forward(self, x):
        out0 = self.block1(x)
        out1 = self.block2(x)
        out2 = self.block3(x)
        out00 = self.block1(self.norm(out0 + out1))
        out11 = self.block2(out1)
        out22 = self.block3(self.norm(out1 + out2))
        out000 = out00 + out11
        out111 = self.ppm(out11)
        out222 = out11 + out22
        out = self.bag(out000, out111, out222)

        return out

class block1(nn.Module):
    def __init__(self):
        super(block1, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU()
        self.norm = nn.BatchNorm2d(64)

    def forward(self, x):
        out0 = x
        out1 = self.lu(self.norm(self.conv1(x)))
        out2 = self.up(self.lu(self.norm(self.conv1(self.conv1(self.down(x))))))
        out3 = self.up(self.up(self.lu(self.norm(self.conv1(self.conv1(self.conv1(self.down(self.down(x)))))))))
        out = self.norm(out0 + out1 + out2 + out3)

        return out

class block2(nn.Module):
    def __init__(self):
        super(block2, self).__init__()
        self.basicblock = BasicBlock()

    def forward(self, x):
        return self.basicblock(self.basicblock(self.basicblock(x)))


class block3(nn.Module):
    def __init__(self):
        super(block3, self).__init__()
        self.bottleneck = Bottleneck()

    def forward(self, x):
        return self.bottleneck(self.bottleneck(x))

class bag(nn.Module):
    def __init__(self):
        super(bag, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.norm = nn.BatchNorm2d(64)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input1, input2, input3):
        sig = self.sigmoid(input3)
        return input1 * sig + input2 * (1 - sig)

class ppm(nn.Module):
    def __init__(self):
        super(ppm, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.conv0 = nn.Conv2d(64, 64, (1,1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(128, 64, (1,1), (1, 1), (0, 0))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.lu = nn.ReLU()
        self.norm = nn.BatchNorm2d(64)

    def forward(self, x):
        out0 = self.conv0(x)
        out10 = self.up(self.lu(self.norm(self.avgpool(x))))
        out11 = self.conv0(x)
        out111 = out10 + out11
        out1 = torch.cat([self.conv1(out111), self.conv0(x)], 1)

        out = out0 + self.conv2(out1)
        return out
        



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64 * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 1, (1, 1), (1, 1), (0, 0))

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        return out2

class SRN(nn.Module):
    def __init__(self):
        super(SRN, self).__init__()
        self.Extraction = FeatureExtraction(level=3)
        self.Reconstruction = ImageReconstruction()

    def forward(self, LR):
        tmp = self.Extraction(LR)
        img = self.Reconstruction(tmp)

        return img

# print(SRN())


# #plot the net
# myNet = SRN()
# x = torch.randn(1, 1, 64, 64)
# y = torch.randn(1, 1, 64, 64)
# output = myNet(x)
# torch.onnx.export(myNet, x, "testnet.onnx", opset_version=11)
# netron.start("testnet.onnx")
# print(SRN())
#
# 计算网络参数
# def init(module):
#     if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
#         nn.init.kaiming_normal_(module.weight.data, 0.25)
#         nn.init.constant_(module.bias.data, 0)
# net = SRN()
# net.apply(init)
# print('net total parameters:', sum(param.numel() for param in net.parameters()))


