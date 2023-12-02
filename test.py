from  __future__ import print_function
import argparse
import os.path
from glob import glob
import imageio
import numpy as np
import torch
from PIL import Image
from scipy import misc
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from evaluation import evaluation
from sklearn import preprocessing
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch jun')

    parser.add_argument('--test_folder', type=str, default='images/mri-spect', help='input image to use')
    # parser.add_argument('--model', type=str, default='./model/model.pth', help='model file to use')
    parser.add_argument('--model', type=str, default='./model/model.pth', help='model file to use')
    parser.add_argument('--cuda', action='store_true', default='true', help='use cuda')

    args = parser.parse_args()
    return args


def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

def main():
    args = parse_args()
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    # images_list0 = glob(args.test_folder + '/MRI*.jpg')
    # images_list1 = glob(args.test_folder + '/SPECT*.jpg')

    name0 = []
    name1 = []
    model = torch.load(args.model)
    index = 0
    if args.cuda:
        model = model.cuda()

    path = './data/MRI-CT/'
    for i in range(1, 31):
        index1 = 'MRI'
        index2 = 'CT'
        path1 = path + str(index1) + str('/') + str(i) + '.png'
        name0.append(path1)
        path2 = path + str(index2) + str('/') + str(i) + '.png'
        name1.append(path2)

    # for T in range(1, 31):
    #     index = 0
    T=30
    for i in range(1, 31):
        img0 = Image.open(name0[index]).convert('L')
        img1 = Image.open(name1[index]).convert('YCbCr')
        y0 = img0
        y1, cb1, cr1 = img1.split()
        LR0 = y0
        LR1 = y1
        LR0 = Variable(ToTensor()(LR0)).view(1, -1, LR0.size[1], LR0.size[0])
        LR1 = Variable(ToTensor()(LR1)).view(1, -1, LR1.size[1], LR1.size[0])
        if args.cuda:
            LR0 = LR0.cuda()
            LR1 = LR1.cuda()
        with torch.no_grad():
            tem0 = model.Extraction(LR0)
            tem1 = model.Extraction(LR1)

            #KD
            k0 = torch.exp((tem0) / T) / (torch.exp((tem0) / T) + torch.exp((tem1) / T))
            k1 = torch.exp((tem1) / T) / (torch.exp((tem0) / T) + torch.exp((tem1) / T))

            tem = tem0 * k0 + tem1 * k1
            tem = model.Reconstruction(tem)
            tem = tem.cpu()
            tem = process(tem, cb1, cr1)
            print(index + 1)

            imageio.imsave('./images/CT' + '\%d.png'%(index + 1), tem)
            index += 1

        # print(f"T={T}")


if __name__ == '__main__':
    main()


