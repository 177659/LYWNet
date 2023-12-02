from __future__ import print_function
import argparse
import os
from math import log10
from os.path import exists, join, basename
from os import makedirs, remove

import numpy as np
import torch
import time
import cv2

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from model import SRN
from data import get_training_set, get_test_set
from mymodel import SRN
from Loss_compute import cosine_loss, edge_loss




# Training settings
parser = argparse.ArgumentParser(description='PyTorch jun')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--weights', type=int, default=[1, 25, 4], help='weights for loss')
parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default='true')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--nIters', type=int, default=1, help='Number of iterations in epoch')
opt = parser.parse_args()
print(opt)

#gpu_device = "cuda:0"

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
test_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model')
model = SRN()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

print(model)
if cuda:
    model = model.cuda()

losses = []
testloss = []
def train(epoch):
    NITERS = opt.nIters
    weights = opt.weights
    avg_loss = 0
    avg_test = 0
    for i in range(NITERS):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            LR = Variable(batch[0])
            if cuda:
                LR = LR.cuda()
            optimizer.zero_grad()
            HR = model(LR)
            # print(LR.shape)
            loss_mse = ((HR - LR)**2).mean()
            loss_cos = cosine_loss(HR, LR)
            loss_ed = edge_loss(HR, LR)
            loss = weights[0] * loss_mse + weights[1] * loss_cos + weights[2] * loss_ed
            avg_loss += loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("===> Epoch {}, Loop{}: Avg. Loss: {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))

        avg_loss += epoch_loss / len(training_data_loader)
        avg_loss /= NITERS
        losses.append(avg_loss)
        epoch_loss = 0
        for batch in testing_data_loader:
            LR = Variable(batch[0])
            if cuda:
                LR = LR.cuda()
            HR = model(LR)
            loss_mse = ((HR - LR) ** 2).mean()
            loss_cos = cosine_loss(HR, LR)
            loss_ed = edge_loss(HR, LR)
            loss = weights[0] * loss_mse + weights[1] * loss_cos + weights[2] * loss_ed
            avg_test += loss
            epoch_loss += loss.item()
        avg_test += epoch_loss / len(testing_data_loader)
        avg_loss /= NITERS
        testloss.append(avg_loss)

lr=opt.lr
psnrs = []
def test():
    avg_psnr1 = 0
    weights = opt.weights
    for batch in testing_data_loader:
        LR = Variable(batch[0])
        if cuda:
            LR = LR.cuda()

        HR = model(LR)
        loss_mse = ((HR - LR) ** 2).mean()
        loss_cos = cosine_loss(HR, LR)
        loss_ed = edge_loss(HR, LR)
        loss = weights[0] * loss_mse + weights[1] * loss_cos + weights[2] * loss_ed
        psnr1 = 10 * log10(1 / loss_mse.item())
        avg_psnr1 += psnr1

    print("===> Avg. PSNR1: {:.4f} dB".format(avg_psnr1 / len(testing_data_loader)))
    return avg_psnr1 / len(testing_data_loader)
total_time = time.time()
num = []
for epoch in range(1, opt.epochs + 1):
    start_time = time.time()
    train(epoch)
    end_time = time.time()
    print(f"each_epoch_time:{end_time - start_time}")
    print(f"total_epoch_time:{end_time - total_time}")
    # test()
    num.append(test())
    if not os.path.exists('./models_spect'):
        os.mkdir('./models_spect')
    # if epoch % 500 == 0:
    #     lr = lr/2
    #     print('new learning rate {}'.format(lr))
    # if epoch % 50 == 0:
    model_out_path = "./model/model.pth".format(epoch)
    torch.save(model, model_out_path)



np.save('losses', torch.tensor(losses).cpu())
np.save('testloss', torch.tensor(testloss).cpu())
print('===> End train')
#np.save('psnrs', psnrs)
