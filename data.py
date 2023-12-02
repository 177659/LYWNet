from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, RandomCrop, RandomHorizontalFlip

from dataset import DatasetFromFolder

crop_size = 128

def LR_transform(crop_size):
    return Compose([
        Scale(crop_size//4),
        ToTensor(),
    ])

def HR_2_transform(crop_size):
    return Compose([
        Scale(crop_size//2),
        ToTensor(),
    ])

def HR_4_transform(crop_size):
    return Compose([
        RandomCrop((crop_size, crop_size)),
        RandomHorizontalFlip(),
    ])


def get_training_set():
    train_dir = './train/mri-spect/SPECT'

    return DatasetFromFolder(train_dir,
                             LR_transform=LR_transform(crop_size),
                             HR_2_transform=HR_2_transform(crop_size),
                             HR_4_transform=HR_4_transform(crop_size))


def get_test_set():
    test_dir = './train/mri-spect/MRI'

    return DatasetFromFolder(test_dir,
                             LR_transform=LR_transform(crop_size),
                             HR_2_transform=HR_2_transform(crop_size),
                             HR_4_transform=HR_4_transform(crop_size))

