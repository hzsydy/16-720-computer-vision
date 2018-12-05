import torch
import torchvision
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from run_q7 import Flatten

trans = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.ToTensor()])

batch_size = 64


def get_dataset_flower17():
    train_dataset = datasets.ImageFolder('../data/oxford-flowers17/train', transform=trans)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder('../data/oxford-flowers17/val', transform=trans)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


pretrained_net = torchvision.models.squeezenet1_1(pretrained=True)
pretrained_net.num_classes = 17
pretrained_net.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, 17, 1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.LogSoftmax(dim=1)
)
for params in pretrained_net.features:
    params.requires_grad = False

print(pretrained_net)

self_net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 256, kernel_size=4),
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(4096, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 17),
    nn.LogSoftmax(dim=1)
)

for layer in self_net:
    try:
        nn.init.kaiming_uniform_(layer.weight)
    except:
        pass

print(self_net)

from run_q7 import main
import run_q7

run_q7.nr_valid = 340
run_q7.nr_train = 680

# main(get_dataset_flower17, pretrained_net, 1e-3, 20)
main(get_dataset_flower17, self_net, 5e-3, 20)
