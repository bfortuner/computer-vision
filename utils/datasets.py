import os
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll
import torch
import torchvision
from torchvision import transforms
import glob
import random

import config as cfg
import utils.metadata as meta
from . import csv_loader
from . import img_loader

# Datasets
# pytorch.org/docs/master/torchvision/datasets.html
# https://github.com/bfortuner/pytorch-cheatsheet/blob/master/pytorch-cheatsheet.ipynb


def get_iris_data():
    fpath = "../data/iris.csv"
    url = "https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv"
    df = csv_loader.load_or_download_df(fpath, url)
    return df


def get_sin_data():
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))
    return X,y


def get_housing_data():
    # https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
    fpath = "../data/housing.csv"
    url = "https://raw.githubusercontent.com/ggallo/boston-housing/master/housing.csv"
    df = csv_loader.load_or_download_df(fpath, url)
    return df


def get_advertising_data():
    fpath = "../data/advertising.csv"
    url = "http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv"
    df = csv_loader.load_or_download_df(fpath, url)
    df = df.drop(df.columns[0], axis=1)
    return df


def get_swiss_roll_data(n_samples=1000):
    noise = 0.2
    X, _ = make_swiss_roll(n_samples, noise)
    X = X.astype('float32')[:, [0, 2]]
    return X, _


def get_swiss_roll_loader(n_samples=1000):
    X, _ = get_swiss_roll_data(n_samples)
    dataset = torch.utils.data.dataset.TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(_))
    loader = torch.utils.data.dataloader.DataLoader(
        dataset, batch_size=100, shuffle=True)
    return loader


def get_mnist_loader():
    MNIST_MEAN = np.array([0.1307,])
    MNIST_STD = np.array([0.3081,])
    normTransform = transforms.Normalize(MNIST_MEAN, MNIST_STD)

    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                            download=True, transform=trainTransform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                        download=True, transform=testTransform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


def get_cifar_loader():
    # https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
    CIFAR_MEAN = np.array([0.49139968, 0.48215827, 0.44653124])
    CIFAR_STD = np.array([0.24703233, 0.24348505, 0.26158768])
    normTransform = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=trainTransform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=testTransform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def get_catsdogs_loader(imgs_dir):
    # Need to download Kaggle cats/dogs competition
    # And move ALL images into single directory
    classes = ['cat','dog']
    class_to_idx, idx_to_class = meta.get_key_int_maps(classes)

    def get_targs_from_fpaths(fpaths):
        targs = []
        for fpath in fpaths:
            classname = fpath.split('/')[-1].split('.')[0]
            # For one-hot sigmoid
            #targ = meta.onehot_encode_class(
            #    class_to_idx, classname)
            targs.append(class_to_idx[classname])
        return targs

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    trainTransform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    testTransform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    fpaths = glob.glob(imgs_dir + '*.jpg')
    random.shuffle(fpaths)
    trn_fpaths = fpaths[:20000]
    val_fpaths = fpaths[20000:]

    trn_targs = get_targs_from_fpaths(trn_fpaths)
    val_targs = get_targs_from_fpaths(val_fpaths)

    img_reader = 'pil'
    trn_dataset = FileDataset(
        trn_fpaths, img_reader, trn_targs, trainTransform)
    val_dataset = FileDataset(
        val_fpaths, img_reader, val_targs, testTransform)

    trn_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=64,
        shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64,
        shuffle=False, num_workers=2)

    return trn_loader, val_loader, classes


loaders = {
    'pil': img_loader.pil_loader,
    'tns': img_loader.tensor_loader,
    'npy': img_loader.numpy_loader,
    'io': img_loader.io_loader
}


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, fpaths,
                 img_loader='pil',
                 targets=None,
                 transform=None,
                 target_transform=None):
        self.fpaths = fpaths
        self.loader = self._get_loader(img_loader)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def _get_loader(self, loader_type):
        return loaders[loader_type]

    def _get_target(self, index):
        if self.targets is None:
            return 1
        target = self.targets[index]
        if self.target_transform is not None:
            return self.target_transform(target)
        return int(target)

    def _get_input(self, index):
        img_path = self.fpaths[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        img_path = self.fpaths[index]
        return input_, target, img_path

    def __len__(self):
        return len(self.fpaths)
