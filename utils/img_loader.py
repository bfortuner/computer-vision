import numpy as np 
from PIL import Image
from skimage import io
import torch


def pil_loader(path):
    return Image.open(path).convert('RGB')


def tensor_loader(path):
    return torch.load(path)


def numpy_loader(path):
    return np.load(path)


def io_loader(path):
    return io.imread(path)
   