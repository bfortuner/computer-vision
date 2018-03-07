import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from . import files
import torchvision


def plot_tensor(img, fs=(10,10), title=""):
    if len(img.size()) == 4:
        img = img.squeeze(dim=0)
    npimg = img.numpy()
    plt.figure(figsize=fs)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.show()

def plot_batch(samples, title="", fs=(10,10)):
    plot_tensor(torchvision.utils.make_grid(samples), fs=fs, title=title)

def plot_metric(trn, tst, title):
    plt.plot(np.stack([trn, tst], 1));
    plt.title(title)
    plt.show()

def load_img_as_arr(img_path):
    return plt.imread(img_path)


def load_img_as_pil(img_path):
    return Image.open(img_path).convert('RGB')


def norm_meanstd(arr, mean, std):
    return (arr - mean) / std


def denorm_meanstd(arr, mean, std):
    return (arr * std) + mean


def norm255_tensor(arr):
    """Given a color image/where max pixel value in each channel is 255
    returns normalized tensor or array with all values between 0 and 1"""
    return arr / 255.


def denorm255_tensor(arr):
    return arr * 255.


def plot_arr(arr, fs=(6,6), title=None):
    if len(arr.shape) == 2:
        plot_gray_arr(arr, fs, title)
    else:
        plot_img_arr(arr, fs, title)


def plot_img_arr(arr, fs=(6,6), title=None):
    plt.figure(figsize=fs)
    plt.imshow(arr.astype('uint8'))
    plt.title(title)
    plt.show()


def plot_gray_arr(arr, fs=(6,6), title=None):
    plt.figure(figsize=fs)
    plt.imshow(arr.astype('float32'), cmap='gray')
    plt.title(title)
    plt.show()


def plot_imgs(imgs, titles=None, dim=(4,4), fs=(6,6)):
    plt.figure(figsize=fs)
    for i,img in enumerate(imgs[:dim[0]*dim[1]]):
        # Tensor
        if type(img) is not np.ndarray:
            img = img.numpy().transpose((0,2,3,1))

        plt.subplot(*dim, i+1)
        if len(img.shape) == 2:
            plt.imshow(img.astype('float32'), cmap='gray')
        else:
            plt.imshow(img.astype('uint8'))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()


def plot_rgb_samples(arr, dim=(4,4), figsize=(6,6)):
    if type(arr) is not np.ndarray:
        arr = arr.numpy().transpose((0,2,3,1))
    plt.figure(figsize=figsize)
    for i,img in enumerate(arr[:16]):
        plt.subplot(*dim, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()


def plot_bw_samples(arr, dim=(4,4), figsize=(6,6)):
    if type(arr) is not np.ndarray:
        arr = arr.numpy()
    arr = arr.reshape(arr.shape[0], 28, 28)
    plt.figure(figsize=figsize)
    for i,img in enumerate(arr[:16]):
        plt.subplot(*dim, i+1)
        plt.imshow(img.astype('float32'), cmap='gray')
        plt.axis('off')
    plt.tight_layout()


def plot_img_from_fpath(img_path, fs=(8,8), title=None):
    plt.figure(figsize=fs)
    plt.imshow(plt.imread(img_path))
    plt.title(title)
    plt.show()


def plot_samples_from_dir(dir_path, shuffle=False, n=6):
    fpaths, fnames = files.get_paths_to_files(dir_path)
    plt.figure(figsize=(16,12))
    start = random.randint(0,len(fpaths)-1) if shuffle else 0
    j = 1
    for idx in range(start, min(len(fpaths), start+n)):
        plt.subplot(2,3,j)
        plt.imshow(plt.imread(fpaths[idx]))
        plt.title(fnames[idx])
        plt.axis('off')
        j += 1


def cut_image(arr, mask, color=(255,255,255)):
    arr = arr.copy()
    mask = format_1D_binary_mask(mask.copy())
    arr[mask > 0] = 255
    return arr


def format_1D_binary_mask(mask):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 0)
    mask = np.stack([mask,mask,mask], axis=1).squeeze().transpose(1,2,0)
    return mask.astype('float32')


def plot_binary_mask(arr, mask, title=None, color=(255,255,255)):
    mask = format_1D_binary_mask(mask.copy())
    for i in range(3):
        arr[:,:,i][mask[:,:,i] > 0] = color[i]
    utils.imgs.plot_img_arr(arr, title=title)


def plot_binary_mask_overlay(img_arr, mask, fs=(18,18), title=None):
    mask = format_1D_binary_mask(mask.copy())
    fig = plt.figure(figsize=fs)
    a = fig.add_subplot(1,2,1)
    a.set_title(title)
    plt.imshow(img_arr.astype('uint8'))
    plt.imshow(mask, cmap='jet', alpha=.5) # interpolation='none'
    plt.show()