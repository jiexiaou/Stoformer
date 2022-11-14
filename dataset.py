import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch, load_gray_img
import random
import cv2

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

##################################################################################################
def padding(img_lq, gt_size=384):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    return img_lq


class DataLoaderTrainGoPro(Dataset):
    def __init__(self, rgb_dir, patchsize, target_transform=None):
        super(DataLoaderTrainGoPro, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.input_filenames = [os.path.join(rgb_dir, input_dir, x) for x in input_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.crop_size = patchsize

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        ps = self.crop_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        input = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))

        clean = clean.permute(2, 0, 1)
        input = input.permute(2, 0, 1)

        # Crop Input and Target
        H = clean.shape[1]
        W = clean.shape[2]

        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        input = input[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        input = getattr(augment, apply_trans)(input)

        return [clean, input]

class DataLoaderTest(Dataset):
    def __init__(self, input_dir):
        super(DataLoaderTest, self).__init__()

        noisy_files = sorted(os.listdir(input_dir))
        self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2, 0, 1)

        return noisy, noisy_filename