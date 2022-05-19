import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
from random import randrange


class DataProcess(torch.utils.data.Dataset):
    def __init__(self, de_root, st_root, mask_root, opt, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        self.Train = False
        self.opt = opt

        if train:
            self.de_paths = sorted(glob('{:s}/*'.format(de_root), recursive=True))
            self.st_paths = sorted(glob('{:s}/*'.format(st_root), recursive=True))
            self.mask_paths = sorted(glob('{:s}/*'.format(mask_root), recursive=True))
            self.Train = True
        self.N_mask = len(self.mask_paths)
        print(self.N_mask)

    def __getitem__(self, index):
        de_img = Image.open(self.de_paths[index])
        st_img = Image.open(self.st_paths[index])
        # if self.Train:  # for Paris Street View dataset
        #     x, y = de_img.size
        #     if x != y:
        #         matrix_length = min(x, y)
        #         x1 = randrange(0, x-matrix_length+1)
        #         y1 = randrange(0, y-matrix_length+1)
        #         de_img = de_img.crop((x1, y1, x1+matrix_length, y1+matrix_length))
        #         st_img = st_img.crop((x1, y1, x1 + matrix_length, y1 + matrix_length))
        mask_img = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        de_img = self.img_transform(de_img.convert('RGB'))
        st_img = self.img_transform(st_img.convert('RGB'))
        mask_img = self.mask_transform(mask_img.convert('RGB'))
        return de_img, st_img, mask_img

    def __len__(self):
        return len(self.de_paths)
