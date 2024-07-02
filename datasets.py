import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from PIL import Image
from misc.data_utils import CDDataAugmentation


def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg','png']])

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


class bcd(Dataset):
    def __init__(self, root, split):
        super(bcd, self).__init__()
        self.img_t0_root = pjoin(root, 'A')
        self.img_t1_root = pjoin(root, 'B')
        self.img_label_root = pjoin(root, 'label')
        self.list_path = pjoin(root, 'list', (split+'.txt'))
        self.filename = load_img_name_list(self.list_path)
        self.img_size = 256

        # self.filename = list(spt(f)[0] for f in os.listdir(self.img_label_root) if check_validness(f))
        self.dataset_size = len(self.filename)
        self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )

    def __getitem__(self, index):
        name = self.filename[index % self.dataset_size]
        t0_path = pjoin(self.img_t0_root, name)
        t1_path = pjoin(self.img_t1_root, name)
        label_path = pjoin(self.img_label_root, name)

        img_t0 = np.asarray(Image.open(t0_path).convert('RGB'))
        img_t1 = np.asarray(Image.open(t1_path).convert('RGB'))
        # label = np.array(Image.open(label_path), dtype=np.uint8)[:, :, np.newaxis]
        label = np.array(Image.open(label_path), dtype=np.uint8)
        
        # label = np.asarray(Image.open(label_path).convert('RGB'))
        label = label // 255

        [img_t0, img_t1], [label] = self.augm.transform([img_t0, img_t1], [label], to_tensor=False)
        
        img_t0 = np.asarray(img_t0).astype('f').transpose(2, 0, 1)
        img_t1 = np.asarray(img_t1).astype('f').transpose(2, 0, 1)
        label = np.asarray(label).astype('f')[:, :, np.newaxis].transpose(2, 0, 1)

        input_ = torch.from_numpy(np.concatenate((img_t0, img_t1), axis=0))
        label_ = torch.from_numpy(label).long()

        return input_, label_

    def __len__(self):
        return self.dataset_size

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)

class bcd_eval(Dataset):
    def __init__(self, root, split):
        super(bcd_eval, self).__init__()
        self.img_t0_root = pjoin(root, 'A')
        self.img_t1_root = pjoin(root, 'B')
        self.img_label_root = pjoin(root, 'label')
        self.list_path = pjoin(root, 'list', (split+'.txt'))
        self.filename = load_img_name_list(self.list_path)
        self.img_size = 256

        # self.filename = list(spt(f)[0] for f in os.listdir(self.img_label_root) if check_validness(f))
        self.dataset_size = len(self.filename)
        self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        name = self.filename[index]
        # print(name)
        t0_path = pjoin(self.img_t0_root, name)
        t1_path = pjoin(self.img_t1_root, name)
        label_path = pjoin(self.img_label_root, name)

        img_t0 = np.asarray(Image.open(t0_path).convert('RGB'))
        img_t1 = np.asarray(Image.open(t1_path).convert('RGB'))
        # label = np.array(Image.open(label_path), dtype=np.uint8)[:, :, np.newaxis]
        label = np.array(Image.open(label_path), dtype=np.uint8)

        w, h, c = img_t0.shape
        w_r = w
        h_r = h
        # label = np.asarray(Image.open(label_path).convert('RGB'))
        label = label // 255

        [img_t0, img_t1], [label] = self.augm.transform([img_t0, img_t1], [label], to_tensor=False)

        img_t0 = np.asarray(img_t0).astype('f').transpose(2, 0, 1)
        img_t1 = np.asarray(img_t1).astype('f').transpose(2, 0, 1)
        label = np.asarray(label).astype('f')[:, :, np.newaxis].transpose(2, 0, 1)

        input_ = torch.from_numpy(np.concatenate((img_t0, img_t1), axis=0))
        label_ = torch.from_numpy(label).long()

        #return img_t0, img_t1, label, w, h, w_r, h_r
        return input_, label_

    def __len__(self):
        return self.dataset_size

    # def get_random_image(self):
    #     idx = np.random.randint(0,len(self))
    #     return self.__getitem__(idx)
       
        
        
        
        


