# coding:utf-8
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as TF
import random
import imageio
import numpy as np
from numpy import genfromtxt
import imgaug.augmenters as iaa
"""
vl_dir, ir_dir, gt_dir : each path of the images
[is_train = True ] Generate training data set
[is_train = False] Generate validation/test data set
"""
class PL_dataset(Dataset):

    def __init__(self, vl_list, ir_list, gt_list, is_train):
        super(PL_dataset, self).__init__()

        self.vl_list  = sorted(vl_list)
        self.ir_list  = sorted(ir_list)
        self.gt_list  = sorted(gt_list)
        self.n_data   = len(self.vl_list)
        self.is_train = is_train

    def transform_train(self, vl, ir, gt):
        vl = TF.to_pil_image(vl)
        ir = TF.to_pil_image(ir)
        # print('ir.type', ir.type())
        gt = TF.to_pil_image(gt)



        if random.random() > 0.5:
            vl = TF.hflip(vl)
            ir = TF.hflip(ir)

            gt = TF.hflip(gt)

        if random.random() > 0.5:
            angle = int(np.random.uniform(-10, 10, 1))
            vl = TF.rotate(vl, angle)
            ir = TF.rotate(ir, angle)
            gt = TF.rotate(gt, angle)
        gt = gt.convert('L')
        vl = transforms.ToTensor()(vl)
        ir = transforms.ToTensor()(ir)
        gt = transforms.ToTensor()(gt)
        # gt = torch.cat([gt < 0.5, gt >= 0.5], dim=0).to(torch.float32)

        return vl, ir, gt


    def transform_test(self, vl, ir, gt):

        vl = TF.to_pil_image(vl)
        ir = TF.to_pil_image(ir)
        gt = TF.to_pil_image(gt)
        gt = gt.convert('L')
        vl = transforms.ToTensor()(vl)
        ir = transforms.ToTensor()(ir)
        gt = transforms.ToTensor()(gt)
        # gt = gt.data.cpu().numpy()

        # gt = torch.cat([gt < 0.5, gt >= 0.5], dim=0).to(torch.float32)
        return vl, ir, gt


    def get_train_item(self, index):
        # print('self.vl_list[index]', self.vl_list[index])
        vl = imageio.imread(self.vl_list[index])

        if random.random() > 0.5:
            image = cv.imread(self.vl_list[index])
            aug_idx = np.random.randint(4)
            if aug_idx == 0:
                h, v, s = 1.0, 0.6, 0.7
                hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                hsv[:, :, 2] = (hsv[:, :, 2] * v).astype('uint8')
                hsv[:, :, 1] = (hsv[:, :, 1] * s).astype('uint8')
                im = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

                im[:, :, 1] = im[:, :, 1] + 15
                im[:, :, 2] = im[:, :, 2] + 15
                vl = im[:, :, ::-1]

            elif aug_idx == 1:
                h, v, s = 1.0, 0.3, 0.3
                hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                hsv[:, :, 2] = (hsv[:, :, 2] * v).astype('uint8')
                hsv[:, :, 1] = (hsv[:, :, 1] * s).astype('uint8')
                im = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

                im[:, :, 0] = im[:, :, 0] + 10
                vl = im[:, :, ::-1]

            elif aug_idx == 2:
                aug = iaa.imgcorruptlike.Fog(severity=1)
                im = aug(images=[image[:, :, ::-1]])[0]
                vl = im

            elif aug_idx == 3:
                aug = iaa.imgcorruptlike.Snow(severity=1)
                im = aug(images=[image[:, :, ::-1]])[0]
                vl = im

        gt = imageio.imread(self.gt_list[index])
        ir = genfromtxt(self.ir_list[index], delimiter=',')
        ir = ir.astype('uint8')
        # print('ir.type', ir.type())

        vl, im, gt = self.transform_train(vl, ir, gt)

        return vl, im, gt

    def get_test_item(self, index):
        vl = imageio.imread(self.vl_list[index])
        gt = imageio.imread(self.gt_list[index])
        # gt = gt.convert('L')
        ir = genfromtxt(self.ir_list[index], delimiter=',')
        ir = ir.astype('uint8')
        vl, im, gt = self.transform_test(vl, ir, gt)
        name = os.path.basename(self.vl_list[index])



        return vl, im, gt, name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data

class test_dataset_VDT:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.jpg')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        # print('self.img_list[self.index]', self.img_list[self.index])
        # image = self.binary_loader(os.path.join(self.image_root, self.img_list[self.index] + '.png'))
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.jpg'))
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.jpg'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

if __name__ == '__main__':
    PL_dataset()