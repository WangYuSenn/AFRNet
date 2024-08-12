import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, label, ti,di):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        ti = ti.transpose(Image.FLIP_LEFT_RIGHT)
        di = di.transpose(Image.FLIP_LEFT_RIGHT)

    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     ti = ti.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, ti,di


def randomCrop(image, label, ti,di):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), ti.crop(random_region),di.crop(random_region)



def randomRotation(image, label, ti,di):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        ti = ti.rotate(random_angle, mode)
        di = di.rotate(random_angle, mode)

    return image, label, ti,di


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized ti maps for training and test. If you use the normalized ti maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, ti_root, di_root,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')or f.endswith('.png')]
        # print(self.images)
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.tis = [ti_root + f for f in os.listdir(ti_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.dis = [di_root + f for f in os.listdir(di_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.tis = sorted(self.tis)
        self.dis = sorted(self.dis)


        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.di_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        ti = self.rgb_loader(self.tis[index])
        di = self.binary_loader(self.dis[index])

        image, gt, ti ,di = cv_random_flip(image, gt, ti,di)
        image, gt, ti ,di = randomCrop(image, gt, ti,di)
        image, gt, ti,di  = randomRotation(image, gt, ti,di)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        ti = self.tis_transform(ti)
        di = self.di_transform(di)

        return image, gt, ti,di

    def filter_files(self):
        print(len(self.images),len(self.gts),len(self.gts),len(self.tis))
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.tis)
        images = []
        gts = []
        tis = []
        bounds = []
        for img_path, gt_path, ti_path in zip(self.images, self.gts, self.tis):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            ti = Image.open(ti_path)
            if img.size == gt.size and gt.size == ti.size:
                images.append(img_path)
                gts.append(gt_path)
                tis.append(ti_path)
        self.images = images
        self.gts = gts
        self.tis = tis

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, ti):
        assert img.size == gt.size and gt.size == ti.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), ti.resize((w, h), Image.NEAREST)
        else:
            return img, gt, ti

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, ti_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, ti_root, trainsize)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    # print(len(data_loader))
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, ti_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.tis = [ti_root + f for f in os.listdir(ti_root) if f.endswith('.jpg')
                       or f.endswith('.csv') ]
        # self.dis = [di_root + f for f in os.listdir(di_root) if f.endswith('.jpg')
        #             or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.tis = sorted(self.tis)
        # self.dis = sorted(self.dis)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.dis_transform = transforms.Compose([
        #     transforms.Resize((self.testsize, self.testsize)),
        #     transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        ti = self.rgb_loader(self.tis[self.index])
        ti = self.tis_transform(ti).unsqueeze(0)
        # di = self.binary_loader(self.dis[self.index])
        # di = self.dis_transform(di).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        # print("hh",image.size(),gt.size(),ti.size(),di.size())
        return image, gt, ti, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

# if __name__ == '__main__':
#     t = SalObjDataset()
#     print( t.images)