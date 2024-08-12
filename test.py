import torch
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

sys.path.append('./models')
import os
import cv2
import yaml
from torch.utils.data import DataLoader
from utils.PL_dataset import PL_dataset
# your model
model = xxx()
from config import opt
print('USE GPU:', opt.gpu_id)
device = torch.device('cuda:0')
def get_files_list(raw_dir):
    files_list = []
    for filepath, dirnames, filenames in os.walk(raw_dir):
        for filename in filenames:
            files_list.append(filepath+'/'+filename)
    return files_list

test_dataset_path = opt.test_path
image_root = get_files_list(test_dataset_path + '/vl')
ti_root = get_files_list(test_dataset_path + '/ir')
gt_root = get_files_list(test_dataset_path + '/gt')
test_dataset = PL_dataset(image_root, ti_root, gt_root, is_train=False)

test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

print('test_dataset', len(test_dataset))
print('test_loader_size', len(test_loader))
model.load_state_dict(torch.load('', map_location='cpu'),strict=False)
model.to(device)
model.eval()
test_datasets = ['1']
# test
test_mae = []
for dataset in test_datasets:
    mae_sum = 0
    save_path = '' + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for n_iter, batch_data in enumerate(test_loader):
        with torch.no_grad():
            image, ti, labels, name = batch_data
            image = image.to(device)
            ti = ti.to(device)
            labels = labels.to(device)
            res = model(image, ti)[0]
            name = str(name).replace('\'', "").replace('[','').replace(']','')
            predict = torch.sigmoid(res)
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
            mae = torch.sum(torch.abs(predict - labels)) * 1.0 / torch.numel(labels)
            mae_sum = mae.item() + mae_sum
        predict = predict.data.cpu().numpy().squeeze()
        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, predict * 256)
    test_mae.append(mae_sum / len(test_loader))
print('Test Done!', 'MAE{}'.format(test_mae))