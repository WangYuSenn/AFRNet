import argparse
import os
from collections import OrderedDict
from utils.PL_dataset import PL_dataset
from utils.util import calc_loss, print_metrics, adjust_lr
import torch
import torchvision.models
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader
# from dataset import get_loader,test_dataset
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
cudnn.benchmark = True
cudnn.enabled = True
import yaml
print('USE GPU:', opt.gpu_id)

# build the model
print("TLD")

# your models
from
from loss.losses_KD import *
from loss.ofd import OFD
cfg = "train"


print(torch.__version__)
teacher_model = xxx()
student_model = xxx()
teacher_model.load_state_dict(torch.load('', map_location='cuda:0'), strict=False)  # 163nei 100  153
teacher_model.eval()

print('model:Distill')

l2 = nn.MSELoss().cuda()
KLD = KLDLoss().cuda()
CL = nn.TripletMarginLoss().cuda()

def dic_loss(pre, mask):
    mask = torch.sigmoid(mask)
    pre = torch.sigmoid(pre)
    intersection = (pre*mask).sum(axis=(2, 3))
    unior = (pre+mask).sum(axis=(2, 3))
    dice = (2*intersection + 1)/(unior + 1)
    dice = torch.mean(1 - dice)
    return dice
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
def contrastive_loss(anchor, positive, negative, alpha=1.0, reg_lamba=0.1):
    cosine_sim_pos = F.cosine_similarity(anchor, positive)
    cosine_sim_neg = F.cosine_similarity(anchor, negative)

    loss = torch.clamp(cosine_sim_neg - cosine_sim_pos + alpha , min=0.0)
    loss = torch.mean(loss)

    l2_1_norm = torch.sqrt(torch.sum(torch.sqrt(torch.sum(anchor**2, dim=-1))))
    loss += reg_lamba * l2_1_norm
    return loss
student_model.cuda()
teacher_model.cuda()
logger = logging.getLogger(__name__)
params = student_model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
print("==> Total params: %.2fM" % ( sum(p.numel() for p in student_model.parameters()) / 1e6))

def get_files_list(raw_dir):
    files_list = []
    for filepath, dirnames, filenames in os.walk(raw_dir):
        for filename in filenames:
            files_list.append(filepath+'/'+filename)
    return files_list

# set the path
train_dataset_path = opt.train_root
image_root = get_files_list(train_dataset_path + '/vl')
ti_root = get_files_list(train_dataset_path + '/ir')
gt_root = get_files_list(train_dataset_path + '/gt')


val_dataset_path = opt.val_root
val_image_root = get_files_list(val_dataset_path + '/vl')
val_ti_root = get_files_list(val_dataset_path + '/ir')
val_gt_root = get_files_list(val_dataset_path + '/gt')

# 保存训练权重
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')

print('Training data set', len(image_root), len(ti_root), len(gt_root))
print('Validation data set', len(val_image_root), len(val_ti_root), len(val_gt_root))
train_dataset = PL_dataset(image_root, ti_root, gt_root, is_train=True)
val_dataset = PL_dataset(image_root, ti_root, gt_root, is_train=False)
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False
    )
train_loader.n_iter = len(train_loader)
val_loader.n_iter = len(val_loader)
total_step = len(train_loader)
print(total_step)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Model: LHDecoder4_small_catorignal2")
logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss().cuda()
step = 0
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

def train(train_loader, student_model, teacher_model, optimizer, epoch, save_path, temperature):
    global step, best_mae, best_epoch
    student_model.train()
    teacher_model.eval()
    metrics = defaultdict(float)
    loss_all = 0
    epoch_step = 0
    mae_sum = 0
    try:
        for i, (images, ti, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = Variable(images).cuda()
            ti = Variable(ti).cuda()
            labels = Variable(labels).cuda()
            if opt.gpu_id >= 0:
                images = images.cuda(opt.gpu_id)
                ti = ti.cuda(opt.gpu_id)
                labels = labels.cuda(opt.gpu_id)
            s = student_model(images)
            with torch.no_grad():
                t = teacher_model(images, ti)
            outputs = s[0].cuda()
            loss1,metrics = calc_loss(outputs, labels,metrics)
            loss2 = KLD(s[5], t[5], 0, labels)
            loss3 = F.binary_cross_entropy_with_logits(torch.sigmoid(s[0]), torch.sigmoid(t[0]))
            loss4 = dic_loss(t[4], s[4])

            loss = loss1 + loss2 + loss3 + loss4
            res = torch.sigmoid(s[0])
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res.cuda() - labels.cuda())) * 1.0 / (torch.numel(labels.cuda()))
            mae_sum = mae_train.item() + mae_sum
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            step = step + 1
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss.item()))
        mae_train = mae_sum / len(train_loader)
        print('Train : Epoch: {} MAE: {} bestmae: {} bestepoch: {}'.format(epoch, mae_train, best_mae, best_epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(student_model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise
# test function
def val(val_loader, student_model, epoch, save_path):
    global best_mae, best_epoch
    student_model.eval()
    with torch.no_grad():
        mae_sum = 0
        for it, (images, ti, labels, name) in enumerate(val_loader):
            image = Variable(images)
            ti = Variable(ti)
            gt = Variable(labels)
            if opt.gpu_id >= 0:
                image = images.cuda(opt.gpu_id)
                ti = ti.cuda(opt.gpu_id)
                gt = labels.cuda(opt.gpu_id)
            res = student_model(image, ti)
            res = torch.sigmoid(res[0])
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
            mae_sum = mae_train.item() + mae_sum
        mae = mae_sum / len(val_loader)
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                print('Val update ---- ', mae)
                best_mae = mae
                best_epoch = epoch
                # if epoch > 150:
                torch.save(student_model.state_dict(), save_path + 'best_{}'.format(epoch)+'_epoch.pth')
                    # torch.save(student_model.state_dict(), save_path + 'best_epoch.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        print('Val Epoch: {} MAE: {} bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))


if __name__ == '__main__':
    print("Start train...")
    global temperature
    temperature = 1
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, student_model, teacher_model, optimizer, epoch, save_path, temperature)
        val(val_loader, student_model, epoch, save_path)
        print('temperature:', temperature)
        print("lr",cur_lr)
