import argparse
from easydict import EasyDict as edict
cfg = edict()
parser = argparse.ArgumentParser()
# train
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.005, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=300, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
# parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--gpu_id', type=int, default=0, help='select gpu id')
parser.add_argument("--test_batch_size", default=1, type=int)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--train_root', type=str, default='/media/xug/shuju/datasets/VITL Dataset/seed0/train',
                    help='the train images root')
parser.add_argument('--save_path', type=str, default='/media/xug/shuju/datasets/Two202394/P2T/KD_StudentNoSRM/',
                    help='the path to save models and logs')
# test(predict)
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--test_path', type=str, default='/media/xug/shuju/datasets/VITL Dataset/seed0/test_change',
                    help='test dataset path')
parser.add_argument('--val_root', type=str, default='/media/xug/shuju/datasets/VITL Dataset/seed0/val_change', help='val dataset path')

# 权重加载
cfg.PRETRAINED_ConvnextS_PATH = "/pth/convnext_small_1k_224_ema.pth"
cfg.PRETRAINED_Resnet34 = "/home/xug/PycharmProjects/TLD/pth/resnet34-333f7ec4.pth"
cfg.PRETRAINED_Resnet18 = "/home/xug/PycharmProjects/TLD/pth/resnet18-5c106cde.pth"
cfg.PRETRAINED_Resnet50 = "/home/xug/PycharmProjects/TLD/pth/resnet50-19c8e357.pth"
cfg.PRETRAINED_SSA = "/home/xug/PycharmProjects/TLD/pth/ckpt_S.pth"
cfg.PRETRAINED_PVTv2B5 = "/home/xug/PycharmProjects/TLD/pth/pvt_v2_b5.pth"
cfg.PRETRAINED_PVTv2B0 = "/home/xug/PycharmProjects/TLD/pth/pvt_v2_b0.pth"
cfg.PRETRAINED_ConvnextB_PATH = "/pth/convnext_base_1k_224_ema.pth"
cfg.PRETRAINED_ShuffleNetv2_PATH = "/pth/shufflenetv2_x1.pth"
cfg.PRETRAINED_MobileVitXs_PATH = "/pth/mobilevit_xs.pth"
cfg.PRETRAINED_MobileVitXxs_PATH = "/pth/mobilevit_xxs.pth"
cfg.PRETRAINED_MobileVitV2_PATH = "/pth/mobilevit-w1.0.pth"
opt = parser.parse_args()

