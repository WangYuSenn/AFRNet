import argparse
# from easydict import EasyDict as edict
# cfg = edict()
parser = argparse.ArgumentParser()
# train
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.0005, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=150, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
# parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--gpu_id', type=int, default=0, help='select gpu id')
parser.add_argument("--test_batch_size", default=1, type=int)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--train_root', type=str, default='',
                    help='the train images root')
parser.add_argument('--save_path', type=str, default='',
                    help='the path to save models and logs')
# test(predict)
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--test_path', type=str, default='',
                    help='test dataset path')
parser.add_argument('--val_root', type=str, default='', help='val dataset path')


opt = parser.parse_args()

