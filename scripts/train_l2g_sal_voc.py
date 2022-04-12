import sys
import os

sys.path.append(os.getcwd())

import torch
import argparse
import time
import shutil
import my_optim
import torch.optim as optim
import models
from torchvision import ops
import torch.nn.functional as F
from utils import AverageMeter
from utils.LoadData import train_l2g_sal_crop_data_loader
from models import vgg
import importlib
import numpy as np
import random


def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of L2G')
    parser.add_argument("--img_dir", type=str, default='./data/VOCdevkit/VOC2012/')
    parser.add_argument("--train_list", type=str, default='./data/voc12/train_cls.txt')
    parser.add_argument("--test_list", type=str, default='./data/voc12/val_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iter_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='61')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='./runs/exp8/model/')
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--att_dir", type=str, default='./runs/exp8/')
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--sal_dir", type=str, default="Sal")
    parser.add_argument("--poly_optimizer", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--kd_weights", type=int, default=15)
    parser.add_argument("--bg_thr", type=float, default=0.001)

    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed & (2**32 - 1))
     random.seed(seed & (2**32 - 1))
     torch.backends.cudnn.deterministic = True

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = vgg.vgg16(pretrained=True, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    model_local = vgg.vgg16(pretrained=True, num_classes=args.num_classes)
    model_local = torch.nn.DataParallel(model_local).cuda()

    param_groups = model.module.get_parameter_groups()
    param_groups_local = model_local.module.get_parameter_groups()

    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups_local[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2 * args.lr},
        {'params': param_groups_local[1], 'lr': 2 * args.lr},
        {'params': param_groups[2], 'lr': 10 * args.lr},
        {'params': param_groups_local[2], 'lr': 10 * args.lr},
        {'params': param_groups[3], 'lr': 20 * args.lr},
        {'params': param_groups_local[3], 'lr': 20 * args.lr}], momentum=0.9, weight_decay=args.weight_decay,
        nesterov=True)
    criterion = torch.nn.MSELoss()

    return model, model_local, optimizer, criterion

def get_resnet38_model(args):
    model_name = "models.resnet38"
    print(model_name)
    model = getattr(importlib.import_module(model_name), 'Net')(args)
    model_local = getattr(importlib.import_module(model_name), 'Net')(args)

    if len(args.load_checkpoint) == 0:
        weights_dict = models.resnet38_base.convert_mxnet_to_torch('./models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
        model.load_state_dict(weights_dict, strict=False)
        model = torch.nn.DataParallel(model).cuda()

        model_local.load_state_dict(weights_dict, strict=False)
        model_local = torch.nn.DataParallel(model_local).cuda()
    else:
        weights_dict = torch.load(args.load_checkpoint)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(weights_dict["state_dict"])

        model_local = torch.nn.DataParallel(model_local).cuda()
        model_local.load_state_dict(weights_dict["local_dict"])

    param_groups = model.module.get_parameter_groups()
    param_groups_local = model_local.module.get_parameter_groups()

    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups_local[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2 * args.lr},
        {'params': param_groups_local[1], 'lr': 2 * args.lr},
        {'params': param_groups[2], 'lr': 10 * args.lr},
        {'params': param_groups_local[2], 'lr': 10 * args.lr},
        {'params': param_groups[3], 'lr': 20 * args.lr},
        {'params': param_groups_local[3], 'lr': 20 * args.lr}], momentum=0.9, weight_decay=args.weight_decay,
        nesterov=True)
    if len(args.load_checkpoint) > 0:
        opt_weights_dict = torch.load(args.load_checkpoint)["optimizer"]
        optimizer.load_state_dict(opt_weights_dict)
    criterion = torch.nn.MSELoss()

    return model, model_local, optimizer, criterion

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    train_loader, val_loader = train_l2g_sal_crop_data_loader(args)
    max_step = total_epoch * len(train_loader)
    args.max_step = max_step
    print('Max step:', max_step)

    model, model_local, optimizer, criterion = get_resnet38_model(args)
    print(model)
    model.train()
    model_local.train()
    end = time.time()

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        if not args.poly_optimizer:
            res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)

        flag = 0
        for idx, dat in enumerate(train_loader):
            img, crop_imgs, crop_sals, boxes, label, local_label, img_name = dat
            crop_sals = crop_sals
            label = label.cuda(non_blocking=True)
            local_label = local_label.cuda(non_blocking=True)

            # dealing with batch size
            bs, bxs, c, h, w = crop_imgs.shape
            _, _, c_s, h_s, w_s = crop_sals.shape
            crop_imgs = crop_imgs.reshape(bs * bxs, c, h, w)
            crop_sals = crop_sals.reshape(bs * bxs, c_s, h_s, w_s)
            local_label = local_label.reshape(bs * bxs, args.num_classes)
            box_ind = torch.cat([torch.zeros(args.patch_size).fill_(i) for i in range(bs)])
            boxes = boxes.reshape(bs * bxs, 5)
            boxes[:, 0] = box_ind

            feat, logits = model(img)
            feat_local, logits_local = model_local(crop_imgs)
            boxes = boxes.cuda(non_blocking=True).type_as(feat)

            # visualize
            feat_local_label = feat_local.clone().detach()  # 4, 20, 224, 224

            # normalize
            ba = logits_local.shape[0]
            feat_local_label[feat_local_label < 0] = 0
            ll_max = torch.max(torch.max(feat_local_label, dim=3)[0], dim=2)[0]
            feat_local_label = feat_local_label / (ll_max.unsqueeze(2).unsqueeze(3) + 1e-8)
            for i in range(bs):
                ind = torch.nonzero(label[i] == 0)
                feat_local_label[i * bxs:(i + 1) * bxs, ind] = 0

            # keep max value among all classes
            n, c, h, w = feat_local_label.shape
            feat_local_label_c = feat_local_label.permute(1, 0, 2, 3).reshape(c, -1)
            ind_f = torch.argsort(-feat_local_label_c, axis=0)
            pos = torch.eye(c)[ind_f[0]].transpose(0, 1).type_as(feat_local_label_c)
            feat_local_label_c = pos * feat_local_label_c
            feat_local_label = feat_local_label_c.reshape(c, n, h, w).permute(1, 0, 2, 3)

            # match the sal label    hyper-parameter
            feat_local_label_bool = (feat_local_label > args.bg_thr).type_as(feat_local_label)
            crop_sals = F.interpolate(crop_sals, (h, w)).type_as(feat_local_label)
            feat_local_label[:, :-1, :, :] = feat_local_label_bool[:, :-1, :, :] * crop_sals.repeat(1, 20, 1, 1)
            feat_local_label[:, -1, :, :] = feat_local_label_bool[:, -1, :, :] * ((1 - crop_sals).squeeze(1))

            # roi align
            feat_aligned = ops.roi_align(feat, boxes, (h, w), 1 / 8.0)
            feat_aligned = F.softmax(feat_aligned, dim=1)
            loss_kd = criterion(feat_aligned, feat_local_label) * args.kd_weights

            # cls loss
            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)

            loss_cls_global = F.multilabel_soft_margin_loss(logits, label) / args.iter_size
            loss_cls_local = F.multilabel_soft_margin_loss(logits_local, local_label) / args.iter_size

            loss_val = loss_kd + loss_cls_local
            loss_val.backward()

            flag += 1
            if flag == args.iter_size:
                optimizer.step()
                optimizer.zero_grad()
                flag = 0

            losses.update(loss_val.data.item(), img.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()

            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t'
                      'Loss_kd {:.4f}\t'
                      'Loss_cls_global {:.4f}\t'
                      'Loss_cls_local {:.4f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], loss_kd, loss_cls_global,
                    loss_cls_local, loss=losses))

        if current_epoch == args.epoch - 1:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict': model.state_dict(),
                                'local_dict': model_local.state_dict(),
                                'optimizer': optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d.pth' % (args.dataset, current_epoch))
        current_epoch += 1


if __name__ == '__main__':
    setup_seed(15742315057023588855)
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
