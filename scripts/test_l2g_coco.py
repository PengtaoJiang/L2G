from pickle import TRUE
import sys
import os
sys.path.append(os.getcwd())

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.LoadData import test_l2g_data_loader_mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import importlib
from torch.multiprocessing import Process

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    
def get_arguments():
    parser = argparse.ArgumentParser(description='L2G')
    parser.add_argument("--save_dir", type=str, default='./runs/exp8/')
    parser.add_argument("--img_dir", type=str, default='./data/VOCdevkit/VOC2012/JPEGImages/')
    parser.add_argument("--test_list", type=str, default='./data/voc12/train_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='voc2012')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--multi_scale", action='store_true', default=False)
    parser.add_argument("--restore_from", type=str, default='./runs/exp7/model/pascal_voc_epoch_14.pth')
    parser.add_argument("--thr", default=0.20, type=float)
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--cam_png", default='./runs/exp7/cam_png/', type=str)
    parser.add_argument("--resume_from", type=int, default=0)
    # parser.add_argument("--window_size", type=int, default=320)
    # parser.add_argument("--window_stride", type=int, default=128)

    return parser.parse_args()


def get_resnet38_model(args):
    model_name = "models.resnet38"
    print(model_name)
    model = getattr(importlib.import_module(model_name), 'Net')(args)

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()

    # print(pretrained_dict.keys())
    # print(model_dict.keys())

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return  model

def validate(args, gpu_id):
    print('\nvalidating ... ', flush=True, end='')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = get_resnet38_model(args)
    model = model.cuda()
    model.eval()
    val_loader = test_l2g_data_loader_mp(args, args.multi_scale, process_id=gpu_id)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
    if args.cam_png is not None:
        os.makedirs(args.cam_png, exist_ok=True)
    

    with torch.no_grad():
        print("process {} start at {}".format(os.getpid(), gpu_id))
        for idx, dat in tqdm(enumerate(val_loader)):
            if idx < args.resume_from:
                continue
            if not args.multi_scale:
                img_name, img, label_in = dat
                cv_im = cv2.imread(img_name[0])
                height, width = cv_im.shape[:2]
                _, logits = model(img.cuda())
                cam_map = model.module.get_heatmaps()
                cam_map = F.interpolate(cam_map, (height, width), mode='bicubic', align_corners=False)[0]
                cam_map_np = cam_map.cpu().data.numpy()
            else:
                pred_cam_list = []
                img_name, msf_img_list, label_in = dat 
                cv_im = cv2.imread(img_name[0])
                height, width = cv_im.shape[:2]

                for m in range(len(msf_img_list)):
                    img = msf_img_list[m]
                    cam_map, logits = model(img.cuda())
                    cam_map = F.interpolate(cam_map, (height, width), mode='bicubic', align_corners=False)[0]

                    cam_map_np = cam_map.cpu().data.numpy()
                    if m % 2 == 1:
                        cam_map_np = np.flip(cam_map_np, axis=-1)
                    pred_cam_list.append(cam_map_np)

                cam_map_np = np.mean(pred_cam_list, axis=0)

            cam_map_np[cam_map_np < 0] = 0
            norm_cam = cam_map_np / (np.max(cam_map_np, (1, 2), keepdims=True) + 1e-8)
            labels = label_in.long().numpy()[0]
            im_name = img_name[0].split('/')[-1][:-4]
            cam_dict = {}
            for j in range(args.num_classes):
                if labels[j] > 1e-5:
                    cam_dict[j] = norm_cam[j]
                    out_name = args.save_dir + im_name + '_{}.png'.format(j)
                    cv2.imwrite(out_name, norm_cam[j]*255.0)


if __name__ == '__main__':
    args = get_arguments()
    processes = []
    for i in range(4):
        proc = Process(target=validate, args=(args, i))
        processes.append(proc)
        proc.start()
    for proc in processes:
        proc.join()
