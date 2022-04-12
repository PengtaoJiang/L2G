import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import json
import cv2
from torch.multiprocessing import Process

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        # recall = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        # precision = np.nanmean(precision)
        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return acc, recall, precision, TP, TN, FP, cls_iu, mean_iu, fwavacc


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='./data/VOCdevkit/VOC2012/SegmentationClassAug/')
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--datalist', type=str, default='./data/train.txt')
    parser.add_argument('--save_path', type=str)
    parser.add_argument("--att_dir", type=str, default='./runs/exp8/')
    parser.add_argument('--dataset', type=str, default='pascal_voc')
    # parser.add_argument("--thr", type=list, default=[0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--thr", type=list, default=[0.25])
    parser.add_argument("--num_workers", type=int, default=8)
    # parser.add_argument("--thr", type=list, default=[0.3, 0.4, 0.5, 0.6])

    return parser.parse_args()

def visual(args, i):
    gt_dir = args.gt_dir
    list_dir = 'ImageSets/Segmentation/'
    ids = [i.split()[0].split('/')[2].split('.')[0].strip() for i in open(args.datalist) if not i.strip() == '']
    with open('./data/voc12/train_cls.txt') as f:
        lines = f.readlines()
    label_lst = [line[:-1].split()[1:] for line in lines]
    # gt_dir = args.gt_dir
    # with open(args.datalist) as f:
    #     lines = f.readlines()
    # ids = [line[:-1].split()[0] for line in lines]
    # label_lst = [line[:-1].split()[1:] for line in lines]

    if 'coco' in args.dataset:
        num_classes = 80
    else:
        num_classes = 20

    classes = np.array(('background',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]
    values = [i for i in range(21)]
    color2val = dict(zip(colormap, values))

    import time

    st = time.time()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    # mIOU = IOUMetric(num_classes=21)
    for ind, img_id in tqdm(enumerate(ids)):
        if not ind % args.num_workers == i:
            continue
        img_path = os.path.join(gt_dir, img_id + '.jpg')
        gt = Image.open(img_path)
        w, h = gt.size[0], gt.size[1]
        # gt = np.array(gt, dtype=np.int32)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary

        pred = []
        pred.append(np.zeros((num_classes+1, h, w), np.float32))
        pred.append(np.zeros((num_classes+1, h, w), np.float32))
        for tt, bg_thr in enumerate(args.thr):
            pred[tt][0] = bg_thr
            for jj in range(len(label_lst[ind])):
                la = int(label_lst[ind][jj])
                att_img_path = os.path.join(args.pred_dir, img_id + '_{}.png'.format(la))
                att = cv2.imread(att_img_path, 0) / 255.0
                pred[tt][la+1] = att
            pred[tt] = np.argmax(pred[tt], axis=0).astype(np.uint8)
        # diff = np.where(pred[0] == pred[1], pred[0], 255).astype(np.uint8)
        # print(diff.shape)
        cv2.imwrite(args.save_path + img_id+".png", pred[0])

if __name__ == '__main__':
    args = get_arguments()
    processes = []
    for i in range(args.num_workers):
        proc = Process(target=visual, args=(args, i))
        processes.append(proc)
        proc.start()
    for proc in processes:
        proc.join()
    
