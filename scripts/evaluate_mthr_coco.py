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
    parser.add_argument("--save_label_path", type=str, default='./runs/exp1/ms/cam_png/')
    parser.add_argument('--dataset', type=str, default='pascal_voc')
    parser.add_argument("--att_dir", type=str, default='./runs/exp8/')
    parser.add_argument("--thr", type=list, default=[0.2, 0.3])
    parser.add_argument("--num_workers", type=int, default=1)

    return parser.parse_args()

def visual(args, i):
    if not os.path.exists(args.save_label_path):
        os.makedirs(args.save_label_path)

    gt_dir = args.gt_dir
    with open(args.datalist) as f:
        lines = f.readlines()
    ids = [line[:-1].split()[0] for line in lines]
    label_lst = [line[:-1].split()[1:] for line in lines]

    if 'coco' in args.dataset:
        num_classes = 80
    else:
        num_classes = 20

    import time

    st = time.time()
    for _, bg_thr in enumerate(args.thr):

        mIOU = IOUMetric(num_classes=num_classes+1)
        for ind, img_id in tqdm(enumerate(ids)):
            if not ind % args.num_workers == i:
                continue
            img_path = os.path.join(gt_dir, img_id + '.png')
            gt = Image.open(img_path)
            w, h = gt.size[0], gt.size[1]
            gt = np.array(gt, dtype=np.int32)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
            
            pred = np.zeros((num_classes+1, h, w), np.float32)
            pred[0] = bg_thr
            for jj in range(len(label_lst[ind])):
                la = int(label_lst[ind][jj])
                att_img_path = os.path.join(args.pred_dir, img_id + '_{}.png'.format(la))
                att = cv2.imread(att_img_path, 0) / 255.0
                pred[la+1] = att
            pred = np.argmax(pred, axis=0).astype(np.uint8)
            mIOU.add_batch(pred, gt)

        acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()

        mean_prec = np.nanmean(precision)
        mean_recall = np.nanmean(recall)

        result = {"Threshold": bg_thr, 
                "Recall": ["{:.2f}".format(i) for i in recall.tolist()],
                "Precision": ["{:.2f}".format(i) for i in precision.tolist()],
                "Mean_Recall": mean_recall,
                "Mean_Precision": mean_prec,
                "IoU": cls_iu,
                "Mean IoU": miou,
                "TP": TP.tolist(),
                "TN": TN.tolist(),
                "FP": FP.tolist()}

        with open(args.save_path, "a") as f:
            json.dump(result, f, indent=4, sort_keys=True)
        print('bg_threshold = %s, mIOU = %s, time = %s s' % (bg_thr, miou, str(time.time() - st)))

if __name__ == '__main__':
    args = get_arguments()
    processes = []
    for i in range(1):
        proc = Process(target=visual, args=(args, i))
        processes.append(proc)
        proc.start()
    for proc in processes:
        proc.join()
    
