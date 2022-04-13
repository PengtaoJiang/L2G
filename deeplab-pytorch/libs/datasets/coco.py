#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   08 February 2019

from __future__ import absolute_import, print_function

import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class COCO(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset
    """

    def __init__(self, year=14, **kwargs):
        super(COCO, self).__init__(**kwargs)

    def _set_files(self):

        if self.split in ["train",  "val"]:
            file_list = osp.join(
               "./data/datasets/coco/", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip().split(" ") for id_ in file_list]
            self.files, self.labels = list(zip(*file_list))
        else:
            raise ValueError("Invalid split name: {}".format(self.split))


    def _load_data(self, index):
        # Set paths
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][1:])
        label_path = osp.join(self.root, self.labels[index][1:])
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_id, image, label