# from torchvision import transforms
from .transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import ResizeShort, RandomResizeLong, get_random_crop_box, crop_with_box
import os
from PIL import Image
import random
import utils.transforms.functional as F



class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx]

        return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels  # , np.array(img_labels, dtype=np.float32)

############# data loader for local refinement ###############

def test_l2g_data_loader(args, ms=True):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    if ms:
        tsfm_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean_vals, std_vals),
                                        ])
        # multi scale inference
        img_test = VOCDatasetMSF_l2g(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes,
                                     transform=tsfm_test, test=True)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        tsfm_test = transforms.Compose([transforms.Resize(args.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_vals, std_vals),
                                        ])
        # single scale inference
        img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test,
                              test=True)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDataset_l2g(Dataset):
    def __init__(self, datalist_file, root_dir, input_size=448, crop_size=224, num_classes=20, transform=None,
                 test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.input_size = input_size
        self.crop_size = crop_size
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        boxes = self.box_generation()

        if self.transform is not None:
            image = self.transform(image)

        if self.testing:
            return img_name, image, self.label_list[idx]

        crop_images = []
        for i in range(len(boxes)):
            box = boxes[i][1:]
            crop_images.append(image[:, box[1]:box[3], box[0]:box[2]].clone()[None])
        crop_images = torch.cat(crop_images, dim=0)

        label = self.label_list[idx]
        label_local = []
        for i in range(4):
            label_local.append(label[None])
        label_local = np.concatenate(label_local, axis=0)
        return image, crop_images, boxes, label, label_local, img_name

    def box_generation(self):
        max_range = self.input_size - self.crop_size
        boxes = []
        for i in range(4):
            ind_h, ind_w = np.random.randint(0, max_range, size=2)
            boxes.append(torch.tensor([0, ind_w, ind_h, ind_w + self.crop_size, ind_h + self.crop_size])[None])
        boxes = torch.cat(boxes, dim=0)

        return boxes  # K, 5

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels  # , np.array(img_labels, dtype=np.float32)


# multi scale inference
class VOCDatasetMSF_l2g(Dataset):
    def __init__(self, datalist_file, root_dir, scales=(0.5, 1.0, 1.5, 2.0), num_classes=20, transform=None,
                 test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.scales = scales
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        ms_img_list = []
        for s in self.scales:
            target_size = (int(round(image.size[0] * s)),
                           int(round(image.size[1] * s)))
            s_img = image.resize(target_size, resample=Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform is not None:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))

        if self.testing:
            return img_name, msf_img_list, self.label_list[idx]

        return msf_img_list, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels  # np.array(img_labels, dtype=np.float32)


######################################################################

def train_l2g_sal_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    img_train = VOCDataset_l2g_sal(args.train_list, root_dir=args.img_dir, sal_dir=args.sal_dir, input_size=input_size, crop_size=crop_size,
                                   num_classes=args.num_classes, patch_num=args.patch_num, transform=tsfm_train, test=False)
    img_test = VOCDataset_l2g(args.test_list, root_dir=args.img_dir, input_size=input_size, crop_size=crop_size,
                              num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


class VOCDataset_l2g_sal(Dataset):
    def __init__(self, datalist_file, root_dir, sal_dir='Sal', input_size=448, crop_size=224, num_classes=20, patch_num=4, transform=None,
                 test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.input_size = input_size
        self.crop_size = crop_size
        self.patch_num = patch_num
        self.image_list, self.sal_list, self.label_list = self.read_labeled_image_list(self.root_dir,
                                                                                       self.datalist_file, sal_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        sal_name = self.sal_list[idx]
        image = Image.open(img_name).convert('RGB')
        boxes = self.box_generation()
        sal = Image.open(sal_name).convert('L')

        if self.transform is not None:
            image = self.transform(image)
            sal = F.to_tensor(F.resize(sal, (self.input_size, self.input_size)))
            sal /= (sal.max() + 1e-10)

        if self.testing:
            return img_name, image, self.label_list[idx]

        crop_images = []
        crop_sals = []
        for i in range(len(boxes)):
            box = boxes[i][1:]
            crop_images.append(image[:, box[1]:box[3], box[0]:box[2]].clone()[None])
            crop_sals.append(sal[:, box[1]:box[3], box[0]:box[2]].clone()[None])

        crop_images = torch.cat(crop_images, dim=0)
        crop_sals = torch.cat(crop_sals, dim=0)

        label = self.label_list[idx]
        label_local = []
        for i in range(self.patch_num):
            label_local.append(label[None])
        label_local = np.concatenate(label_local, axis=0)

        return image, crop_images, crop_sals, boxes, label, label_local, img_name

    def box_generation(self):
        max_range = self.input_size - self.crop_size
        boxes = []
        for i in range(self.patch_num):
            ind_h, ind_w = np.random.randint(0, max_range, size=2)
            boxes.append(torch.tensor([0, ind_w, ind_h, ind_w + self.crop_size, ind_h + self.crop_size])[None])
        boxes = torch.cat(boxes, dim=0)

        return boxes  # K, 5

    def read_labeled_image_list(self, data_dir, data_list, sal_dir):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        sal_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            sal_image = fields[0] + '.png'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, 'JPEGImages', image))
            sal_name_list.append(os.path.join(data_dir, sal_dir, sal_image))
            img_labels.append(labels)
        return img_name_list, sal_name_list, img_labels  # , np.array(img_labels, dtype=np.float32)


## use crop to have a higher resolution
class VOCDataset_l2g_sal_crop(Dataset):
    def __init__(self, datalist_file, root_dir, sal_dir='Sal', input_size=448, crop_size=224, num_classes=20, patch_size=4,
                 transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.input_size = input_size
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.extra_transform = ResizeShort(512)
        self.image_list, self.sal_list, self.label_list = self.read_labeled_image_list(self.root_dir,
                                                                                       self.datalist_file, sal_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        sal_name = self.sal_list[idx]
        image = Image.open(img_name).convert('RGB')
        boxes = self.box_generation()
        sal = Image.open(sal_name).convert('L')

        if self.transform is not None:
            # target_short = random.randint(448, 672)
            # self.extra_transform.short_size = target_short
            image = self.extra_transform(image)
            sal = self.extra_transform(sal, isimg=False)

            box_f = get_random_crop_box(image.size, self.input_size)
            image = image.crop((box_f[6], box_f[4], box_f[7], box_f[5]))
            sal = sal.crop((box_f[6], box_f[4], box_f[7], box_f[5]))
            image = self.transform(image)
            sal = F.to_tensor(sal)
            sal /= (sal.max() + 1e-10)

        if self.testing:
            return img_name, image, self.label_list[idx]

        crop_images = []
        crop_sals = []
        for i in range(len(boxes)):
            box = boxes[i][1:]
            crop_images.append(image[:, box[1]:box[3], box[0]:box[2]].clone()[None])
            crop_sals.append(sal[:, box[1]:box[3], box[0]:box[2]].clone()[None])

        crop_images = torch.cat(crop_images, dim=0)
        crop_sals = torch.cat(crop_sals, dim=0)

        label = self.label_list[idx]
        label_local = []
        for i in range(self.patch_size):
            label_local.append(label[None])
        label_local = np.concatenate(label_local, axis=0)

        return image, crop_images, crop_sals, boxes, label, label_local, img_name

    def box_generation(self):
        max_range = self.input_size - self.crop_size
        assert(self.patch_size >= 4)
        boxes = []
        boxes = [torch.tensor([0, 0, 0, self.crop_size, self.crop_size])[None],
                torch.tensor([0, 0, max_range, self.crop_size, max_range + self.crop_size])[None],
                torch.tensor([0, max_range, 0, max_range + self.crop_size, self.crop_size])[None],
                torch.tensor([0, max_range, max_range, max_range + self.crop_size, max_range + self.crop_size])[None]
               ]
        for i in range(self.patch_size - 4):
            ind_h, ind_w = np.random.randint(0, max_range, size=2)
            boxes.append(torch.tensor([0, ind_w, ind_h, ind_w + self.crop_size, ind_h + self.crop_size])[None])
        boxes = torch.cat(boxes, dim=0)

        return boxes  # K, 5

    def read_labeled_image_list(self, data_dir, data_list, sal_dir):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        sal_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            sal_image = fields[0] + '.png'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, 'JPEGImages', image))
            sal_name_list.append(os.path.join(data_dir, sal_dir, sal_image))
            img_labels.append(labels)
        return img_name_list, sal_name_list, img_labels  # , np.array(img_labels, dtype=np.float32)

def train_l2g_sal_crop_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    img_train = VOCDataset_l2g_sal_crop(args.train_list, root_dir=args.img_dir, sal_dir=args.sal_dir, input_size=input_size, crop_size=crop_size,
                                   num_classes=args.num_classes, transform=tsfm_train, test=False, patch_size=args.patch_size)
    img_test = VOCDataset_l2g(args.test_list, root_dir=args.img_dir, input_size=input_size, crop_size=crop_size,
                              num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

class VOCDataset_l2g_crop(Dataset):
    def __init__(self, datalist_file, root_dir, input_size=448, crop_size=224, num_classes=20, patch_size=4,
                 transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.input_size = input_size
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.extra_transform = ResizeShort(512)
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir,
                                                                                       self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        boxes = self.box_generation()

        if self.transform is not None:
            target_short = random.randint(448, 672)
            self.extra_transform.short_size = target_short
            image = self.extra_transform(image)

            box_f = get_random_crop_box(image.size, self.input_size)
            image = image.crop((box_f[6], box_f[4], box_f[7], box_f[5]))
            image = self.transform(image)

        if self.testing:
            return img_name, image, self.label_list[idx]

        crop_images = []
        for i in range(len(boxes)):
            box = boxes[i][1:]
            crop_images.append(image[:, box[1]:box[3], box[0]:box[2]].clone()[None])

        crop_images = torch.cat(crop_images, dim=0)

        label = self.label_list[idx]
        label_local = []
        for i in range(self.patch_size):
            label_local.append(label[None])
        label_local = np.concatenate(label_local, axis=0)

        return image, crop_images, boxes, label, label_local, img_name

    def box_generation(self):
        max_range = self.input_size - self.crop_size
        assert(self.patch_size >= 4)
        # boxes = []
        boxes = [torch.tensor([0, 0, 0, self.crop_size, self.crop_size])[None],
                torch.tensor([0, 0, max_range, self.crop_size, max_range + self.crop_size])[None],
                torch.tensor([0, max_range, 0, max_range + self.crop_size, self.crop_size])[None],
                torch.tensor([0, max_range, max_range, max_range + self.crop_size, max_range + self.crop_size])[None]
               ]
        for i in range(self.patch_size - 4):
            ind_h, ind_w = np.random.randint(0, max_range, size=2)
            boxes.append(torch.tensor([0, ind_w, ind_h, ind_w + self.crop_size, ind_h + self.crop_size])[None])
        boxes = torch.cat(boxes, dim=0)

        return boxes  # K, 5

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, 'JPEGImages', image))
            img_labels.append(labels)
        return img_name_list, img_labels  # , np.array(img_labels, dtype=np.float32)

def train_l2g_crop_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    img_train = VOCDataset_l2g_crop(args.train_list, root_dir=args.img_dir, input_size=input_size, crop_size=crop_size,
                                   num_classes=args.num_classes, transform=tsfm_train, test=False, patch_size=args.patch_size)
    img_test = VOCDataset_l2g(args.test_list, root_dir=args.img_dir, input_size=input_size, crop_size=crop_size,
                              num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

## Enable multi process
class VOCDatasetMSF_l2g_MP(Dataset):
    def __init__(self, datalist_file, root_dir, scales=(0.5, 1.0, 1.5, 2.0), num_classes=20, transform=None,
                 test=False, total_process=4, process_id=0):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file = datalist_file
        self.scales = scales
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        split_num = len(self.image_list)//total_process
        if process_id == total_process - 1:
            self.image_list = self.image_list[split_num * process_id:]
            self.label_list = self.label_list[split_num * process_id:]
        else:
            self.image_list = self.image_list[split_num * process_id:split_num * (1+process_id)]
            self.label_list = self.label_list[split_num * process_id:split_num * (1+process_id)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        ms_img_list = []
        for s in self.scales:
            target_size = (int(round(image.size[0] * s)),
                           int(round(image.size[1] * s)))
            s_img = image.resize(target_size, resample=Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform is not None:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))

        if self.testing:
            return img_name, msf_img_list, self.label_list[idx]

        return msf_img_list, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels  # np.array(img_labels, dtype=np.float32)

def test_l2g_data_loader_mp(args, ms=True, process_id=0, process_num=4):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    if ms:
        tsfm_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean_vals, std_vals),
                                        ])
        # multi scale inference
        img_test = VOCDatasetMSF_l2g_MP(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes,
                                     transform=tsfm_test, test=True, process_id=process_id, total_process=process_num)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        tsfm_test = transforms.Compose([transforms.Resize(args.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_vals, std_vals),
                                        ])
        # single scale inference
        img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test,
                              test=True)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader