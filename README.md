# Local to Global
The Official PyTorch code for ["L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation"](https://arxiv.org/abs/2204.03206), which is implemented based on the code of [OAA-PyTorch](https://github.com/PengtaoJiang/OAA-PyTorch). 
The segmentation framework is borrowed from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch).

## Installation
Use the following command to prepare your enviroment.
```
pip install -r requirements.txt
```

Download the [PASCAL VOC dataset](https://drive.google.com/file/d/1jnHE6Sau0tHI7X6JQKhzHov-vseYbrf9/view?usp=sharing) and MS COCO dataset, respectively.

## L2G
Before training your L2G model, you need to check whether you set the correct path to dataset in train_l2g_sal_voc.sh:
```
...
CUDA_VISIBLE_DEVICES=1 python3 ./scripts/${RUN_FILE} \
    --img_dir=[PATH] \
...
```
[PATH] is the path to your VOC2012 dataset or COCO dataset.

Our code also need to take use of off-the-shelf saliency maps. Copy them to a folder named **saliency_aug** in your VOC2012 dataset. 
Take VOC2012 as an example, the folder structure should be like:
```
VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── SegmentationClass_Aug
│   │   │   ├── saliency_aug
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
```

To train a L2G model on dataset VOC2012, you need to:
```
cd L2G/
./train_l2g_sal_voc.sh 
```
And the same for COCO:
```
cd L2G/
./train_l2g_sal_coco.sh 
```
After the training process, you will need the following command to generate pseudo label and check their quality:
```
./test_l2g.sh
```

## Weakly Supervised Segmentation
To train a segmentation model, you need to generate pseudo segmentation labels first by 
```
./gen_gt_voc.sh
```
This code will generate pseudo segmentation labels in './data/VOCdevkit/VOC2012/pseudo_seg_labels/'.
For coco, it should be
```
./gen_gt_coco.sh
```


Then you can train the [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) model as follows:  
```
cd deeplab-pytorch
bash scripts/setup_caffemodels.sh
python convert.py --dataset coco
python convert.py --dataset voc12
```
Train the segmentation model by
```
python main.py train \
      --config-path configs/voc2012.yaml
```
Test the segmentation model by 
```
python main.py test \
    --config-path configs/voc12.yaml \
    --model-path data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
```
Apply the crf post-processing by 
```
python main.py crf \
    --config-path configs/voc12.yaml
```
## Performance
Method |mIoU(val) | mIoU (test)  
--- |:---:|:---:
OAA(VOC)  | 72.1 | 71.7
OAA(COCO) | 44.2 | ---


If you have any question about L2G, please feel free to contact [Me](https://pengtaojiang.github.io/) (pt.jiang AT mail DOT nankai.edu.cn). 

## Citation
If you use these codes and models in your research, please cite:


## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for NonCommercial use only. Any commercial use should get formal permission first.
  
