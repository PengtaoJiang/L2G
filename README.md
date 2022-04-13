# Local to Global
The Official PyTorch code for ["L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation"](https://arxiv.org/abs/2204.03206), which is implemented based on the code of [OAA-PyTorch](https://github.com/PengtaoJiang/OAA-PyTorch). 
The segmentation framework is borrowed from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch).

## Installation
Use the following command to prepare your enviroment.
```
pip install -r requirements.txt
```

Download the PASCAL VOC dataset and MS COCO dataset, respectively. 
- [PASCAL VOC 2012](https://drive.google.com/file/d/1jnHE6Sau0tHI7X6JQKhzHov-vseYbrf9/view?usp=sharing)
- [MS COCO 2014](https://cocodataset.org/#home)  

L2G uses the off-the-shelf saliency maps generated from PoolNet. Download them and move to a folder named **Sal**.
- [Saliency maps for PASCAL VOC 2012](https://drive.google.com/file/d/1ZBLZ3YFw6yDIRWo0Apd4znOozg-Buj4A/view?usp=sharing)
- [Saliency maps for MS COCO 2014](https://drive.google.com/file/d/1IN6qQK0kL4_x8yzF7jvS6hNIFXsrR6XV/view?usp=sharing)  

The data folder structure should be like:
```
L2G
├── models
├── scripts
├── utils
├── data
│   ├── voc12
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── SegmentationClassAug
│   │   ├── ImageSets
│   │   ├── Sal
│   ├── coco14
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── ImageSets
│   │   ├── Sal

```
Download the [pretrained model](https://drive.google.com/file/d/15F13LEL5aO45JU-j45PYjzv5KW5bn_Pn/view) 
to initialize the classification network and put it to `./models/`.

## L2G
To train a L2G model on dataset VOC2012, you need to implement the following commands:
```
cd L2G/
./train_l2g_sal_voc.sh 
```
For COCO:
```
cd L2G/
./train_l2g_sal_coco.sh 
```
We provide the pretrained classification models on PASCAL VOC and MS COCO, respectively.
- [Pretrained models for VOC](https://drive.google.com/file/d/1Yc-LZ4bTM_1arpPBId6CMP9I2gOrDkdi/view?usp=sharing)
- [Pretrained models for COCO](https://drive.google.com/file/d/1i3b35g4GJO448kVdibBa5aL-yG6G2Huc/view?usp=sharing)  

After the training process, you will need the following command to generate pseudo labels 
and check their qualities.   
For VOC:
```
./test_l2g_voc.sh
```
For COCO:
```
./test_l2g_coco.sh
```
## Weakly Supervised Segmentation
To train a segmentation model, you need to generate pseudo segmentation labels first by 
```
./gen_gt_voc.sh
```
This code will generate pseudo segmentation labels in './data/voc12/pseudo_seg_labels/'.  
For COCO
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
Dataset | mIoU(val) | mIoU (test)  
--- |:---:|:---:
PASCAL VOC  | 72.1 | 71.7
MS COCO     | 44.2 | ---


If you have any question about L2G, please feel free to contact [Me](https://pengtaojiang.github.io/) (pt.jiang AT mail DOT nankai.edu.cn). 

## Citation
If you use our codes and models in your research, please cite:
```
@inproceedings{jiang2022l2g,
  title={L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation},
  author={Jiang, Peng-Tao and Yang, Yuqi and Hou, Qibin and Wei, Yunchao},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2022}
}
```

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for NonCommercial use only. Any commercial use should get formal permission first.
 
## Acknowledgement
Some parts of this code are borrowed from a nice work, [EPS](https://github.com/PengtaoJiang/EPS).
