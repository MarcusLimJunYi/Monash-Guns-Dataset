# Updating repo now, apologize for the delay
Hold your horses fellow researchers i'm currently updating the repo now and will release the dataset within the coming days. Hey, it feels great to have your first Q1 journal paper accepted, why not chill abit ayy. Just kidding, im actually working on a second journal paper to be submitted coming February hence the delay. No worries folks, im workin on it now and open to assist you fellas on any bugs/fixes. 
  
## Monash Guns Dataset
Our journal paper, "Deep multi-level feature pyramids: Application for non-canonical firearm detection in video surveillance" has been accepted by EAAI and can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0952197620303456). 

## Conference paper
A brief summary of our dataset and model was recently presented at the 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA-ASC).

The title of the conference paper: *Gun Detection in Surveillance Videos using Deep Neural Networks* [Link to paper](https://marcuslimjunyi.github.io/papers/Gun%20Detection%20in%20Surveillance%20Videos%20using%20Deep%20Neural%20Networks.pdf)

## Sample Training Images in our Dataset
<img src="/images/pistol_1.jpg" width="200"> <img src="/images/pistol_1268.jpg" width="200"> <img src="/images/pistol_1476.jpg" width="200"> <img src="/images/pistol_1511.jpg" width="200"> <img src="/images/pistol_1574.jpg" width="200"> <img src="/images/pistol_1659.jpg" width="200"> <img src="/images/pistol_1931.jpg" width="200"> <img src="/images/pistol_2023.jpg" width="200"> <img src="/images/pistol_2038.jpg" width="200"> <img src="/images/pistol_2078.jpg" width="200"> <img src="/images/pistol_2730.jpg" width="200"> <img src="/images/pistol_318.jpg" width="200">

### Details of the dataset:
1) All images are scaled down from 1920x1080 pixels to 512x512 for training and testing purposes.
2) Data format and annotations follow the PASCAL VOC format.

## Getting Started
- Prepare python environment using [Anaconda3](https://www.anaconda.com/download/).
- Install deeplearning framework, i.e., pytorch, torchvision and other libs.

```Shell
conda install pytorch==1.7.1 torchvision -c pytorch
pip install opencv-python tqdm
```
- Clone this repository.
```Shell
git clone https://github.com/MarcusLimJunYi/Monash-Guns-Dataset.git
```
- Compile the nms and coco tools:

```Shell
sh make.sh
```

## Dataset Preparation
Download the [MGD]() dataset. Organize them in `dataset` folder as follows:

    ~~~
    |-- dataset/
    |   |-- MGD/
    |       |-- MGD2020
    |           |-- Annotations
    |           |-- ImageSets
    |           |-- JPEGImages
    ~~~

## Training and Testing
### Training

As simple as [demo](#Demo) and [evaluation](#Evaluation), Just use the train script:
```Shell
  CUDA_VISIBLE_DEVICES=0,1 python train.py -c=configs/m2det512_resnet101.py --ngpu 2 -t True
```
All training configs and model configs are written well in configs/*.py.

### Test
We provide evaluation script for M2Det:
```Shell
  python test.py -c=configs/m2det512_resnet101.py -m=weights/MGD_GIoU_FL_FFMv3.pth
```

## Citation
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

~~~
@article{lim97deep,
  title={Deep multi-level feature pyramids: Application for non-canonical firearm detection in video surveillance},
  author={Lim, JunYi and Al Jobayer, Md Istiaque and Baskaran, Vishnu Monn and Lim, Joanne MunYee and See, John and Wong, KokSheik},
  journal={Engineering Applications of Artificial Intelligence},
  volume={97},
  pages={104094},
  publisher={Elsevier}
}
~~~

## Acknowledgement
In our paper, we employ M2Det with GIoU Loss and Focal Loss to improve bounding box regression and address foreground-background class imbalance during training.

All credits of M2Det go to the authors Qijie Zhao, Tao Sheng, Yongtao Wang, Zhi Tang, Ying Chen, Ling Cai and Haibing Ling.

AAAI2019 "M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network"[Paper Link](https://qijiezhao.github.io/imgs/m2det.pdf)

Github: https://github.com/qijiezhao/M2Det

