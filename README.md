# Monash Guns Dataset
The Monash Guns Dataset journal paper is currently being peer reviewed. 

The full dataset and model will be released here upon completion of review.

## Conference paper
A brief summary of our dataset and model was recently presented at the 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA-ASC).

The title of the conference paper: *Gun Detection in Surveillance Videos using Deep Neural Networks* [Link to paper](https://marcuslimjunyi.github.io/papers/Gun%20Detection%20in%20Surveillance%20Videos%20using%20Deep%20Neural%20Networks.pdf)

## Sample Training Images in our Dataset
<img src="/images/pistol_1.jpg" width="200"> <img src="/images/pistol_1268.jpg" width="200"> <img src="/images/pistol_1476.jpg" width="200"> <img src="/images/pistol_1511.jpg" width="200"> <img src="/images/pistol_1574.jpg" width="200"> <img src="/images/pistol_1659.jpg" width="200"> <img src="/images/pistol_1931.jpg" width="200"> <img src="/images/pistol_2023.jpg" width="200"> <img src="/images/pistol_2038.jpg" width="200"> <img src="/images/pistol_2078.jpg" width="200"> <img src="/images/pistol_2730.jpg" width="200"> <img src="/images/pistol_318.jpg" width="200">

### Details of the dataset:
1) All images are scaled down from 1920x1080 pixels to 512x512 for training and testing purposes.
2) Data format and annotations follow PASCAL VOC format

## Proposed Gun Detector
### M2Det
In our paper, we employ M2Det with Focal Loss to address foreground-background class imbalance during training.

All credits of M2Det go to the authors Qijie Zhao, Tao Sheng, Yongtao Wang, Zhi Tang, Ying Chen, Ling Cai and Haibing Ling.

AAAI2019 "M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network"[Paper Link](https://qijiezhao.github.io/imgs/m2det.pdf)

Github: https://github.com/qijiezhao/M2Det

