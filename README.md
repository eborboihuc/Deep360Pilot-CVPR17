
# Deep 360 Pilot: Learning a Deep Agent for Piloting through 360° Sports Video

Hou-Ning Hu\*, Yen-Chen Lin\*, Ming-Yu Liu, Hsien-Tzu Cheng, Yung-Ju Chang, Min Sun
(\*indicate equal contribution)

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017 (oral presentation)

Official Implementation of CVPR 2017 Oral paper "Deep 360 Pilot: Learning a Deep Agent for Piloting through 360◦ Sports Videos" in Tensorflow.

![](https://user-images.githubusercontent.com/7057863/28415179-980e0d34-6d1c-11e7-87ae-8d190f7cdd2f.gif)

Project page: [https://aliensunmin.github.io/project/360video/](https://aliensunmin.github.io/project/360video/)

Paper: [High resolution](https://drive.google.com/file/d/0B2dg5RanEUBQRkJYZDc1Mmh2bmM/view), [ArXiv pre-print](https://arxiv.org/abs/1705.01759), [Open access](http://openaccess.thecvf.com/content_cvpr_2017/html/Hu_Deep_360_Pilot_CVPR_2017_paper.html)

# Prerequisites

- Linux
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- Python 2.7 with numpy
- [Tensorflow](https://www.tensorflow.org/) 0.12.1


## Getting Started
- Change the version:

  We provide both `0.12` and `1.2.1` version of Tensorflow implementation
You may choose the ideal version to use

- Clone [this](https://github.com/eborboihuc/Deep360Pilot-CVPR17) repo and [another](https://github.com/yenchenlin/Deep360Pilot-optical-flow) for formating the input data:
```bash
git clone git@github.com:eborboihuc/Deep360Pilot-CVPR17.git
cd Deep360Pilot/misc

git clone git@github.com:yenchenlin/Deep360Pilot-optical-flow.git
```
- Download our [dataset](#dataset) and [pre-trained model](#pre-trained-model)

After run the scripts
```bash
python require.py
```
You can download our model and dataset and place it under `./checkpoint` and `./data`, respectively.


# Usage
To train a model with downloaded dataset:
```bash
python main.py --mode train --gpu 0 -d bmx -l 10 -b 16 -p classify --opt Adam
```
Then
```bash
python main.py --mode train --gpu 0 -d bmx -l 10 -b 16 -p regress --opt Adam --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model
```

To test with an existing model:
```bash
python main.py --mode test --gpu 0 -d bmx -l 10 -b 16 -p classify --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model
```
Or,
```bash
python main.py --mode test --gpu 0 -d bmx -l 10 -b 16 -p regress --model checkpoint/bmx_16boxes_lam10.0/bmx_lam10.0_regress_best_model
```

To get prediction with an existing model:
```bash
python main.py --mode pred --model checkpoint/bmx_16boxes_lam10.0/bmx_lam10.0_regress_best_model --gpu 0 -d bmx -l 10 -b 16 -p regress -n zZ6FlZRLvek_6
```

## Pre-trained Model
Please download the trained model [here](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjNWdFbnVYbG9kNm8).
You can use `--model {model_path}` in `main.py` to load the model. 

## Dataset
If you want to test on our dataset, please download the dataset [here](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjWnF3LV9WUXdZMzA) and place it under `./data`.
You can use code from [here](https://github.com/yenchenlin/Deep360Pilot-optical-flow) to convert it to our input format.

## Author
Hou-Ning Hu / [@eborboihuc](https://eborboihuc.github.io/) and Yen-Chen Lin / [@yenchenlin](https://yclin.me)

## Cite
If you find our code useful for your research, please cite
```bibtex
@InProceedings{Hu_2017_CVPR,
author = {Hu, Hou-Ning and Lin, Yen-Chen and Liu, Ming-Yu and Cheng, Hsien-Tzu and Chang, Yung-Ju and Sun, Min},
title = {Deep 360 Pilot: Learning a Deep Agent for Piloting Through 360deg Sports Videos},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```
