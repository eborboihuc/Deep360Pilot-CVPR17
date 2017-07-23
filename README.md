
# Deep 360 Pilot in Tensorflow

Official Implementation of CVPR 2017 Oral paper "Deep 360 Pilot: Learning a Deep Agent for Piloting through 360◦ Sports Videos" in Tensorflow.

![](https://user-images.githubusercontent.com/7057863/28415179-980e0d34-6d1c-11e7-87ae-8d190f7cdd2f.gif)

# Prerequisites

- Linux
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- Python 2.7 with numpy
- [Tensorflow](https://www.tensorflow.org/) 0.12.1


## Getting Started
- Clone this repo:
```bash
git clone git@github.com:eborboihuc/Deep360Pilot.git
cd Deep360Pilot/misc

git clone git@github.com:yenchenlin/Deep360Pilot-optical-flow.git
```

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

## Author

Hou-Ning Hu / [@eborboihuc](https://eborboihuc.github.io/) and Yen-Chen Lin / [@yenchenlin](https://yclin.me)
