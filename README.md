
# Deep 360 Pilot in Tensorflow

Tensorflow implementation of CVPR 2017 Oral paper "Deep 360 Pilot: Learning a Deep Agent for Piloting through 360◦ Sports Videos".

# Prerequisites

- Linux
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- Python 2.7 with numpy
- [Tensorflow](https://www.tensorflow.org/) 0.12.1


## Getting Started
- Clone this repo:
```bash
git clone git@github.com:eborboihuc/Deep360Pilot.git
cd Deep360Pilot
```



# Usage
To train a model with downloaded dataset:
```bash
python main.py --mode train --gpu 0 -d bmx -l 10 -b 16 -p classify --opt Adam
```

To test with an existing model:
```bash
python main.py --mode test --gpu 0 -d bmx -l 10 -b 16 -p classify --model bmx_lam1_classify_best_model
```

To get prediction with an existing model:
```bash
python main.py --mode pred --model bmx_lam10.0_regress_best_model --gpu 0 -d bmx -l 10 -b 16 -p regress -n zZ6FlZRLvek_6

## Author

Hou-Ning Hu / [@eborboihuc](https://eborboihuc.github.io/) and Yen-Chen Lin / [@yenchenlin](https://yclin.me)
