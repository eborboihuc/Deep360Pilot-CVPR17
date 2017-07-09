#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hou-Ning Hu

import os
import argparse
import numpy as np
import tensorflow as tf
from model import Deep360Pilot
from train import train
from test import test
from demo import video_base

# Usage:
# python main.py --mode train --gpu 0 -d bmx -l 10 -b 16 -p classify --opt Adam
# python main.py --mode train -d bmx -l 10 -b 16 -p regress --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model
# python main.py --mode test -d bmx -l 10 -b 16 -p classify --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model
# python main.py --mode vid --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model -d bmx -l 10 -b 16 -p classify -n zZ6FlZRLvek_6
# python main.py --mode pred --model checkpoint/parkour_16boxes_lam10.0/parkour_lam1_classify_best_model -d bmx -l 10 -b 16 -p classify -n zZ6FlZRLvek_6 --data ./data2/

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Deep360Pilot')
    parser.add_argument('--opt', dest='opt_method', help='[Adam, Adadelta, RMSProp]', default='Adam')
    parser.add_argument('--root', dest='root_path', help='root path of data', default='./')
    parser.add_argument('--data', dest='data_path', help='data path of data', default='./data/')
    parser.add_argument('--mode', dest='mode', help='[train, test, vid, pred]', required=True)
    parser.add_argument('--model', dest='model_path', help='model path to load')
    parser.add_argument('--gpu', dest='gpu', help='Choose which gpu to use', default='0')
    parser.add_argument('-n', '--name', dest='video_name', help='youtube_id + _ + part')
    parser.add_argument('-d', '--domain', dest='domain', help='skate, skiing, ...', required=True)
    parser.add_argument('-l', '--lambda', dest='lam', help='movement tradeoff lambda, the higher the smoother.', type=float, required=True)
    parser.add_argument('-b', '--boxnum', dest='boxnum', help='boxes number, Use integer, [8, 16, 32]', type=int, required=True)
    parser.add_argument('-p', '--phase', dest='phase', help='phase [classify, regress]', required=True)
    parser.add_argument('-s', '--save', dest='save', help='save images for debug', default=False)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--debug', dest='debug', help='Start debug mode or not', action='store_true')

    args = parser.parse_args()

    return args, parser


if __name__ == '__main__':

    args, parser = parse_args()
    
    # Setup visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Init Deep 360 Pilot
    Agent = Deep360Pilot(args)
    
    # Switch Modes
    if args.mode == 'train':
        train(Agent)
    elif args.mode in ['vid', 'pred']:
        video_base(Agent, args.domain, args.video_name)
    elif args.mode == 'test':
        test(Agent)
    else:
        parser.print_help()
