import argparse
import os

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    args = parser.parse_args()

    return args

args = parse_args()
DIRs = ['train', 'test']
FEATUREs = ['hof', 'divide_area_pruned_boxes', 'onehot',
            'pruned_roisavg', 'roisavg', 'roislist', 'label',
            'batch_clips']
for dir in DIRs:
    for feature in FEATUREs:
        dir_path = args.domain + '/' + dir + '/' + feature
        print dir_path
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
