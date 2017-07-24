import os
import glob
import argparse
import numpy as np
import config as cfg
from numpy.testing import assert_array_almost_equal

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes', type=int)
    args = parser.parse_args()

    return args

args = parse_args()
DOMAIN = args.domain
FEATURE_PATH = os.path.join(cfg.FEATURE_PATH, 'feature_' + DOMAIN + '_' + str(args.n_boxes) + 'boxes/')
print FEATURE_PATH
LABEL_PATH = './parsed_label/' + DOMAIN + '/'

# test training batches
batch_paths = glob.glob('batch_data/' + DOMAIN + '/train/batch_clips/*.npy')
n_batches = len(batch_paths)

FEATUREs = [
    'hof', 'divide_area_pruned_boxes', 'pruned_roisavg', 'roislist', 'roisavg',
            'label', 'onehot', 'avg_motion', 'avg_flow', 'motion_inclusion']

for _ in xrange(5):
    batch_id = np.random.randint(1, n_batches + 1)
    print "Testing batch_{}".format(batch_id)
    batch = {}

    batch_clips = np.load('batch_data/' + DOMAIN + '/train/batch_clips/batch_{}.npy'.format(batch_id))

    for feature in FEATUREs:
        batch[feature] = np.load('batch_data/' + DOMAIN + '/train/' + feature + \
                                 '/batch_{}.npy'.format(batch_id))

    for (i, clip) in enumerate(batch_clips):
        name, part = clip.rsplit('_', 1)
        correct = {}
        for feature in FEATUREs:
            if feature == 'label':
                correct[feature] = np.load(LABEL_PATH + name + '/label' + str(part) + '.npy')
            else:
                correct[feature] = np.load(FEATURE_PATH + name + '/' + feature + str(part) + '.npy')

        for feature in FEATUREs:
            assert_array_almost_equal(batch[feature][i], correct[feature], 2)

print "Congrats! Generated batches are correct!"
