import os
import glob
import argparse
import numpy as np
import config as cfg
from random import shuffle
from joblib import Parallel, delayed

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes', type=int)
    args = parser.parse_args()

    return args

args = parse_args()
DOMAIN = args.domain

metadata = np.load('./metadata/metadata_' + DOMAIN + '.npy').item()
NAMEs = metadata.keys()

LABEL_PATH = './parsed_label/' + DOMAIN + '/'
FEATURE_PATH = cfg.FEATURE_PATH + 'feature_' \
    + DOMAIN + '_' + str(args.n_boxes) + 'boxes/'
LEN_CLIP = 50

clips = {'train': [], 'test': []}
for name in NAMEs:
    info = metadata[name]
    video_id = info['id']
    video_type = 'train' if info['type'] == 'training' else 'test'

    n_clips = int(np.ceil(info['n_frames'] / float(LEN_CLIP))) - 1 # drop last 50 frames since it may contain null

    # verify the correctness of n_clips
    feature_path = FEATURE_PATH + name + '/'
    roisavg_paths = glob.glob(feature_path + 'roisavg*.npy')
    assert n_clips == len(roisavg_paths) - 1, "{}, {}, {}".format(n_clips, len(roisavg_paths), roisavg_paths)

    for i in xrange(1, n_clips + 1):
        clips[video_type].append('{}_{:04d}'.format(name, i))

shuffle(clips['train'])
shuffle(clips['test'])

batch_size = 10
n_boxes = args.n_boxes

def gen_batch(i):
    print "Generating " + video_type + " batch {} / {}".format(i, n_batches)
    start = (i - 1) * batch_size

    batches = {'hof': np.zeros((10, 50, n_boxes, 12), dtype=np.float16),
               'divide_area_pruned_boxes': np.zeros((10, 50, n_boxes, 4), dtype=np.float16),
               'pruned_roisavg': np.zeros((10, 50, n_boxes, 512), dtype=np.float16),
               'roislist': np.zeros((10, 50, n_boxes, 4), dtype=np.float16),
               'roisavg': np.zeros((10, 50, n_boxes, 512), dtype=np.float16),
               'label': np.zeros((10, 50, 3), dtype=np.float16),
               'onehot': np.zeros((10, 50, n_boxes), dtype=np.float16),
               }
    
    # make dir

    batch_clips = []
    for (j, clip) in enumerate(clips[video_type][start:start + batch_size]):
        name, part = clip.rsplit('_', 1)
        batch_clips.append(clip)

        for feature in batches.keys():
            if feature == 'label':
                path = LABEL_PATH + name + '/label' + str(part) + '.npy'
            else:
                path = FEATURE_PATH + name + '/' + feature + str(part) + '.npy'

            if feature == 'conv5_':
                batches[feature][j] = np.load(path).reshape(50, 10 * 20, 512)
            else:
                batches[feature][j] = np.load(path)

    for feature in batches.keys():
        if feature == 'conv5_':
            np.save('batch_data/' + DOMAIN + '/' + video_type + '/conv5' + '/batch_{}'.format(i), batches[feature])
        else:
            np.save('batch_data/' + DOMAIN + '/' + video_type + '/' + feature + '/batch_{}'.format(i), batches[feature])

    np.save('batch_data/' + DOMAIN + '/' + video_type + '/batch_clips/batch_{}'.format(i), batch_clips)

# generating training / testing batch
for video_type in ['train', 'test']:
    n_clips = len(clips[video_type])
    n_batches = n_clips / 10
    print "n_" + video_type + "_clips", n_clips
    print "n_" + video_type + "_batches", n_batches

    # NOTE: batch start from 1
    Parallel(n_jobs=5)(delayed(gen_batch)(i) for i in xrange(1, n_batches + 1))
