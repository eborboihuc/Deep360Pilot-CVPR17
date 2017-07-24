import glob
import os
import argparse
import numpy as np
import config as cfg

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes')
    args = parser.parse_args()

    return args

args = parse_args()

DOMAIN = args.domain
metadata = np.load('metadata/metadata_' + DOMAIN + '.npy').item()
NAMEs = sorted(metadata.keys())

n_boxes = args.n_boxes
LABEL_PATH = cfg.FEATURE_PATH + 'label_' + DOMAIN + '/'
FEATURE_PATH = cfg.FEATURE_PATH + 'feature_' + \
    DOMAIN + '_' + str(n_boxes) + 'boxes/'
LEN_CLIP = 50

def parse_raw_label():
	for NAME in NAMEs:
		print NAME
		# feature path for each video
		feature_path = FEATURE_PATH + NAME + '/'

		info = metadata[NAME]
		n_clips = int( np.ceil(info['n_frames'] / float(LEN_CLIP)) )

		# verify clips number
		roislist_paths = glob.glob(feature_path + 'roislist*.npy')
		assert n_clips == len(roislist_paths)

		# raw label path for each video
		label_paths = glob.glob(LABEL_PATH + NAME + '/*.npy')

		# assume we only have one label for each video
		assert len(label_paths) == 1
		label = np.load(label_paths[0])

		# label should have shape (n_frames, 3)
		assert info['n_frames'] == len(label)
		n_frames = len(label)

		# split label into size of LEN_CLIP for each clip
		for i in xrange(n_clips):
			start = i * LEN_CLIP

			# check if dir exists or create a new dir for each video
			output_path = 'parsed_label/' + DOMAIN + '/' + NAME + '/'
			if not os.path.isdir(output_path):
				os.mkdir(output_path)

			# ex: label0001, label0002, ...
			filename = output_path + 'label' + '{:04d}'.format(i+1)

			# if is last clip & need padding
			if i == n_clips - 1 and n_frames % LEN_CLIP != 0:
				# label for each clip has size (50, 3)
				padded_labels = np.zeros((50, 3))
				print
				padded_labels[:(n_frames % LEN_CLIP)] = label[start:]
				np.save(filename, padded_labels)
			else:
				np.save(filename, label[start : start + LEN_CLIP])

if __name__ == '__main__':
	parse_raw_label()
