import csv
import numpy as np
import argparse

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', dest='domain')
    args = parser.parse_args()

    return args

args = parse_args()
DOMAIN = args.domain

metadata = {}
with open('dataset/dataset_' + DOMAIN + '.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')

	for row in spamreader:

		# skip first row
		if row[0] == 'ID':
			continue

		# csv format: 0   1      2       3
		#             ID, Video, Frames, Type
		#
		video_id = int(row[0])
		name = row[1]
		n_frames = int(row[2])
		video_type = row[3]
		metadata[name] = {'id':video_id, 'n_frames':n_frames, 'type':video_type}
np.save('metadata/metadata_' + DOMAIN, metadata)
