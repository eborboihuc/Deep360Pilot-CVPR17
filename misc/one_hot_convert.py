import os
import sys
import time
import glob
import argparse
import numpy as np
import config as cfg

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='basic_LSTM')
    parser.add_argument('-m', '--mode', dest='mode', help='convert [data] or [batch] path, or [test] its function, and also [copy] to path of yenchen', default='test')
    parser.add_argument('-d', '--domain', dest='domain', help='skate, skiing, ...', required=True)
    parser.add_argument('-t', '--times', dest='times', help='how many times we generate testcase, default = 1', default=1, type=int)
    parser.add_argument('-s', '--size', dest='size', help='upper bound range of testcase, default = 1.0', default=1.0, type=float)
    parser.add_argument('-b', '--boxnum', dest='boxnum', help='Number of boxes, [8, 16, 32]', required=True, type=int)
    args = parser.parse_args()

    return args

args = parse_args()

# Path
domain = args.domain

# Scale
W = 1920.0
H = 960.0

# Parameters
n_frames = 50
n_detection = args.boxnum
n_classes = 2

# Flag
DEBUG = False #True

# For data
label_path = os.path.join(cfg.FEATURE_PATH, 'label_' + domain + '/')
fea_path = os.path.join(cfg.FEATURE_PATH, 'feature_' + domain + '_' + str(n_detection) + 'boxes/')
out_prefix = 'onehot' 

# For batch
train_path = os.path.join(cfg.BATCH_PATH, 'train_' + domain + '_' + str(n_detection) + 'boxes/')
test_path = os,path.join(cfg.BATCH_PATH, 'test_' + domain + '_' + str(n_detection) + 'boxes/')
label_suffix = 'label/'
batch_out_suffix = 'one_hot_label/'

train_num = len(glob.glob(train_path+'roisavg/*.npy'))
test_num = len(glob.glob(test_path+'roisavg/*.npy'))

print train_num, test_num


def nearest_box(yloc, boxes):
    """ Generate one hot y that one indicates the nearset box """
    """ yloc (50, 2), boxes (50, n_detection, 2) """
    y_dist = l2_dist_360(boxes, yloc[:,np.newaxis,:], 2, W) 
    y_true = np.argmin(y_dist, 1).flatten()
    y_one_hot = convert_to_one_hot(y_true, n_detection)
    
    if DEBUG:
        print y_one_hot[0]
        print y_dist[0].flatten()
        print np.argmin(y_dist[0].flatten())

    return y_one_hot


def test(times=1, size=1.0):
    """ Test """
    label = np.random.rand(times,2)
    boxes = np.random.rand(times,n_detection,2)
    if size != 1.0:
        label[:,0] = label[:,0]*size
        label[:,1] = label[:,1]*size/2
        boxes[:,:,0] = boxes[:,:,0]*size
        boxes[:,:,1] = boxes[:,:,1]*size/2

    one_hot = nearest_box(label, boxes)
    for i in xrange(times):
        loc = np.argmin(np.sum(abs(label[i]-boxes[i]),1))
        onehot_loc = np.where(one_hot[i])
        print label[i]
        print boxes[i]
        print loc
        print one_hot[i]
        print "Pass" if loc == onehot_loc else "Fail"


def data_convert():
    """ data iteration wrapper """
    if os.path.isdir(fea_path) == False:
        os.mkdir(fea_path)

    for dataname in sorted(os.listdir(label_path)):
        data_y, data_box = load_data(dataname)
        
        if data_y is None or data_box is None:
            continue
        
        # Skip the last one since we don't use it in yenchen's code
        for i in xrange(data_y.shape[0]/50):
            idx = i*50
            print idx, data_y.shape[0], data_box.shape[0]
            data_one_hot = nearest_box(data_y[idx:idx+50], data_box[idx:idx+50])
            npy_save(fea_path + dataname + '/' +  out_prefix + str(i+1).zfill(4), data_one_hot)   


def batch_convert(num, batch_path):
    """ batch iteration wrapper """
    batch_out_path = batch_path + batch_out_suffix
    
    if os.path.isdir(batch_out_path) == False:
        os.mkdir(batch_out_path)
    
    for batch in xrange(1,num+1):
        batch_y, batch_box = load_batch_data(batch_path, batch)
        batch_one_hot = np.zeros((batch_y.shape[0], batch_y.shape[1], n_detection))
        for i in xrange(batch_y.shape[0]):
            batch_one_hot[i] = nearest_box(batch_y[i], batch_box[i])
        
        npy_save(batch_out_path + 'batch_' + str(batch), batch_one_hot)   


def load_data(dirname):
    """ Load data from path """
    # label part
    filename = os.listdir(label_path + dirname)
    assert filename != [], "Empty dir at {}".format(dirname)
    assert len(filename) == 1, "More than one label at {}/{}".format(dirname,filename)
    filename = filename[0]
    labels = np.load(label_path + dirname + '/' + filename)

    # box part
    #assert os.path.isdir(fea_path + dirname), "Not batch path: {}".format(fea_path + dirname)
    if not os.path.isdir(fea_path + dirname): print "Not batch path: {}".format(fea_path + dirname); return labels, None
    
    boxnames = sorted(os.listdir(fea_path + dirname))
    boxes = np.empty((0,n_detection,4))
    for boxname in boxnames:
        if boxname.startswith('roislist'):
            boxes = np.vstack((boxes, np.load(fea_path + dirname + '/' + boxname)))
    
    # boxes [xmin, ymin, xmax, ymax] -> boxes [xcenter, ycenter]
    boxes[:,:,0] = (boxes[:,:,0] + boxes[:,:,2])/2
    boxes[:,:,1] = (boxes[:,:,1] + boxes[:,:,3])/2

    # labels [x, y, flag] -> labels [x, y]
    return labels[:,:2], boxes[:,:,:2]


def load_batch_data(path, num_batch):
    """ Load batch data from path and num """
    labels = np.load(path + 'label/batch_' + str(num_batch)+'.npy')
    boxes = np.load(path + 'roislist/batch_' + str(num_batch)+'.npy')

    # boxes [xmin, ymin, xmax, ymax] -> boxes [xcenter, ycenter]
    boxes[:,:,:,0] = (boxes[:,:,:,0] + boxes[:,:,:,2])/2
    boxes[:,:,:,1] = (boxes[:,:,:,1] + boxes[:,:,:,3])/2

    # labels [x, y, flag] -> labels [x, y]
    return labels[:,:,:2], boxes[:,:,:,:2]


def catData(totalData, newData):
    """ Concat data from scratch """
    if totalData is None:
        totalData = newData[np.newaxis].copy()
    else:
        totalData = np.concatenate((totalData, newData[np.newaxis]))
    
    return totalData


def convert_to_one_hot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    
    return result.astype(int)


def l2_dist_360(pA, pB, _axis=2, W=1920.0):
    """ pA is Nx1x2 , pB is Nxn_detectionx2, this will return a n_detectionx1 L2-norm in 360 video form """
    dist = (pA-pB).astype(np.float64)
    xdist, ydist = np.split(dist, 2, axis=_axis)
    xdist[xdist>W/2] = -W + xdist[xdist>W/2]
    xdist[xdist<-W/2] = -W - xdist[xdist<-W/2]
    
    return np.sqrt((np.square(xdist)+np.square(ydist)))


def npy_save(filename, arr):
    """ save arr in filename """
    assert arr is not None, "Array is empty."
    
    filename = filename if filename.endswith('.npy') else filename + '.npy'
    print "Saving {}".format(filename)
    if not DEBUG: np.save(filename, arr)


if __name__ == '__main__':

    if args.mode.lower() == 'batch':
        batch_convert(train_num, train_path)       
        batch_convert(test_num, test_path)
    elif args.mode.lower() == 'data':
        data_convert()
    elif args.mode.lower() == 'test':
        test(args.times, args.size)
    else:
        print "Wrong mode choice"
