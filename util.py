#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import pdb
import numpy as np
from os.path import join
from ops import l2_dist_360
from MeanOverlap import MeanOverlap


def catData(totalData, newData):
    """ Concat data from scratch """
    if totalData is None:
        totalData = newData[np.newaxis].copy()
    else:
        totalData = np.concatenate((totalData, newData[np.newaxis]))
    
    return totalData


def score(Agent, seq1, seq2, _full=True):
    """ Calculate IoU """
    acc = 0.0
    total_num = 0
    MO = MeanOverlap(Agent.W, Agent.H)
    for batch in xrange(Agent.batch_size):
        for i in xrange(Agent.n_frames):
            if not _full and np.sum(seq2[batch][i]) == 0:
                continue
            acc += MO.IOU((seq1[batch, i, 0], seq1[batch, i, 1]), (seq2[batch, i, 0], seq2[batch, i, 1]))
            total_num += 1

    return (acc / total_num) if total_num != 0 else 0 #(n_frames*batch_size)


def printAcc(threshold, targetFrNum, totalFrNum):
    """ Fetch accuracy and print out """
    print "Acc is:"
    for th in threshold:
        print ("%d" %(th)),        
    print
    for i, types in enumerate(targetFrNum):
        print i if i < 4 else (i-4),
        for j, th in enumerate(threshold): 
            print ("%.5f" %(types[j] / (totalFrNum if totalFrNum > 0 else 1))),
        print


def cal_accuracy(Agent, pred, gt, targetFrNum, totalFrNum):
    """ Calculate and return accuracy """
    if np.sum(gt) == 0:
        return targetFrNum, totalFrNum
    
    l2_dist = l2_dist_360(pred, gt, Agent.W)
    l2_dist = np.tile(l2_dist,(len(Agent.threshold), 1))
    
    """ if l2_dist(10 x 50) <= thres(1 x 10), then targetFrNum(8types x 10thres) += 1 """
    thres = np.sum(l2_dist <= np.tile(np.reshape(Agent.threshold, (-1, 1)), (1, l2_dist.shape[-1])), axis=1)
    center = np.array([Agent.W/2, Agent.H/2])
    for th, i in enumerate(thres):
        if np.min(np.linalg.norm(gt - center, axis=1)) > 100: 
            targetFrNum[i,th] += 1
        else:
            targetFrNum[i+4,th] += 1

    totalFrNum += 1
    
    return targetFrNum, totalFrNum


def load_batch_data(Agent, path, num_batch, _copy=False, _augment=False):
    """ Load batch data from path and normalize them, use copy to preserve raw data """
    
    data = np.load(join(path, 'pruned_roisavg/batch_{}.npy'.format(num_batch))) #[0:Agent.batch_size,0:Agent.n_frames,0:Agent.n_detection,0:Agent.n_input]
    
    oracle_viewangle = np.load(join(path, 'label/batch_{}.npy'.format(num_batch))) #[0:Agent.batch_size,0:Agent.n_frames,0:Agent.n_classes+1]
    
    one_hot_labels = np.load(join(path, 'onehot/batch_{}.npy'.format(num_batch))) #[0:Agent.batch_size,0:Agent.n_frames,0:Agent.n_detection]
    
    hof = np.load(join(path, 'hof/batch_{}.npy'.format(num_batch))) #[0:Agent.batch_size,0:Agent.n_frames,0:Agent.n_detection,0:Agent.n_bin_size]
    
    box_center = np.load(join(path, 'divide_area_pruned_boxes/batch_{}.npy'.format(num_batch))) #[0:Agent.batch_size,0:Agent.n_frames,0:Agent.n_detection,0:]
    
    img = np.zeros((Agent.batch_size), dtype=np.float16)

    if _augment is True:
        data, oracle_viewangle, box_center = augment_data(data, oracle_viewangle, box_center)

    if _copy is True:
        box = np.copy(box_center)
        gt = np.copy(oracle_viewangle)[:, :, :2]
    else:
        box = None
        gt = None

    box_center[:,:,:,0] = (box_center[:,:,:,0]/Agent.W + box_center[:,:,:,2]/Agent.W)/2
    box_center[:,:,:,1] = (box_center[:,:,:,1]/Agent.H + box_center[:,:,:,3]/Agent.H)/2
    box_center = box_center[:, :, :, :2]
    oracle_viewangle[:,:,0] = oracle_viewangle[:,:,0]/Agent.W
    oracle_viewangle[:,:,1] = oracle_viewangle[:,:,1]/Agent.H
    oracle_viewangle = oracle_viewangle[:, :, :2]

    return data, one_hot_labels, oracle_viewangle, box_center, hof, img, box, gt


def visual_gaze(Agent, img_name, gt, pred, alphas, box):
    """
        [Deprecated]
        Draw and plot visual gaze contains boxes, gt gazes, and prediction 
    """
    print Agent.img_path + img_name + '.jpg'
    img = cv2.imread(Agent.img_path + img_name + '.jpg',3)
    
    if img is None:
        print 'No image is found.'
        return 1
    img = cv2.resize(img, (int(W),int(H)))
    
    W = Agent.W
    H = Agent.H

    # Box
    idx = 0
    transparent = 0.90
    for xmin, ymin, xmax, ymax in box.astype(np.int32):
        if xmax > W: xmax = int(W)
        if ymax > H: ymax = int(H)
        print xmin, ymin, xmax, ymax, alphas[idx]
        #if alphas[idx] > 0.0:
        cv2.rectangle(img,(xmin, ymin),(xmax, ymax), (255,255,255), 2)
        img[ymin:ymax,xmin:xmax,:] = img[ymin:ymax,xmin:xmax,:]*0.95 + np.ones((ymax-ymin,xmax-xmin,3))*0.05
        cv2.putText(img, ("{0:.2f}").format(alphas[idx]), (int((xmax+xmin)/2)+1 , int((ymax+ymin)/2)+1), cv2.FONT_HERSHEY_SIMPLEX, 1.50, (0,0,0), 2)
        cv2.putText(img, ("{0:.2f}").format(alphas[idx]), (int((xmax+xmin)/2) , int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.50, (255,255,255), 2)
        idx += 1
        
    # Predicted gaze
    ll = 3
    # Desire gaze
    color = [(255, 0, 0), (0,255,0),(0,255,255),(0,0,255)] # Green, Yellow, Red
    i = 2
    u, v = gt.astype(np.int32)
    img[v-ll:v+ll,u-ll:u+ll,1] = 255 
    cv2.circle(img,(u,v),10,color[i],2) # desize gaze centers

    xmin = u - int(W/4) if u > W/4 else 0
    xmax = u + int(W/4) if u < 3*W/4 else int(W)
    ymin = v - int(H/4) if v > H/4 else 0
    ymax = v + int(H/4) if v < 3*H/4 else int(H)

    cv2.rectangle(img,(xmin, ymin),(xmax, ymax), color[i], 2)
    img[ymin:ymax,xmin:xmax,:] = img[ymin:ymax,xmin:xmax,:]*transparent + \
                np.tile(np.array([clr for clr in color[i]])*(1-transparent),(ymax-ymin,xmax-xmin,1))
    print ("gt: ({}, {})").format(u, v)

    # Predicted gaze
    i = 0
    u, v = int(pred[0]), int(pred[1])
    img[v-ll:v+ll,u-ll:u+ll,2] = 255
    cv2.circle(img,(u,v),10,(255,0,0),2) # predicted gaze center
    
    xmin = u - int(W/4) if u > W/4 else 0
    xmax = u + int(W/4) if u < 3*W/4 else int(W)
    ymin = v - int(H/4) if v > H/4 else 0
    ymax = v + int(H/4) if v < 3*H/4 else int(H)

    cv2.rectangle(img,(xmin, ymin),(xmax, ymax), color[i], 2)
    img[ymin:ymax,xmin:xmax,:] = img[ymin:ymax,xmin:xmax,:]*transparent + \
                np.tile(np.array([clr for clr in color[i]])*(1-transparent),(ymax-ymin,xmax-xmin,1))
    print ("pred: ({}, {})").format(u, v)

    img = cv2.resize(img, (800,400))
    if Agent._save_img:
        cv2.imwrite(save_path+img_name+'.jpg', img)
    else:
        cv2.imshow("gaze", img)


    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        return -1
    elif key == ord('q'):
        return -2
    elif key == ord('s'):
        return -3
    elif key == ord('c'):
        Agent._save_img = not Agent._save_img
        return 0
    else:
        return 0
