#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
 

def tf_360_constraint(viewangle):
    """ 360 constraint, x[x > 1.0] = x[x > 1.0] - 1.0, x[x < 0.0] = x[x < 0.0] + 1.0, 0.0 < y < 1.0 """
    xpart, ypart = tf.split(1, 2, viewangle)
    _greater = tf.greater(xpart, 1.0) # (n_batch, 1)
    _less = tf.less(xpart, 0.0) # (n_batch, 1)
    xpart = xpart - tf.to_float(_greater)
    xpart = xpart + tf.to_float(_less)
    ypart = tf.clip_by_value(ypart, 0.0, 1.0)
    
    return tf.concat(1, [xpart, ypart]) # b x self.n_output


def tf_l2_dist(Gt, pred,_axis):
    """ pred should be the same type and shape as Gt, _axis indicates the axis to work with, return l2diff """
    l2diff = tf.sub(Gt, pred)
    l2diff = tf.reduce_sum(tf.square(l2diff), reduction_indices=_axis)
    l2diff = tf.maximum(l2diff, 1e-10)
    l2diff = tf.sqrt(l2diff) # b x n_class -> b x 1

    return l2diff


# NOTE: Input should be in the range of [-1, 1]
def tf_l2_dist_360(Gt, pred, _axis):
    """ pred should be the same type and shape as Gt, _axis indicates the axis to work with, return l2diff in 360 video form """
    l2diff = tf_dist_360_classify(Gt, pred, _axis)
    l2diff = tf.reduce_sum(tf.square(l2diff), reduction_indices=_axis)
    l2diff = tf.maximum(l2diff, 1e-10)
    l2diff = tf.sqrt(l2diff) # b x n_class -> b x 1

    return l2diff


# NOTE: Input should be in the range of [-1, 1]
def tf_dist_360_classify(Gt, pred, _axis):
    """ pred is current position, Gt is target position, and diff_360 finds shortest path.
        e.g., we will have xdiff_output=-0.4 when xdiff_input=0.6 > 0.5, and xdiff_output=0.4 when xdiff_input=-0.6 < -0.5
        Deminish the gap between left most and right most by function : { max(X,0.5)*(X-1) + min(X,0.5)*X } or { max(X,-0.5)*(X+1) + min(X,-0.5)*X } 
    """
    diff = tf.sub(Gt, pred)
    xdiff, ydiff = tf.split(_axis, 2, diff)
    xdiff = tf.cast(tf.greater(xdiff,0.5),tf.float32) * (xdiff - 1) + tf.cast(tf.less_equal(xdiff,0.5),tf.float32) * xdiff
    xdiff = tf.cast(tf.less(xdiff,-0.5),tf.float32) * (xdiff + 1) + tf.cast(tf.greater_equal(xdiff,-0.5),tf.float32) * xdiff
    l2diff = tf.concat(_axis,[xdiff,ydiff])

    return l2diff


def l2_dist_360(pA, pB, W):
    """ pA is 1x2 , pB is 3x2, this will return a 3x1 L2-norm in 360 video form"""
    dist = (pA-pB).astype(np.float64)
    xdist = abs(dist[:,0])
    xdist[xdist>W/2] = W - xdist[xdist>W/2]
    ydist = abs(dist[:,1])
    
    return np.sqrt((np.square(xdist)+np.square(ydist)))


# =========================================================================================
# NOTE: Not use anymore
'''
def tf_dist_360(pred, Gt, _axis):
    """ pred is current position, Gt is target position, and diff_360 finds shortest path. """
    diff = tf.sub(Gt, pred)
    xdiff, ydiff = tf.split(_axis, 2, diff)

    xdiff = xdiff + 0.5
    x_exceed_value = tf.floordiv(xdiff, 1.0)
    xdiff = xdiff - 0.5 - x_exceed_value

    ydiff = tf.clip_by_value(ydiff, 0.0, 1.0)
    l2diff = tf.concat(_axis,[xdiff,ydiff])

    return l2diff
'''

