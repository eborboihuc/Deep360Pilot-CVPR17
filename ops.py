#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
 
"""
All the position are aligned to pixel coordinate
0.0 < X < 1.0,
0.0 < Y < 1.0

X distance is confined in [-0.5, 0.5] because of the natural constraint of 360 degree
-0.5 < delta X < 0.5,
-1.0 < delta Y < 1.0
"""

def tf_mov_coef(vec_a, vec_b):
    """ Moving Coefficient, input value between -1 to 1, output value between 0 to 1 """
    cos_sim = tf_cosine_sim(vec_a, vec_b)
    return 0.5 * (cos_sim + 1)


def tf_cosine_sim(vec_a, vec_b):
    """ Cosine similarity of two vectors, output value between -1 to 1 """
    normalize_a = tf.nn.l2_normalize(vec_a, 0)
    normalize_b = tf.nn.l2_normalize(vec_b, 0)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b))


# NOTE: Input should be in the range of [-1, 1]
def tf_dist_360(Gt, pred, _axis):
    """ Distance wrapper """
    diff = tf.subtract(Gt, pred)
    return tf_360_shortest_dist(diff, _axis)


def tf_360_shortest_dist(diff, _axis=1):
    """ tf_diff_360 finds shortest path.
        e.g., we will have xdiff_output=-0.4 when xdiff_input=0.6 > 0.5, and xdiff_output=0.4 when xdiff_input=-0.6 < -0.5
        Deminish the gap between left most and right most by function : 
        {max(X,0.5)*(X-1) + min(X,0.5)*X } or { max(X,-0.5)*(X+1) + min(X,-0.5)*X } 
    """
    xdiff, ydiff = tf.split(axis=_axis, num_or_size_splits=2, value=diff)
    xdiff = tf.cast(tf.greater(xdiff, 0.5),tf.float32) * (xdiff - 1) + tf.cast(tf.less_equal(xdiff, 0.5),tf.float32) * xdiff
    xdiff = tf.cast(tf.less(xdiff, -0.5),tf.float32) * (xdiff + 1) + tf.cast(tf.greater_equal(xdiff, -0.5),tf.float32) * xdiff
    ydiff = tf.clip_by_value(ydiff, -1.0, 1.0)
    l2diff = tf.concat(axis=_axis,values=[xdiff,ydiff])

    return l2diff


def tf_viewangle_360_constraint(viewangle, _axis=1):
    """ 360 constraint, x[x > 1.0] = x[x > 1.0] - 1.0, x[x < 0.0] = x[x < 0.0] + 1.0, 0.0 < y < 1.0 """
    xpart, ypart = tf.split(axis=_axis, num_or_size_splits=2, value=viewangle)
    _greater = tf.greater(xpart, 1.0) # (n_batch, 1)
    _less = tf.less(xpart, 0.0) # (n_batch, 1)
    xpart = xpart - tf.to_float(_greater)
    xpart = xpart + tf.to_float(_less)
    ypart = tf.clip_by_value(ypart, 0.0, 1.0)
    
    return tf.concat(axis=1, values=[xpart, ypart]) # (n_batch, n_output)


def l2_dist_360(pA, pB, W=1.0):
    """ [Numpy version] pA is 1x2 , pB is 3x2, this will return a 3x1 L2-norm in 360 video form"""
    dist = (pA-pB).astype(np.float64)
    xdist = abs(dist[:,0])
    xdist[xdist>W/2] = W - xdist[xdist>W/2]
    ydist = abs(dist[:,1])
    
    return np.sqrt((np.square(xdist)+np.square(ydist)))

