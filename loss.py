#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from ops import tf_l2_dist_360

def total_cost(_phase, loss, l2loss, smoothloss, residual_loss):
    # Cost in Classification phase (True) or Regression phase (False)
    return tf.cond(_phase, \
            lambda: l2loss + loss + smoothloss + residual_loss, \
            lambda: l2loss + loss + smoothloss + residual_loss)


def residual_loss(residual, weight=10):
    """ Smooth residual prediction """
    return weight * tf.nn.l2_loss(residual)


def accurate_loss(cur_viewangle, oracle_viewangle, _axis=1):
    """ Prediction to oracle_viewangle """
    l2diff = tf_l2_dist_360(cur_viewangle, oracle_viewangle, _axis) # (b, n_output) -> (b, 1)
    
    return tf.reduce_mean(l2diff)


def smooth_loss(_phase, cls_lambda, reg_lambda, angle_diff, vel_diff):
    # Classification phase (True) or Regression phase (False)
    reg_l2diff = tf.cond(_phase, \
            lambda: cls_lambda * angle_diff, \
            lambda: reg_lambda * vel_diff ) # b x n_output -> b x 1
    
    return tf.reduce_mean(reg_l2diff)


def policy_loss(sal_box_prob, oracle_action):
    # Pred to one hot
    loss = tf.nn.softmax_cross_entropy_with_logits(sal_box_prob, oracle_action)
    
    return tf.reduce_mean(loss)



