#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from ops import tf_dist_360

def tf_l2_loss(Gt, pred,_axis):
    """ pred should be the same type and shape as Gt, _axis indicates the axis to work with, return l2diff """
    l2diff = tf.subtract(Gt, pred)
    l2loss = tf.reduce_sum(tf.square(l2diff), axis=_axis)
    l2loss = tf.maximum(l2loss, 1e-10)
    l2loss = tf.sqrt(l2loss) # (n_batch, n_class) -> (n_batch, 1)

    return l2loss


# NOTE: Input should be in the range of [-1, 1]
def tf_l2_loss_360(Gt, pred, _axis):
    """ pred should be the same type and shape as Gt, _axis indicates the axis to work with, return l2diff in 360 video form """
    l2diff = tf_dist_360(Gt, pred, _axis)
    l2loss = tf.reduce_sum(tf.square(l2diff), axis=_axis)
    l2loss = tf.maximum(l2loss, 1e-10)
    l2loss = tf.sqrt(l2loss) # (n_batch, n_class) -> (n_batch, 1)

    return l2loss


def total_cost(_phase, policyloss, l2loss, smoothloss, residual_loss):
    # Cost in Classification phase (True) or Regression phase (False)
    return tf.cond(_phase, \
            lambda: l2loss + policyloss + smoothloss + residual_loss, \
            lambda: l2loss + smoothloss + residual_loss
            )


def residual_loss(residual, weight=10):
    """ Smooth residual prediction """
    return weight * tf.nn.l2_loss(residual)


def accurate_loss(cur_viewangle, oracle_viewangle, _axis=1):
    """ Prediction to oracle_viewangle """
    l2diff = tf_l2_loss_360(cur_viewangle, oracle_viewangle, _axis) # (n_batch, n_output) -> (n_batch, 1)
    
    return tf.reduce_mean(l2diff)


def smooth_loss(_phase, cls_lambda, reg_lambda, angle_diff, vel_diff):
    # Classification phase (True) or Regression phase (False)
    reg_l2diff = tf.cond(_phase, \
            lambda: cls_lambda * angle_diff, \
            lambda: cls_lambda * angle_diff + reg_lambda * vel_diff 
            ) # (n_batch, n_output) -> (n_batch, 1)

    return tf.reduce_mean(reg_l2diff)


def policy_loss(sal_box_prob, oracle_action, sample_weights):
    """ Action select an object, Reward encourages going closer """
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=sal_box_prob, labels=oracle_action)
    
    return tf.reduce_mean(loss) * (1.0 - sample_weights / 10.0)


def get_reward(distance, ang_range=40.9/360.0):
    """ distance between pred_viewing_angle and oracle_viewing_angle """
    if_inside = tf.cast(tf.less_equal(distance, ang_range), tf.float32)
    reward = (1.0 - distance/ang_range) * if_inside \
            + tf.constant(-1.0) * (1.0 - if_inside)
    #reward = tf.cond(distance <= ang_range, \
            #        lambda: 1.0 - distance/ang_range, \
            #        lambda: tf.constant(-1.0)
    #        )
    return tf.reduce_mean(reward)


