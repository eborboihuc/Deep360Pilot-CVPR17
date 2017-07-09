#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import tensorflow as tf
from test import test_all
from util import load_batch_data, score
from MeanVelocityDiff import MeanVelocityDiff
from tensorflow.python import debug as tf_debug

def train(Agent):
    """ Training wrapper """

    # Summary
    if not os.path.isdir(Agent.save_path):
        os.mkdir(Agent.save_path)
 
    # Init MVD
    MVD = MeanVelocityDiff(W=Agent.W)

    # Initial Session
    with tf.Session(config = Agent.sess_config) as sess:
        if Agent.Debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph
        sess.run(init)
        saver = tf.train.Saver()
        
        writer = tf.summary.FileWriter(os.path.join(Agent.save_path, 'logs'), sess.graph)
    
        # Load model and continue
        if Agent.restore_path and tf.train.checkpoint_exists(Agent.restore_path):
            saver.restore(sess, Agent.restore_path)
            print "Your model restored!!!"
        else:
            print "Train from scratch!!!"
    
        # Saver condition to save model
        best_iou = 0.0
        best_vel = 999999.0

        # Initial predition location
        init_viewangle_value = np.ones([Agent.batch_size, Agent.n_output])/2

        # Keep training until reach max iterations
        for epoch in range(Agent.n_epochs):
            
            # random shuffle batch order
            n_batchs = np.arange(1, Agent.train_num+1)
            np.random.shuffle(n_batchs)
            
            epoch_loss = 0.0
            delta_loss = 0.0
            acc = 0.0
            iou = 0.0
            vel_diff = 0.0
            
            tStart_epoch = time.time()
            for batch in range(Agent.train_num):
                
                batch_xs, batch_ys, batch_o_viewangle, batch_box_center, batch_hof, _, _, gt = load_batch_data(Agent, Agent.train_path, n_batchs[batch], True)
        
                # Fit training using batch data
                _, summary_out, batch_loss, deltaloss, viewangle_out, sal_box_out, lr_out = sess.run(
                        [Agent.opt, Agent.merged, Agent.cost, Agent.delta, Agent.viewangle, Agent.sal_box_prob, Agent.lr], 
                        feed_dict={
                            Agent.obj_app: batch_xs, 
                            Agent.oracle_actions: batch_ys, 
                            Agent.oracle_viewangle: batch_o_viewangle, 
                            Agent.box_center: batch_box_center, 
                            Agent.hof: batch_hof, 
                            Agent.keep_prob: 1-Agent.trainDropPr, 
                            Agent.init_viewangle: init_viewangle_value, 
                            Agent._phase: Agent.bool_two_phase
                        }
                )
                
                writer.add_summary(summary_out, epoch*len(n_batchs)+batch)

                viewangle_out[:,:,0] = (viewangle_out[:,:,0]*Agent.W).astype(int)
                viewangle_out[:,:,1] = (viewangle_out[:,:,1]*Agent.H).astype(int)

                sys.stdout.flush()
                epoch_loss += batch_loss/Agent.n_frames
                delta_loss += deltaloss/Agent.n_frames
                acc        += float(np.sum(np.logical_and(batch_ys, sal_box_out))) / (Agent.batch_size*Agent.n_frames)
                iou        += score(Agent, viewangle_out, gt)
                # convert into degree form (* 360 / 1920 / n_frames)
                vel_diff   += MVD.batch_vel_diff(viewangle_out) * 0.1875 / (Agent.n_frames)

            # Print one epoch
            tStop_epoch = time.time()
            print "Epoch: {:3d} | Time: {:.2f}s | Loss: {:.3f} DeltaLoss: {:.3f}, lr: {:.2e}, IoU: {:.3f}, Acc: {:.3f}, Vel_diff: {:.3f}".format(
                    epoch, round(tStop_epoch - tStart_epoch,2), epoch_loss/Agent.train_num, delta_loss/Agent.train_num, lr_out, iou/Agent.train_num, acc/Agent.train_num, vel_diff/Agent.train_num)

            print "Pred: {}, GT: {}".format(viewangle_out[0, -1, :], gt[0, -1, :2])
            sys.stdout.flush()
            
            if iou/Agent.train_num > best_iou and Agent.bool_two_phase:
                best_iou = iou/Agent.train_num
                
                # Save lam1_classify_best_model
                saver.save(sess, os.path.join(Agent.save_path, '{}_lam{}_{}_best_model'.format(Agent.domain, 1, Agent.two_phase)))
                Agent.Best_score={'epoch':epoch, 'loss':epoch_loss/Agent.train_num, 'smooth_loss':delta_loss/Agent.train_num, \
                            'lr':lr_out, 'iou':iou/Agent.train_num, 'acc':acc/Agent.train_num, 'vel_diff':vel_diff/Agent.train_num}
            
            if vel_diff/Agent.train_num < best_vel and not Agent.bool_two_phase:
                best_vel = vel_diff/Agent.train_num
                
                # Save lam{}_regress_model
                saver.save(sess, os.path.join(Agent.save_path, '{}_lam{}_{}_best_model'.format(Agent.domain, Agent.regress_lmbda, Agent.two_phase)))
                Agent.Best_score={'epoch':epoch, 'loss':epoch_loss/Agent.train_num, 'smooth_loss':delta_loss/Agent.train_num, \
                            'lr':lr_out, 'iou':iou/Agent.train_num, 'acc':acc/Agent.train_num, 'vel_diff':vel_diff/Agent.train_num}
                
            if (epoch+1) % Agent.display_step == 0:
                test_all(sess, Agent, is_train=True)
                test_all(sess, Agent, is_train=False)
        
        print "Optimization Finished!"
        test_all(sess, Agent, is_train=True)
        test_all(sess, Agent, is_train=False)
 
        # Save log
        np.save(
            os.path.join(Agent.save_path, 'logs', '{}_lam{}_{}_best_model_log'.format(
                    Agent.domain, 
                    1 if Agent.bool_two_phase else Agent.regress_lmbda, 
                    Agent.two_phase)
            ), 
            Agent.Best_score
        )

        # Save model
        saver.save(sess, os.path.join(Agent.save_path, 'final_model'))


