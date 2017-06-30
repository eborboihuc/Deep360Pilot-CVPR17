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

def train(Agent):
    """ Training wrapper """

    # Summary
    if not os.path.isdir(Agent.save_path):
        os.mkdir(Agent.save_path)
    if Agent.restore_path and not os.path.isdir(Agent.restore_path):
        os.mkdir(Agent.restore_path)
 
    # Init MVD
    MVD = MeanVelocityDiff(W=Agent.W)

    # Initial Session
    with tf.Session(config = Agent.sess_config) as sess:
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph
        sess.run(init)
        saver = tf.train.Saver()
        
        writer = tf.summary.FileWriter(os.path.join(Agent.save_path, 'logs'), sess.graph)
    
        # Load model and continue
        if Agent.restore_path and os.path.isfile(Agent.restore_path):
            saver.restore(sess, Agent.restore_path)
            print "Your model restored!!!"
        else:
            print "Train from scratch!!!"
    
        # Saver condition to save model
        best_iou = 0.0
        best_vel = 999999.0

        pred_init_value = np.ones([Agent.batch_size, Agent.n_output])/2

        # Keep training until reach max iterations
        for epoch in range(Agent.n_epochs):
            # random_chose batch.npz
            n_batchs = np.arange(1,Agent.train_num+1)
            np.random.shuffle(n_batchs)
            
            epoch_loss = 0.0
            delta_loss = 0.0
            acc = 0.0
            iou = 0.0
            vel_diff = 0.0
            num = len(n_batchs)
            
            tStart_epoch = time.time()
            for batch in range(num):
                
                batch_xs, batch_ys, batch_y_loc, batch_box_center, batch_inclusion, batch_hof, _, _, gt = load_batch_data(Agent, Agent.train_path, n_batchs[batch], True)
                
                # Fit training using batch data
                _, summary_out, batch_loss, deltaloss, pred_out, alpha_out, lr_out = sess.run(
                        [Agent.opt, Agent.merged, Agent.cost, Agent.delta, Agent.pred, Agent.alphas, Agent.lr], 
                        feed_dict={
                            Agent.obj_app: batch_xs, 
                            Agent.y: batch_ys, 
                            Agent.y_loc: batch_y_loc, 
                            Agent.box_center: batch_box_center[:,:,:,:Agent.n_output], 
                            Agent.inclusion: batch_inclusion, 
                            Agent.hof: batch_hof, 
                            Agent.keep_prob: 1-Agent.trainDropPr, 
                            Agent.pred_init: pred_init_value, 
                            Agent._phase: Agent.bool_two_phase
                        }
                )
                
                writer.add_summary(summary_out, epoch*len(n_batchs)+batch)

                pred_out[:,:,0] = (pred_out[:,:,0]*Agent.W).astype(int)
                pred_out[:,:,1] = (pred_out[:,:,1]*Agent.H).astype(int)

                sys.stdout.flush()
                epoch_loss += batch_loss/Agent.n_frames
                delta_loss += deltaloss/Agent.n_frames
                acc        += float(np.sum(np.logical_and(batch_ys, alpha_out))) / (Agent.batch_size*Agent.n_frames)
                iou        += score(Agent, pred_out, gt[:,:,:2])
                vel_diff   += MVD.batch_vel_diff(pred_out) * 0.1875 / (Agent.n_frames)

            # Print one epoch
            print "Epoch: {} done. Loss: {:.3f} DeltaLoss: {:.3f}, lr: {:.3f}, IoU: {:.3f}, Acc: {:.3f}, Vel_diff: {:.3f}".format(
                    epoch, epoch_loss/num, delta_loss/num, lr_out, iou/num, acc/num, vel_diff/num)

            tStop_epoch = time.time()
            print "Epoch Time Cost: {}s".format(round(tStop_epoch - tStart_epoch,2))
            sys.stdout.flush()
            
            if iou/num > best_iou and Agent.bool_two_phase:
                best_iou = iou/num
                
                # Save lam1_classify_best_model
                saver.save(sess, os.path.join(Agent.save_path, '{}_lam{}_{}_best_model'.format(Agent.domain, 1, Agent.two_phase)))
                Agent.Best_score={'epoch':epoch, 'loss':epoch_loss/num, 'smooth_loss':delta_loss/num, \
                            'lr':lr_out, 'iou':iou/num, 'acc':acc/num, 'vel_diff':vel_diff/num}
            
            if vel_diff/num < best_vel and Agent.bool_two_phase:
                best_vel = vel_diff/num
                
                # Save lam{}_regress_model
                saver.save(sess, os.path.join(Agent.save_path, '{}_lam{}_{}_best_model'.format(Agent.domain, Agent.regress_lmbda, Agent.two_phase)))
                Agent.Best_score={'epoch':epoch, 'loss':epoch_loss/num, 'smooth_loss':delta_loss/num, \
                            'lr':lr_out, 'iou':iou/num, 'acc':acc/num, 'vel_diff':vel_diff/num}
                
            if (epoch+1) % Agent.display_step == 0:
                print "Training"
                test_all(sess, Agent, is_train=True)
                print "Testing"
                test_all(sess, Agent, is_train=False)
        print "Optimization Finished!"

        print "Total Training"
        test_all(sess, Agent, is_train=True)
        print "Total Testing"
        test_all(sess, Agent, is_train=False)
        
        np.save(
            os.path.join(Agent.save_path, 'logs', '{}_lam{}_{}_best_model_log'.format(
                    Agent.domain, 
                    1 if Agent.bool_two_phase else Agent.regress_lmbda, 
                    Agent.two_phase)
            ), 
            Agent.Best_score
        )
        saver.save(sess, os.path.join(Agent.save_path, 'final_model'))


