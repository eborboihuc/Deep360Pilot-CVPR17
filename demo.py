#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from glob import glob
from model import Deep360Pilot
from util import *
from MeanVelocityDiff import MeanVelocityDiff



def video_base(Agent, vid_domain, vid_name):
    """ Run test as a whole video, instead of cropped batches """
    
    # Initialization
    FEATURE_PATH = os.path.join(Agent.data_path, 'feature_{}_{}boxes'.format(vid_domain, Agent.n_detection), vid_name)
    print FEATURE_PATH

    total_loss = 0.0
    total_deltaloss = 0.0
    iou = 0.0
    acc = 0.0
    vel_diff = 0.0
    total_pred = None
    
    pred_init_value = np.ones([Agent.batch_size, Agent.n_output])/2
    
    # Init MVD
    MVD = MeanVelocityDiff(W=Agent.W)

    # calc n_clips
    n_clips = len(glob(os.path.join(FEATURE_PATH, 'roisavg*.npy')))
    assert n_clips > 0, "There is no feature file at {}".format(FEATURE_PATH)
    print n_clips

    # n_clips - 1 since we drop last batch which may contain null data.
    n_clips = n_clips - 1
    
    # Initial Session
    with tf.Session(config = Agent.sess_config) as sess:
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph
        sess.run(init)
        saver = tf.train.Saver()
        
        # Load model and continue
        if Agent.restore_path and os.path.isdir(Agent.save_path):
            saver.restore(sess, Agent.restore_path)
            print "Your model restored!!!"
        else:
            print "Model Not Found!!!"
            return False
    

        # generate roislist and roisavg of specified video
        # from 1 to n_clips only, abandon last one clip
        for count in xrange(1, n_clips + 1):
            
            # load test_data
            inclusion_batch = np.zeros((1, 50, Agent.n_detection, 3), dtype=np.float16)

            #box_center = np.load(os.path.join(FEATURE_PATH, 'divide_area_pruned_boxes{:04d}.npy'.format(count)))
            box_center = np.load(os.path.join(FEATURE_PATH, 'roislist{:04d}.npy'.format(count)))
            #roisavg_batch = np.load(os.path.join(FEATURE_PATH, 'pruned_roisavg{:04d}.npy'.format(count)))
            roisavg_batch = np.load(os.path.join(FEATURE_PATH, 'roisavg{:04d}.npy'.format(count)))
            #inclusion_batch[0, :, :, 0] = np.load(os.path.join(FEATURE_PATH, 'avg_motion{:04d}.npy'.format(count)))
            #inclusion_batch[0, :, :, 1:] = np.load(os.path.join(FEATURE_PATH, 'avg_flow{:04d}.npy'.format(count)))
            hof_batch = np.load(os.path.join(FEATURE_PATH, 'hof{:04d}.npy'.format(count)))

            box_center = np.tile(np.expand_dims(box_center, 0), [Agent.batch_size, 1, 1, 1])
            roisavg_batch = np.tile(np.expand_dims(roisavg_batch, 0), [Agent.batch_size, 1, 1, 1])
            hof_batch = np.tile(np.expand_dims(hof_batch, 0), [Agent.batch_size, 1, 1, 1])
            inclusion_batch = np.tile(inclusion_batch, [Agent.batch_size, 1, 1, 1])
            
            label_batch = np.zeros([Agent.batch_size, Agent.n_frames, Agent.n_output+1])
            one_hot_label_batch = np.zeros([Agent.batch_size, Agent.n_frames, Agent.n_detection])

            box = box_center.copy()
            gt = label_batch.copy()
            
            box_center[:,:,:,0] = (box_center[:,:,:,0]/Agent.W + box_center[:,:,:,2]/Agent.W)/2
            box_center[:,:,:,1] = (box_center[:,:,:,1]/Agent.H + box_center[:,:,:,3]/Agent.H)/2

            label_batch[:,:,0] = label_batch[:,:,0]/Agent.W
            label_batch[:,:,1] = label_batch[:,:,1]/Agent.H

            # TODO: add data level inclusion
            [_, loss, deltaloss, pred_out, alpha_out] = sess.run([Agent.opt, Agent.cost, Agent.delta, Agent.pred, Agent.alphas], \
                    feed_dict={Agent.obj_app: roisavg_batch, Agent.y: one_hot_label_batch, Agent.y_loc: label_batch, \
                            Agent.box_center: box_center[:,:,:,:Agent.n_output], Agent.inclusion: inclusion_batch, \
                            Agent.hof: hof_batch, Agent.keep_prob:1.0, Agent.pred_init: pred_init_value, Agent._phase: Agent.bool_two_phase})
            
            total_loss += loss/Agent.n_frames #(Agent.batch_size*Agent.n_frames)
            total_deltaloss += deltaloss/Agent.n_frames

            # Feed in init value to next batch
            pred_init_value = pred_out[:,-1,:].copy()

            pred_out[:,:,0] = (pred_out[:,:,0]*Agent.W).astype(int)
            pred_out[:,:,1] = (pred_out[:,:,1]*Agent.H).astype(int)
            
            ac = float(np.sum(np.logical_and(one_hot_label_batch, alpha_out))) / (Agent.batch_size * Agent.n_frames)
            iu = score(Agent, pred_out, gt[:,:,:2], False)

            # only one row in batch are used, average to get result.
            # convert into degree form (* 360 / 1920 / Agent.n_frames)
            vd = MVD.batch_vel_diff(pred_out) * 0.1875 / (Agent.n_frames)
            
            acc += ac
            iou += iu
            vel_diff += vd
            print "Acc: {:.3f}, IoU: {:.3f}, Vel_diff:{:.3f}".format(ac, iu, vd)
            print "Corr: {}".format(np.sum(np.logical_and(one_hot_label_batch, alpha_out)))
            print np.where(one_hot_label_batch[0])
            print "----------------------------------------------------------------"
            print np.where(alpha_out[0])

            if total_pred is None:
                total_pred = pred_out[0].copy()
            else:
                total_pred = np.vstack((total_pred, pred_out[0].copy()))
            
            ret = 0
            if Agent._show:
                nimages = (count-1)*Agent.n_frames
                for nimage in xrange(Agent.n_frames):
                    vidname = vid_name + '/' + str(nimages+nimage+1).zfill(6)
                    if Agent._save_img and not os.path.isdir(Agent.save_path + vid_name):
                        print 'Make dir at ' + Agent.save_path + vid_name
                        os.makedirs(Agent.save_path + vid_name) # mkdir recursively
                    if Agent._show:
                        print 
                        print ("num_batch: {}, video: {}, count: {}, nimage: {}").format(n_clips, vidname, count, nimage)
                        ret = visual_gaze(Agent, vidname, gt[0,nimage,:2], pred_out[0,nimage, :], alpha_out[0,nimage, :], box[0,nimage, :, :])
                    if ret == -1 or ret == -2 or ret == -3:
                        break
                if ret == -1 or ret == -2:
                    break
            if ret == -1:
                break    

        print "Loss = {:.3f}".format(total_loss/n_clips) # 40/20, number of training/testing set
        print "DeltaLoss = {:.3f}".format(total_deltaloss/n_clips)
        print "IOU = {:.3f}".format(iou/n_clips)
        print "Acc = {:.3f}".format(acc/n_clips)
        print "Velocity Diff = {:.3f}".format(vel_diff/n_clips)

        if Agent._save_pred:
            print total_pred.shape
            out_path = Agent.save_path + vid_name 
            print "Save prediction of vid {} to {}".format(vid_name, Agent.save_path)
            np.save(out_path + "_{}_{}_lam{}_{}_best_model".format(Agent.domain, Agent.n_detection, Agent.regress_lmbda, Agent.two_phase), total_pred)
            with open(out_path + "_{}_{}_lam{}_{}_best_model".format(Agent.domain, Agent.n_detection, Agent.regress_lmbda, Agent.two_phase) + '.txt', 'w') as f:
                f.write("Loss = {:.5f}\n".format(total_loss/n_clips))
                f.write("DeltaLoss = {:.5f}\n".format(total_deltaloss/n_clips))
                f.write("IOU = {:.5f}\n".format(iou/n_clips))
                f.write("Acc = {:.5f}\n".format(acc/n_clips))
                f.write("Velocity Diff = {:.5f}\n".format(vel_diff/n_clips))


