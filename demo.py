#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from util import *
from glob import glob
from model import Deep360Pilot
from MeanVelocityDiff import MeanVelocityDiff


def video_base(Agent, vid_domain, vid_name):
    """ Run test as a whole video, instead of cropped batches """
    
    # Initialization
    FEATURE_PATH = os.path.join(Agent.data_path, 'feature_{}_{}boxes'.format(vid_domain, Agent.n_detection), vid_name)
    print FEATURE_PATH

    iou = 0.0
    acc = 0.0
    vel_diff = 0.0
    total_loss = 0.0
    total_deltaloss = 0.0
    
    # Init prediction
    view_trajectory = None
    init_viewangle_value = np.ones([Agent.batch_size, Agent.n_output])/2
    
    # Init MVD
    MVD = MeanVelocityDiff(W=Agent.W)

    # calc n_clips
    n_clips = len(glob(os.path.join(FEATURE_PATH, 'roisavg*.npy')))
    assert n_clips > 0, "There is no feature file at {}".format(FEATURE_PATH)
    print "Found {} clips in {}".format(n_clips, FEATURE_PATH)

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
        if Agent.restore_path and tf.train.checkpoint_exists(Agent.restore_path):
            saver.restore(sess, Agent.restore_path)
            print "Your model restored!!!"
        else:
            print "Model Not Found!!!"
            return False
    

        # generate roislist and roisavg of specified video
        # from 1 to n_clips only, abandon last one clip
        for count in xrange(1, n_clips + 1):
            
            # load test_data
            box_center = np.load(os.path.join(FEATURE_PATH, 'divide_area_pruned_boxes{:04d}.npy'.format(count)))
            roisavg_batch = np.load(os.path.join(FEATURE_PATH, 'pruned_roisavg{:04d}.npy'.format(count)))
            hof_batch = np.load(os.path.join(FEATURE_PATH, 'hof{:04d}.npy'.format(count)))

            box_center = np.tile(np.expand_dims(box_center, 0), [Agent.batch_size, 1, 1, 1])
            roisavg_batch = np.tile(np.expand_dims(roisavg_batch, 0), [Agent.batch_size, 1, 1, 1])
            hof_batch = np.tile(np.expand_dims(hof_batch, 0), [Agent.batch_size, 1, 1, 1])
            
            oracle_viewangle_batch = np.zeros([Agent.batch_size, Agent.n_frames, Agent.n_output])
            one_hot_label_batch = np.zeros([Agent.batch_size, Agent.n_frames, Agent.n_detection])

            box = box_center.copy()
            gt = oracle_viewangle_batch.copy()
            
            box_center[:,:,:,0] = (box_center[:,:,:,0]/Agent.W + box_center[:,:,:,2]/Agent.W)/2
            box_center[:,:,:,1] = (box_center[:,:,:,1]/Agent.H + box_center[:,:,:,3]/Agent.H)/2
            box_center = box_center[:, :, :, :2]

            oracle_viewangle_batch[:,:,0] = oracle_viewangle_batch[:,:,0]/Agent.W
            oracle_viewangle_batch[:,:,1] = oracle_viewangle_batch[:,:,1]/Agent.H

            [loss, deltaloss, viewangle_out, sal_box_out] = sess.run(
                    [Agent.cost, Agent.delta, Agent.viewangle, Agent.sal_box_prob], \
                    feed_dict={
                        Agent.obj_app: roisavg_batch, 
                        Agent.oracle_actions: one_hot_label_batch, 
                        Agent.oracle_viewangle: oracle_viewangle_batch, \
                        Agent.box_center: box_center, 
                        Agent.hof: hof_batch, 
                        Agent.keep_prob:1.0, 
                        Agent.init_viewangle: init_viewangle_value, 
                        Agent._phase: Agent.bool_two_phase
                    }
            )
            
            total_loss += loss/Agent.n_frames
            total_deltaloss += deltaloss/Agent.n_frames

            # Feed in init value to next batch
            init_viewangle_value = viewangle_out[:,-1,:].copy()

            viewangle_out[:,:,0] = (viewangle_out[:,:,0]*Agent.W).astype(int)
            viewangle_out[:,:,1] = (viewangle_out[:,:,1]*Agent.H).astype(int)
            
            corr = np.sum(np.logical_and(one_hot_label_batch, sal_box_out))
            ac = float(corr) / (Agent.batch_size * Agent.n_frames)
            iu = score(Agent, viewangle_out, gt[:,:,:2], False)

            # only one row in batch are used, average to get result.
            # convert into degree form (* 360 / 1920 / Agent.n_frames)
            vd = MVD.batch_vel_diff(viewangle_out) * 0.1875 / (Agent.n_frames)
            
            acc += ac
            iou += iu
            vel_diff += vd
            print "Video: {:3d} | Corr: {:3d}, IoU: {:.3f}, Acc: {:.3f}, Vel_diff: {:.3f}".format(
                count, corr, iu, ac, vd)
            print "Oracle: ", np.where(one_hot_label_batch[0])
            print "----------------------------------------------------------------"
            print "Prediction: ", np.where(sal_box_out[0])

            if view_trajectory is None:
                view_trajectory = viewangle_out[0].copy()
            else:
                view_trajectory = np.vstack((view_trajectory, viewangle_out[0].copy()))
            
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
                        ret = visual_gaze(Agent, vidname, gt[0,nimage,:2], viewangle_out[0,nimage, :], sal_box_out[0,nimage, :], box[0,nimage, :, :])
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
            print view_trajectory.shape
            out_path = '{}{}_{}_{}_lam{}_{}_best_model'.format(
                    Agent.save_path, 
                    vid_name, 
                    Agent.domain, 
                    Agent.n_detection, 
                    Agent.regress_lmbda, 
                    Agent.two_phase) 
            print "Save prediction of vid {} to {}".format(vid_name, out_path)
            np.save(out_path, view_trajectory)
            with open(out_path + '.txt', 'w') as f:
                f.write("Loss = {:.5f}\n".format(total_loss/n_clips))
                f.write("DeltaLoss = {:.5f}\n".format(total_deltaloss/n_clips))
                f.write("IOU = {:.5f}\n".format(iou/n_clips))
                f.write("Acc = {:.5f}\n".format(acc/n_clips))
                f.write("Velocity Diff = {:.5f}\n".format(vel_diff/n_clips))


