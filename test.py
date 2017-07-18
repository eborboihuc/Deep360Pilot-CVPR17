#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from util import *
from MeanVelocityDiff import MeanVelocityDiff


def test(Agent):
    """ Test wrapper """
    
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
            test_all(sess, Agent, is_train=True)
            test_all(sess, Agent, is_train=False)
        else:
            print "Model ({}) Not Found!!!".format(Agent.restore_path)


def test_all(sess, Agent, is_train):
    """ Testing whole batches of training/testing set """
    
    # Init MVD
    MVD = MeanVelocityDiff(W=Agent.W)

    # Init parameters
    num = Agent.train_num if is_train else Agent.test_num
    path = Agent.train_path if is_train else Agent.test_path
    dropPr = Agent.trainDropPr if is_train else Agent.testDropPr
    
    iou = 0.0
    acc = 0.0
    vel_diff = 0.0
    total_loss = 0.0
    total_deltaloss = 0.0
    init_viewangle_value = np.ones([Agent.batch_size, Agent.n_output])/2
    
    print "-"*35, "Training" if is_train else "Testing ", "-"*35

    # Go through each batch
    for batch_num in range(1,num+1):
        
        # load test_data
        batch_data, batch_oracle_actions, batch_oracle_viewangle, batch_box_center, batch_hof, batch_img, box, gt = load_batch_data(Agent, path, batch_num, True)
        

        [loss, deltaloss, viewangle_out, sal_box_out] = sess.run(
                [Agent.cost, Agent.delta, Agent.viewangle, Agent.sal_box_prob], 
                feed_dict={
                    Agent.obj_app: batch_data, 
                    Agent.oracle_actions: batch_oracle_actions, 
                    Agent.oracle_viewangle: batch_oracle_viewangle, 
                    Agent.box_center: batch_box_center, 
                    Agent.hof: batch_hof, 
                    Agent.keep_prob:1.0-dropPr, 
                    Agent.init_viewangle: init_viewangle_value, 
                    Agent._phase: Agent.bool_two_phase
                }
        )
        
        total_loss      += loss/Agent.n_frames #(batch_size*Agent.n_frames)
        total_deltaloss += deltaloss/Agent.n_frames

        viewangle_out[:,:,0] = (viewangle_out[:,:,0]*Agent.W).astype(int)
        viewangle_out[:,:,1] = (viewangle_out[:,:,1]*Agent.H).astype(int)
        
        # NOTE: Use same scale
        corr = np.sum(np.logical_and(batch_oracle_actions, sal_box_out))
        ac = float(corr) / (Agent.batch_size * Agent.n_frames)
        iu = score(Agent, viewangle_out, gt)

        # convert into degree form (* 360 / 1920 / n_frames)
        vd = MVD.batch_vel_diff(viewangle_out) * 0.1875 / (Agent.n_frames)
        
        acc += ac
        iou += iu
        vel_diff += vd
        print "Batch: {:3d} | Corr: {:3d}, IoU: {:.3f}, Acc: {:.3f}, Vel_diff: {:.3f}".format(
                batch_num, corr, iu, ac, vd)

        if Agent._show:
            ret = 0
            for count, name in enumerate(batch_img):
                vid_name = name[:13]
                nimages = (int(name[14:])-1)*Agent.n_frames

                if Agent._save_pred:
                    print viewangle_out.shape
                    out_path = Agent.root_path + ('output/test/' if 'test' in path else 'output/train/') +  name + '.npy'
                    print "Save prediction of vid {} to {}".format(name, out_path)
                    np.save(out_path, viewangle_out[count].astype(int))

                for nimage in xrange(Agent.n_frames):
                    vidname = os.path.join(vid_name, str(nimages + nimage + 1).zfill(6))
                    if Agent._save_img and not os.path.isdir(save_path + vid_name):
                        print 'Make dir at ' + save_path + ('test/' if 'test' in path else 'train/') + vid_name
                        os.makedirs(save_path + ('test/' if 'test' in path else 'train/') + vid_name) # mkdir recursively
                    
                    if Agent._show:
                        print 
                        print ("batch_num: {}, video: {}, vid: {}, count: {}, nimage: {}").format(batch_num, name, vid_name, count, nimage)
                        ret = visual_gaze(Agent, vidname, gt[count, nimage, :], viewangle_out[count, nimage,:], sal_box_out[count, nimage,:], box[count, nimage,:,:])
                    if ret == -1 or ret == -2 or ret == -3:
                        break
                if ret == -1 or ret == -2:
                    break
            if ret == -1:
                break    

    print "Loss = {:.3f}".format(total_loss/num) # number of training/testing set
    print "DeltaLoss = {:.3f}".format(total_deltaloss/num)
    print "IOU = {:.3f}".format(iou/num)
    print "Acc = {:.3f}".format(acc/num)
    print "Velocity Diff = {:.3f}".format(vel_diff/num)
    print "-"*80

