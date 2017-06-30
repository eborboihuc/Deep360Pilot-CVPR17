#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
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
        if Agent.restore_path and os.path.isdir(Agent.save_path):
            saver.restore(sess, Agent.restore_path)
            print "Your model restored!!!"
            test_all(sess, Agent, is_train=True)
            test_all(sess, Agent, is_train=False)
        else:
            print "Model ({}) Not Found!!!".format(Agent.restore_path)


def test_all(sess, Agent, is_train):
    """ Testing whole batches of training/testing set """
    num = Agent.train_num if is_train else Agent.test_num
    path = Agent.train_path if is_train else Agent.test_path
    
    total_loss = 0.0
    total_deltaloss = 0.0
    iou = 0.0
    acc = 0.0
    vel_diff = 0.0
    pred_init_value = np.ones([Agent.batch_size, Agent.n_output])/2
    dropPr = Agent.trainDropPr if is_train else Agent.testDropPr
    
    # Init MVD
    MVD = MeanVelocityDiff(W=Agent.W)

    for num_batch in range(1,num+1):
        
        # load test_data
        batch_data, batch_label, batch_y_loc, batch_box_center, batch_inclusion, batch_hof, batch_img, box, gt = load_batch_data(Agent, path, num_batch, True)
        
        # NOTE: Change this after feature changed
        gt = gt[:,:,:2]

        [_, loss, deltaloss, pred_out, alpha_out] = sess.run([Agent.opt, Agent.cost, Agent.delta, Agent.pred, Agent.alphas], 
                feed_dict={Agent.obj_app: batch_data, Agent.y: batch_label, Agent.y_loc: batch_y_loc, 
                Agent.box_center: batch_box_center[:,:,:,:Agent.n_output], Agent.inclusion: batch_inclusion, 
                Agent.hof: batch_hof, Agent.keep_prob:1.0-dropPr, 
                Agent.pred_init: pred_init_value, Agent._phase: Agent.bool_two_phase})
        
        total_loss      += loss/Agent.n_frames #(batch_size*Agent.n_frames)
        total_deltaloss += deltaloss/Agent.n_frames

        pred_out[:,:,0] = (pred_out[:,:,0]*Agent.W).astype(int)
        pred_out[:,:,1] = (pred_out[:,:,1]*Agent.H).astype(int)
        
        # NOTE: Use same scale
        ac = float(np.sum(np.logical_and(batch_label, alpha_out))) / (Agent.batch_size * Agent.n_frames)
        iu = score(Agent, pred_out, gt)
        vd = MVD.batch_vel_diff(pred_out) * 0.1875 / (Agent.n_frames)
        acc += ac
        iou += iu
        vel_diff += vd
        print "Acc: {:.3f}, IoU: {:.3f}, Vel_diff: {:.3f}".format(ac, iu, vd)
        print "Corr: {}".format(np.sum(np.logical_and(batch_label, alpha_out)))

        if Agent._show:
            ret = 0
            for count, name in enumerate(batch_img):
                vid_name = name[:13]
                nimages = (int(name[14:])-1)*Agent.n_frames

                if Agent._save_pred:
                    print pred_out.shape
                    out_path = Agent.root_path + ('output/test/' if 'test' in path else 'output/train/') +  name + '.npy'
                    print "Save prediction of vid {} to {}".format(name, out_path)
                    np.save(out_path, pred_out[count].astype(int))

                for nimage in xrange(Agent.n_frames):
                    vidname = os.path.join(vid_name, str(nimages + nimage + 1).zfill(6))
                    if Agent._save_img and not os.path.isdir(save_path + vid_name):
                        print 'Make dir at ' + save_path + ('test/' if 'test' in path else 'train/') + vid_name
                        os.makedirs(save_path + ('test/' if 'test' in path else 'train/') + vid_name) # mkdir recursively
                    
                    if Agent._show:
                        print 
                        print ("num_batch: {}, video: {}, vid: {}, count: {}, nimage: {}").format(num_batch, name, vid_name, count, nimage)
                        ret = visual_gaze(Agent, vidname, gt[count, nimage, :], pred_out[count, nimage,:], alpha_out[count, nimage,:], box[count, nimage,:,:])
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


