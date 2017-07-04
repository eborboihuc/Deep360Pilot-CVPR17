#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb
import tensorflow as tf

rnn = tf.nn.rnn_cell #from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from glob import glob
from os.path import join
from ops import *

class Deep360Pilot(object):
    
    def __init__(self, flag):

        # Arguments
        self.two_phase          = flag.phase
        self.opt_method         = flag.opt_method
        self.bool_two_phase     = (flag.phase == 'classify')
        
        # Path
        self.domain             = flag.domain
        self.root_path          = flag.root_path
        self.data_path          = flag.data_path
        self.img_path           = join(flag.data_path, 'frame_{}'.format(flag.domain))
        self.train_path         = join(flag.data_path, '{}_{}boxes'.format(flag.domain, flag.boxnum), 'train')
        self.test_path          = join(flag.data_path, '{}_{}boxes'.format(flag.domain, flag.boxnum), 'test')
        self.save_path          = join(flag.root_path, 'checkpoint', 
                                            '{}_{}boxes_lam{}'.format(flag.domain, flag.boxnum, flag.lam))
        self.restore_path       = join(flag.root_path, 'checkpoint', flag.model_path) if flag.model_path else None
    
        # Batch Number
        self.train_num          = len(glob(join(self.train_path, 'roisavg/*.npy'))) 
        self.test_num           = len(glob(join(self.test_path, 'roisavg/*.npy')))
        assert self.train_num   > 0 or flag.mode not in ['train', 'test'], "Found 0 files in {}".format(self.train_path)
        assert self.test_num    > 0 or flag.mode not in ['train', 'test'], "Found 0 files in {}".format(self.test_path)

        # Flag
        self.Debug              = flag.debug
        self._save_img          = flag.save
        self._show              = flag.mode == 'vid'
        self._save_pred         = flag.mode == 'pred'

        # Parameters
        self.l2_beta            = 1.0
        self.testDropPr         = 0.0
        self.trainDropPr        = 0.5
        self.classify_lmbda     = 1.0
        self.regress_lmbda      = flag.lam
        self.init_learning_rate = 1e-5

        # Network Parameters
        self.W                  = 1920.0
        self.H                  = 960.0
        self.n_epochs           = 400
        self.batch_size         = 10
        self.display_step       = 10
        self.n_input            = 512 # Conv5
        self.n_hidden           = 256 # Num of features in hidden layer
        self.n_output           = 2 # u and v
        self.n_frames           = 50
        self.n_bin_size         = 12
        self.n_detection        = flag.boxnum
        self.n_onebox           = self.n_hidden/self.n_detection

        # Best loader
        self.Best_score         = { 'epoch': 0, 
                                    'loss' : -1.0, 
                                    'smooth_loss': -1.0, 
                                    'lr': self.init_learning_rate, 
                                    'iou': -1.0, 
                                    'acc': -1.0, 
                                    'vel_diff':-1.0
                                    }

        # GPU memory usage fraction
        self.gpuUsage           = 0.5
        self.sess_config        = tf.ConfigProto()
        self.sess_config.allow_soft_placement = True
        self.sess_config.gpu_options.allow_growth = (self.gpuUsage!=1.0)
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = self.gpuUsage
        
        # Initial model
        self.build_model()


    def build_model(self):
        """ Build up a model based on RNN(~) down below """
        
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        
        # tf Graph input
        self.obj_app    = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, self.n_input])
        self.y          = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection])
        self.y_loc      = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_output+1])
        self.box_center = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, self.n_output])
        self.inclusion  = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, 3])
        self.hof        = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, self.n_bin_size])
        self.keep_prob  = tf.placeholder("float")
        self.pred_init  = tf.placeholder("float", [self.batch_size, self.n_output])
        self._phase     = tf.placeholder("bool", name="trainphase")   
        
        # Initial prediction
        self.prev_pred_init = tf.convert_to_tensor(self.pred_init)
        
        # Initial variables
        self.init_vars()

        # RNN 
        self.RNN()

        # Define optimizer
        self.opt = self.optimizer(name=self.opt_method).minimize(self.cost, global_step=self.global_step)

        # show on the tensorborad
        loss_summary = tf.summary.scalar("Loss", self.cost)
        self.merged = tf.summary.merge_all()
        

    def init_vars(self):
        
        # Define self.weights
        self.weights = {
            'em_att': tf.get_variable('em_att', shape=[self.n_input, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_w': tf.get_variable('att_w', shape=[self.n_hidden+4, self.n_hidden+2], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_wa': tf.get_variable('att_wa', shape=[self.n_hidden, self.n_hidden+2], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_ua': tf.get_variable('att_ua', shape=[self.n_hidden+2, self.n_hidden+2], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'gaze': tf.get_variable('gaze', shape=[self.n_hidden, self.n_detection], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'onebox': tf.get_variable('onebox', shape=[self.n_hidden+2, self.n_onebox], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
        }
        self.biases = {
            'em_att': tf.get_variable('em_att_b', shape=[1, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_ba': tf.get_variable('att_ba', shape=[1, self.n_hidden+2], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'gaze' : tf.get_variable('gaze_b', shape=[1, self.n_detection], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            
        }
        self.regressor_w = {    
            'rnn_velo': tf.Variable(tf.convert_to_tensor([[1.0, 0.0],[0.0, 1.0]])),
            'rnn_h': tf.Variable(tf.convert_to_tensor([[0.0, 0.0],[0.0, 0.0]])),
            'pred': tf.Variable(tf.convert_to_tensor([[1.0, 0.0],[0.0, 1.0]])),
            
            'flow': tf.Variable(tf.convert_to_tensor([[1.0, 0.0],[0.0, 1.0]])),
        }
        self.regressor_b = {
            'pred': tf.Variable(tf.zeros([self.batch_size, 2])),
        }      
        
        self.rnn_cell = rnn.LSTMCell(self.n_hidden, 
                initializer=xavier_initializer(), 
                use_peepholes=True)
        
        self.rnn_cell = rnn.DropoutWrapper(self.rnn_cell, 
                input_keep_prob=self.keep_prob, 
                output_keep_prob=self.keep_prob)
        
        self.rnn_state = self.rnn_cell.zero_state(self.batch_size, 'float32')


    def optimizer(self, *args, **kwargs):
        """ Define Optimizer to use """

        # Learning rate decays every 50 epochs
        if kwargs['name'] in ['Adam', 'Adadelta']:
            self.lr = tf.constant(self.init_learning_rate)
        else:
            self.lr = tf.train.exponential_decay(self.init_learning_rate, self.global_step,
                                            5*(11*self.train_num + self.test_num), 0.9, staircase=True)

        # Change optimizer base on name
        if kwargs['name'] == 'Adam':
            return tf.train.AdamOptimizer(learning_rate=self.lr, *args, **kwargs)
        elif kwargs['name'] == 'Adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=self.lr, *args, **kwargs)
        elif kwargs['name'] == 'RMSProp':
            return tf.train.RMSPropOptimizer(momentum=0.1, learning_rate=self.lr, *args, **kwargs)
        elif kwargs['name'] == 'Momentum':
            return tf.train.MomentumOptimizer(momentum=0.1, use_nesterov=True, learning_rate=self.lr, *args, **kwargs)
        else:
            raise IOError("Optimizer {} not Found.".format(kwargs['name']))


    def RNN(self):
        """ Define Selector RNN and Regressor RNN here """

        prev_pred       = self.prev_pred_init
        reg_hidden      = tf.zeros([self.batch_size, self.n_output])
        prev_velocity   = tf.zeros([self.batch_size, self.n_output])
        h_prev          = tf.ones([self.batch_size, self.n_hidden]) / 2.0
        
        predpr_array    = tf.TensorArray(dtype=tf.float32, size=self.n_frames)
        pred_array      = tf.TensorArray(dtype=tf.float32, size=self.n_frames)
        
        
        def recurrent_body(i, predpr_array, pred_array, prev_pred, prev_velocity, rnn_output, rnn_state, reg_hidden, cost, deltaloss):

            # Input
            cur_box_center = tf.transpose(self.box_center[:, i, :, :],[1, 0, 2]) # n_det x b x self.n_output
            X = tf.transpose(self.obj_app[:, i, :, :], [1, 0, 2])  # permute n_dets and self.batch_size (n_det x b x h)
            #X = tf.transpose(self.obj_app[:,i,:,:] * tf.expand_dims(self.inclusion[:,i,:,0], 2), [1, 0, 2])  # permute n_dets and self.batch_size (n_det x b x h)
            
            # Object embedded
            X = tf.reshape(X, [-1, self.n_input]) # (n_dets*self.batch_size, self.n_input)
            X = tf.matmul(X, self.weights['em_att']) + self.biases['em_att'] # (n_det x b) x h
            X = tf.reshape(X, [self.n_detection, self.batch_size, self.n_hidden]) # n_det x b x h
            
            prev_loc = tf.tile(tf.expand_dims(prev_pred, 0), [self.n_detection, 1, 1]) # n_det x b x self.n_output
            X = tf.concat(2, [X, tf_dist_360(prev_loc, cur_box_center, 2)]) # n_det x b x h+2

            # Object attention
            brcst_w = tf.tile(tf.expand_dims(self.weights['att_w'], 0), [self.n_detection, 1, 1]) # n_det x h+4 x h+2
            image_part = tf.batch_matmul(X, tf.tile(tf.expand_dims(self.weights['att_ua'], 0), [self.n_detection, 1, 1])) + self.biases['att_ba'] # n_det x b x h+2
            e = tf.tanh(tf.matmul(rnn_output, self.weights['att_wa']) + image_part) # n_det x b x h+2
            
            # TODO:motion attention
            inclusion_emb = tf.expand_dims(tf.transpose(self.inclusion[:, i, :, 0], [1, 0]), 2)
            inclusion_emb = tf.tile(inclusion_emb, [1, 1, 2]) # n_det x b x 2
            e = tf.concat(2, [e, inclusion_emb]) # n_det x b x h+4
            e = tf.exp(tf.reduce_sum(tf.batch_matmul(e, brcst_w), 2)) # n_det x b
            
            # Eliminate the empty box
            denomin = tf.reduce_sum(e,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0)) # avoid nan

            # Soft attention : alphas
            alphas = tf.div(e, tf.tile(tf.expand_dims(denomin, 0), [self.n_detection, 1])) # n_det x b
            
            attention_list = tf.mul(tf.tile(tf.expand_dims(alphas, 2),[1, 1, self.n_hidden+2]), X) # n_det x b x h+2
            attention = tf.batch_matmul(attention_list, tf.tile(tf.expand_dims(self.weights['onebox'], 0), [self.n_detection, 1, 1])) # n_det x b x h
            attention = tf.reshape(tf.transpose(attention, [1, 0, 2]), [self.batch_size, -1])
            
            # Selector RNN
            rnn_output, rnn_state = self.rnn_cell(attention, rnn_state)
            
            # Gaze prediction
            predpr = tf.matmul(rnn_output, self.weights['gaze']) + self.biases['gaze'] # b x self.n_detection

            # Gaze location
            amax = tf.argmax(tf.nn.log_softmax(predpr), 1) # b
            amaxDense = tf.one_hot(amax, self.n_detection, 1.0, 0.0, 0) # self.n_detection x b
            amaxDenseBatch = tf.tile(tf.expand_dims(amaxDense, 2), [1, 1, self.n_output]) # self.n_detection x b x self.n_output
            predloc = tf.reduce_sum(tf.mul(cur_box_center, amaxDenseBatch), 0) # b x self.n_output
            
            # Zero box location handle: Add prev_pred at where predloc is zerow
            zero_box = tf.reduce_sum(predloc, 1) # b x 1
            zero_box_mask = tf.expand_dims(tf.cast(tf.equal(zero_box, 0.0), tf.float32),1) # b x 1
            predloc = predloc + tf.mul(prev_pred, zero_box_mask) # b x self.n_output

            # Gaze regression RNN
            pred_diff = tf_dist_360_classify(prev_pred, predloc, 1) # b x self.n_output
            reg_output = tf.matmul(pred_diff, self.regressor_w['rnn_velo']) + tf.matmul(reg_hidden, self.regressor_w['rnn_h'])
            pred = prev_pred + tf.matmul(reg_output, self.regressor_w['pred']) + self.regressor_b['pred']
            
            # Get velocity
            current_velocity = tf_dist_360_classify(prev_pred, pred, 1) # b x self.n_output

            # 360 constraint, x[x > 1.0] = x[x > 1.0] - 1.0, x[x < 0.0] = x[x < 0.0] + 1.0, 0.0 < y < 1.0
            xpart, ypart = tf.split(1, 2, pred)
            _greater = tf.greater(xpart, 1.0) # b x 1
            _less = tf.less(xpart, 0.0) # b x 1
            xpart = xpart - tf.to_float(_greater)
            xpart = xpart + tf.to_float(_less)
            ypart = tf.clip_by_value(ypart, 0.0, 1.0)
            pred = tf.concat(1,[xpart, ypart]) # b x self.n_output

            # Send out pred and one_hot
            pred_array = pred_array.write(i, pred)
            #predpr_array = predpr_array.write(i, predpr)
            predpr_array = predpr_array.write(i, amaxDense)

            # Pred to one hot
            loss = tf.nn.softmax_cross_entropy_with_logits(predpr, self.y[:, i, :])
            loss = tf.reduce_mean(loss)

            # Pred to y_loc
            l2diff = tf_l2_dist_360(pred, self.y_loc[:, i, :2], 1) # b x n_output -> b x 1
            l2loss = tf.reduce_mean(l2diff)

            """ Regulized by current prediction and previous ones
                Deminish the gap between left most and right most by function : 
                { max(X,0.5)-X + min(X,0.5)*2*X }=={ X[X < 0.5] = 1 - X[X< 0.5]] }
            """
            # Classification phase (True) or Regression phase (False)
            reg_l2diff = tf.cond(self._phase, \
                    lambda: tf_l2_dist_360(pred, prev_pred, 1), \
                    lambda: tf_l2_dist_360(current_velocity, prev_velocity, 1)) # b x n_output -> b x 1
            
            deltaloss = tf.reduce_mean(reg_l2diff)

            # Cost in Classification phase (True) or Regression phase (False)
            cost += tf.cond(self._phase, \
                    lambda: loss + self.classify_lmbda * deltaloss, \
                    lambda: l2loss + self.regress_lmbda * deltaloss)
            
            return i+1, predpr_array, pred_array, pred, current_velocity, rnn_output, rnn_state, reg_hidden, cost, deltaloss


        # While loop over self.n_frames
        _, self.predpr, self.pred, cur_pred, cur_vel, rnn_output, rnn_state, reg_state, self.cost, self.delta = tf.while_loop(
                cond = lambda i, *_: i < self.n_frames,
                body = recurrent_body, # i, predpr, pred, cur_vel, rnn_output, rnn_state, reg_hidden, total_loss, delta_loss
                loop_vars = (
                    tf.constant(0, tf.int32), 
                    predpr_array, 
                    pred_array,
                    prev_pred, 
                    prev_velocity, 
                    self.rnn_state[1], 
                    self.rnn_state, 
                    reg_hidden,
                    tf.constant(0, tf.float32),
                    tf.constant(0, tf.float32)
                    )
        )
        
        #self.alphas = tf.transpose(self.predpr.pack(), [1, 0, 2]) # predpr
        self.alphas = tf.transpose(self.predpr.pack(), [2, 0, 1]) # amaxDense
        self.pred = tf.transpose(self.pred.pack(), [1, 0, 2])


