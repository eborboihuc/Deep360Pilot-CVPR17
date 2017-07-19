#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from glob import glob
from os.path import join
rnn = tf.nn.rnn_cell #from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from ops import tf_viewangle_360_constraint, tf_360_shortest_dist, tf_dist_360, tf_mov_coef
from loss import tf_l2_loss, tf_l2_loss_360, total_cost, residual_loss, accurate_loss, smooth_loss, policy_loss, get_reward

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
        self.restore_path       = join(flag.root_path, flag.model_path) if flag.model_path else None
    
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
        self.l2_beta            = 1e-2
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
        
        # Initial variables
        self.init_vars()

        # Selector and Regressor RNN 
        self.deep360pilot_rnn()

        # Define optimizer
        self.opt = self.optimizer(name=self.opt_method).minimize(self.cost, global_step=self.global_step)

        # show on the tensorborad
        loss_summary = tf.summary.scalar("Loss", self.cost)
        self.merged = tf.summary.merge_all()
        

    def init_vars(self):
        """ Declare graph inputs and variables used in RNN() down below """
        
        # Steps
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        
        # tf Graph input
        self.obj_app            = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, self.n_input])
        self.oracle_actions     = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection])
        self.oracle_viewangle   = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_output])
        self.box_center         = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, self.n_output])
        self.hof                = tf.placeholder("float", [self.batch_size, self.n_frames, self.n_detection, self.n_bin_size])
        self.keep_prob          = tf.placeholder("float")
        self.init_viewangle     = tf.placeholder("float", [self.batch_size, self.n_output])
        self._phase             = tf.placeholder("bool", name="trainphase")   
        
        # Initial prediction
        self.prev_viewangle_init = tf.convert_to_tensor(self.init_viewangle)
        
        # Define weights and biases
        self.weights = {
            'em_att': tf.get_variable('em_att', shape=[self.n_input, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_w': tf.get_variable('att_w', shape=[self.n_hidden, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_wa': tf.get_variable('att_wa', shape=[self.n_hidden, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_ua': tf.get_variable('att_ua', shape=[self.n_hidden+self.n_output+self.n_bin_size, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'gaze': tf.get_variable('gaze', shape=[self.n_hidden, self.n_detection], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'onebox': tf.get_variable('onebox', shape=[self.n_hidden+self.n_output+self.n_bin_size, self.n_onebox], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
        }

        self.biases = {
            'em_att': tf.get_variable('em_att_b', shape=[1, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'att_ba': tf.get_variable('att_ba', shape=[1, self.n_hidden], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            'gaze' : tf.get_variable('gaze_b', shape=[1, self.n_detection], initializer=xavier_initializer(), regularizer=l2_regularizer(self.l2_beta)),
            
        }
        
        init = tf.constant_initializer([[0.0, 0.0]]*(self.n_bin_size + self.n_output))
        self.regressor_w = tf.get_variable('pred', shape=[self.n_output+self.n_bin_size, self.n_output], initializer=init, regularizer=l2_regularizer(self.l2_beta))
        
        init_b = tf.constant_initializer([[0.0, 0.0]]*self.batch_size)
        self.regressor_b = tf.get_variable('pred_b', shape=[self.batch_size, self.n_output], initializer=init_b, regularizer=l2_regularizer(self.l2_beta))
        
        with tf.variable_scope('Selector'):
            # state_is_tuple == true : output, new_state is the return of the rnn_cell. Where new_state is (cell state, output) actually.
            self.rnn_cell_s = rnn.LSTMCell(self.n_hidden, 
                    initializer=xavier_initializer(), 
                    use_peepholes=True)

            self.rnn_cell_s = rnn.DropoutWrapper(self.rnn_cell_s, 
                    input_keep_prob=self.keep_prob, 
                    output_keep_prob=self.keep_prob)
            
            self.rnn_state_s = self.rnn_cell_s.zero_state(self.batch_size, 'float32')
        
        with tf.variable_scope('Regressor'):
            self.rnn_cell_r = rnn.LSTMCell(self.n_output + self.n_bin_size, 
                    initializer=xavier_initializer(), 
                    use_peepholes=True)
            
            self.rnn_cell_r = rnn.DropoutWrapper(self.rnn_cell_r, 
                    input_keep_prob=self.keep_prob, 
                    output_keep_prob=self.keep_prob)
            
            self.rnn_state_r = self.rnn_cell_r.zero_state(self.batch_size, 'float32')


    def optimizer(self, *args, **kwargs):
        """ Define Optimizer to use """

        # Learning rate decays every 50 epochs
        if kwargs['name'] in ['Adam', 'Adadelta']:
            self.lr = tf.constant(self.init_learning_rate)
        else:
            self.lr = tf.train.exponential_decay(self.init_learning_rate, self.global_step,
                                            5*(self.display_step*self.train_num), 0.9, staircase=True)

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


    def deep360pilot_rnn(self):
        """ Define Selector RNN and Regressor RNN here """

        prev_viewangle  = self.prev_viewangle_init
        prev_velocity   = tf.zeros([self.batch_size, self.n_output])
        
        sal_box_prob_array    = tf.TensorArray(dtype=tf.float32, size=self.n_frames)
        pred_array      = tf.TensorArray(dtype=tf.float32, size=self.n_frames)
        
        def recurrent_body(i, sal_box_prob_array, pred_array, prev_viewangle, prev_velocity, prev_rnn_state_s, prev_rnn_state_r, cost, deltaloss):
            #####################
            #       Input       #
            #####################
            O_t = self.obj_app[:, i, :, :] # (n_batch, n_det, n_input)
            M_t = self.hof[:, i, :, :] # (n_batch, n_det, n_bin_size)
            P_t = self.box_center[:, i, :, :] # (n_batch, n_det, n_output)

            # Location encode
            prev_loc = tf.expand_dims(prev_viewangle, 1) # (n_batch, 1, n_output)
            relative_position = tf_dist_360(P_t, prev_loc, 2) # (n_batch, n_det, n_output)

            # Object embedded
            O_t = tf.reshape(O_t, [-1, self.n_input]) # (n_batch * n_det, n_input)
            O_t = tf.matmul(O_t, self.weights['em_att']) + self.biases['em_att'] # (n_batch * n_det, n_hidden)
            O_t = tf.reshape(O_t, [self.batch_size, self.n_detection, self.n_hidden]) # (n_batch, n_det, n_input)
            
            # Object level Feature
            V_t = tf.concat(axis=2, values=[O_t, relative_position, M_t]) # (n_batch, n_det, n_hidden+n_output+n_bin_size)

            #####################
            #   Selector RNN    #
            #####################
            
            obj_observation = tf.matmul(tf.reshape(V_t, [-1, self.n_hidden+self.n_output+self.n_bin_size]), 
                    self.weights['att_ua']) + self.biases['att_ba'] # (n_batch * n_det, n_hidden)
            obj_observation = tf.reshape(obj_observation, [self.batch_size, self.n_detection, self.n_hidden])
            e = tf.tanh(tf.expand_dims(tf.matmul(prev_rnn_state_s[1], self.weights['att_wa']), 1) + obj_observation) # (n_batch, n_det, n_hidden)
            
            e = tf.matmul(tf.reshape(e, [-1, self.n_hidden]), self.weights['att_w']) # (n_batch * n_det, n_hidden)
            e = tf.reshape(e, [self.batch_size, self.n_detection, self.n_hidden]) # (n_batch, n_det, n_hidden)
            e = tf.exp(tf.reduce_sum(e, 2)) # (n_batch, n_det)
            
            # Eliminate the empty box
            denomin = tf.reduce_sum(e, 1) # (n_batch, )
            denomin = denomin + tf.to_float(tf.equal(denomin, 0)) # avoid nan

            # Soft attention : alphas
            alphas = tf.div(e, tf.expand_dims(denomin, 1)) # (n_batch, n_det)
            
            attention_list = tf.multiply(tf.expand_dims(alphas, 2), V_t) # (n_batch, n_det, n_hidden+self.n_output+self.n_bin_size)
            attention = tf.matmul(tf.reshape(attention_list, [-1, self.n_hidden+self.n_output+self.n_bin_size]), self.weights['onebox']) # (n_batch * n_det, n_onebox)
            attention = tf.reshape(attention, [self.batch_size, self.n_hidden]) # (n_batch, n_hidden)

            # Selector RNN
            with tf.variable_scope('Selector'):
                rnn_output_s, rnn_state_s = self.rnn_cell_s(attention, prev_rnn_state_s)
            
            # Gaze prediction
            sal_box_prob = tf.matmul(rnn_output_s, self.weights['gaze']) + self.biases['gaze'] # (n_batch, n_det)

            # Gaze location
            amax = tf.argmax(tf.nn.log_softmax(sal_box_prob), 1) # n_batch
            amaxDense = tf.one_hot(amax, self.n_detection, 1.0, 0.0, 1) # (n_batch, n_det)
            amaxDenseBatch = tf.expand_dims(amaxDense, 2) # (n_batch, n_det, 1)
            cur_select_angle = tf.reduce_sum(tf.multiply(P_t, amaxDenseBatch), 1) # (n_batch, n_output)
            cur_object_motion = tf.reduce_sum(tf.multiply(M_t, amaxDenseBatch), 1) # (n_batch, n_bin_size)
            
            # Zero box location handle: Add prev_viewangle at where predloc is zero
            zero_box = tf.reduce_sum(cur_select_angle, 1) # n_batch
            zero_box_mask = tf.expand_dims(tf.cast(tf.equal(zero_box, 0.0), tf.float32), 1) # (n_batch, 1)
            cur_select_angle = cur_select_angle + tf.multiply(prev_viewangle, zero_box_mask) # (n_batch, n_output)

            #####################
            #   Regressor RNN   #
            #####################

            # Regressor RNN
            displacement = tf_dist_360(cur_select_angle, prev_viewangle, 1) # (n_batch, n_output)
            disp_inc = tf_360_shortest_dist(prev_velocity + displacement)
            mov_coef = tf_mov_coef(prev_velocity, displacement)
            reg_input = tf.concat(axis=1, values=[disp_inc, cur_object_motion]) # (n_batch, n_output+n_bin_size)
            with tf.variable_scope('Regressor'):
                rnn_output_r, rnn_state_r = self.rnn_cell_r(reg_input, prev_rnn_state_r)
            
            residual = mov_coef * (tf.matmul(rnn_output_r, self.regressor_w) + self.regressor_b) # (n_batch, 2)
            
            # Current viewangle generation
            cur_viewangle = prev_viewangle + displacement + residual
            cur_viewangle = tf_viewangle_360_constraint(cur_viewangle)

            #####################
            #   Viewpoint traj  #
            #####################
            
            # Send out pred and one_hot
            pred_array = pred_array.write(i, cur_viewangle)
            sal_box_prob_array = sal_box_prob_array.write(i, amaxDense)
            #sal_box_prob_array = sal_box_prob_array.write(i, sal_box_prob)

            #####################
            #        Cost       #
            #####################
            # Get velocity
            current_velocity = tf_dist_360(cur_viewangle, prev_viewangle, 1) # (n_batch, n_output)
            vel_diff = tf_l2_loss(current_velocity, prev_velocity, 1) # (n_batch, n_output)
            angle_diff = tf_l2_loss_360(cur_viewangle, prev_viewangle, 1) # (n_batch, n_output)
            
            # Pred to oracle_viewangle
            l2loss = accurate_loss(cur_viewangle, self.oracle_viewangle[:, i, :], 1) # (n_batch, n_output) -> (n_batch, 1)

            # Pred to oracle actions
            policyloss = policy_loss(sal_box_prob, self.oracle_actions[:, i, :], get_reward(l2loss))

            # Classification phase (True) or Regression phase (False)
            deltaloss = smooth_loss(self._phase, self.classify_lmbda, self.regress_lmbda, angle_diff, vel_diff) # (n_batch, n_output) -> (n_batch, 1)
            
            # Smooth prediction
            residualloss = residual_loss(residual)

            # Cost in Classification phase (True) or Regression phase (False)
            cost += total_cost(self._phase, policyloss, l2loss, deltaloss, residualloss)

            return i+1, sal_box_prob_array, pred_array, cur_viewangle, current_velocity, rnn_state_s, rnn_state_r, cost, tf.reduce_mean(vel_diff)


        # While loop over self.n_frames
        _, sal_box_prob, viewangle, cur_pred, cur_vel, rnn_state_s, rnn_state_r, self.cost, self.delta = tf.while_loop(
                cond = lambda i, *_: i < self.n_frames,
                body = recurrent_body, # i, sal_box_prob, pred, cur_vel, rnn_output_s, rnn_state_s, rnn_output_r, rnn_state_r, total_loss, delta_loss
                loop_vars = (
                    tf.constant(0, tf.int32), 
                    sal_box_prob_array, 
                    pred_array,
                    prev_viewangle, 
                    prev_velocity, 
                    self.rnn_state_s, 
                    self.rnn_state_r, 
                    tf.constant(0, tf.float32),
                    tf.constant(0, tf.float32)
                    )
        )
        
        #self.sal_box_prob = tf.transpose(sal_box_prob.stack(), [1, 0, 2]) # sal_box_prob
        self.sal_box_prob = tf.transpose(sal_box_prob.stack(), [1, 0, 2]) # amaxDense
        self.viewangle = tf.transpose(viewangle.stack(), [1, 0, 2])


