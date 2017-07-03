#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hou-Ning Hu

import numpy as np

class MeanVelocityDiff(object):
    """ Class for Mean Overlap metric via a simplified IoU (Only for relative comparison)
        
        Init:
            W               - Panoramic image width, int, float, in pixel format
            FPS             - FPS in video, float, in degree format
        
        Input: 
            Traj            - An array with shape (n_frames, 2), each frame has viewpoint (x, y), int, float, in pixel format

        Functions:
            vel_diff        - Calculate IOU in degree format
            batch_vel_diff  - Calculate IOU in pixel format

        Output:
            mean_vel_diff   - Mean Velocity Difference
    """

    def __init__(self, W=1920.0, FPS=1.0):
        self.W = W
        self.FPS = FPS

    def batch_vel_diff(self, batch_trajs):
        n_trajs = batch_trajs.shape[0]
        acc_vel_diff = 0.0
        for i in xrange(n_trajs):
            acc_vel_diff += self.vel_diff(batch_trajs[i])

        return acc_vel_diff / n_trajs


    def vel_diff(self, traj):
        """ Traj should be an array with shape (n_frames, 2)
            which consists of prediction (x, y) of each frame.
        """
        n_frames = traj.shape[0]

        acc_velocity_diff = 0.0
        last_velocity = 0.0

        for frame in xrange(1, n_frames): # start computing velocity from 2nd prediction
            x_diff = traj[frame][0] - traj[frame-1][0]
            y_diff = traj[frame][1] - traj[frame-1][1]

            # handle x boundary
            if abs(x_diff) >= self.W/2:
                if x_diff >= 0:
                    x_diff -= self.W
                else:
                    x_diff += self.W

            velocity = np.array([x_diff, y_diff]) / self.FPS

            if not frame == 1: # start computing velocity difference from 3rd prediction
                acc_velocity_diff += np.sqrt(np.sum(np.square(velocity - last_velocity)))

            last_velocity = velocity

        return acc_velocity_diff

if __name__ == '__main__':
    MVD = MeanVelocityDiff()
    
    assert MVD.vel_diff(np.array([[1, 1], [1, 1], [1, 1]])) == 0
    assert MVD.vel_diff(np.array([[1, 2], [1, 3], [1, 4]])) == 0
    assert MVD.vel_diff(np.array([[1919, 2], [1920, 2], [1, 2]])) == 0
    assert MVD.vel_diff(np.array([[1, 2], [1920, 2], [1919, 2]])) == 0
    assert MVD.vel_diff(np.array([[1, 2], [1, 4], [1, 2]])) == (2 + 2) / MVD.FPS
    
    print "Success!"
